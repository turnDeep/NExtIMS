"""Focused invariants and restart tests for EXP-V019."""

import json
from pathlib import Path
from contextlib import closing
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch_geometric.data import Batch

import run_generalization_algorithms as NEW


class AlgorithmTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_fusion_and_model_unchanged(self):
        args = NEW.parser().parse_args([])
        config = NEW.BASE.model_config(args)
        self.assertEqual(config["spectral_feedback_fusion_weight"], .3)
        self.assertEqual(config["refinement_steps"], 4)
        self.assertEqual(config["node_dim"], 34)
        self.assertTrue(config["use_act"] and config["spectral_residual_enabled"])

    def test_depth_distribution_and_restoration(self):
        class Model:
            refinement_steps = 4
        model = Model()
        generator = torch.Generator().manual_seed(42)
        counts = {2: 0, 3: 0, 4: 0}
        for _ in range(1000):
            with NEW.training_depth(model, "random_depth", generator) as depth:
                counts[depth] += 1
            self.assertEqual(model.refinement_steps, 4)
        self.assertTrue(150 < counts[2] < 250 and 450 < counts[4] < 550)
        with self.assertRaises(RuntimeError):
            with NEW.training_depth(model, "random_depth", generator):
                raise RuntimeError("probe")
        self.assertEqual(model.refinement_steps, 4)

    def test_sam_restores_exact_weights_on_error(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.RAdam(model.parameters(), lr=5e-5)
        args = NEW.parser().parse_args([])
        raw = {key: value.detach().clone() for key, value in model.state_dict().items()}
        calls = 0
        def loss_function(model, batch, args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("second-forward failure")
            prediction = model(torch.ones(2, 2))
            loss = prediction.square().mean()
            return loss, prediction, {}, {}
        with self.assertRaisesRegex(RuntimeError, "second-forward"):
            NEW.sam_update(model, optimizer, None, args, loss_function)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, raw[key], rtol=0, atol=0)
        self.assertEqual(len(optimizer.state), 0)

    def test_sam_rho_zero_is_regular_update(self):
        NEW.BASE.set_seed(42)
        model = nn.Sequential(nn.Linear(3, 5), nn.Dropout(.2), nn.Linear(5, 2))
        other = nn.Sequential(nn.Linear(3, 5), nn.Dropout(.2), nn.Linear(5, 2))
        other.load_state_dict(model.state_dict())
        args = NEW.parser().parse_args(["--sam-rho", "0"])
        optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr)
        other_optimizer = torch.optim.RAdam(other.parameters(), lr=args.lr)
        rng = NEW.BASE.cpu_rng_state()
        def loss_function(model, batch, args):
            prediction = model(torch.ones(2, 3))
            return prediction.square().mean(), prediction, {}, {}
        NEW.sam_update(model, optimizer, None, args, loss_function)
        after = torch.get_rng_state().clone()
        NEW.BASE.restore_rng(rng)
        loss_function(other, None, args)[0].backward()
        torch.nn.utils.clip_grad_norm_(other.parameters(), args.grad_clip)
        other_optimizer.step()
        self.assertTrue(torch.equal(after, torch.get_rng_state()))
        for first, second in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_teacher_cannot_contain_validation_or_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "teacher.pth"
            torch.save({"ids": torch.tensor([1, 2]), "spectra": torch.rand(2, 1000)}, path)
            teacher = NEW.TrainTeacher(path, [1, 2])
            self.assertEqual(teacher.get(torch.tensor([2, 1]), "cpu").shape, (2, 1000))
            with self.assertRaises(KeyError):
                teacher.get(torch.tensor([3]), "cpu")
            with self.assertRaisesRegex(ValueError, "training IDs"):
                NEW.TrainTeacher(path, [1, 3])
            torch.save({"ids": torch.tensor([1, 1]), "spectra": torch.rand(2, 1000)}, path)
            with self.assertRaises(ValueError):
                NEW.TrainTeacher(path, [1, 2])

    def test_distillation_does_not_modify_real_target(self):
        args = NEW.parser().parse_args(["--hidden-dim", "16", "--num-layers", "1", "--num-heads", "4"])
        model = NEW.COMMON.QCGN2oEI_Minimal(**NEW.BASE.model_config(args))
        graphs = [NEW.BASE.make_graph("CCO"), NEW.BASE.make_graph("CCC")]
        for index, graph in enumerate(graphs):
            graph.y = torch.rand(1, 1000)
            graph.sample_id = torch.tensor([index + 1])
        batch = Batch.from_data_list(graphs)
        target = batch.y.clone()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "teacher.pth"
            torch.save({"ids": torch.tensor([1, 2]), "spectra": torch.rand(2, 1000)}, path)
            teacher = NEW.TrainTeacher(path, [1, 2])
            loss, prediction, stats, terms = NEW.training_loss(model, batch, args, teacher)
            self.assertTrue(torch.isfinite(loss))
            self.assertIn("distillation_loss", terms)
            torch.testing.assert_close(batch.y, target, rtol=0, atol=0)
            loss.backward()

    def test_cycle_schedule_not_validation_dependent(self):
        args = NEW.parser().parse_args([])
        self.assertEqual(NEW.cycle_lr(args, 40), args.lr)
        self.assertEqual(NEW.cycle_lr(args, 41), args.lr * 2)
        self.assertAlmostEqual(NEW.cycle_lr(args, 50), args.lr * .2)
        self.assertEqual(NEW.cycle_lr(args, 51), NEW.cycle_lr(args, 41))

    def test_prediction_needs_no_measured_spectrum(self):
        args = NEW.parser().parse_args(["--hidden-dim", "16", "--num-layers", "1", "--num-heads", "4"])
        model = NEW.COMMON.QCGN2oEI_Minimal(**NEW.BASE.model_config(args)).eval()
        batch = Batch.from_data_list([NEW.BASE.make_graph("CCO")])
        with torch.inference_mode():
            first, _, _ = NEW.BASE.forward(model, batch)
            batch.y = torch.rand(1, 1000)
            second, _, _ = NEW.BASE.forward(model, batch)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_cpu_restart_matches_all_training_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "dataset.sqlite"
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("CREATE TABLE samples(sample_id INTEGER PRIMARY KEY,smiles TEXT,peaks TEXT,split TEXT)")
                for sample_id, smiles in enumerate(("CC", "CCC", "CCO", "CCN", "CO", "CN", "CCCC", "CCCO"), 1):
                    split = "train" if sample_id <= 4 else "validation" if sample_id <= 6 else "test"
                    db.execute("INSERT INTO samples VALUES(?,?,?,?)", (sample_id, smiles, json.dumps([[15, 1.], [29, .3]]), split))
            panel = {"train": [1, 2, 3, 4], "validation": [5, 6], "test": [7, 8]}
            teacher_path = root / "teacher.pth"
            torch.save({"ids": torch.tensor(panel["train"]), "spectra": torch.rand(4, 1000)}, teacher_path)
            for algorithm in ("sam", "random_depth", "distill", "cycle_swa"):
                with self.subTest(algorithm=algorithm):
                    common = ["--hidden-dim", "16", "--num-layers", "1", "--num-heads", "4", "--batch-size", "2",
                              "--epochs", "4", "--checkpoint-interval", "1", "--swa-start-epoch", "0", "--swa-cycle-epochs", "2",
                              "--graph-cache-size", "16", "--min-free-ram-gb", "0", "--teacher-train-predictions", str(teacher_path)]
                    def args_for(name, resume=False):
                        args = NEW.parser().parse_args(common)
                        args.output = root / f"{algorithm}_{name}"
                        args.resume = resume
                        return args
                    whole, partial = args_for("whole"), args_for("partial")
                    NEW.run(whole, path, {"fingerprint": "test"}, panel, algorithm, 42, torch.device("cpu"))
                    real_copy = NEW.BASE.atomic_copy
                    last_variant = "cycle_swa" if algorithm == "cycle_swa" else algorithm
                    def interrupted_copy(source, destination):
                        real_copy(source, destination)
                        if destination.name == "durable_best.pth" and destination.parent.name == last_variant:
                            state = torch.load(destination.parent.parent / "last_checkpoint.pth", weights_only=False)
                            if state["epoch"] == 2:
                                raise InterruptedError("synthetic epoch2 stop")
                    with patch.object(NEW.BASE, "atomic_copy", interrupted_copy):
                        with self.assertRaises(InterruptedError):
                            NEW.run(partial, path, {"fingerprint": "test"}, panel, algorithm, 42, torch.device("cpu"))
                    NEW.run(args_for("partial", True), path, {"fingerprint": "test"}, panel, algorithm, 42, torch.device("cpu"))
                    def saved(args):
                        return torch.load(args.output / f"{algorithm}_seed42/last_checkpoint.pth", weights_only=False)
                    first, second = saved(whole), saved(partial)
                    self.assertEqual(first["updates"], second["updates"])
                    for key, value in first["model"].items():
                        torch.testing.assert_close(value, second["model"][key], rtol=0, atol=0)
                    self.assertTrue(torch.equal(first["depth_rng"], second["depth_rng"]))


if __name__ == "__main__":
    unittest.main()
