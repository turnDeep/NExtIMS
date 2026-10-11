"""Focused tests for the reversible V007 generalization extension."""

import tempfile
import unittest
from pathlib import Path
from contextlib import closing
import sqlite3
from unittest.mock import patch

import torch
from torch import nn

import run_feedback_generalization_experiment as NEW


class GeneralizationTests(unittest.TestCase):
    def test_cache_exact_inputs_and_mutation_isolation(self):
        cache = NEW.GraphCache(2, 2**20)
        graph = NEW.BASE.make_graph("CCO")
        cache.put(("source", 1), graph)
        first = cache.get(("source", 1))
        for key, value in graph.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(first[key], value, rtol=0, atol=0)
        first.x.zero_()
        torch.testing.assert_close(cache.get(("source", 1)).x, graph.x, rtol=0, atol=0)

    def test_cache_lru_count_and_payload_caps(self):
        graph = NEW.BASE.make_graph("CCO")
        size = sum(value.numel() * value.element_size() for value in graph.values()
                   if isinstance(value, torch.Tensor))
        cache = NEW.GraphCache(2, size * 3)
        cache.put(1, graph)
        cache.put(2, graph)
        cache.get(1)
        cache.put(3, graph)
        self.assertIsNone(cache.get(2))
        self.assertEqual(len(cache.entries), 2)
        cache = NEW.GraphCache(10, size)
        cache.put(1, graph)
        cache.put(2, graph)
        self.assertEqual(len(cache.entries), 1)
        self.assertLessEqual(cache.payload_bytes, size)
        too_small = NEW.GraphCache(10, size - 1)
        too_small.put(1, graph)
        self.assertEqual(len(too_small.entries), 0)
        self.assertEqual(NEW.parser().parse_args([]).graph_cache_size, 0)

    def test_cache_rejects_non_cpu_tensors(self):
        cache = NEW.GraphCache(2, 2**20)
        graph = NEW.BASE.make_graph("CCO")
        graph.x = torch.empty_like(graph.x, device="meta")
        with self.assertRaisesRegex(ValueError, "CPU"):
            cache.put(1, graph)

    def test_recipes_preserve_fusion_and_feedback(self):
        args = NEW.parser().parse_args([])
        for recipe in NEW.RECIPES:
            config = NEW.recipe_config(args, recipe)
            self.assertEqual(config["spectral_feedback_fusion_weight"], .3)
            self.assertTrue(config["spectral_feedback_enabled"])
            self.assertTrue(config["spectral_residual_enabled"])
            self.assertTrue(config["use_act"])
            self.assertEqual(config["dropout"], .2 if recipe == "dropout" else .1)

    def test_ema_eval_restores_raw_weights_on_error(self):
        model = nn.Linear(2, 1)
        ema = NEW.make_ema(model, 8.)
        ema.update_parameters(model)
        with torch.no_grad():
            model.weight.add_(1.)
        raw = model.weight.detach().clone()
        with self.assertRaises(RuntimeError):
            with NEW.evaluation_weights(model, ema):
                self.assertFalse(torch.equal(raw, model.weight))
                raise RuntimeError("probe")
        torch.testing.assert_close(raw, model.weight, rtol=0, atol=0)

    def test_ema_decay_is_epoch_half_life_and_resumeable(self):
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        ema = NEW.make_ema(model, 2.)
        ema.update_parameters(model)
        with torch.no_grad():
            model.weight.fill_(1.)
        ema.update_parameters(model)
        ema.update_parameters(model)
        self.assertAlmostEqual(ema.module.weight.item(), .5, places=6)
        other = NEW.make_ema(model, 2.)
        other.load_state_dict(ema.state_dict())
        self.assertEqual(int(other.n_averaged), int(ema.n_averaged))

    def test_plateau_reduces_lr_only_on_stagnation(self):
        args = NEW.parser().parse_args(["--lr-patience", "1"])
        optimizer = torch.optim.RAdam(nn.Linear(1, 1).parameters(), lr=args.lr)
        scheduler = NEW.make_scheduler(optimizer, args, "plateau")
        for value in (.4, .5, .5, .5):
            scheduler.step(value)
        self.assertEqual(optimizer.param_groups[0]["lr"], args.lr * .5)
        self.assertIsNone(NEW.make_scheduler(optimizer, args, "baseline"))

    def test_csv_is_append_only(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "history.csv"
            NEW.append_csv(path, {"epoch": 1, "value": .1})
            NEW.append_csv(path, {"epoch": 2, "value": .2})
            self.assertEqual(len(path.read_text().splitlines()), 3)

    def test_source_hash_keys_are_platform_independent(self):
        self.assertTrue(all("\\" not in key for key in NEW.source_hashes()))

    def test_prediction_does_not_read_measured_target(self):
        from torch_geometric.data import Batch
        args = NEW.parser().parse_args(["--hidden-dim", "16", "--num-layers", "1", "--num-heads", "4"])
        model = NEW.QCGN2oEI_Minimal(**NEW.recipe_config(args, "baseline")).eval()
        batch = Batch.from_data_list([NEW.BASE.make_graph("CCO"), NEW.BASE.make_graph("Cc1ccccc1")])
        with torch.inference_mode():
            first, _, _ = NEW.BASE.forward(model, batch)
            batch.y = torch.rand(batch.num_graphs, 1000)
            second, _, _ = NEW.BASE.forward(model, batch)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_ensemble_is_scale_invariant(self):
        import predict_spectral_ensemble as ENS
        prediction = torch.rand(3, 1000)
        expected = torch.nn.functional.normalize(prediction, dim=-1)
        torch.testing.assert_close(ENS.combine([prediction, prediction * 5]), expected)

    def test_ensemble_cosine_is_not_below_component_mean(self):
        import predict_spectral_ensemble as ENS
        first, second, target = torch.rand(3, 1000), torch.rand(3, 1000), torch.rand(3, 1000)
        cosine = torch.nn.functional.cosine_similarity
        actual = cosine(ENS.combine([first, second]), target)
        reference = (cosine(first, target) + cosine(second, target)) * .5
        self.assertTrue(bool(torch.all(actual >= reference - 1e-6)))

    def test_cpu_resume_matches_uninterrupted_ema(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "data.sqlite"
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("CREATE TABLE samples(sample_id INTEGER PRIMARY KEY, smiles TEXT, peaks TEXT, split TEXT)")
                for sample_id, smiles in enumerate(("CC", "CCC", "CCO", "CCN", "CO", "CN", "CCCC", "CCCO"), 1):
                    split = "train" if sample_id <= 4 else "validation" if sample_id <= 6 else "test"
                    db.execute("INSERT INTO samples VALUES (?, ?, ?, ?)",
                               (sample_id, smiles, "[[15,1.0],[29,0.5],[43,0.2]]", split))
            common = ["--device", "cpu", "--hidden-dim", "16", "--num-layers", "1",
                      "--num-heads", "4", "--batch-size", "2", "--epochs", "2",
                      "--checkpoint-interval", "1", "--ema-start-epoch", "1",
                      "--train-diagnostic-size", "0", "--min-free-ram-gb", "0",
                      "--graph-cache-size", "16"]
            uninterrupted = NEW.parser().parse_args(common + ["--output", str(root / "full")])
            NEW.train_recipe(uninterrupted, path, {"fingerprint": "synthetic"}, "ema", 42, torch.device("cpu"))
            resumed = NEW.parser().parse_args(common + ["--output", str(root / "resume")])
            original = NEW.BASE.epoch_run

            def interrupt(model, loader, args, device, condition, optimizer=None, epoch=0):
                if optimizer is not None and epoch == 2:
                    raise RuntimeError("test interruption after durable epoch1")
                return original(model, loader, args, device, condition, optimizer, epoch)

            with patch.object(NEW.BASE, "epoch_run", interrupt), self.assertRaises(RuntimeError):
                NEW.train_recipe(resumed, path, {"fingerprint": "synthetic"}, "ema", 42, torch.device("cpu"))
            resumed.resume = True
            NEW.train_recipe(resumed, path, {"fingerprint": "synthetic"}, "ema", 42, torch.device("cpu"))
            left = torch.load(root / "full/runs/ema_seed42/best_model.pth", weights_only=False)
            right = torch.load(root / "resume/runs/ema_seed42/best_model.pth", weights_only=False)
            self.assertEqual(left["epoch"], right["epoch"])
            for key, value in left["model_state_dict"].items():
                torch.testing.assert_close(value, right["model_state_dict"][key], atol=1e-7, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
