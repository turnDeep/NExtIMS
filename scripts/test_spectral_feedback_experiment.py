"""Portable tests and an optional tiny CUDA smoke, not a performance study."""

import argparse
import csv
from contextlib import closing
import json
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from rdkit import Chem
import torch
from torch_geometric.data import Batch

import run_spectral_feedback_experiment as runner
from spectral_feedback_support.data import FEATURIZER, connect, discover_inputs, iter_msp, make_graph, prepare_database
from spectral_feedback_support.losses import stage_teacher_losses


def fixture(root):
    mol_dir = root / "data" / "NIST17.MOL"
    mol_dir.mkdir(parents=True)
    msp = root / "data" / "NIST17.MSP"
    with msp.open("w", encoding="utf-8") as handle:
        for identifier in range(1, 65):
            # 32 unique canonical structures, two replicate spectra each.
            smiles = "C" * ((identifier - 1) % 32 + 2) + "O"
            Chem.MolToMolFile(Chem.MolFromSmiles(smiles), str(mol_dir / f"ID{identifier}.MOL"))
            handle.write(f"Name: fixture-{identifier}\nID: {identifier}\nNum Peaks: 4\n15 30\n29 100\n43 45\n57 20\n\n")
    return msp, mol_dir


def real_fixture(root, source_root):
    source = next(source_root.rglob("NIST17.MSP"))
    mol_source = source.parent / "NIST17.MOL"
    target = root / "data"
    (target / "NIST17.MOL").mkdir(parents=True)
    count = 0
    with (target / "NIST17.MSP").open("w", encoding="utf-8") as handle:
        for entry in iter_msp(source):
            identifier = entry.get("id", "")
            path = mol_source / f"ID{identifier}.MOL"
            mol = Chem.MolFromMolFile(str(path), sanitize=True, removeHs=False) if path.exists() else None
            if not FEATURIZER.validate_molecule(mol)[0] or not entry["peak_count_ok"] or not entry["peaks"]:
                continue
            shutil.copyfile(path, target / "NIST17.MOL" / path.name)
            handle.write(f"Name: {entry['name']}\nID: {identifier}\nNum Peaks: {len(entry['peaks'])}\n")
            for mz, intensity in entry["peaks"].items():
                handle.write(f"{mz} {intensity}\n")
            handle.write("\n")
            count += 1
            if count == 64:
                break
    if count != 64:
        raise ValueError("Could not construct a 64-entry real-data fixture")


class ExperimentTests(unittest.TestCase):
    def test_repeat_schedule_validation(self):
        args = runner.parser().parse_args([])
        self.assertEqual(args.updates_per_batch, 1)
        runner.validate_update_schedule(args)
        args.updates_per_batch = 4
        with self.assertRaisesRegex(ValueError, "requires --grad-accumulation 1"):
            runner.validate_update_schedule(args)
        args.grad_accumulation = 1
        runner.validate_update_schedule(args)
        args.updates_per_batch = 0
        with self.assertRaisesRegex(ValueError, "must be positive"):
            runner.validate_update_schedule(args)

    def test_four_updates_recompute_same_batch_and_evaluate_once(self):
        args = runner.parser().parse_args(["--updates-per-batch", "4", "--grad-accumulation", "1",
                                          "--min-free-ram-gb", "0"])
        loader = [Batch.from_data_list([make_graph("CCO", [(29, 1.0)]) for _ in range(n)])
                  for n in (2, 1)]
        model = torch.nn.Linear(1, 1000)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        observed = []

        def toy_loss(model, batch, args, condition, training):
            observed.append((id(batch), model.weight.detach().clone()))
            prediction = model(torch.ones(batch.num_graphs, 1)).softmax(-1)
            loss = (1 - torch.nn.functional.cosine_similarity(prediction, batch.y)).mean()
            return loss, prediction, {"ponder_steps": torch.ones(batch.num_graphs)}, {"main_loss": loss}

        with patch.object(runner, "loss_terms", toy_loss), patch.object(optimizer, "step", wraps=optimizer.step) as step:
            train = runner.epoch_run(model, loader, args, torch.device("cpu"), "teacher", optimizer)
            self.assertEqual(step.call_count, 8)
            self.assertEqual(train["optimizer_updates"], 8)
            self.assertEqual(train["samples"], 12)
            self.assertEqual(train["dataset_samples"], 3)
            self.assertEqual(train["source_batches"], 2)
            for start in (0, 4):
                self.assertEqual(len({row[0] for row in observed[start:start + 4]}), 1)
                for index in range(start + 1, start + 4):
                    self.assertFalse(torch.equal(observed[index - 1][1], observed[index][1]))
            val = runner.epoch_run(model, loader, args, torch.device("cpu"), "teacher")
            self.assertEqual(len(observed), 10)
            self.assertEqual(step.call_count, 8)
            self.assertEqual(val["samples"], 3)
            self.assertEqual(val["replays_per_batch"], 1)
            self.assertEqual(val["optimizer_updates"], 0)

    def test_single_update_accumulation_partial_batch_normalization(self):
        args = runner.parser().parse_args(["--grad-accumulation", "2", "--min-free-ram-gb", "0"])
        loader = [Batch.from_data_list([make_graph("CCO", [(29, 1.0)]) for _ in range(n)])
                  for n in (2, 1, 1)]
        model = torch.nn.Linear(1, 1000)
        expected = torch.nn.Linear(1, 1000)
        expected.load_state_dict(model.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        expected_optimizer = torch.optim.SGD(expected.parameters(), lr=0.1)

        def toy_loss(model, batch, args, condition, training):
            prediction = model(torch.ones(batch.num_graphs, 1)).softmax(-1)
            loss = (1 - torch.nn.functional.cosine_similarity(prediction, batch.y)).mean()
            return loss, prediction, {"ponder_steps": torch.ones(batch.num_graphs)}, {"main_loss": loss}

        for size in (3, 1):
            batch = Batch.from_data_list([make_graph("CCO", [(29, 1.0)]) for _ in range(size)])
            expected_optimizer.zero_grad(set_to_none=True)
            loss, _, _, _ = toy_loss(expected, batch, args, "teacher", True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(expected.parameters(), args.grad_clip)
            expected_optimizer.step()
        with patch.object(runner, "loss_terms", toy_loss):
            metrics = runner.epoch_run(model, loader, args, torch.device("cpu"), "teacher", optimizer)
        self.assertEqual(metrics["optimizer_updates"], 2)
        self.assertEqual(metrics["samples"], 4)
        self.assertEqual(metrics["dataset_samples"], 4)
        for actual, reference in zip(model.parameters(), expected.parameters()):
            self.assertTrue(torch.allclose(actual, reference, atol=1e-7))

    def test_flat_mol_folder_and_group_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            args = (root / "preparation", root / "data", None, None, 0, 42, 0.2, 0.2)
            path, summary = prepare_database(*args)
            self.assertEqual(summary["valid_entries"], 64)
            self.assertEqual(summary["indexed_mols"], 64)
            with closing(connect(path)) as db:
                leakage = db.execute("SELECT COUNT(*) FROM (SELECT smiles FROM samples WHERE selected=1 GROUP BY smiles HAVING COUNT(DISTINCT split)>1)").fetchone()[0]
            self.assertEqual(leakage, 0)
            self.assertEqual(prepare_database(*args)[1], summary)
            self.assertTrue(all(summary["split_counts"].get(s, 0) > 0 for s in ("train", "validation", "test")))

    def test_recursive_layout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            data = root / "data"
            nested = data / "1-50000"
            nested.mkdir()
            (data / "NIST17.MSP").rename(nested / "NIST17.MSP")
            (data / "NIST17.MOL").rename(nested / "NIST17.MOL")
            _, sources, _ = discover_inputs(data)
            self.assertEqual(sources, [nested / "NIST17.MSP"])
            path, summary = prepare_database(root / "prep", data, None, None, 0, 42, 0.2, 0.2)
            self.assertEqual(summary["valid_entries"], 64)

    def test_native_act_and_teacher_gradients(self):
        args = runner.parser().parse_args(["--hidden-dim", "32", "--num-layers", "1", "--num-heads", "4"])
        torch.manual_seed(42)
        model = runner.QCGN2oEI_Minimal(**runner.model_config(args)).eval()
        batch = Batch.from_data_list([make_graph("CCO", [(29, 1.0), (15, 0.3)])])
        prediction, stats, aux = runner.forward(model, batch)
        before = prediction.detach().clone()
        cosine, topk = stage_teacher_losses(aux["act_exit_spectra"], aux["act_exit_weights"], batch.y)
        (cosine * 0.05 + topk * 0.02).backward()
        residual = model.spectrum_residual_head.weight.grad
        feedback = model.spectral_feedback_encoder[0].weight.grad
        self.assertTrue(torch.isfinite(residual).all() and residual.abs().sum() > 0)
        self.assertTrue(torch.isfinite(feedback).all() and feedback.abs().sum() > 0)
        self.assertTrue(torch.allclose(aux["act_exit_weights"].sum(1), torch.ones(1)))
        with torch.no_grad():
            after, _, _ = runner.forward(model, batch)
        self.assertTrue(torch.equal(before, after))

    def test_neutral_target_axis(self):
        target = torch.zeros(1, 1000)
        target[0, 315] = 1.0
        target[0, 297] = 0.5
        converted = runner.neutral_target(target, torch.tensor([316]))
        self.assertEqual(converted[0, 0].item(), 1.0)
        self.assertEqual(converted[0, 18].item(), 0.5)

    def test_interrupted_resume_matches_uninterrupted(self):
        self.check_resume_equality(repeats=1, accumulation=2)

    def test_four_updates_interrupted_resume_matches_uninterrupted(self):
        self.check_resume_equality(repeats=4, accumulation=1)

    def check_resume_equality(self, repeats, accumulation):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            path, preparation = prepare_database(root / "prep", root / "data", None, None, 0, 42, 0.2, 0.2)
            options = ["--hidden-dim", "32", "--num-layers", "1", "--num-heads", "4",
                       "--epochs", "2", "--batch-size", "8", "--grad-accumulation", str(accumulation),
                       "--updates-per-batch", str(repeats),
                       "--checkpoint-interval", "1", "--min-free-ram-gb", "0"]
            args = runner.parser().parse_args(options)
            args.output = root / "continuous"
            runner.train_condition(args, path, preparation, "teacher", 42, torch.device("cpu"))
            args.output = root / "interrupted"
            original = runner.epoch_run

            def interrupt(*positional, **kwargs):
                optimizer = positional[5] if len(positional) > 5 else kwargs.get("optimizer")
                epoch = positional[6] if len(positional) > 6 else kwargs.get("epoch")
                if optimizer is not None and epoch == 2:
                    raise RuntimeError("deliberate interruption after durable epoch 1")
                return original(*positional, **kwargs)

            with patch.object(runner, "epoch_run", interrupt):
                with self.assertRaisesRegex(RuntimeError, "deliberate interruption"):
                    runner.train_condition(args, path, preparation, "teacher", 42, torch.device("cpu"))
            args.resume = True
            runner.train_condition(args, path, preparation, "teacher", 42, torch.device("cpu"))
            checkpoints = [torch.load(root / name / "runs/teacher_seed42/last_checkpoint.pth", weights_only=False, map_location="cpu") for name in ("continuous", "interrupted")]
            for key, value in checkpoints[0]["model_state_dict"].items():
                self.assertTrue(torch.equal(value, checkpoints[1]["model_state_dict"][key]), key)
            self.assertEqual(checkpoints[0]["total_optimizer_updates"], checkpoints[1]["total_optimizer_updates"])
            self.assertEqual(checkpoints[0]["training_schedule"]["updates_per_batch"], repeats)


def smoke(output, device, real_data=None, updates_per_batch=1):
    output.mkdir(parents=True, exist_ok=True)
    real_fixture(output, real_data) if real_data else fixture(output)
    entrypoint = Path(runner.__file__).with_name("run_spectral_feedback_repeat4.py") if updates_per_batch == 4 else Path(runner.__file__)
    command = [sys.executable, str(entrypoint), "--data-dir", str(output / "data"),
               "--output", str(output / "run"), "--max-samples", "0", "--seeds", "42",
               "--epochs", "2", "--batch-size", "4", "--grad-accumulation", str(1 if updates_per_batch > 1 else 2),
               "--updates-per-batch", str(updates_per_batch),
               "--hidden-dim", "32", "--num-layers", "1", "--num-heads", "4",
               "--val-fraction", "0.2", "--test-fraction", "0.2",
               "--checkpoint-interval", "1", "--bootstrap-repeats", "100",
               "--min-free-ram-gb", "0", "--device", device]
    subprocess.run(command, check=True)
    subprocess.run([*command, "--resume"], check=True)
    checkpoint = output / "run/runs/teacher_seed42/best_model.pth"
    subprocess.run([sys.executable, str(Path(runner.__file__)), "--mode", "predict",
                    "--checkpoint", str(checkpoint), "--predict-smiles", "CCO", "--device", device,
                    "--prediction-output", str(output / "prediction.csv")], check=True)
    with (output / "run/evaluation/summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4 and all(int(r["samples"]) > 0 for r in rows)
    for row in rows:
        path = output / "run/runs" / f"{row['condition']}_seed{row['seed']}" / "evaluation" / f"{row['split']}_per_sample.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            samples = list(csv.DictReader(handle))
        assert len(samples) == int(row["samples"])
        for metric in runner.METRICS:
            actual = sum(float(s[metric]) for s in samples) / len(samples)
            assert abs(actual - float(row[metric])) < 1e-10
    assert all(Path(p).exists() for p in [output / "run/evaluation/condition_means.csv", output / "run/evaluation/paired_deltas.csv"])
    for condition in ("reference", "teacher"):
        directory = output / "run/runs" / f"{condition}_seed42"
        with (directory / "history.csv").open(newline="", encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
        total_updates = 0
        for row in history:
            source_batches = int(row["train_source_batches"])
            accumulation = 1 if updates_per_batch > 1 else 2
            expected_updates = (source_batches * updates_per_batch + accumulation - 1) // accumulation
            self_updates = int(row["train_optimizer_updates"])
            assert self_updates == expected_updates
            assert int(row["train_samples"]) == int(row["train_dataset_samples"]) * updates_per_batch
            assert int(row["validation_replays_per_batch"]) == 1
            total_updates += self_updates
            assert int(row["total_optimizer_updates"]) == total_updates
        saved = torch.load(directory / "last_checkpoint.pth", map_location="cpu", weights_only=False)
        assert saved["total_optimizer_updates"] == total_updates
        assert saved["training_schedule"]["updates_per_batch"] == updates_per_batch
        assert saved["training_schedule"]["state_carry_between_updates"] is False
    import zipfile
    with zipfile.ZipFile(output / "run/analysis_results.zip") as archive:
        assert "evaluation/paired_deltas.csv" in archive.namelist()
        assert not any(n.endswith((".pth", ".sqlite")) for n in archive.namelist())
    (output / "smoke_receipt.json").write_text(json.dumps({"passed": True, "device": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None, "torch": torch.__version__,
        "updates_per_batch": updates_per_batch, "runner_version": runner.RUNNER_VERSION, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke-output", type=Path)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--real-data", type=Path)
    p.add_argument("--updates-per-batch", type=int, default=1)
    args, unittest_args = p.parse_known_args()
    if args.smoke_output:
        smoke(args.smoke_output.resolve(), args.device, args.real_data, args.updates_per_batch)
    else:
        unittest.main(argv=[sys.argv[0], *unittest_args])
