"""No silent component leakage, fixed-stage teaching, and adapter resume invariants."""

import json
from pathlib import Path
from contextlib import closing
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Batch

import run_component_ablation as NEW


class ComponentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.args = NEW.GEN.parser().parse_args(["--hidden-dim", "16", "--num-layers", "1", "--num-heads", "4"])
        self.batch = Batch.from_data_list([NEW.BASE.make_graph(s) for s in ("CCO", "CC(=O)OC")])
        self.batch.y = torch.rand(2, 1000)

    def model(self, name="full_sam"):
        NEW.BASE.set_seed(42)
        return NEW.ComponentModel(**NEW.ORIGINAL_CONFIG(self.args), component_config=NEW.component_recipe(name))

    def test_full_matches_original_predictions_and_training_loss(self):
        model = self.model().eval()
        NEW.BASE.set_seed(42)
        old = NEW.ORIGINAL_MODEL(**NEW.ORIGINAL_CONFIG(self.args)).eval()
        with torch.inference_mode():
            torch.testing.assert_close(NEW.BASE.forward(model, self.batch)[0], NEW.BASE.forward(old, self.batch)[0], rtol=0, atol=0)
        model.train()
        NEW.BASE.set_seed(421)
        first = NEW.BASE.loss_terms(model, self.batch, self.args, "teacher", True)
        NEW.BASE.set_seed(421)
        second = NEW.loss_terms(model, self.batch, self.args, "teacher", True)
        torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)

    def test_common_weights_initialize_identically(self):
        reference = self.model().state_dict()
        for name in NEW.CHANGES:
            with self.subTest(name=name):
                state = self.model(name).state_dict()
                self.assertEqual(reference.keys(), state.keys())
                for key in reference:
                    torch.testing.assert_close(reference[key], state[key], rtol=0, atol=0)

    def test_fixed4_output_and_all_computed_stage_teachers(self):
        model = self.model("fixed4").eval()
        prediction, stats, aux = NEW.BASE.forward(model, self.batch)
        self.assertEqual(aux["act_exit_spectra"].size(1), 4)
        torch.testing.assert_close(prediction, aux["act_exit_spectra"][:, -1], rtol=0, atol=0)
        self.assertTrue(torch.all(stats["ponder_steps"] == 4))
        _, _, _, terms = NEW.loss_terms(model, self.batch, self.args, "teacher", True)
        expected = NEW.BASE.stage_teacher_losses(aux["act_exit_spectra"], torch.ones_like(aux["act_exit_weights"]), self.batch.y, self.args.stage_rank_margin)
        torch.testing.assert_close(terms["stage_cosine_loss"], expected[0])
        torch.testing.assert_close(terms["stage_topk_loss"], expected[1])

    def test_act_cap2(self):
        model = self.model("act2").eval()
        _, stats, aux = NEW.BASE.forward(model, self.batch)
        self.assertLessEqual(aux["act_exit_spectra"].size(1), 2)
        self.assertTrue(torch.all(stats["ponder_steps"] <= 2.000001))

    def test_no_residual_has_no_residual_head_gradient(self):
        model = self.model("no_residual").train()
        NEW.loss_terms(model, self.batch, self.args, "teacher", True)[0].backward()
        self.assertIsNone(model.spectrum_residual_head.weight.grad)
        self.assertIsNotNone(model.spectrum_head[-2].weight.grad)

    def test_feedback_injection_and_halt_diagnostics_are_separate(self):
        for name in ("no_feedback_input", "no_feedback_all"):
            with self.subTest(name=name):
                model = self.model(name).eval()
                diagnostics, gates = [], []
                model.spectral_halt_head.register_forward_pre_hook(lambda module, inp: diagnostics.append(inp[0][:, -3:].detach()))
                model.spectral_feedback_gate.register_forward_hook(lambda module, inp, output: gates.append(output.detach()))
                NEW.BASE.forward(model, self.batch)
                if name == "no_feedback_input":
                    self.assertTrue(all(torch.count_nonzero(gate) == 0 for gate in gates))
                    self.assertTrue(any(torch.count_nonzero(value) > 0 for value in diagnostics))
                else:
                    self.assertTrue(all(torch.count_nonzero(value) == 0 for value in diagnostics))

    def test_no_stage_teacher_retains_final_and_neutral_losses(self):
        model = self.model("no_stage_teacher").eval()
        loss, _, stats, terms = NEW.loss_terms(model, self.batch, self.args, "teacher", True)
        self.assertEqual(float(terms["stage_cosine_loss"]), 0)
        self.assertEqual(float(terms["stage_topk_loss"]), 0)
        expected = terms["main_loss"] + self.args.neutral_loss_weight * terms["neutral_loss"] + self.args.act_loss_weight * stats["act_loss"]
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)

    def test_fusion_only_has_zero_refinement_and_exact_initial_fusion(self):
        model = self.model("fusion_only").eval()
        direct, neutral, refinements = [], [], []
        model.spectrum_head.register_forward_hook(lambda module, inp, output: direct.append(output))
        model.neutral_loss_head.register_forward_hook(lambda module, inp, output: neutral.append(output))
        model.refinement_block.update.register_forward_hook(lambda module, inp, output: refinements.append(output))
        prediction, stats, _ = NEW.BASE.forward(model, self.batch)
        self.assertEqual(len(direct), 1)
        self.assertEqual(len(refinements), 0)
        torch.testing.assert_close(prediction, NEW.fuse_neutral_loss_prediction(direct[0], neutral[0], self.batch.molecular_ion_mz, .3), rtol=0, atol=0)
        self.assertTrue(torch.all(stats["ponder_steps"] == 0))
        prediction.sum().backward()
        self.assertIsNone(model.refinement_block.update[0].weight.grad)

    def test_predictions_do_not_read_measured_targets(self):
        for name in NEW.CHANGES:
            with self.subTest(name=name), torch.inference_mode():
                model = self.model(name).eval()
                first = NEW.BASE.forward(model, self.batch)[0]
                altered = self.batch.clone()
                altered.y = torch.rand_like(altered.y)
                torch.testing.assert_close(first, NEW.BASE.forward(model, altered)[0], rtol=0, atol=0)

    def test_adapter_restores_globals_after_failure(self):
        old = NEW.BASE.model_config, NEW.BASE.loss_terms, NEW.COMMON.QCGN2oEI_Minimal, NEW.GEN.source_hashes
        with self.assertRaises(RuntimeError):
            with NEW.installed(NEW.component_recipe("fixed4")):
                raise RuntimeError("expected")
        self.assertEqual(old, (NEW.BASE.model_config, NEW.BASE.loss_terms, NEW.COMMON.QCGN2oEI_Minimal, NEW.GEN.source_hashes))

    def test_cpu_resume_and_zero_depth_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "dataset.sqlite"
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("CREATE TABLE samples(sample_id INTEGER PRIMARY KEY,smiles TEXT,peaks TEXT,split TEXT)")
                for i, smiles in enumerate(("CC", "CCC", "CO", "CN"), 1):
                    db.execute("INSERT INTO samples VALUES(?,?,?,?)", (i, smiles, json.dumps([[15, 1.], [29, .3]]), "train" if i < 3 else "validation" if i == 3 else "test"))
            panel = dict(train=[1, 2], validation=[3], test=[4])
            for name in ("fixed4", "no_feedback_input", "no_stage_teacher", "fusion_only"):
                with self.subTest(name=name):
                    def args_for(output, resume=False):
                        args = NEW.GEN.parser().parse_args(["--hidden-dim", "16", "--num-layers", "1", "--num-heads", "4", "--batch-size", "2", "--epochs", "2", "--checkpoint-interval", "1", "--min-free-ram-gb", "0"])
                        args.output, args.resume = root / output, resume
                        return args
                    NEW.run_component(args_for("whole"), path, {"fingerprint": "test"}, panel, name, 42, torch.device("cpu"))
                    original_copy = NEW.BASE.atomic_copy
                    def interrupt(source, destination):
                        original_copy(source, destination)
                        if destination.name == "durable_best.pth":
                            state = torch.load(destination.parent.parent / "last_checkpoint.pth", weights_only=False)
                            if state["epoch"] == 1:
                                raise InterruptedError("test")
                    with patch.object(NEW.BASE, "atomic_copy", interrupt), self.assertRaises(InterruptedError):
                        NEW.run_component(args_for("partial"), path, {"fingerprint": "test"}, panel, name, 42, torch.device("cpu"))
                    NEW.run_component(args_for("partial", True), path, {"fingerprint": "test"}, panel, name, 42, torch.device("cpu"))
                    states = [torch.load(root / output / name / "sam_seed42/last_checkpoint.pth", weights_only=False) for output in ("whole", "partial")]
                    for key in states[0]["model"]:
                        torch.testing.assert_close(states[0]["model"][key], states[1]["model"][key], rtol=0, atol=0)
                    if name == "fusion_only":
                        result = json.loads((root / "whole" / name / "sam_seed42/sam/evaluation/validation_summary.json").read_text())
                        self.assertEqual(result["stop_depth"], 0)


if __name__ == "__main__":
    unittest.main()
