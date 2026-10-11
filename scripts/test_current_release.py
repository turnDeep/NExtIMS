"""Current CLI and checkpoint compatibility; no accuracy claims."""
import tempfile
from pathlib import Path
import unittest

import torch
from torch_geometric.data import Batch
import nextims
import predict_component_model as predictor
import run_component_ablation as component
from test_spectral_feedback_experiment import fixture


class ReleaseTests(unittest.TestCase):
    def test_current_defaults_and_recipe(self):
        args = nextims.parser().parse_args([])
        self.assertEqual((args.hidden_dim, args.num_layers, args.num_heads), (768, 14, 24))
        self.assertEqual((args.grad_accumulation, args.updates_per_batch, args.max_samples), (1, 4, 0))
        self.assertFalse(nextims.FULL.RECIPE["feedback_input"])
        self.assertFalse(nextims.FULL.RECIPE["feedback_diagnostics"])

    def test_checkpoint_hashes_accept_training_only_extra(self):
        original = component.source_hashes()
        predictor.verify_algorithm_sources(original)
        predictor.verify_algorithm_sources({**original, "run_full_feedback_off_experiment.py": "historical-runner"})
        changed = {**original, "run_component_ablation.py": "wrong"}
        with self.assertRaises(ValueError):
            predictor.verify_algorithm_sources(changed)
        with self.assertRaises(ValueError):
            predictor.verify_algorithm_sources({**original, "unknown-model.py": "wrong"})

    def test_portable_prepare_and_predict(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            args = nextims.parser().parse_args(["--data-dir", str(root / "data"), "--output", str(root / "run")])
            path = nextims.prepare(args)
            self.assertTrue(path.exists())
            args.hidden_dim, args.num_layers, args.num_heads = 24, 1, 4
            config = {**component.ORIGINAL_CONFIG(args), "component_config": nextims.FULL.RECIPE}
            model = component.ComponentModel(**config).eval()
            graph = component.BASE.make_graph("CCO")
            with torch.inference_mode():
                before, _, _ = component.BASE.forward(model, Batch.from_data_list([graph]))
            checkpoint = root / "model.pth"
            torch.save(dict(model_config=config, model_state_dict=model.state_dict(), epoch=1,
                            source_sha256=component.BASE.source_hashes(),
                            algorithm_source_sha256=nextims.FULL.source_hashes()), checkpoint)
            loaded, _ = predictor.load_model(checkpoint, torch.device("cpu"))
            with torch.inference_mode():
                after, _, _ = component.BASE.forward(loaded, Batch.from_data_list([graph]))
            torch.testing.assert_close(before, after)
            args.checkpoint, args.device, args.predict_smiles = checkpoint, "cpu", "CCO"
            args.prediction_output = root / "predicted.msp"
            nextims.predict(args)
            text = args.prediction_output.read_text()
            self.assertIn("not measured", text)
            self.assertIn("Num Peaks:", text)
            nextims.main(["--mode", "evaluate", "--device", "cpu", "--checkpoint", str(checkpoint),
                          "--prepared-db", str(path), "--output", str(root / "new_eval")])
            self.assertTrue((root / "new_eval/evaluation/test_summary.json").is_file())

    def test_legacy_graph_import_without_bondnet(self):
        from src.data.graph_generator_minimal import MinimalGraphGenerator
        graph = MinimalGraphGenerator().smiles_to_graph("CCO")
        self.assertTrue(torch.equal(graph.edge_attr[:, 1], torch.full_like(graph.edge_attr[:, 1], .5)))
        with self.assertRaises(ValueError):
            MinimalGraphGenerator(use_bde=True)


if __name__ == "__main__":
    unittest.main()
