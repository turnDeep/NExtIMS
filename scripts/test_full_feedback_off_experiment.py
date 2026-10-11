"""Stopping-rule tests do not train or subsample the full experiment."""

import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from run_full_feedback_off_experiment import BASE, SaturationMonitor, capacity_subset, write_checkpoint


def metrics(cosine=.5, top5=.3, top10=.35, top20=.4):
    return dict(cosine=cosine, top5=top5, top10=top10, top20=top20)


class SaturationTests(unittest.TestCase):
    def test_minimum_epochs(self):
        tracker = SaturationMonitor(patience=3, minimum_epochs=8)
        for epoch in range(1, 8):
            self.assertFalse(tracker.update(epoch, metrics())["stop"])
        self.assertTrue(tracker.update(8, metrics())["stop"])

    def test_topk_improvement_prevents_stop(self):
        tracker = SaturationMonitor(patience=3, minimum_epochs=1)
        tracker.update(1, metrics())
        tracker.update(2, metrics())
        tracker.update(3, metrics())
        result = tracker.update(4, metrics(top20=.403))
        self.assertFalse(result["stop"])
        self.assertEqual(result["meaningful_improvements"], ["top20"])
        self.assertTrue(tracker.update(5, metrics(top20=.403))["stale_epochs"] == 1)

    def test_tiny_cosine_maxima_do_not_reset_plateau(self):
        tracker = SaturationMonitor(patience=3, minimum_epochs=1)
        tracker.update(1, metrics())
        tracker.update(2, metrics(cosine=.5001))
        tracker.update(3, metrics(cosine=.5002))
        self.assertTrue(tracker.update(4, metrics(cosine=.5003))["stop"])
        self.assertEqual(tracker.maxima["cosine"], .5003)

    def test_incremental_improvements_accumulate(self):
        tracker = SaturationMonitor(patience=10, minimum_epochs=1)
        tracker.update(1, metrics())
        tracker.update(2, metrics(cosine=.5002))
        result = tracker.update(3, metrics(cosine=.5006))
        self.assertEqual(result["stale_epochs"], 0)

    def test_resume_reproduces_stop(self):
        first = SaturationMonitor(patience=3, minimum_epochs=1)
        first.update(1, metrics())
        first.update(2, metrics())
        second = SaturationMonitor(patience=3, minimum_epochs=1, state=first.state_dict())
        for epoch in (3, 4):
            self.assertEqual(first.update(epoch, metrics()), second.update(epoch, metrics()))

    def test_resume_saturated_checkpoint_needs_no_extra_epoch(self):
        tracker = SaturationMonitor(patience=2, minimum_epochs=1)
        for epoch in range(1, 4):
            tracker.update(epoch, metrics())
        resumed = SaturationMonitor(patience=2, minimum_epochs=1, state=tracker.state_dict())
        self.assertTrue(resumed.saturated())

    def test_validation_nan_is_failure_not_saturation(self):
        tracker = SaturationMonitor()
        with self.assertRaises(FloatingPointError):
            tracker.update(1, metrics(cosine=float("nan")))

    def test_missing_epoch_is_rejected(self):
        with self.assertRaises(ValueError):
            SaturationMonitor().update(2, metrics())


class CapacityTests(unittest.TestCase):
    class WrappedDataset:
        def __init__(self):
            self.ids = np.asarray([101, 202, 303, 404])

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, position):
            return int(self.ids[position])

    def test_cached_wrapper_ids_are_not_mutated(self):
        dataset = self.WrappedDataset()
        chosen = [404, 202]
        subset = capacity_subset(dataset, chosen)
        self.assertEqual(len(subset), 2)
        self.assertEqual(list(subset), chosen)
        self.assertEqual(list(dataset.ids), [101, 202, 303, 404])

    def test_capacity_cannot_include_nontrain_or_duplicate_ids(self):
        for chosen in ([101, 101], [999]):
            with self.assertRaises(ValueError):
                capacity_subset(self.WrappedDataset(), chosen)


class CheckpointTests(unittest.TestCase):
    def test_atomic_checkpoint_carries_weights_optimizer_monitor_and_loader_state(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.RAdam(model.parameters(), lr=.01)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tracker = SaturationMonitor()
        tracker.update(1, metrics())
        generator = torch.Generator().manual_seed(22)
        with tempfile.TemporaryDirectory() as directory, patch.object(BASE, "cpu_rng_state", return_value={"cpu_test": True}):
            path = Path(directory) / "last.pth"
            write_checkpoint(path, model, optimizer, 1, 12, tracker, .5, 1, {"train": generator})
            saved = torch.load(path, map_location="cpu", weights_only=False)
            restored = torch.nn.Linear(2, 1)
            restored.load_state_dict(saved["model"])
            restored_optimizer = torch.optim.RAdam(restored.parameters(), lr=.01)
            restored_optimizer.load_state_dict(saved["optimizer"])
            self.assertTrue(torch.equal(model.weight, restored.weight))
            self.assertTrue(torch.equal(generator.get_state(), saved["loader_rng"]["train"]))
            self.assertEqual(saved["monitor"], tracker.state_dict())
            self.assertEqual((saved["epoch"], saved["total_updates"], saved["best_epoch"]), (1, 12, 1))
            self.assertEqual(len(restored_optimizer.state), len(optimizer.state))


if __name__ == "__main__":
    unittest.main()
