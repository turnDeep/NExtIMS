"""Audit published aggregate evidence and its scientific comparison boundaries."""
import csv
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent.parent / "docs/research/evidence"


class EvidenceTests(unittest.TestCase):
    def test_all_exported_hashes_and_no_compound_rows(self):
        manifest = json.loads((ROOT / "manifest.json").read_text())
        for item in manifest["records"]:
            path = ROOT / item["output"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["exported_sha256"])
            if path.suffix == ".csv":
                with path.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    self.assertNotIn("smiles", reader.fieldnames)
                    self.assertNotIn("sample_id", reader.fieldnames)
                    self.assertEqual(len(list(reader)), item["rows"])

    def test_large_fusion_result_and_denominator(self):
        with (ROOT / "EXP-V001/test_evaluation_summary.csv").open(newline="") as handle:
            results = {row["experiment"]: row for row in csv.DictReader(handle)}
        control = results["baseline_direct"]
        fusion = results["act_nl_fusion_030"]
        self.assertEqual(int(fusion["count"]), 30294)
        self.assertAlmostEqual(float(fusion["cosine"]) - float(control["cosine"]), .0790505195564616)
        for key in ("cosine", "top5", "top10", "top20"):
            self.assertGreater(float(fusion[key]), float(control[key]))


if __name__ == "__main__":
    unittest.main()
