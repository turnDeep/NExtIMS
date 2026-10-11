#!/usr/bin/env python3
"""Export aggregate research evidence, never NIST compound rows or model weights."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess

STUDIES = {
    "EXP-V008": "version_comparison_20261002",
    "EXP-V009": "v004_v007_update4000_20261003_r02",
    "EXP-V010": "fresh_updates_accuracy_20261006",
    "EXP-V011": "act_factorial_20261006",
    "EXP-V012": "chemical_ion_branches_20261006",
    "EXP-V013": "residual_charge_factorial_20261006",
    "EXP-V014": "v013_regularization_20261006",
    "EXP-V015": "fragment_generalization_20261006",
    "EXP-V016": "batch_accumulation_20261007_r03",
    "EXP-V018": "v018_generalization_20261010_r02",
    "EXP-V019": "v019_single_model_generalization_20261010",
    "EXP-V020": "v020_component_ablation_20261010",
    "EXP-V021": "v021_feedback_off_act_fixed_20261011",
    "HIST-BDE": "bde-real-20260611",
    "HIST-FEATURE": "feature-ablation-20260611",
    "HIST-FUSION50K": "neutral-fusion-random50k-20260620",
    "HIST-GATE50K": "neutral-fusion-gate-random50k-20260620",
    "HIST-MOTIF": "consensus-motif-50k-20260620",
    "HIST-DOCKER10": "docker-recheck-20260627",
    "HIST-HSHIFT": "hydrogen-shift-docker-20260628",
    "HIST-CASCADE": "fragment-cascade-docker-20260628",
}


def export_csv(source, target):
    raw = source.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    fields, rows = reader.fieldnames, list(reader)
    if not fields or len(rows) > 500 or any("smiles" in k.lower() or k.lower() in {"sample_id", "inchikey"} for k in fields):
        raise ValueError(f"Not aggregate evidence: {source.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            # Aggregate reports can contain local checkpoint/log paths.
            row = {k: ("[local artifact path]" if re.search(r"[A-Za-z]:[\\/]|/workspace/", v or "") else v)
                   for k, v in row.items()}
            writer.writerow(row)
    return dict(source_sha256=hashlib.sha256(raw).hexdigest(), rows=len(rows),
                exported_sha256=hashlib.sha256(target.read_bytes()).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verification-root", type=Path, required=True)
    p.add_argument("--paper-summary", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("docs/research/evidence"))
    args = p.parse_args()
    manifest, missing = [], []
    names = {"comparison.csv", "comparison_all.csv", "comparison_val.csv", "condition_means.csv",
             "gap_means.csv", "additional_condition_means.csv", "comparison_with_previous.csv",
             "edge_compare_summary.csv", "full_edge_occlusion_results.csv", "feature_ablation_results.csv"}
    names.update({"fusion_train_eval_best_summary.csv", "fusion_train_same_eval_summary.csv",
                  "fixed_vs_bounded_variable_comparison.csv", "hydrogen_shift_proxy_summary.csv",
                  "fragment_cascade_proxy_summary.csv"})
    for version, directory in STUDIES.items():
        folder = args.verification_root / directory
        candidates = [*folder.glob("*.csv"), *(folder / "evaluation").glob("*.csv")]
        candidates = [f for f in candidates if f.name in names or f.name.endswith("comparison_summary.csv")
                      or f.name.endswith("sweep_summary.csv") or f.name.endswith("10ideas_summary.csv")]
        if not candidates:
            missing.append(dict(version=version, source_directory=directory, note="No allowlisted aggregate CSV; see narrative history"))
        for file in sorted(candidates):
            target = args.output / version / file.name
            record = export_csv(file, target)
            manifest.append(dict(version=version, source=str(file.relative_to(args.verification_root)).replace("\\", "/"),
                                 output=str(target.relative_to(args.output)).replace("\\", "/"), **record))
    target = args.output / "EXP-V001" / "test_evaluation_summary.csv"
    manifest.append(dict(version="EXP-V001", source="RTX5090/experiments/test_evaluation_summary.csv",
                         output="EXP-V001/test_evaluation_summary.csv", **export_csv(args.paper_summary, target)))
    reports = {
        "HIST-BDE/bde_real_summary.md": "bde-real-20260611/bde_real_verification_summary.md",
        "HIST-FEATURE/node_feature_group_summary.md": "node-feature-groups-20260611/node_feature_group_verification_summary.md",
    }
    for relative, source_name in reports.items():
        source = args.verification_root / source_name
        raw = source.read_bytes()
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        header = "> Historical report: not current defaults; see ../../HISTORY.md for later reversals.\n\n"
        target.write_text(header + raw.decode("utf-8-sig").replace("\r\n", "\n"), encoding="utf-8", newline="\n")
        manifest.append(dict(version=relative.split("/")[0], source=source_name, output=relative,
                             source_sha256=hashlib.sha256(raw).hexdigest(),
                             exported_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    payload = dict(as_of="2026-10-11", scope="Aggregate saved results, not a new accuracy rerun",
                   private_compound_data_exported=False, records=manifest, missing=missing)
    (args.output / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    history = subprocess.check_output(["git", "log", "HEAD", "--date=short",
                                      "--format=%H%x09%ad%x09%s"], text=True, encoding="utf-8")
    with (args.output / "git_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["commit", "date", "subject"])
        writer.writerows(line.split("\t", 2) for line in history.splitlines())
    print(json.dumps(dict(aggregate_files=len(manifest), narrative_only=missing)))


if __name__ == "__main__":
    main()
