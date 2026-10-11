#!/usr/bin/env python3
"""
Package NExtIMS paper-run CSV files for analysis on another PC.

This collects:
- experiments/test_evaluation_summary.csv
- experiments/**/evaluation/*.csv

It intentionally excludes model weights, NPZ prediction dumps, MSP files, logs,
and prepared metadata unless explicitly added later.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from typing import Iterable, List


REQUIRED_RELATIVE_FILES = (
    Path("experiments") / "test_evaluation_summary.csv",
)


def iter_evaluation_csvs(output_dir: Path) -> Iterable[Path]:
    csv_files: List[Path] = []
    for eval_dir in output_dir.rglob("evaluation"):
        if not eval_dir.is_dir():
            continue
        csv_files.extend(sorted(path for path in eval_dir.glob("*.csv") if path.is_file()))
    return csv_files


def default_zip_path(output_dir: Path) -> Path:
    return output_dir.parent / f"{output_dir.name}_analysis_csvs.zip"


def package_results(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    if not output_dir.is_dir():
        raise NotADirectoryError(f"--output-dir must be a directory: {output_dir}")

    current_summary = output_dir / "evaluation_summary.csv"
    required_files = ([current_summary] if current_summary.is_file() else
                      [output_dir / rel_path for rel_path in REQUIRED_RELATIVE_FILES])
    missing_required = [path for path in required_files if not path.exists()]
    if missing_required:
        missing_text = "\n".join(str(path) for path in missing_required)
        raise FileNotFoundError(f"Required analysis file(s) not found:\n{missing_text}")

    evaluation_csvs = list(iter_evaluation_csvs(output_dir))
    if not evaluation_csvs:
        raise FileNotFoundError(
            f"No evaluation CSV files found under: {output_dir / 'experiments'}"
        )

    diagnostics = [output_dir / name for name in ("history.csv", "EXPERIMENT.json", "training_complete.json")
                   if (output_dir / name).is_file()]
    files = sorted(set(required_files + evaluation_csvs + diagnostics))
    zip_path = Path(args.zip).resolve() if args.zip else default_zip_path(output_dir).resolve()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists() and not args.overwrite:
        raise FileExistsError(f"ZIP already exists. Use --overwrite: {zip_path}")

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(output_dir).as_posix())

    print(f"Created: {zip_path}")
    print(f"Files:   {len(files)}")
    print(f"Size:    {zip_path.stat().st_size:,} bytes")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zip NExtIMS test_evaluation_summary.csv and evaluation/*.csv files."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="NExtIMS paper pipeline output directory, e.g. paper_runs/nextims_act_neutral_loss_306k",
    )
    parser.add_argument(
        "--zip",
        default=None,
        help="Destination ZIP path. Default: <output-dir-parent>/<output-dir-name>_analysis_csvs.zip",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination ZIP if it already exists.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        package_results(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
