#!/usr/bin/env python3
"""Save installation, CLI, unit and full-size synthetic smoke evidence."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    commands = {
        "installation": [sys.executable, "-m", "pip", "install", "--no-deps", "-e", "."],
        "console_help": ["nextims", "--help"],
        "unit_tests": [sys.executable, "-m", "unittest", "discover", "-s", "scripts", "-p", "test_*.py", "-q"],
        "pipeline": [sys.executable, "scripts/smoke_current_pipeline.py", "--output", str(root / "pipeline")],
    }
    results = {}
    for label, command in commands.items():
        with (root / f"{label}.log").open("w", encoding="utf-8") as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
        results[label] = dict(returncode=result.returncode, log=f"{label}.log")
        print(json.dumps({label: results[label]}), flush=True)
        if result.returncode:
            raise RuntimeError(f"Release QA failed: {label}; see {root / (label + '.log')}")
    if importlib.util.find_spec("dgl") or importlib.util.find_spec("bondnet"):
        raise RuntimeError("Clean release QA must run without DGL/BonDNet")
    (root / "receipt.json").write_text(json.dumps(dict(passed=True, results=results,
                            scope="runtime correctness, not scientific efficacy"), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
