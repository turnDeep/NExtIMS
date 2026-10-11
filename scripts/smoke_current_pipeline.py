#!/usr/bin/env python3
"""Synthetic full-size CLI smoke; NOT an accuracy validation or NIST benchmark."""
import argparse
import importlib.util
import json
from pathlib import Path

import torch
import nextims
from spectral_feedback_support.data import split_for_smiles
from rdkit import Chem


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.output.resolve()
    if root.exists():
        raise ValueError("Use a new smoke output directory")
    data = root / "data"
    data.mkdir(parents=True)
    structures = ["C" * n + terminal for n in range(1, 11) for terminal in ("O", "N", "F")]
    groups = {split_for_smiles(Chem.MolToSmiles(Chem.MolFromSmiles(s)), 42, .1, .1) for s in structures}
    if groups != {"train", "validation", "test"}:
        raise ValueError("Synthetic fixture lacks a split")
    with (data / "NIST17.MSP").open("w", encoding="utf-8") as handle:
        for i, smiles in enumerate(structures):
            handle.write(f"Name: synthetic-{i}\nSMILES: {smiles}\nNum Peaks: 4\n15 30\n29 100\n43 40\n57 20\n\n")
    run = root / "run"
    argv = ["--device", "cpu", "--data-dir", str(data), "--output", str(run),
            "--epochs", "1", "--minimum-epochs", "1", "--batch-size", "64",
            "--min-free-ram-gb", "0", "--graph-cache-size", "0", "--train-diagnostic-size", "64"]
    nextims.main(argv)
    nextims.main([*argv, "--resume"])
    nextims.main(["--mode", "predict", "--device", "cpu", "--checkpoint", str(run / "best_model.pth"),
                  "--predict-smiles", "CCO", "--prediction-output", str(root / "predicted.msp")])
    nextims.main(["--mode", "evaluate", "--device", "cpu", "--checkpoint", str(run / "best_model.pth"),
                  "--prepared-db", str(run / "preparation/dataset.sqlite"), "--output", str(root / "reevaluation")])
    receipt = dict(passed=True, scope="synthetic 768/14/24 CPU CLI smoke, not efficacy",
                   torch=torch.__version__, dgl_present=importlib.util.find_spec("dgl") is not None,
                   bondnet_present=importlib.util.find_spec("bondnet") is not None,
                   artifacts=["run/best_model.pth", "run/evaluation_summary.csv", "predicted.msp",
                              "reevaluation/evaluation/test_per_step_summary.csv"])
    for relative in receipt["artifacts"]:
        if not (root / relative).is_file():
            raise ValueError(f"Missing smoke artifact: {relative}")
    (root / "smoke_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
