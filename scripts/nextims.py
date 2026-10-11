#!/usr/bin/env python3
"""Current NExtIMS: fusion0.30, ACT4, residual, teacher, SAM; no feedback/BDE."""

import csv
import json
from pathlib import Path
import sys

# Frozen research runners use sibling imports; preserve their checkpoint hashes.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import run_full_feedback_off_experiment as FULL
from predict_component_model import load_model
from spectral_feedback_support.data import DiskDataset, prepare_database


def parser():
    p = FULL.parser()
    p.description = __doc__
    p.set_defaults(output=Path("paper_runs/current"), data_dir=Path("data"))
    p._option_string_actions["--mode"].choices = ["prepare", "train", "predict", "evaluate"]
    p.add_argument("--evaluation-split", choices=["train", "validation", "test"], default="test")
    p.add_argument("--input-csv", type=Path, help="CSV with name and smiles; prediction only")
    return p


def prepare(args):
    if args.max_samples != 0:
        raise ValueError("Current full-data recipe uses --max-samples 0; use research runners for pilots")
    path, manifest = prepare_database(args.output / "preparation", args.data_dir, args.msp,
                                     args.mol_dir, 0, args.split_seed,
                                     args.val_fraction, args.test_fraction)
    print(json.dumps({"prepared_db": str(path.resolve()), "counts": manifest["split_counts"]}))
    return path


def predict(args):
    if not args.checkpoint:
        raise ValueError("--checkpoint is required")
    model, saved = load_model(args.checkpoint, FULL.configure_device(args))
    if args.input_csv:
        with args.input_csv.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or any(not row.get("smiles") for row in rows):
            raise ValueError("CSV must contain a nonempty smiles column")
    elif args.predict_smiles:
        rows = [dict(name="prediction", smiles=args.predict_smiles)]
    else:
        raise ValueError("Supply --predict-smiles or --input-csv")
    output = args.prediction_output
    output.parent.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    with output.open("w", newline="", encoding="utf-8") as handle, torch.inference_mode():
        writer = None if output.suffix.lower() == ".msp" else csv.writer(handle)
        if writer:
            writer.writerow(["name", "smiles", "mz", "intensity"])
        for row in rows:
            graph = FULL.BASE.make_graph(row["smiles"])
            values, _, _ = FULL.BASE.forward(model, Batch.from_data_list([graph]).to(device))
            if not torch.isfinite(values).all():
                raise FloatingPointError("Nonfinite spectrum")
            values = (100 * values[0] / values[0].max().clamp_min(1e-8)).cpu().tolist()
            name = row.get("name") or row["smiles"]
            if any(c in name + row["smiles"] for c in "\r\n"):
                raise ValueError("Names and SMILES cannot contain line breaks")
            if writer:
                writer.writerows((name, row["smiles"], mz, v) for mz, v in enumerate(values, 1))
            else:
                peaks = [(mz, v) for mz, v in enumerate(values, 1) if v >= .01]
                handle.write(f"Name: {name}\nSMILES: {row['smiles']}\nMW: {int(graph.molecular_ion_mz)}\n")
                handle.write(f"Comment: NExtIMS predicted EI; epoch {saved['epoch']}; not measured\nNum Peaks: {len(peaks)}\n")
                handle.writelines(f"{mz} {v:.6f}\n" for mz, v in peaks)
                handle.write("\n")
    FULL.BASE.write_json(output.with_suffix(output.suffix + ".json"), {
        "checkpoint": str(args.checkpoint), "epoch": saved["epoch"], "records": len(rows),
        "component_config": model.component_config, "requires_target": False,
        "note": "Predicted spectra, not measured references"})
    print(f"Prediction: {output.resolve()}")


def main(argv=None):
    args = parser().parse_args(argv)
    args.output = args.output.resolve()
    if args.mode == "predict":
        return predict(args)
    if args.mode == "evaluate":
        if not args.prepared_db or not args.checkpoint:
            raise ValueError("Evaluation requires --prepared-db and --checkpoint")
        args.output.mkdir(parents=True, exist_ok=True)
        device = FULL.configure_device(args)
        model, saved = load_model(args.checkpoint, device)
        dataset = DiskDataset(args.prepared_db.resolve(), args.evaluation_split)
        try:
            loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
            result = FULL.COMPONENT.evaluate(model, loader, args.output, "current", saved.get("seed", 42),
                                             args.evaluation_split, saved["epoch"], device)
            print(json.dumps(result))
        finally:
            dataset.close()
        return
    if args.mode == "prepare":
        prepare(args)
        return
    if (args.hidden_dim, args.num_layers, args.num_heads) != (768, 14, 24):
        raise ValueError("Current full-data recipe preserves 768/14/24")
    if args.max_samples != 0 or args.grad_accumulation != 1 or args.refinement_steps != 4 or args.fusion_weight != .3:
        raise ValueError("Requires max-samples0, accumulation1, ACT4, fusion0.30")
    if args.sam_rho <= 0 or not 0 < args.cuda_memory_fraction <= 1:
        raise ValueError("Invalid SAM rho or CUDA memory fraction")
    if min(args.epochs, args.batch_size, args.checkpoint_interval) < 1 or args.epochs < args.minimum_epochs:
        raise ValueError("Invalid epoch, batch or checkpoint settings")
    FULL.BASE.validate_update_schedule(args)
    FULL.SaturationMonitor(args.patience, args.minimum_epochs, args.cosine_min_delta, args.topk_min_delta)
    if not args.prepared_db:
        args.prepared_db = prepare(args)
    args.prepared_db = args.prepared_db.resolve()
    if args.full_mode == "audit":
        FULL.audit(args)
    elif args.full_mode == "capacity":
        FULL.capacity(args)
    else:
        FULL.run(args)


def train_main():
    main(["--mode", "train", *sys.argv[1:]])


def evaluate_main():
    main(["--mode", "evaluate", *sys.argv[1:]])


def predict_main():
    main(["--mode", "predict", *sys.argv[1:]])


if __name__ == "__main__":
    main()
