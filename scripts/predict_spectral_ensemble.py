"""A03: fixed equal-weight unit-spectrum ensemble of compatible predictors."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

import run_feedback_generalization_experiment as NEW


def load_models(paths, device):
    expected = {key.replace("\\", "/"): value for key, value in NEW.BASE.source_hashes().items()}
    models, configs = [], []
    for path in paths:
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(saved, dict) or not {"source_sha256", "model_config", "model_state_dict"} <= saved.keys():
            raise ValueError("This is not a compatible V007 research checkpoint")
        recorded = {key.replace("\\", "/"): value for key, value in saved["source_sha256"].items()}
        if recorded != expected:
            raise ValueError("Checkpoint predictor source differs; use its matching support files")
        config = saved["model_config"]
        if configs and config != configs[0]:
            raise ValueError("Ensemble requires identical architecture/feature/fusion configuration")
        model = NEW.QCGN2oEI_Minimal(**config).to(device).eval()
        model.load_state_dict(saved["model_state_dict"])
        models.append(model)
        configs.append(config)
    if len(models) < 2:
        raise ValueError("At least two checkpoints are required")
    return models


def combine(predictions):
    if not predictions or any(value.shape != predictions[0].shape for value in predictions):
        raise ValueError("Predictions must have identical shapes")
    if any(not torch.isfinite(value).all() for value in predictions):
        raise FloatingPointError("Nonfinite ensemble input")
    return torch.stack([F.normalize(value, p=2, dim=-1) for value in predictions]).mean(dim=0)


def forward(models, batch):
    predictions = [NEW.BASE.forward(model, batch)[0] for model in models]
    return combine(predictions), predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, default=Path("ensemble_prediction.csv"))
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(2)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    models = load_models(args.checkpoints, device)
    batch = Batch.from_data_list([NEW.BASE.make_graph(args.smiles)]).to(device)
    with torch.inference_mode():
        prediction, _ = forward(models, batch)
    values = prediction[0].cpu().numpy()
    values = values / max(float(values.max()), 1e-8) * 100
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mz", "intensity"])
        writer.writerows(enumerate(values.tolist(), start=1))
    args.output.with_suffix(".json").write_text(json.dumps({"method": "equal L2-normalized predicted spectra",
        "models": len(models), "checkpoints": [str(path) for path in args.checkpoints],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "note": "Two independent predictors, not two dropout passes or single-model EMA"}, indent=2), encoding="utf-8")
    print(f"Prediction: {args.output.resolve()}")


if __name__ == "__main__":
    main()
