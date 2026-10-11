"""Predict with an EXP-V020 research checkpoint and its saved component recipe."""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch_geometric.data import Batch

import run_component_ablation as STUDY


def verify_algorithm_sources(saved_sources):
    normalize = lambda values: {key.replace("\\", "/"): value for key, value in values.items()}
    saved_sources = normalize(saved_sources)
    expected = normalize(STUDY.source_hashes())
    # V022 adds a training-only runner; shared model/adapter hashes must match.
    allowed_extra = {"run_full_feedback_off_experiment.py"}
    if any(saved_sources.get(key) != value for key, value in expected.items()):
        raise ValueError("Component adapter sources differ from the checkpoint")
    if set(saved_sources) - set(expected) - allowed_extra:
        raise ValueError("Unknown checkpoint algorithm source; use its original predictor")


def load_model(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    required = {"source_sha256", "model_config", "model_state_dict"}
    if not isinstance(saved, dict) or not required <= saved.keys():
        raise ValueError("Not a compatible spectral-feedback research checkpoint")
    normalize = lambda values: {key.replace("\\", "/"): value for key, value in values.items()}
    if normalize(saved["source_sha256"]) != normalize(STUDY.BASE.source_hashes()):
        raise ValueError("Predictor support sources differ from the checkpoint")
    config = saved["model_config"]
    if "component_config" in config:
        verify_algorithm_sources(saved["algorithm_source_sha256"])
    else:
        expected = dict(use_recurrent_refinement=True, use_act=True, refinement_steps=4,
                        spectral_feedback_pilot=True, spectral_residual_enabled=True)
        if any(config.get(key) != value for key, value in expected.items()) or not config.get("spectral_feedback_enabled", True):
            raise ValueError("Legacy checkpoint is not a full native-ACT4 reference; use its own predictor")
    model = STUDY.ComponentModel(**config).to(device).eval()
    model.load_state_dict(saved["model_state_dict"])
    return model, saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    model, saved = load_model(args.checkpoint, device)
    batch = Batch.from_data_list([STUDY.BASE.make_graph(args.smiles)]).to(device)
    with torch.inference_mode():
        prediction, stats, _ = STUDY.BASE.forward(model, batch)
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite prediction")
    values = prediction[0].cpu()
    values = (values / values.max().clamp_min(1e-8) * 100).tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("mz", "intensity"))
        writer.writerows(enumerate(values, start=1))
    STUDY.BASE.write_json(args.output.with_suffix(".json"), dict(
        checkpoint=str(args.checkpoint), selected_epoch=saved["epoch"],
        component_config=model.component_config, smiles=args.smiles,
        expected_depth=float(stats["ponder_steps"][0]), target_required=False,
        note="Research checkpoint prediction; does not validate full-scale accuracy"))
    print(f"Prediction: {args.output.resolve()}")


if __name__ == "__main__":
    main()
