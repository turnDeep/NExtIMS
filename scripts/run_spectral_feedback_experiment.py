#!/usr/bin/env python3
"""Portable RTX5090 residual-feedback ACT / weak stage-teacher experiment.

Run from the NExtIMS root; see SPECTRAL_FEEDBACK_RTX5090.md beside this file.
Production src/ and scripts/train_gnn_minimal.py are not imported or modified.
"""

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import zipfile

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from spectral_feedback_support.data import DiskDataset, connect, make_graph, prepare_database
from spectral_feedback_support.losses import stage_teacher_losses
from spectral_feedback_support.model import QCGN2oEI_Minimal

try:
    import psutil
except ImportError:
    psutil = None

METRICS = ("cosine", "top5", "top10", "top20")
SCRIPT_ROOT = Path(__file__).resolve().parent
RUNNER_VERSION = "EXP-V007/r01"


class BatchReplayLoader:
    """Replay only the current batch; do not cache the dataset or model state."""

    def __init__(self, loader, repeats):
        self.loader, self.repeats = loader, repeats

    def __len__(self):
        return len(self.loader) * self.repeats

    def __iter__(self):
        for batch in self.loader:
            for _ in range(self.repeats):
                yield batch


def validate_update_schedule(args):
    repeats = getattr(args, "updates_per_batch", 1)
    if repeats < 1 or args.grad_accumulation < 1:
        raise ValueError("Updates per batch and gradient accumulation must be positive")
    if repeats > 1 and args.grad_accumulation != 1:
        raise ValueError("--updates-per-batch > 1 requires --grad-accumulation 1: each replay must update the weights")


def training_schedule(args):
    return {"runner_version": RUNNER_VERSION,
            "updates_per_batch": getattr(args, "updates_per_batch", 1),
            "physical_batch_size": args.batch_size,
            "grad_accumulation": args.grad_accumulation,
            "effective_batch_per_optimizer_update": args.batch_size * args.grad_accumulation,
            "state_carry_between_updates": False,
            "evaluation_replays": 1}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def source_hashes():
    files = [Path(__file__), *sorted((SCRIPT_ROOT / "spectral_feedback_support").glob("*.py"))]
    return {str(p.relative_to(SCRIPT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def model_config(args):
    return dict(node_dim=34, edge_dim=10, hidden_dim=args.hidden_dim,
                num_layers=args.num_layers, num_heads=args.num_heads, output_dim=1000,
                dropout=0.1, use_edge_attr=True, use_graph_features=True,
                graph_feature_dim=17, pooling_mode="mean_add_max",
                use_neutral_loss_head=True, use_recurrent_refinement=True,
                refinement_steps=args.refinement_steps, refinement_max_steps=max(8, args.refinement_steps),
                refinement_dropout=0.1, use_lti_stabilization=True, use_loop_embedding=True,
                use_act=True, act_epsilon=0.01, act_loss_weight=args.act_loss_weight,
                spectral_feedback_pilot=True, spectral_feedback_enabled=True,
                spectral_residual_enabled=True, spectral_feedback_fusion_weight=args.fusion_weight)


def forward(model, batch):
    # Native ACT exits, NOT return_intermediate=True (which bypasses ACT).
    prediction, stats, aux = model(
        batch.x, batch.edge_index, batch.edge_attr, batch.batch,
        graph_features=batch.graph_features, molecular_ion_mz=batch.molecular_ion_mz,
        return_act_stats=True, return_aux_outputs=True,
    )
    if not aux["spectrum_already_fused"].item():
        raise RuntimeError("Expected native fused ACT spectra; do not apply fusion twice.")
    return prediction, stats, aux


def neutral_target(target, parent_mz):
    mz = torch.arange(1, target.size(-1) + 1, device=target.device).view(1, -1)
    losses = parent_mz.view(-1, 1) - mz
    valid = (losses >= 0) & (losses < target.size(-1))
    converted = torch.zeros_like(target)
    converted.scatter_add_(1, losses.clamp(0, target.size(-1) - 1), target * valid)
    return converted / converted.amax(dim=-1, keepdim=True).clamp_min(1e-8)


def metric_vectors(prediction, target):
    result = {"cosine": F.cosine_similarity(prediction, target, dim=-1)}
    for k in (5, 10, 20):
        p = prediction.topk(k, dim=-1).indices
        t = target.topk(k, dim=-1).indices
        result[f"top{k}"] = (p.unsqueeze(-1) == t.unsqueeze(1)).any(dim=1).float().mean(dim=-1)
        # Supplemental sparse-target recall; primary numbers retain repository definition.
        valid = target.gather(1, t) > 0
        hits = (p.unsqueeze(-1) == t.unsqueeze(1)).any(dim=1)
        result[f"top{k}_positive_recall"] = (hits & valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)
    return result


def loss_terms(model, batch, args, condition, training):
    prediction, stats, aux = forward(model, batch)
    main = (1 - F.cosine_similarity(prediction, batch.y, dim=-1)).mean()
    converted = neutral_target(batch.y, batch.molecular_ion_mz)
    neutral = (1 - F.cosine_similarity(aux["neutral_loss_spectrum"], converted, dim=-1)).mean()
    stage_cosine, stage_topk = main.new_zeros(()), main.new_zeros(())
    if training and condition == "teacher":
        stage_cosine, stage_topk = stage_teacher_losses(
            aux["act_exit_spectra"], aux["act_exit_weights"], batch.y, args.stage_rank_margin)
    loss = (main + args.neutral_loss_weight * neutral + args.act_loss_weight * stats["act_loss"]
            + args.stage_cosine_weight * stage_cosine + args.stage_topk_weight * stage_topk)
    return loss, prediction, stats, {"main_loss": main, "neutral_loss": neutral,
                                    "stage_cosine_loss": stage_cosine, "stage_topk_loss": stage_topk}


def memory_guard(args):
    if psutil is None:
        return {}
    vm = psutil.virtual_memory()
    available = vm.available / 2**30
    if args.min_free_ram_gb > 0 and available < args.min_free_ram_gb:
        raise MemoryError(f"RAM safeguard: only {available:.2f} GiB available. Reduce --batch-size; keep workers=0. Last periodic checkpoint can be resumed.")
    return {"available_ram_gb": available, "process_rss_gb": psutil.Process().memory_info().rss / 2**30}


def epoch_run(model, loader, args, device, condition, optimizer=None, epoch=0):
    training = optimizer is not None
    if training:
        validate_update_schedule(args)
    repeats = getattr(args, "updates_per_batch", 1) if training else 1
    source_batches = len(loader)
    if repeats > 1:
        loader = BatchReplayLoader(loader, repeats)
    model.train(training)
    totals, count = {}, 0
    dataset_samples, optimizer_updates = 0, 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    accumulation_samples = 0
    bar = tqdm(loader, desc=f"Epoch {epoch} {'train' if training else 'validation'}", unit="batch")
    for index, cpu_batch in enumerate(bar):
        if index % 20 == 0:
            memory_guard(args)
        batch = cpu_batch.to(device)
        with torch.set_grad_enabled(training):
            loss, prediction, stats, components = loss_terms(model, batch, args, condition, training)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, batch {index}")
        n = batch.num_graphs
        if training:
            # Normalize by actual samples, also for a partial last accumulation group.
            (loss * n).backward()
            accumulation_samples += n
            if (index + 1) % args.grad_accumulation == 0 or index + 1 == len(loader):
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulation_samples)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer_updates += 1
                optimizer.zero_grad(set_to_none=True)
                accumulation_samples = 0
        vectors = metric_vectors(prediction.detach(), batch.y)
        for key, vector in vectors.items():
            totals[key] = totals.get(key, 0.0) + vector.sum().item()
        scalar = {"loss": loss, "expected_depth": stats["ponder_steps"].mean(), **components}
        for key, value in scalar.items():
            totals[key] = totals.get(key, 0.0) + value.detach().item() * n
        count += n
        if index % repeats == 0:
            dataset_samples += n
        bar.set_postfix(cos=f"{totals['cosine']/count:.4f}", top20=f"{totals['top20']/count:.4f}",
                        updates=optimizer_updates)
        # Release this forward's graph before recomputing with the updated weights.
        del loss, prediction, stats, components, vectors, scalar
    if count == 0:
        raise ValueError("No samples in epoch")
    return {key: value / count for key, value in totals.items()} | {
        "samples": count, "dataset_samples": dataset_samples, "source_batches": source_batches,
        "replays_per_batch": repeats, "optimizer_updates": optimizer_updates}


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_copy(source, destination):
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def cpu_rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def make_loaders(path, args, seed):
    datasets = {s: DiskDataset(path, s) for s in ("train", "validation", "test")}
    generator = torch.Generator().manual_seed(seed)
    loaders = {s: DataLoader(d, batch_size=args.batch_size, shuffle=(s == "train"),
                            num_workers=0, pin_memory=False, generator=generator if s == "train" else None)
               for s, d in datasets.items()}
    return datasets, loaders, generator


def train_condition(args, path, preparation, condition, seed, device):
    validate_update_schedule(args)
    directory = args.output / "runs" / f"{condition}_seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    config = model_config(args)
    settings = {k: v for k, v in vars(args).items() if k not in {
        "output", "resume", "conditions", "seeds", "mode", "device", "checkpoint", "predict_smiles", "prediction_output"}}
    schedule = training_schedule(args)
    signature = {"settings": settings, "model_config": config, "condition": condition, "seed": seed,
                 "training_schedule": schedule,
                 "dataset_fingerprint": preparation["fingerprint"], "source_sha256": source_hashes()}
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["signature"] != json.loads(json.dumps(signature, default=str)):
            raise ValueError(f"Run configuration changed: {directory}. Use a new output directory.")
        if not args.resume:
            raise ValueError(f"Run exists: {directory}. Use --resume or a NEW --output.")
    else:
        write_json(manifest_path, {"signature": signature, "created_utc": datetime.now(timezone.utc).isoformat(),
                                  "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                                  "torch": torch.__version__, "selection": "maximum sample-weighted validation cosine",
                                  "teachers": "training targets in loss ONLY; ACT active masks detached; equally weighted active exits",
                                  "numerics": "float32; no TF32; no spectrum augmentation; workers=0; no graph cache"})
    set_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = QCGN2oEI_Minimal(**config).to(device)
    datasets, loaders, generator = make_loaders(path, args, seed)
    optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start, best_cosine, best_epoch, stale = 1, -math.inf, 0, 0
    total_optimizer_updates = 0
    checkpoint_path = directory / "last_checkpoint.pth"
    complete = directory / "training_complete.json"
    if args.resume and complete.exists():
        print(f"Training complete, evaluating: {directory.name}", flush=True)
    else:
        if args.resume and checkpoint_path.exists():
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(saved["model_state_dict"])
            optimizer.load_state_dict(saved["optimizer"])
            start, best_cosine, best_epoch, stale = saved["epoch"] + 1, saved["best_cosine"], saved["best_epoch"], saved["stale"]
            total_optimizer_updates = saved["total_optimizer_updates"]
            restore_rng(saved["rng"])
            generator.set_state(saved["loader_rng"])
            del saved
            durable_best = torch.load(directory / "resume_best_model.pth", map_location="cpu", weights_only=False)
            if durable_best["epoch"] != best_epoch:
                raise ValueError("Resume files were interrupted while saving. Use a new output; do not mix inconsistent checkpoints.")
            del durable_best
            atomic_copy(directory / "resume_best_model.pth", directory / "best_model.pth")
        elif args.resume and (directory / "history.csv").exists():
            raise ValueError("Interrupted before first resume checkpoint. Use a NEW output or --checkpoint-interval 1 initially.")
        history_path = directory / "history.csv"
        # Resume only through the periodic checkpoint, dropping later incomplete progress.
        if history_path.exists():
            with history_path.open(newline="", encoding="utf-8") as handle:
                history = [r for r in csv.DictReader(handle) if int(r["epoch"]) < start]
            if history:
                with history_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(history[0]))
                    writer.writeheader()
                    writer.writerows(history)
        last_epoch = start - 1
        for epoch in range(start, args.epochs + 1):
            last_epoch = epoch
            print(f"\n{condition} seed={seed}, epoch={epoch}/{args.epochs}", flush=True)
            train = epoch_run(model, loaders["train"], args, device, condition, optimizer, epoch)
            total_optimizer_updates += train["optimizer_updates"]
            val = epoch_run(model, loaders["validation"], args, device, condition, epoch=epoch)
            row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                   "total_optimizer_updates": total_optimizer_updates,
                   **{f"train_{k}": v for k, v in train.items()}, **{f"validation_{k}": v for k, v in val.items()},
                   "cosine_gap": train["cosine"] - val["cosine"]}
            if device.type == "cuda":
                row["peak_vram_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
            row.update(memory_guard(args))
            existed = history_path.exists() and history_path.stat().st_size > 0
            with history_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if not existed:
                    writer.writeheader()
                writer.writerow(row)
            payload = {"model_state_dict": model.state_dict(), "model_config": config, "epoch": epoch,
                       "condition": condition, "seed": seed, "validation": val, "source_sha256": source_hashes(),
                       "training_schedule": schedule, "total_optimizer_updates": total_optimizer_updates}
            if val["cosine"] > best_cosine:
                best_cosine, best_epoch, stale = val["cosine"], epoch, 0
                atomic_save(payload, directory / "best_model.pth")
            else:
                stale += 1
            stopping = args.patience > 0 and stale >= args.patience
            if epoch % args.checkpoint_interval == 0 or epoch == args.epochs or stopping:
                atomic_save({**payload, "optimizer": optimizer.state_dict(),
                             "best_cosine": best_cosine, "best_epoch": best_epoch, "stale": stale,
                             "rng": cpu_rng_state(), "loader_rng": generator.get_state()}, checkpoint_path)
                # Keep best weights consistent with the durable resume point.
                atomic_copy(directory / "best_model.pth", directory / "resume_best_model.pth")
            print(f"Validation cosine={val['cosine']:.6f}; Top5/10/20={val['top5']:.4f}/{val['top10']:.4f}/{val['top20']:.4f}; best epoch={best_epoch}; optimizer updates={train['optimizer_updates']} (total {total_optimizer_updates})", flush=True)
            if stopping:
                break
        write_json(complete, {"best_epoch": best_epoch, "best_validation_cosine": best_cosine,
                              "last_epoch": last_epoch, "total_optimizer_updates": total_optimizer_updates,
                              "training_schedule": schedule})
    saved = torch.load(directory / "best_model.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    selected_epoch = saved["epoch"]
    selected_updates = saved["total_optimizer_updates"]
    del saved, optimizer
    summaries = []
    for split in ("validation", "test"):
        summary = evaluate(model, loaders[split], directory, condition, seed, split, selected_epoch, device)
        summary.update(runner_version=RUNNER_VERSION, updates_per_batch=schedule["updates_per_batch"],
                       selected_optimizer_updates=selected_updates)
        write_json(directory / "evaluation" / f"{split}_summary.json", summary)
        summaries.append(summary)
    for dataset in datasets.values():
        dataset.close()
    del model, datasets, loaders
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summaries


def evaluate(model, loader, directory, condition, seed, split, epoch, device):
    model.eval()
    evaluation_dir = directory / "evaluation"
    evaluation_dir.mkdir(exist_ok=True)
    totals, count, stage_totals = {}, 0, {}
    primary = ["sample_id", "cosine", "top5", "top10", "top20", "top5_positive_recall",
               "top10_positive_recall", "top20_positive_recall", "expected_depth", "stop_depth"]
    stage_fields = ["sample_id", "step", "active", "exit_weight", "cosine", "top5", "top10", "top20",
                    "cosine_delta", "top5_delta", "top10_delta", "top20_delta", "spectrum_l1_delta"]
    with (evaluation_dir / f"{split}_per_sample.csv").open("w", newline="", encoding="utf-8") as sample_file, \
         (evaluation_dir / f"{split}_per_step.csv").open("w", newline="", encoding="utf-8") as stage_file, torch.inference_mode():
        sw, tw = csv.DictWriter(sample_file, fieldnames=primary), csv.DictWriter(stage_file, fieldnames=stage_fields)
        sw.writeheader()
        tw.writeheader()
        for cpu_batch in tqdm(loader, desc=f"Evaluate {condition} {split}"):
            batch = cpu_batch.to(device)
            prediction, stats, aux = forward(model, batch)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Nonfinite evaluation spectrum")
            weights = aux["act_exit_weights"]
            if not torch.allclose(weights.sum(dim=1), torch.ones(batch.num_graphs, device=device), atol=1e-5):
                raise RuntimeError("ACT exit weights do not sum to one")
            vectors = {k: v.cpu().numpy() for k, v in metric_vectors(prediction, batch.y).items()}
            depths = stats["ponder_steps"].cpu().numpy()
            w = weights.cpu().numpy()
            ids = batch.sample_id.cpu().tolist()
            for index, sample_id in enumerate(ids):
                stop = int(np.flatnonzero(w[index] > 0)[-1] + 1)
                row = {k: float(v[index]) for k, v in vectors.items()} | {
                    "sample_id": sample_id, "expected_depth": float(depths[index]), "stop_depth": stop}
                sw.writerow(row)
                for key in primary[1:]:
                    totals[key] = totals.get(key, 0.0) + row[key]
            previous_metrics, previous_spectrum = None, None
            for step in range(aux["act_exit_spectra"].size(1)):
                spectrum = aux["act_exit_spectra"][:, step]
                values = {k: v.cpu().numpy() for k, v in metric_vectors(spectrum, batch.y).items() if k in METRICS}
                l1 = (spectrum - previous_spectrum).abs().sum(-1).cpu().numpy() if previous_spectrum is not None else None
                for index, sample_id in enumerate(ids):
                    active = int(w[index, step] > 0)
                    row = {"sample_id": sample_id, "step": step + 1, "active": active,
                           "exit_weight": float(w[index, step]), **{k: float(v[index]) for k, v in values.items()},
                           **{f"{k}_delta": float(values[k][index] - previous_metrics[k][index]) if previous_metrics else "" for k in METRICS},
                           "spectrum_l1_delta": float(l1[index]) if l1 is not None else ""}
                    tw.writerow(row)
                    stat = stage_totals.setdefault(step + 1, {"computed_samples": 0, "active_samples": 0, **dict.fromkeys(METRICS, 0.0)})
                    stat["computed_samples"] += 1
                    stat["active_samples"] += active
                    for key in METRICS:
                        stat[key] += row[key] * active
                previous_metrics, previous_spectrum = values, spectrum
            count += batch.num_graphs
    summary = {"condition": condition, "seed": seed, "split": split, "best_epoch": epoch, "samples": count,
               **{k: v / count for k, v in totals.items()}}
    write_json(evaluation_dir / f"{split}_summary.json", summary)
    with (evaluation_dir / f"{split}_per_step_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "computed_samples", "active_samples", *METRICS])
        writer.writeheader()
        for step, stat in sorted(stage_totals.items()):
            writer.writerow({"step": step, **stat, **{k: stat[k] / stat["active_samples"] if stat["active_samples"] else "" for k in METRICS}})
    return summary


def comparison_and_archive(args, summaries):
    evaluation = args.output / "evaluation"
    evaluation.mkdir(exist_ok=True)
    with (evaluation / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    means = []
    for condition in args.conditions:
        for split in ("validation", "test"):
            rows = [r for r in summaries if r["condition"] == condition and r["split"] == split]
            means.append({"condition": condition, "split": split, "seeds": len(rows),
                          **{f"{k}_mean": float(np.mean([r[k] for r in rows])) for k in METRICS},
                          **{f"{k}_seed_std": float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else "" for k in METRICS}})
    with (evaluation / "condition_means.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(means[0]))
        writer.writeheader()
        writer.writerows(means)
    paired = []
    for seed in args.seeds:
        for split in ("validation", "test"):
            files = [args.output / "runs" / f"{c}_seed{seed}" / "evaluation" / f"{split}_per_sample.csv" for c in ("reference", "teacher")]
            if not all(p.is_file() for p in files):
                continue
            data = []
            for path in files:
                with path.open(newline="", encoding="utf-8") as handle:
                    data.append({int(r["sample_id"]): r for r in csv.DictReader(handle)})
            if data[0].keys() != data[1].keys():
                raise RuntimeError("Paired comparison does not contain identical molecules")
            ids = sorted(data[0])
            # Replicates of a structure are one statistical sampling unit.
            db = connect(args.output / "preparation" / "dataset.sqlite")
            groups = {}
            for i in ids:
                smiles = db.execute("SELECT smiles FROM samples WHERE sample_id=?", (i,)).fetchone()[0]
                groups.setdefault(smiles, []).append(i)
            db.close()
            for metric in METRICS:
                delta = {i: float(data[1][i][metric]) - float(data[0][i][metric]) for i in ids}
                group_sums = np.array([sum(delta[i] for i in group) for group in groups.values()])
                group_counts = np.array([len(group) for group in groups.values()])
                rng = np.random.default_rng(seed)
                boot = np.empty(args.bootstrap_repeats)
                for repeat in range(args.bootstrap_repeats):
                    draws = rng.integers(0, len(groups), len(groups))
                    boot[repeat] = group_sums[draws].sum() / group_counts[draws].sum()
                low, high = np.quantile(boot, [0.025, 0.975])
                paired.append({"seed": seed, "split": split, "metric": metric, "samples": len(ids),
                               "canonical_groups": len(groups), "teacher_minus_reference": float(np.mean(list(delta.values()))),
                               "cluster_bootstrap_ci_low": low, "cluster_bootstrap_ci_high": high})
    if paired:
        with (evaluation / "paired_deltas.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)
    archive_path = args.output / "analysis_results.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in args.output.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".csv", ".json", ".jsonl"}:
                archive.write(path, path.relative_to(args.output))
    print(f"\nCompleted. Transfer for analysis: {archive_path}\nLarge models and dataset.sqlite are NOT included.", flush=True)


def predict(args, device):
    if not args.checkpoint or not args.predict_smiles:
        raise ValueError("predict mode requires --checkpoint and --predict-smiles")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("source_sha256") != source_hashes():
        raise ValueError("Checkpoint implementation differs. Use the matching script/support package.")
    model = QCGN2oEI_Minimal(**checkpoint["model_config"]).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    graph = make_graph(args.predict_smiles)
    from torch_geometric.data import Batch
    with torch.inference_mode():
        values, _, _ = forward(model, Batch.from_data_list([graph]).to(device))
    values = values[0].cpu().numpy()
    values = values / max(float(values.max()), 1e-8) * 100
    args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
    with args.prediction_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mz", "intensity"])
        writer.writerows(enumerate(values.tolist(), start=1))
    print(f"Predicted spectrum: {args.prediction_output.resolve()}")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["train", "prepare", "predict"], default="train")
    p.add_argument("--data-dir", type=Path, default=SCRIPT_ROOT.parent / "data")
    p.add_argument("--msp", type=Path)
    p.add_argument("--mol-dir", type=Path, help="Directory of individual MOL files, e.g. data/NIST17.MOL")
    p.add_argument("--output", type=Path, default=SCRIPT_ROOT.parent / "paper_runs" / "spectral_feedback_stage_teacher")
    p.add_argument("--max-samples", type=int, default=50000, help="Uniform hash sampling over ALL valid MSP entries; 0=all")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    p.add_argument("--conditions", choices=["reference", "teacher"], nargs="+", default=["reference", "teacher"])
    p.add_argument("--hidden-dim", type=int, default=768)
    p.add_argument("--num-layers", type=int, default=14)
    p.add_argument("--num-heads", type=int, default=24)
    p.add_argument("--refinement-steps", type=int, default=4)
    p.add_argument("--fusion-weight", type=float, default=0.30)
    p.add_argument("--stage-cosine-weight", type=float, default=0.05)
    p.add_argument("--stage-topk-weight", type=float, default=0.02)
    p.add_argument("--stage-rank-margin", type=float, default=0.2)
    p.add_argument("--neutral-loss-weight", type=float, default=0.05)
    p.add_argument("--act-loss-weight", type=float, default=0.001)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accumulation", type=int, default=4)
    p.add_argument("--updates-per-batch", type=int, default=1,
                   help="Train repeatedly on the same batch, recomputing predictions after every optimizer update; >1 requires --grad-accumulation 1. Validation/test are never replayed.")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=0, help="0=run all epochs; otherwise validation early stopping")
    p.add_argument("--checkpoint-interval", type=int, default=5)
    p.add_argument("--min-free-ram-gb", type=float, default=2.0)
    p.add_argument("--bootstrap-repeats", type=int, default=2000)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--predict-smiles")
    p.add_argument("--prediction-output", type=Path, default=Path("spectral_feedback_prediction.csv"))
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    validate_update_schedule(args)
    if any(v <= 0 for v in [args.batch_size, args.grad_accumulation, args.epochs, args.hidden_dim,
                           args.num_layers, args.num_heads, args.refinement_steps, args.checkpoint_interval, args.bootstrap_repeats]):
        raise ValueError("Batch size, model dimensions, epochs, steps, and intervals must be positive")
    if args.hidden_dim % args.num_heads or args.max_samples < 0 or len(set(args.seeds)) != len(args.seeds) or len(set(args.conditions)) != len(args.conditions):
        raise ValueError("Invalid dimensions/sample count or duplicate seeds/conditions")
    if not (0 < args.val_fraction < 1 and 0 < args.test_fraction < 1 and args.val_fraction + args.test_fraction < 1):
        raise ValueError("Validation/test fractions must be positive and leave a nonempty training fraction")
    if not 0 <= args.fusion_weight <= 1 or min(args.stage_cosine_weight, args.stage_topk_weight,
                                              args.stage_rank_margin, args.neutral_loss_weight, args.act_loss_weight) < 0:
        raise ValueError("Invalid loss/fusion weights")
    if args.lr <= 0 or args.grad_clip <= 0 or min(args.weight_decay, args.patience, args.min_free_ram_gb) < 0:
        raise ValueError("Invalid optimizer, early-stopping, or RAM safeguard settings")
    args.output = args.output.resolve()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Check the RTX5090 container PyTorch/GPU environment.")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # Fail before a lengthy 306k preparation if the container cannot run CUDA.
        probe = torch.ones((8, 8), device=device)
        _ = (probe @ probe).sum().item()
        del probe
    print(f"Device={device}; torch={torch.__version__}; float32; workers=0; graph cache=OFF", flush=True)
    print(json.dumps(training_schedule(args)), flush=True)
    if psutil is None:
        print("Warning: psutil missing; RAM safeguard unavailable. Install psutil to enable it.", flush=True)
    if args.mode == "predict":
        predict(args, device)
        return
    path, preparation = prepare_database(args.output / "preparation", args.data_dir, args.msp,
                                         args.mol_dir, args.max_samples, args.split_seed, args.val_fraction, args.test_fraction)
    print(json.dumps({k: preparation[k] for k in ("valid_entries", "rejected_entries", "selected_entries", "split_counts", "canonical_groups")}, indent=2), flush=True)
    if args.mode == "prepare":
        return
    summaries = []
    for seed in args.seeds:
        for condition in args.conditions:
            summaries.extend(train_condition(args, path, preparation, condition, seed, device))
    comparison_and_archive(args, summaries)


if __name__ == "__main__":
    main()
