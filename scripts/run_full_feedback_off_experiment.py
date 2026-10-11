#!/usr/bin/env python3
"""EXP-V022: all accepted NIST records, feedback OFF, ACT cap4, FP32 SAM."""

import argparse
from contextlib import closing
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import run_component_ablation as COMPONENT

GEN, BASE, COMMON = COMPONENT.GEN, COMPONENT.BASE, COMPONENT.COMMON
VERSION = "EXP-V022/r01"
RECIPE = COMPONENT.component_recipe("no_feedback_all")
METRICS = ("cosine", "top5", "top10", "top20")


class SaturationMonitor:
    """Separate exact checkpoint maxima from meaningful plateau improvements."""

    def __init__(self, patience=10, minimum_epochs=20, cosine_delta=.0005,
                 topk_delta=.001, state=None):
        if patience < 1 or minimum_epochs < 1 or min(cosine_delta, topk_delta) < 0:
            raise ValueError("Invalid saturation settings")
        self.patience, self.minimum_epochs = patience, minimum_epochs
        self.deltas = dict(cosine=cosine_delta, top5=topk_delta,
                           top10=topk_delta, top20=topk_delta)
        self.anchors = dict.fromkeys(METRICS, -math.inf)
        self.maxima = dict.fromkeys(METRICS, -math.inf)
        self.last_meaningful_epoch, self.last_epoch = 0, 0
        if state is not None:
            self.anchors = state["anchors"].copy()
            self.maxima = state["maxima"].copy()
            self.last_meaningful_epoch = state["last_meaningful_epoch"]
            self.last_epoch = state["last_epoch"]

    def update(self, epoch, metrics):
        if epoch != self.last_epoch + 1:
            raise ValueError("Monitor requires consecutive epoch records")
        values = {key: float(metrics[key]) for key in METRICS}
        if not all(math.isfinite(value) for value in values.values()):
            raise FloatingPointError("Nonfinite validation metric")
        improved = []
        for key, value in values.items():
            self.maxima[key] = max(self.maxima[key], value)
            if value > self.anchors[key] + self.deltas[key]:
                self.anchors[key] = value
                improved.append(key)
        if improved:
            self.last_meaningful_epoch = epoch
        self.last_epoch = epoch
        stale = epoch - self.last_meaningful_epoch
        return dict(meaningful_improvements=improved, stale_epochs=stale,
                    stop=epoch >= self.minimum_epochs and stale >= self.patience)

    def state_dict(self):
        return dict(anchors=self.anchors.copy(), maxima=self.maxima.copy(),
                    last_epoch=self.last_epoch,
                    last_meaningful_epoch=self.last_meaningful_epoch)

    def saturated(self):
        return (self.last_epoch >= self.minimum_epochs and
                self.last_epoch - self.last_meaningful_epoch >= self.patience)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes():
    return {**COMPONENT.source_hashes(), Path(__file__).name: sha(__file__)}


def status(args, **record):
    BASE.write_json(args.output / "status.json", dict(version=VERSION, updated_utc=utc(), **record))


def audit(args):
    path = args.prepared_db.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    preparation = json.loads((path.parent / "dataset_manifest.json").read_text())
    if preparation["max_samples"] != 0:
        raise ValueError("This experiment requires the full, uncapped prepared index")
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        integrity = db.execute("PRAGMA quick_check").fetchone()[0]
        counts = dict(db.execute("SELECT split, COUNT(*) FROM samples WHERE selected=1 GROUP BY split"))
        all_count = db.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        cross = db.execute("SELECT COUNT(*) FROM (SELECT smiles FROM samples WHERE selected=1 "
                           "GROUP BY smiles HAVING COUNT(DISTINCT split)>1)").fetchone()[0]
        groups = db.execute("SELECT COUNT(DISTINCT smiles) FROM samples WHERE selected=1").fetchone()[0]
    if integrity != "ok" or cross or counts != preparation["split_counts"]:
        raise ValueError("Index integrity, exact-structure splitting, or counts failed")
    if sum(counts.values()) != all_count or all_count != preparation["valid_entries"]:
        raise ValueError("Some accepted records would be omitted")
    result = dict(version=VERSION, audited_utc=utc(), sqlite_sha256=sha(path),
                  all_accepted_records=all_count, canonical_isomeric_groups=groups,
                  counts=counts, quick_check=integrity, cross_split_exact_groups=cross,
                  rejected_entries=preparation["rejected_entries"], preparation=preparation,
                  split_scope="Existing canonical-isomeric-SMILES grouping; not a scaffold split",
                  historical_scope="The NIST development and test pools have been used in earlier studies",
                  data_policy="All accepted records, no random subsampling; Test labels unused during training")
    args.output.mkdir(parents=True, exist_ok=True)
    BASE.write_json(args.output / "data_audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "preparation"}), flush=True)
    return result


def configure_device(args):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    requested = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
        torch.cuda.reset_peak_memory_stats()
    return device


def configured_model(args, device):
    BASE.set_seed(args.seed)
    config = {**COMPONENT.ORIGINAL_CONFIG(args), "component_config": RECIPE.copy()}
    return COMPONENT.ComponentModel(**config).to(device), config


def capacity_subset(dataset, chosen):
    positions = {int(value): index for index, value in enumerate(dataset.ids)}
    if len(chosen) != len(set(chosen)) or not set(chosen) <= set(positions):
        raise ValueError("Capacity IDs must be unique training records")
    return Subset(dataset, [positions[value] for value in chosen])


def capacity(args):
    """Disposable training-only batches; these weights never enter the full run."""
    record = audit(args)
    device = configure_device(args)
    datasets, _, _ = COMMON.make_loaders(args.prepared_db, args, args.seed)
    dataset = datasets["train"]
    ids = np.asarray(dataset.ids)
    rng = np.random.default_rng(22022)
    selected = rng.choice(ids, size=min(len(ids), args.batch_size * 3), replace=False)
    sets = {"random_train": selected.tolist()}
    if args.capacity_stress_json:
        stress = json.loads(args.capacity_stress_json.read_text())["stress_ids"]
        allowed = set(map(int, ids))
        selected_stress = [int(value) for value in stress if int(value) in allowed]
        if len(selected_stress) >= args.batch_size:
            count = min(len(selected_stress) // args.batch_size, 3) * args.batch_size
            sets["large_train_molecules"] = selected_stress[:count]
    outcomes = []
    try:
        for label, chosen in sets.items():
            loader = DataLoader(capacity_subset(dataset, chosen), batch_size=args.batch_size,
                                shuffle=False, num_workers=0)
            model, config = configured_model(args, device)
            # Capacity only: keep all four stages active even before training the halt head.
            with torch.no_grad():
                model.spectral_halt_head[1].weight.zero_()
                model.spectral_halt_head[1].bias.fill_(-10.)
            optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            elapsed, losses, visited = [], [], []
            model.train()
            with COMPONENT.installed(RECIPE):
                for index, batch in enumerate(loader):
                    visited.extend(batch.sample_id.tolist())
                    batch = batch.to(device)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    start = time.monotonic()
                    loss, prediction, stats, terms = GEN.sam_update(
                        model, optimizer, batch, args, GEN.training_loss)
                    if stats["halt_probs"].shape[1] != args.refinement_steps:
                        raise ValueError("Capacity probe did not compute all four stages")
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    elapsed.append(time.monotonic() - start)
                    losses.append(float(loss))
                    del batch, loss, prediction, stats, terms
            if visited != chosen:
                raise ValueError("Capacity loader fetched incorrect training IDs")
            timed = elapsed[1:] or elapsed
            mean_seconds = float(np.mean(timed))
            outcome = dict(selection=label, outcome="passed", batch_size=args.batch_size,
                           measured_sam_updates=len(elapsed), mean_sam_update_seconds=mean_seconds,
                           visited_ids=visited, forced_computed_stages=4,
                           loss_values=losses, model_parameters=sum(p.numel() for p in model.parameters()),
                           approximate_train_epoch_hours=math.ceil(record["counts"]["train"] / args.batch_size)
                           * args.updates_per_batch * mean_seconds / 3600,
                           peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None,
                           peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30 if device.type == "cuda" else None)
            outcomes.append(outcome)
            print(json.dumps(outcome), flush=True)
            del model, optimizer
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
    except torch.OutOfMemoryError as error:
        outcomes.append(dict(selection=label, outcome="cuda_oom", batch_size=args.batch_size, error=str(error)))
    finally:
        for item in datasets.values():
            item.close()
    receipt = dict(version=VERSION, completed_utc=utc(), numerics="FP32, TF32 OFF", recipe=RECIPE,
                   results=outcomes, training_updates_are_discarded=True,
                   estimate_scope="Few train-only batches; excludes feature I/O and validation, not a completion ETA")
    BASE.write_json(args.output / f"capacity_batch{args.batch_size}.json", receipt)
    if any(row["outcome"] != "passed" for row in outcomes):
        raise RuntimeError("Capacity probe failed; full training was not launched")


def train_epoch(model, optimizer, loader, args, device, epoch, previous_updates):
    model.train()
    totals, count, updates = {}, 0, 0
    start = time.monotonic()
    last_status = start
    replay = BASE.BatchReplayLoader(loader, args.updates_per_batch)
    progress = tqdm(replay, total=len(replay), desc=f"Full train epoch {epoch}", mininterval=10)
    for index, cpu_batch in enumerate(progress):
        if index % 20 == 0:
            BASE.memory_guard(args)
        batch = cpu_batch.to(device)
        loss, prediction, stats, terms = GEN.sam_update(model, optimizer, batch, args, GEN.training_loss)
        vectors = BASE.metric_vectors(prediction, batch.y)
        n = batch.num_graphs
        for key, value in vectors.items():
            totals[key] = totals.get(key, 0.) + float(value.sum())
        for key, value in dict(loss=loss, expected_depth=stats["ponder_steps"].mean(), **terms).items():
            totals[key] = totals.get(key, 0.) + float(value.detach()) * n
        count += n
        updates += 1
        progress.set_postfix(cos=f"{totals['cosine'] / count:.4f}", top20=f"{totals['top20'] / count:.4f}", refresh=False)
        if time.monotonic() - last_status >= 30:
            status(args, phase="training", epoch=epoch, replay_batches_completed=index + 1,
                   replay_batches_total=len(replay), total_optimizer_updates=previous_updates + updates,
                   online_train_cosine=totals["cosine"] / count,
                   note="Online training is dropout ON; not the frozen train-validation gap")
            last_status = time.monotonic()
        del batch, loss, prediction, stats, terms, vectors
    expected = math.ceil(len(loader.dataset) / args.batch_size) * args.updates_per_batch
    if updates != expected or count != len(loader.dataset) * args.updates_per_batch:
        raise ValueError("Full training coverage/update count mismatch")
    return {key: value / count for key, value in totals.items()} | dict(
        samples=count, distinct_record_visits=len(loader.dataset), optimizer_updates=updates,
        forward_backward_passes=2 * updates, seconds=time.monotonic() - start)


def write_checkpoint(path, model, optimizer, epoch, updates, monitor, best, best_epoch, generators):
    BASE.atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                         total_updates=updates, monitor=monitor.state_dict(), best=best, best_epoch=best_epoch,
                         rng=BASE.cpu_rng_state(), loader_rng={k: g.get_state() for k, g in generators.items()}), path)


def finish_report(args, complete, summaries):
    by_split = {item["split"]: item for item in summaries}
    fields = ("train", "validation", "test")
    lines = [f"# {VERSION}: full-data feedback-OFF ACT", "", "## Execution",
             f"- Completed epochs: {complete['completed_epochs']} / {args.epochs}",
             f"- Stop reason: {complete['stop_reason']}", f"- Best Val Cosine epoch: {complete['best_epoch']}",
             f"- Model: hidden {args.hidden_dim}, layers {args.num_layers}, heads {args.num_heads}",
             f"- Physical batch: {args.batch_size}; same-batch updates: {args.updates_per_batch}; SAM rho: {args.sam_rho}",
             "- All accepted records were visited in every completed training epoch.", "", "## Selected checkpoint",
             "| Split | Records | Cosine | Top5 | Top10 | Top20 |", "|---|---:|---:|---:|---:|---:|"]
    for split in fields:
        item = by_split[split]
        lines.append(f"| {split} | {item['samples']} | " + " | ".join(f"{item[k]:.6f}" for k in METRICS) + " |")
    gap = {key: by_split["train"][key] - by_split["validation"][key] for key in METRICS}
    lines += ["", "Frozen full-train minus validation: " + json.dumps(gap), "", "## Scope",
              "One trained seed. Canonical-isomeric structure grouping, not a scaffold split.",
              "The NIST split pools have been used in past studies. This is not wholly untouched external validation.",
              "Test did not control early stopping, model selection, or hyperparameters in this run.",
              "No matched full-scale baseline was trained in this run; historical scores are not causal controls."]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    BASE.write_json(args.output / "evaluation_summary.json", dict(version=VERSION, completed_utc=utc(),
                    execution=complete, metrics=summaries, frozen_train_validation_gap=gap))
    with (args.output / "evaluation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["split", "samples", *METRICS, "expected_depth", "stop_depth"]
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)


def run(args):
    audit_record = audit(args)
    device = configure_device(args)
    manifest_path = args.output / "EXPERIMENT.json"
    signature = dict(version=VERSION, source_sha256=source_hashes(), sqlite_sha256=audit_record["sqlite_sha256"],
                     recipe=RECIPE, settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                                             if k not in {"mode", "resume", "capacity_stress_json"}})
    if manifest_path.exists():
        if not args.resume or json.loads(manifest_path.read_text())["signature"] != signature:
            raise ValueError("Run exists or settings/source changed; use --resume with identical settings")
    else:
        BASE.write_json(manifest_path, dict(signature=signature, created_utc=utc(),
                        gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
                        numerics="FP32; TF32 OFF; no AMP", selection="Exact maximum full-validation Cosine",
                        saturation="All four validation metrics stagnant by declared thresholds",
                        production_changed=False))
    model, config = configured_model(args, device)
    datasets, loaders, generators = COMMON.make_loaders(args.prepared_db, args, args.seed)
    optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    monitor = SaturationMonitor(args.patience, args.minimum_epochs, args.cosine_min_delta, args.topk_min_delta)
    checkpoint_path, best_path = args.output / "last_checkpoint.pth", args.output / "best_model.pth"
    durable_best = args.output / "durable_best.pth"
    complete_path, history_path = args.output / "training_complete.json", args.output / "history.csv"
    best, best_epoch, updates, start_epoch = -math.inf, 0, 0, 1
    if args.resume and checkpoint_path.exists() and not complete_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        monitor = SaturationMonitor(args.patience, args.minimum_epochs, args.cosine_min_delta,
                                    args.topk_min_delta, saved["monitor"])
        best, best_epoch, updates, start_epoch = saved["best"], saved["best_epoch"], saved["total_updates"], saved["epoch"] + 1
        BASE.restore_rng(saved["rng"])
        for name, generator in generators.items():
            generator.set_state(saved["loader_rng"][name])
        BASE.atomic_copy(durable_best, best_path)
        with history_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns, rows = reader.fieldnames, [r for r in reader if int(r["epoch"]) < start_epoch]
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        del saved
    elif args.resume and history_path.exists() and not complete_path.exists():
        raise ValueError("No durable resume checkpoint")
    with COMPONENT.installed(RECIPE):
        if not complete_path.exists():
            stop_reason = "validation_saturation" if monitor.saturated() else "max_epochs"
            epoch = start_epoch - 1
            remaining = () if monitor.saturated() else range(start_epoch, args.epochs + 1)
            for epoch in remaining:
                online = train_epoch(model, optimizer, loaders["train"], args, device, epoch, updates)
                updates += online["optimizer_updates"]
                status(args, phase="validation", epoch=epoch, total_optimizer_updates=updates)
                val = BASE.epoch_run(model, loaders["validation"], args, device, "teacher", epoch=epoch)
                diagnostic = BASE.epoch_run(model, loaders["train_diagnostic"], args, device, "teacher", epoch=epoch)
                plateau = monitor.update(epoch, val)
                if val["cosine"] > best:
                    best, best_epoch = val["cosine"], epoch
                    BASE.atomic_save(dict(model_state_dict=model.state_dict(), model_config=config,
                                          source_sha256=BASE.source_hashes(), algorithm_source_sha256=source_hashes(),
                                          epoch=epoch, seed=args.seed, version=VERSION, condition="teacher",
                                          total_optimizer_updates=updates), best_path)
                row = dict(epoch=epoch, total_optimizer_updates=updates, total_forward_backward_passes=2 * updates,
                           best_epoch=best_epoch, saturation_stale_epochs=plateau["stale_epochs"],
                           meaningful_improvements=";".join(plateau["meaningful_improvements"]),
                           **{f"online_train_{k}": v for k, v in online.items()},
                           **{f"validation_{k}": v for k, v in val.items()},
                           **{f"frozen_train_diagnostic_{k}": v for k, v in diagnostic.items()},
                           **{f"diagnostic_gap_{k}": diagnostic[k] - val[k] for k in METRICS},
                           **BASE.memory_guard(args))
                COMMON.append_csv(history_path, row)
                print(f"{VERSION} seed={args.seed} epoch={epoch}/{args.epochs} Val={val['cosine']:.6f} "
                      f"Top5/10/20={val['top5']:.4f}/{val['top10']:.4f}/{val['top20']:.4f} "
                      f"best={best_epoch} stagnant={plateau['stale_epochs']}/{args.patience} updates={updates}", flush=True)
                if epoch % args.checkpoint_interval == 0 or plateau["stop"] or epoch == args.epochs:
                    write_checkpoint(checkpoint_path, model, optimizer, epoch, updates, monitor, best, best_epoch, generators)
                    BASE.atomic_copy(best_path, durable_best)
                status(args, phase="epoch_complete", **row)
                if plateau["stop"]:
                    stop_reason = "validation_saturation"
                    break
            complete = dict(version=VERSION, completed_utc=utc(), completed_epochs=epoch, maximum_epochs=args.epochs,
                            stop_reason=stop_reason, best_epoch=best_epoch, best_validation_cosine=best,
                            optimizer_updates=updates, forward_backward_passes=2 * updates,
                            counts=audit_record["counts"], saturation_state=monitor.state_dict())
            BASE.write_json(complete_path, complete)
        else:
            complete = json.loads(complete_path.read_text())
        del optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        saved = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model_state_dict"])
        epoch = saved["epoch"]
        del saved
        full_train = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=False, num_workers=0)
        summaries = []
        for split, loader in (("train", full_train), ("validation", loaders["validation"]), ("test", loaders["test"])):
            status(args, phase="selected_checkpoint_evaluation", split=split, best_epoch=epoch)
            summaries.append(BASE.evaluate(model, loader, args.output, "feedback_off_act4_sam", args.seed, split, epoch, device))
        finish_report(args, complete, summaries)
        for dataset in datasets.values():
            dataset.close()
    status(args, phase="completed", execution=complete,
           summary_file="evaluation_summary.csv", report_file="REPORT.md")
    print(json.dumps(summaries), flush=True)


def parser():
    p = GEN.parser()
    p.description = __doc__
    p.set_defaults(output=BASE.SCRIPT_ROOT.parent / "paper_runs/full_feedback_off_act4_sam",
                   epochs=200, hidden_dim=768, num_layers=14, num_heads=24,
                   max_samples=0, batch_size=16, grad_accumulation=1, updates_per_batch=4,
                   conditions=["teacher"], patience=10, checkpoint_interval=1,
                   graph_cache_size=2048, graph_cache_max_mib=64., min_free_ram_gb=3.,
                   train_diagnostic_size=2048)
    p.add_argument("--full-mode", choices=["audit", "capacity", "train"], default="train")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--minimum-epochs", type=int, default=20)
    p.add_argument("--cosine-min-delta", type=float, default=.0005)
    p.add_argument("--topk-min-delta", type=float, default=.001)
    p.add_argument("--cuda-memory-fraction", type=float, default=.80)
    p.add_argument("--capacity-stress-json", type=Path)
    return p


def main():
    args = parser().parse_args()
    if not args.prepared_db or args.grad_accumulation != 1 or args.max_samples != 0:
        raise ValueError("Requires full prepared DB, no sampling, and accumulation1 for native SAM")
    if (args.hidden_dim, args.num_layers, args.num_heads) != (768, 14, 24):
        raise ValueError("EXP-V022 preserves the requested 768/14/24 model")
    if args.refinement_steps != 4 or args.fusion_weight != .3 or args.sam_rho <= 0:
        raise ValueError("Requires ACT cap4, fusion0.30, and SAM")
    if not 0 < args.cuda_memory_fraction <= 1 or min(args.epochs, args.batch_size, args.checkpoint_interval) < 1:
        raise ValueError("Invalid compute settings")
    if args.epochs < args.minimum_epochs:
        raise ValueError("Maximum epochs cannot be below the minimum early-stop epoch")
    BASE.validate_update_schedule(args)
    SaturationMonitor(args.patience, args.minimum_epochs, args.cosine_min_delta, args.topk_min_delta)
    args.output = args.output.resolve()
    try:
        if args.full_mode == "audit":
            audit(args)
        elif args.full_mode == "capacity":
            capacity(args)
        else:
            run(args)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        status(args, phase="failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
