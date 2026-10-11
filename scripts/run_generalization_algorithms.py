#!/usr/bin/env python3
"""EXP-V019: fusion-preserving, single-predictor generalization experiments."""

from contextlib import closing, contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel
from torch_geometric.loader import DataLoader

import run_feedback_generalization_experiment as COMMON

BASE = COMMON.BASE
VERSION = "EXP-V019/r01"
ALGORITHMS = ("baseline", "sam", "random_depth", "distill", "cycle_swa")


def source_hashes():
    return {**COMMON.source_hashes(), Path(__file__).name:
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


class TrainTeacher:
    """CPU predictions indexed by training ID; never used during inference."""

    def __init__(self, path, allowed_ids):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        self.ids, self.spectra = saved["ids"], saved["spectra"]
        if self.ids.ndim != 1 or len(self.ids) != len(self.spectra):
            raise ValueError("Invalid teacher tensor shape")
        if set(self.ids.tolist()) != set(map(int, allowed_ids)) or self.ids.unique().numel() != len(self.ids):
            raise ValueError("Teacher must contain exactly the training IDs, no Val/Test")
        if self.spectra.shape != (len(self.ids), 1000) or not torch.isfinite(self.spectra).all():
            raise ValueError("Invalid teacher spectra")
        self.positions = {int(value): index for index, value in enumerate(self.ids)}

    def get(self, sample_ids, device):
        positions = [self.positions[int(value)] for value in sample_ids.detach().cpu().tolist()]
        return self.spectra[positions].to(device)


@contextmanager
def training_depth(model, algorithm, generator):
    original = model.refinement_steps
    depth = original
    if algorithm == "random_depth":
        candidates = torch.tensor([max(1, original - 2), max(1, original - 1), original])
        index = torch.multinomial(torch.tensor([.2, .3, .5]), 1, generator=generator).item()
        depth = int(candidates[index])
    try:
        model.refinement_steps = depth
        yield depth
    finally:
        model.refinement_steps = original


def training_loss(model, batch, args, teacher=None):
    loss, prediction, stats, terms = BASE.loss_terms(model, batch, args, "teacher", True)
    if teacher is not None:
        soft = teacher.get(batch.sample_id, prediction.device)
        distillation = (1 - F.cosine_similarity(prediction, soft, dim=-1)).mean()
        # Replace part of the main target loss, keeping its total weight at one.
        loss = loss + args.distill_weight * (distillation - terms["main_loss"])
        terms = {**terms, "distillation_loss": distillation}
    return loss, prediction, stats, terms


def sam_update(model, optimizer, batch, args, loss_function):
    optimizer.zero_grad(set_to_none=True)
    before = BASE.cpu_rng_state()
    loss, prediction, stats, terms = loss_function(model, batch, args)
    loss.backward()
    after = BASE.cpu_rng_state()
    parameters = [p for p in model.parameters() if p.grad is not None]
    norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in parameters]))
    if not torch.isfinite(loss) or not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite SAM loss or gradient")
    backups = [p.detach().clone() for p in parameters]
    try:
        with torch.no_grad():
            for parameter in parameters:
                parameter.add_(parameter.grad, alpha=args.sam_rho / (float(norm) + 1e-12))
        optimizer.zero_grad(set_to_none=True)
        BASE.restore_rng(before)
        sharp_loss, _, _, _ = loss_function(model, batch, args)
        if not torch.isfinite(sharp_loss):
            raise FloatingPointError("Nonfinite perturbed SAM loss")
        sharp_loss.backward()
        del sharp_loss
    finally:
        with torch.no_grad():
            for parameter, backup in zip(parameters, backups):
                parameter.copy_(backup)
        BASE.restore_rng(after)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.detach(), prediction.detach(), stats, {k: v.detach() for k, v in terms.items()}


def train_epoch(model, loader, args, optimizer, device, algorithm, teacher, generator, epoch):
    model.train()
    totals, count, updates, depth_sum = {}, 0, 0, 0
    replay = BASE.BatchReplayLoader(loader, args.updates_per_batch)
    start = time.monotonic()
    for index, cpu_batch in enumerate(replay):
        if index % 20 == 0:
            BASE.memory_guard(args)
        batch = cpu_batch.to(device)
        with training_depth(model, algorithm, generator) as depth:
            if algorithm == "sam":
                loss, prediction, stats, terms = sam_update(model, optimizer, batch, args, training_loss)
            else:
                optimizer.zero_grad(set_to_none=True)
                loss, prediction, stats, terms = training_loss(model, batch, args, teacher)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        n = batch.num_graphs
        vectors = BASE.metric_vectors(prediction.detach(), batch.y)
        for key, value in vectors.items():
            totals[key] = totals.get(key, 0.) + float(value.sum())
        for key, value in {"loss": loss, "expected_depth": stats["ponder_steps"].mean(), **terms}.items():
            totals[key] = totals.get(key, 0.) + float(value.detach()) * n
        count += n
        depth_sum += depth * n
        updates += 1
        del loss, prediction, stats, terms, vectors
    if not count:
        raise ValueError("Empty train loader")
    return {key: value / count for key, value in totals.items()} | {
        "samples": count, "optimizer_updates": updates, "forward_backward_passes": updates * (2 if algorithm == "sam" else 1),
        "training_depth_cap": depth_sum / count, "seconds": time.monotonic() - start}


def cycle_lr(args, epoch):
    if epoch <= args.swa_start_epoch:
        return args.lr
    phase = (epoch - args.swa_start_epoch - 1) % args.swa_cycle_epochs
    fraction = phase / max(args.swa_cycle_epochs - 1, 1)
    return args.lr * (args.cycle_lr_high + (args.cycle_lr_low - args.cycle_lr_high) * fraction)


def run(args, path, preparation, panel, algorithm, seed, device):
    directory = args.output / f"{algorithm}_seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                if key not in {"resume", "algorithms", "seeds"}}
    signature = {"version": VERSION, "algorithm": algorithm, "seed": seed, "settings": settings,
                 "source_sha256": source_hashes(), "panel": panel,
                 "data_fingerprint": preparation["fingerprint"]}
    if algorithm == "distill":
        if args.teacher_train_predictions is None:
            raise ValueError("Distillation requires train-only teacher predictions")
        signature["teacher_sha256"] = hashlib.sha256(args.teacher_train_predictions.read_bytes()).hexdigest()
    if manifest_path.exists():
        if not args.resume or json.loads(manifest_path.read_text())["signature"] != signature:
            raise ValueError("Existing run or changed signature; use another output")
    else:
        BASE.write_json(manifest_path, {"signature": signature, "created_utc": datetime.now(timezone.utc).isoformat(),
            "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
            "selection": "maximum Validation Cosine, once per epoch for each declared predictor",
            "scope": "single-predictor inference; internal small screen, not full-data efficacy"})
    BASE.set_seed(seed)
    config = BASE.model_config(args)
    model = COMMON.QCGN2oEI_Minimal(**config).to(device)
    datasets, loaders, generators = COMMON.make_loaders(path, args, seed, panel)
    teacher = TrainTeacher(args.teacher_train_predictions, datasets["train"].ids) if algorithm == "distill" else None
    optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    swa = AveragedModel(model, device=torch.device("cpu"), use_buffers=True) if algorithm == "cycle_swa" else None
    depth_generator = torch.Generator().manual_seed(seed + 19019)
    variants = ("cycle_raw", "cycle_swa") if swa is not None else (algorithm,)
    best = {name: -math.inf for name in variants}
    best_epochs = dict.fromkeys(variants, 0)
    for name in variants:
        (directory / name).mkdir(exist_ok=True)
    updates, passes, start_epoch = 0, 0, 1
    checkpoint_path = directory / "last_checkpoint.pth"
    complete_path = directory / "training_complete.json"
    if args.resume and checkpoint_path.exists() and not complete_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        if swa is not None:
            swa.load_state_dict(saved["swa"])
        best, best_epochs = saved["best"], saved["best_epochs"]
        updates, passes, start_epoch = saved["updates"], saved["passes"], saved["epoch"] + 1
        BASE.restore_rng(saved["rng"])
        depth_generator.set_state(saved["depth_rng"])
        for name, generator in generators.items():
            generator.set_state(saved["loader_rng"][name])
        for name in variants:
            BASE.atomic_copy(directory / name / "durable_best.pth", directory / name / "best_model.pth")
            import csv
            history_path = directory / name / "history.csv"
            with history_path.open(newline="", encoding="utf-8") as handle:
                history = [row for row in csv.DictReader(handle) if int(row["epoch"]) < start_epoch]
            with history_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
        del saved
    elif args.resume and not complete_path.exists() and any((directory / name / "history.csv").exists() for name in variants):
        raise ValueError("No durable resume checkpoint")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if not complete_path.exists():
        for epoch in range(start_epoch, args.epochs + 1):
            if swa is not None:
                for group in optimizer.param_groups:
                    group["lr"] = cycle_lr(args, epoch)
            online = train_epoch(model, loaders["train"], args, optimizer, device, algorithm, teacher, depth_generator, epoch)
            updates += online["optimizer_updates"]
            passes += online["forward_backward_passes"]
            if swa is not None and epoch > args.swa_start_epoch and (epoch - args.swa_start_epoch) % args.swa_cycle_epochs == 0:
                swa.update_parameters(model)
            for variant in variants:
                average = swa if variant == "cycle_swa" and int(swa.n_averaged) > 0 else None
                with COMMON.evaluation_weights(model, average):
                    val = BASE.epoch_run(model, loaders["validation"], args, device, "teacher", epoch=epoch)
                    frozen = BASE.epoch_run(model, loaders["train_diagnostic"], args, device, "teacher", epoch=epoch)
                    if val["cosine"] > best[variant]:
                        best[variant], best_epochs[variant] = val["cosine"], epoch
                        BASE.atomic_save({"model_state_dict": model.state_dict(), "model_config": config,
                            "source_sha256": BASE.source_hashes(), "algorithm_source_sha256": source_hashes(),
                            "epoch": epoch, "seed": seed, "condition": "teacher", "variant": variant,
                            "version": VERSION, "total_optimizer_updates": updates}, directory / variant / "best_model.pth")
                COMMON.append_csv(directory / variant / "history.csv", {"epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"], "total_optimizer_updates": updates,
                    "total_forward_backward_passes": passes, "swa_snapshots": int(swa.n_averaged) if swa is not None else 0,
                    **{f"online_train_{k}": v for k, v in online.items()},
                    **{f"validation_{k}": v for k, v in val.items()},
                    **{f"frozen_train_diagnostic_{k}": v for k, v in frozen.items()},
                    **{f"frozen_diagnostic_gap_{k}": frozen[k] - val[k] for k in BASE.METRICS},
                    **BASE.memory_guard(args)})
                print(f"{variant} seed{seed} epoch{epoch}/{args.epochs} Val={val['cosine']:.6f} "
                      f"Top5/10/20={val['top5']:.4f}/{val['top10']:.4f}/{val['top20']:.4f} "
                      f"diagnostic_gap={frozen['cosine'] - val['cosine']:.6f} updates={updates}", flush=True)
            if epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
                BASE.atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "swa": swa.state_dict() if swa is not None else None, "epoch": epoch,
                    "best": best, "best_epochs": best_epochs, "updates": updates, "passes": passes,
                    "rng": BASE.cpu_rng_state(), "depth_rng": depth_generator.get_state(),
                    "loader_rng": {k: gen.get_state() for k, gen in generators.items()}}, checkpoint_path)
                for name in variants:
                    BASE.atomic_copy(directory / name / "best_model.pth", directory / name / "durable_best.pth")
        BASE.write_json(complete_path, {"version": VERSION, "optimizer_updates": updates,
            "forward_backward_passes": passes, "epochs": args.epochs, "best_epochs": best_epochs})
    del optimizer, swa, teacher
    summaries = []
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=False, num_workers=0,
                              generator=torch.Generator().manual_seed(seed + 40000))
    for variant in variants:
        destination = directory / variant
        saved = torch.load(destination / "best_model.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model_state_dict"])
        selected = []
        for split, loader in (("train", train_loader), ("validation", loaders["validation"]), ("test", loaders["test"])):
            if split == "test" and args.defer_test:
                continue
            summary = BASE.evaluate(model, loader, destination, variant, seed, split, saved["epoch"], device)
            summary.update(variant=variant, version=VERSION)
            selected.append(summary)
        BASE.write_json(destination / "metrics.json", {"variant": variant, "seed": seed, "selected_epoch": saved["epoch"],
            "metrics": selected, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None})
        summaries.extend(selected)
    for dataset in datasets.values():
        dataset.close()
    del model, datasets, loaders, train_loader, saved
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summaries


def parser():
    p = COMMON.parser()
    p.description = __doc__
    p.set_defaults(output=BASE.SCRIPT_ROOT.parent / "paper_runs/v019_generalization", epochs=80, patience=0)
    p.add_argument("--algorithms", choices=ALGORITHMS, nargs="+", default=["baseline"])
    p.add_argument("--sam-rho", type=float, default=.02)
    p.add_argument("--distill-weight", type=float, default=.25)
    p.add_argument("--teacher-train-predictions", type=Path)
    p.add_argument("--swa-start-epoch", type=int, default=40)
    p.add_argument("--swa-cycle-epochs", type=int, default=10)
    p.add_argument("--cycle-lr-high", type=float, default=2.)
    p.add_argument("--cycle-lr-low", type=float, default=.2)
    p.add_argument("--defer-test", action="store_true", help="Study orchestration evaluates Test only after all recipes are frozen and trained")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.mode != "train" or not args.prepared_db or not args.panel_json:
        raise ValueError("Research runner requires --mode train --prepared-db --panel-json")
    if args.grad_accumulation != 1 or args.patience != 0:
        raise ValueError("Matched-budget research runner uses accumulation1 and no early stopping")
    BASE.validate_update_schedule(args)
    if not 0 <= args.distill_weight <= 1 or args.sam_rho < 0 or args.swa_cycle_epochs < 2 or args.swa_start_epoch < 0:
        raise ValueError("Invalid algorithm setting")
    if min(args.lr, args.cycle_lr_high, args.cycle_lr_low, args.graph_cache_max_mib, args.batch_size, args.epochs,
           args.checkpoint_interval) <= 0 or args.graph_cache_size < 0:
        raise ValueError("Invalid training/cache budget")
    if args.conditions != ["teacher"] or args.fusion_weight != .3:
        raise ValueError("This experiment preserves 7:3 fusion and the V007 stage teachers")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.algorithms)) != len(args.algorithms):
        raise ValueError("Duplicate seeds/algorithms")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    path = args.prepared_db.resolve()
    preparation = json.loads((path.parent / "dataset_manifest.json").read_text())
    with closing(COMMON.connect(path)) as db:
        if dict(db.execute("SELECT key,value FROM meta")).get("complete") != "1":
            raise ValueError("Prepared database incomplete")
    panel = json.loads(args.panel_json.read_text())["ids"]
    for seed in args.seeds:
        for algorithm in args.algorithms:
            run(args, path, preparation, panel, algorithm, seed, device)


if __name__ == "__main__":
    main()
