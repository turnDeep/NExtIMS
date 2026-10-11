#!/usr/bin/env python3
"""EXP-V018: isolate V007 generalization changes without replacing its predictor."""

from collections import OrderedDict
from contextlib import closing, contextmanager
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch_geometric.loader import DataLoader

import run_spectral_feedback_experiment as BASE
from spectral_feedback_support.data import DiskDataset, connect, prepare_database
from spectral_feedback_support.model import QCGN2oEI_Minimal

VERSION = "EXP-V018/r02"
RECIPES = ("baseline", "dropout", "ema", "plateau")


class GraphCache:
    """Shared CPU LRU, bounded by both entry count and tensor payload bytes."""

    def __init__(self, max_entries, max_bytes):
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self.entries = OrderedDict()
        self.payload_bytes = 0

    def get(self, key):
        if key not in self.entries:
            return None
        self.entries.move_to_end(key)
        return self.entries[key][0].clone()

    def put(self, key, graph):
        tensors = [value for value in graph.values() if isinstance(value, torch.Tensor)]
        if any(value.device.type != "cpu" for value in tensors):
            raise ValueError("Graph cache accepts CPU tensors only")
        size = sum(value.numel() * value.element_size() for value in tensors)
        if self.max_entries <= 0 or size > self.max_bytes:
            return
        if key in self.entries:
            self.payload_bytes -= self.entries.pop(key)[1]
        while self.entries and (len(self.entries) >= self.max_entries or
                               self.payload_bytes + size > self.max_bytes):
            _, (_, evicted_size) = self.entries.popitem(last=False)
            self.payload_bytes -= evicted_size
        self.entries[key] = (graph.clone(), size)
        self.payload_bytes += size


class CachedDataset:
    def __init__(self, dataset, cache):
        self.dataset, self.cache = dataset, cache
        self.ids, self.path = dataset.ids, dataset.path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        key = (str(self.path), int(self.ids[index]))
        graph = self.cache.get(key)
        if graph is None:
            graph = self.dataset[index]
            self.cache.put(key, graph)
        return graph

    def close(self):
        self.dataset.close()


def source_hashes():
    return {**{key.replace("\\", "/"): value for key, value in BASE.source_hashes().items()}, Path(__file__).name:
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def predict(args, device):
    if not args.checkpoint or not args.predict_smiles:
        raise ValueError("predict mode requires --checkpoint and --predict-smiles")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    recorded = {key.replace("\\", "/"): value for key, value in saved["source_sha256"].items()}
    current = {key.replace("\\", "/"): value for key, value in BASE.source_hashes().items()}
    if recorded != current:
        raise ValueError("Checkpoint predictor source differs; use its matching support files")
    model = QCGN2oEI_Minimal(**saved["model_config"]).to(device).eval()
    model.load_state_dict(saved["model_state_dict"])
    from torch_geometric.data import Batch
    with torch.inference_mode():
        prediction, _, _ = BASE.forward(model, Batch.from_data_list(
            [BASE.make_graph(args.predict_smiles)]).to(device))
    values = prediction[0].cpu().numpy()
    values = values / max(float(values.max()), 1e-8) * 100
    args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
    with args.prediction_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mz", "intensity"])
        writer.writerows(enumerate(values.tolist(), start=1))


def recipe_config(args, recipe):
    config = BASE.model_config(args)
    if recipe == "dropout":
        config.update(dropout=args.regularized_dropout,
                      refinement_dropout=args.regularized_dropout)
    return config


def make_scheduler(optimizer, args, recipe):
    if recipe != "plateau":
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.lr_patience,
        threshold=args.lr_min_delta, threshold_mode="abs", cooldown=0,
        min_lr=args.lr * 0.1)


def make_ema(model, half_life):
    # CPU shadow plus temporary copying keeps a second predictor out of VRAM.
    return AveragedModel(model, device=torch.device("cpu"),
                         multi_avg_fn=get_ema_multi_avg_fn(0.5 ** (1.0 / half_life)),
                         use_buffers=True)


@contextmanager
def evaluation_weights(model, ema):
    if ema is None:
        yield model
        return
    raw = {key: value.detach().to("cpu", copy=True)
           for key, value in model.state_dict().items()}
    was_training = model.training
    try:
        model.load_state_dict(ema.module.state_dict())
        yield model
    finally:
        model.load_state_dict(raw)
        model.train(was_training)


def make_loaders(path, args, seed, panel=None):
    datasets = {split: DiskDataset(path, split) for split in ("train", "validation", "test")}
    if panel is not None:
        for split, dataset in datasets.items():
            allowed = set(map(int, dataset.ids))
            requested = list(map(int, panel[split]))
            if len(requested) != len(set(requested)) or not set(requested) <= allowed:
                raise ValueError(f"Invalid IDs or wrong split in panel: {split}")
            dataset.ids = np.asarray(requested, dtype=np.int64)
    if any(len(dataset) == 0 for dataset in datasets.values()):
        raise ValueError("Empty train/validation/test split")
    generators = {split: torch.Generator().manual_seed(seed + offset)
                  for split, offset in (("train", 0), ("validation", 10000),
                                        ("test", 20000), ("train_diagnostic", 30000))}
    diagnostic = DiskDataset(path, "train")
    ids = datasets["train"].ids
    if args.train_diagnostic_size > 0 and len(ids) > args.train_diagnostic_size:
        rng = np.random.default_rng(args.split_seed + 999)
        ids = np.sort(rng.choice(ids, size=args.train_diagnostic_size, replace=False))
    diagnostic.ids = np.asarray(ids, dtype=np.int64)
    datasets["train_diagnostic"] = diagnostic
    if args.graph_cache_size:
        cache = GraphCache(args.graph_cache_size, int(args.graph_cache_max_mib * 2**20))
        datasets = {split: CachedDataset(dataset, cache) for split, dataset in datasets.items()}
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size,
               shuffle=split == "train", num_workers=0, pin_memory=False,
               generator=generators[split]) for split, dataset in datasets.items()}
    return datasets, loaders, generators


def append_csv(path, row):
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_recipe(args, path, preparation, recipe, seed, device, panel=None):
    directory = args.output / "runs" / f"{recipe}_seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    config = recipe_config(args, recipe)
    signature = {"version": VERSION, "recipe": recipe, "seed": seed,
                 "model_config": config, "source_sha256": source_hashes(),
                 "dataset_fingerprint": preparation["fingerprint"],
                 "panel": panel,
                 "settings": {key: value for key, value in vars(args).items()
                              if key not in {"resume", "recipes", "seeds", "mode"}}}
    signature = json.loads(json.dumps(signature, default=str))
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not args.resume or old["signature"] != signature:
            raise ValueError("Run exists or signature changed; use a new output directory")
    else:
        BASE.write_json(manifest_path, {"signature": signature,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "selection": "maximum validation Cosine, one candidate per epoch",
            "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
            "torch": torch.__version__, "numerics": "FP32; TF32 OFF; workers0",
            "graph_cache": {"enabled": args.graph_cache_size > 0, "max_entries": args.graph_cache_size,
                            "max_tensor_payload_mib": args.graph_cache_max_mib,
                            "limit": "Python/RDKit overhead is additional; global RAM guard retained"}})
    BASE.set_seed(seed)
    model = QCGN2oEI_Minimal(**config).to(device)
    datasets, loaders, generators = make_loaders(path, args, seed, panel)
    optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args, recipe)
    ema = make_ema(model, args.ema_half_life_epochs) if recipe == "ema" else None
    start, best, best_epoch, updates, stale = 1, -math.inf, 0, 0, 0
    last_path = directory / "last_checkpoint.pth"
    history_path = directory / "history.csv"
    completed_path = directory / "training_complete.json"
    if args.resume and last_path.exists() and not completed_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        if ema is not None:
            ema.load_state_dict(saved["ema"])
        if scheduler is not None:
            scheduler.load_state_dict(saved["scheduler"])
        start, best, best_epoch = saved["epoch"] + 1, saved["best"], saved["best_epoch"]
        updates, stale = saved["total_optimizer_updates"], saved["stale"]
        BASE.restore_rng(saved["rng"])
        for key, generator in generators.items():
            generator.set_state(saved["loader_rng"][key])
        durable = torch.load(directory / "resume_best_model.pth", map_location="cpu", weights_only=False)
        if durable["epoch"] != best_epoch:
            raise ValueError("Inconsistent durable resume files; use a new output")
        del durable, saved
        BASE.atomic_copy(directory / "resume_best_model.pth", directory / "best_model.pth")
        with history_path.open(newline="", encoding="utf-8") as handle:
            history = [row for row in csv.DictReader(handle) if int(row["epoch"]) < start]
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    elif args.resume and history_path.exists() and not completed_path.exists():
        raise ValueError("No durable checkpoint; use a new output")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if not completed_path.exists():
        for epoch in range(start, args.epochs + 1):
            print(f"{recipe} seed={seed} epoch={epoch}/{args.epochs}", flush=True)
            online = BASE.epoch_run(model, loaders["train"], args, device, "teacher", optimizer, epoch)
            updates += online["optimizer_updates"]
            if ema is not None:
                if epoch <= args.ema_start_epoch:
                    ema.n_averaged.zero_()
                ema.update_parameters(model)
            used_lr = optimizer.param_groups[0]["lr"]
            with evaluation_weights(model, ema):
                val = BASE.epoch_run(model, loaders["validation"], args, device, "teacher", epoch=epoch)
                frozen = BASE.epoch_run(model, loaders["train_diagnostic"], args, device, "teacher", epoch=epoch)
                payload = {"model_state_dict": model.state_dict(), "model_config": config,
                           "epoch": epoch, "condition": "teacher", "recipe": recipe, "seed": seed,
                           "validation": val, "source_sha256": BASE.source_hashes(),
                           "generalization_source_sha256": source_hashes(), "version": VERSION,
                           "total_optimizer_updates": updates,
                           "training_schedule": BASE.training_schedule(args),
                           "evaluation_weight_kind": "epoch_EMA" if ema is not None else "raw"}
                if val["cosine"] > best:
                    best, best_epoch, stale = val["cosine"], epoch, 0
                    BASE.atomic_save(payload, directory / "best_model.pth")
                else:
                    stale += 1
                del payload
            if scheduler is not None:
                scheduler.step(val["cosine"])
            row = {"epoch": epoch, "lr": used_lr, "next_lr": optimizer.param_groups[0]["lr"],
                   "total_optimizer_updates": updates, "best_epoch": best_epoch,
                   **{f"online_train_{key}": value for key, value in online.items()},
                   **{f"validation_{key}": value for key, value in val.items()},
                   **{f"frozen_train_diagnostic_{key}": value for key, value in frozen.items()},
                   **{f"frozen_diagnostic_gap_{key}": frozen[key] - val[key] for key in BASE.METRICS},
                   "ema_snapshots": int(ema.n_averaged) if ema is not None else 0,
                   **BASE.memory_guard(args)}
            append_csv(history_path, row)
            stopping = args.patience > 0 and stale >= args.patience
            if epoch % args.checkpoint_interval == 0 or epoch == args.epochs or stopping:
                raw = {"model_state_dict": model.state_dict(), "model_config": config, "epoch": epoch,
                       "optimizer": optimizer.state_dict(), "best": best, "best_epoch": best_epoch,
                       "stale": stale, "total_optimizer_updates": updates, "rng": BASE.cpu_rng_state(),
                       "loader_rng": {key: gen.get_state() for key, gen in generators.items()},
                       "ema": ema.state_dict() if ema is not None else None,
                       "scheduler": scheduler.state_dict() if scheduler is not None else None}
                BASE.atomic_save(raw, last_path)
                del raw
                BASE.atomic_copy(directory / "best_model.pth", directory / "resume_best_model.pth")
            print(f"Val={val['cosine']:.6f} Top5/10/20={val['top5']:.4f}/{val['top10']:.4f}/{val['top20']:.4f} "
                  f"frozen diagnostic gap={frozen['cosine']-val['cosine']:.6f} updates={updates}", flush=True)
            if stopping:
                break
        BASE.write_json(completed_path, {"version": VERSION, "best_epoch": best_epoch,
            "best_validation_cosine": best, "last_epoch": epoch, "total_optimizer_updates": updates})
    del optimizer, ema, scheduler
    saved = torch.load(directory / "best_model.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    summaries = []
    # Evaluate every training record once, without replay or Dropout, at the selected weights.
    full_train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=False,
                                  num_workers=0, generator=torch.Generator().manual_seed(seed + 40000))
    for split, loader in (("train", full_train_loader), ("validation", loaders["validation"]),
                          ("test", loaders["test"])):
        summary = BASE.evaluate(model, loader, directory, recipe, seed, split, saved["epoch"], device)
        summary.update(recipe=recipe, version=VERSION, selected_optimizer_updates=saved["total_optimizer_updates"])
        BASE.write_json(directory / "evaluation" / f"{split}_summary.json", summary)
        summaries.append(summary)
    BASE.write_json(directory / "metrics.json", {"recipe": recipe, "seed": seed,
        "selected_epoch": saved["epoch"], "metrics": summaries,
        "graph_cache_entries": len(datasets["train"].cache.entries) if args.graph_cache_size else 0,
        "graph_cache_payload_mib": datasets["train"].cache.payload_bytes / 2**20 if args.graph_cache_size else 0,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None})
    for dataset in datasets.values():
        dataset.close()
    del model, saved, datasets, loaders, full_train_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summaries


def parser():
    p = BASE.parser()
    p.description = __doc__
    p.set_defaults(output=BASE.SCRIPT_ROOT.parent / "paper_runs" / "v018_generalization",
                   conditions=["teacher"], max_samples=0, grad_accumulation=1, updates_per_batch=4)
    p.add_argument("--recipes", choices=RECIPES, nargs="+", default=["baseline"])
    p.add_argument("--prepared-db", type=Path, help="Reuse a complete indexed dataset without rewriting it")
    p.add_argument("--panel-json", type=Path, help="Research-only fixed sample IDs within existing splits")
    p.add_argument("--regularized-dropout", type=float, default=0.2)
    p.add_argument("--ema-start-epoch", type=int, default=10)
    p.add_argument("--ema-half-life-epochs", type=float, default=8.)
    p.add_argument("--lr-patience", type=int, default=5)
    p.add_argument("--lr-min-delta", type=float, default=0.001)
    p.add_argument("--train-diagnostic-size", type=int, default=1024,
                   help="Fixed train subset per epoch; 0=all. Final best-model Train always uses all.")
    p.add_argument("--graph-cache-size", type=int, default=0,
                   help="Optional shared CPU LRU entry cap; 0=OFF (default, suitable for full data).")
    p.add_argument("--graph-cache-max-mib", type=float, default=256.,
                   help="Tensor payload cap for the optional cache; Python overhead is additional.")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    BASE.validate_update_schedule(args)
    if args.conditions != ["teacher"]:
        raise ValueError("This comparison maintains V007 stage teachers; use --conditions teacher")
    if len(set(args.recipes)) != len(args.recipes) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate recipes or seeds")
    if not 0 <= args.regularized_dropout < 1 or args.ema_half_life_epochs <= 0:
        raise ValueError("Invalid Dropout or EMA half-life")
    if args.ema_start_epoch < 1 or args.lr_patience < 0 or args.lr_min_delta < 0 or args.train_diagnostic_size < 0:
        raise ValueError("Invalid scheduler/diagnostic settings")
    if args.graph_cache_size < 0 or not math.isfinite(args.graph_cache_max_mib) or args.graph_cache_max_mib <= 0:
        raise ValueError("Invalid CPU graph cache bounds")
    if args.epochs < 1 or args.checkpoint_interval < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError("Invalid training budget")
    if args.hidden_dim % args.num_heads or min(args.num_layers, args.refinement_steps) < 1:
        raise ValueError("Invalid model dimensions")
    if not 0 <= args.fusion_weight <= 1 or min(args.stage_cosine_weight, args.stage_topk_weight,
            args.neutral_loss_weight, args.act_loss_weight, args.weight_decay,
            args.patience, args.min_free_ram_gb, args.max_samples) < 0:
        raise ValueError("Invalid fusion/loss/optimizer settings")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    if args.mode == "predict":
        predict(args, device)
        return
    if args.prepared_db:
        path = args.prepared_db.resolve()
        preparation = json.loads((path.parent / "dataset_manifest.json").read_text(encoding="utf-8"))
        with closing(connect(path)) as db:
            if dict(db.execute("SELECT key, value FROM meta")).get("complete") != "1":
                raise ValueError("Incomplete prepared dataset")
        if args.max_samples != 0:
            raise ValueError("Prepared dataset uses its recorded selection; specify --max-samples 0")
    else:
        path, preparation = prepare_database(args.output / "preparation", args.data_dir, args.msp,
            args.mol_dir, args.max_samples, args.split_seed, args.val_fraction, args.test_fraction)
    panel = json.loads(args.panel_json.read_text(encoding="utf-8"))["ids"] if args.panel_json else None
    BASE.write_json(args.output / "preparation_reference.json", {"path": path, "manifest": preparation,
                                                                "panel": args.panel_json})
    if args.mode == "prepare":
        return
    summaries = []
    for seed in args.seeds:
        for recipe in args.recipes:
            summaries.extend(train_recipe(args, path, preparation, recipe, seed, device, panel))
    with (args.output / "evaluation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Completed EXP-V018: {args.output}", flush=True)


if __name__ == "__main__":
    main()
