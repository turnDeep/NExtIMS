#!/usr/bin/env python3
"""
NExtIMS v1.0: Minimal GNN Training Script
Trains QCGN2oEI_Minimal model on NIST17 EI-MS dataset.
This version supports:
- Stable ACT Recurrent Spectral Refinement
- LTI-style recurrent stabilization
- Loop embedding
- ACT ponder loss
- Optional intermediate spectra auxiliary loss
- Fixed-loop evaluation
"""
import os
import sys
import argparse
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch_geometric.data import Batch
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.qcgn2oei_minimal import QCGN2oEI_Minimal
from src.training.losses import cosine_similarity_loss
from src.data.graph_generator_minimal import GRAPH_FEATURE_DIM, MinimalGraphGenerator
from src.data.nist_dataset import parse_msp_file, peaks_to_spectrum
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
class NISTGraphDataset(Dataset):
    """
    NIST EI-MS Dataset for PyG graphs.
    Loads NIST MSP data and converts molecules to PyG graphs on-the-fly.
    """
    def __init__(
        self,
        msp_path: str,
        bde_cache_path: Optional[str] = None,
        bondnet_model: Optional[str] = None,
        mol_files_dir: str = "data/mol_files",
        use_bde: bool = True,
        use_mass_features: bool = True,
        use_formula_features: bool = True,
        use_fragment_candidates: bool = False,
        fragment_mask_window: int = 2,
        max_samples: int = 0,
        min_mz: int = 1,
        max_mz: int = 1000,
    ):
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.graph_gen = MinimalGraphGenerator(
            bde_cache_path=bde_cache_path,
            use_bde_calculator=(bondnet_model is not None),
            bondnet_model=bondnet_model,
            use_bde=use_bde,
            use_mass_features=use_mass_features,
            use_formula_features=use_formula_features,
            use_fragment_candidates=use_fragment_candidates,
            fragment_output_dim=max_mz - min_mz + 1,
            fragment_mask_window=fragment_mask_window,
            default_bde=85.0,
        )
        logger.info(f"Loading NIST data from {msp_path}")
        entries = parse_msp_file(
            msp_path,
            mol_files_dir=mol_files_dir,
            max_entries=max_samples,
        )
        if max_samples > 0:
            entries = entries[:max_samples]
        logger.info(f"Loaded {len(entries):,} entries")
        self.data = []
        invalid_count = 0
        for entry in tqdm(entries, desc="Processing entries"):
            if "smiles" not in entry or "peaks" not in entry:
                invalid_count += 1
                continue
            smiles = entry["smiles"]
            peaks = entry["peaks"]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid_count += 1
                continue
            spectrum = peaks_to_spectrum(peaks, min_mz=min_mz, max_mz=max_mz)
            self.data.append(
                {
                    "smiles": smiles,
                    "spectrum": spectrum,
                    "name": entry.get("name", "Unknown"),
                    "mol": mol,
                }
            )
        logger.info(f"Valid entries: {len(self.data):,}")
        logger.info(f"Invalid entries: {invalid_count:,}")
        self.graph_gen.print_stats()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            metadata = {}
            if "max_train_similarity" in item:
                metadata["max_train_similarity"] = torch.tensor(
                    [item["max_train_similarity"]],
                    dtype=torch.float32,
                )
            graph = self.graph_gen.mol_to_graph(
                mol=item["mol"],
                smiles=item["smiles"],
                spectrum=item["spectrum"],
                compound_name=item["name"],
                **metadata,
            )
            return graph
        except ValueError:
            return None
def collate_pyg_graphs(batch_list):
    """
    Collate PyG graphs into a batch.
    """
    batch_list = [item for item in batch_list if item is not None]
    if len(batch_list) == 0:
        return None
    return Batch.from_data_list(batch_list)


class SpectrumAugmentedDataset(Dataset):
    """
    Applies light target-spectrum augmentation to the training subset only.
    The graph itself is cloned, so validation and cached dataset entries stay unchanged.
    """

    def __init__(
        self,
        dataset: Dataset,
        intensity_jitter: float = 0.0,
        peak_dropout: float = 0.0,
        mz_spread: float = 0.0,
    ):
        self.dataset = dataset
        self.intensity_jitter = max(float(intensity_jitter), 0.0)
        self.peak_dropout = min(max(float(peak_dropout), 0.0), 1.0)
        self.mz_spread = min(max(float(mz_spread), 0.0), 1.0)

    def __len__(self):
        return len(self.dataset)

    def _augment_spectrum(self, spectrum: torch.Tensor) -> torch.Tensor:
        original_shape = spectrum.shape
        y = spectrum.view(-1).clone()
        positive = y > 0

        if self.intensity_jitter > 0 and positive.any():
            noise = torch.randn_like(y) * self.intensity_jitter
            y = torch.where(positive, y * (1.0 + noise), y)
            y = torch.clamp(y, min=0.0)

        if self.peak_dropout > 0 and positive.any():
            max_intensity = torch.max(y)
            minor_peak = positive & (y < 0.25 * max_intensity)
            keep = torch.rand_like(y) >= self.peak_dropout
            y = torch.where(minor_peak & ~keep, torch.zeros_like(y), y)

        if self.mz_spread > 0 and positive.any():
            spread = torch.zeros_like(y)
            spread[1:] += y[:-1] * (self.mz_spread * 0.5)
            spread[:-1] += y[1:] * (self.mz_spread * 0.5)
            y = y * (1.0 - self.mz_spread) + spread

        max_intensity = torch.max(y)
        if max_intensity > 0:
            y = y / max_intensity
        return y.view(original_shape)

    def __getitem__(self, idx):
        graph = self.dataset[idx]
        if graph is None or not hasattr(graph, "y"):
            return graph
        graph = graph.clone()
        graph.y = self._augment_spectrum(graph.y)
        return graph


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fingerprint_generator(radius: int, n_bits: int):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def _morgan_fingerprints(
    dataset: NISTGraphDataset,
    indices: Sequence[int],
    radius: int,
    n_bits: int,
):
    fpgen = _fingerprint_generator(radius=radius, n_bits=n_bits)
    fps = []
    valid_indices = []
    for idx in indices:
        mol = dataset.data[idx]["mol"]
        try:
            fps.append(fpgen.GetFingerprint(mol))
            valid_indices.append(idx)
        except Exception:
            continue
    return valid_indices, fps


def annotate_validation_similarity(
    dataset: NISTGraphDataset,
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    radius: int,
    n_bits: int,
    max_train_refs: int,
    seed: int,
) -> Dict[str, int]:
    """
    Add max_train_similarity to validation entries using Morgan/Tanimoto.
    This is diagnostic metadata only and is not fed to the model.
    """
    train_indices = list(train_indices)
    if max_train_refs > 0 and len(train_indices) > max_train_refs:
        rng = random.Random(seed)
        train_indices = rng.sample(train_indices, max_train_refs)
        logger.info(
            "Similarity bucket annotation uses %d sampled train references.",
            len(train_indices),
        )

    _, train_fps = _morgan_fingerprints(
        dataset,
        train_indices,
        radius=radius,
        n_bits=n_bits,
    )
    if not train_fps:
        logger.warning("No train fingerprints were available for similarity buckets.")
        return {}

    bucket_counts: Counter = Counter()
    fpgen = _fingerprint_generator(radius=radius, n_bits=n_bits)
    for idx in val_indices:
        item = dataset.data[idx]
        try:
            val_fp = fpgen.GetFingerprint(item["mol"])
            sims = DataStructs.BulkTanimotoSimilarity(val_fp, train_fps)
            max_sim = float(max(sims)) if sims else 0.0
        except Exception:
            max_sim = 0.0
        item["max_train_similarity"] = max_sim
        if max_sim >= 0.8:
            bucket_counts["near"] += 1
        elif max_sim >= 0.6:
            bucket_counts["mid"] += 1
        else:
            bucket_counts["far"] += 1
    return dict(bucket_counts)


def build_similarity_balanced_sampler(
    dataset: NISTGraphDataset,
    train_indices: Sequence[int],
    radius: int,
    n_bits: int,
    threshold: float,
    exact_max_samples: int,
    strength: float,
    seed: int,
) -> Optional[WeightedRandomSampler]:
    """
    Build a cluster-balanced sampler from Morgan fingerprints.
    For small/medium subsets it uses exact Butina clustering. For larger sets it
    falls back to deterministic on-bit buckets to avoid O(n^2) memory/time.
    """
    train_indices = list(train_indices)
    valid_indices, fps = _morgan_fingerprints(
        dataset,
        train_indices,
        radius=radius,
        n_bits=n_bits,
    )
    if len(fps) != len(train_indices):
        logger.warning(
            "Similarity sampler skipped: fingerprint count mismatch (%d/%d).",
            len(fps),
            len(train_indices),
        )
        return None

    cluster_ids: List[int] = [0] * len(fps)
    if len(fps) <= exact_max_samples:
        distances = []
        for i in range(1, len(fps)):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            distances.extend(1.0 - sim for sim in sims)
        clusters = Butina.ClusterData(
            distances,
            len(fps),
            distThresh=1.0 - threshold,
            isDistData=True,
        )
        for cluster_idx, members in enumerate(clusters):
            for member in members:
                cluster_ids[member] = cluster_idx
        logger.info(
            "Similarity-balanced sampler: exact Butina clusters=%d threshold=%.2f",
            len(clusters),
            threshold,
        )
    else:
        buckets: Dict[Tuple[int, ...], int] = {}
        for local_idx, fp in enumerate(fps):
            on_bits = tuple(sorted(int(bit) // 64 for bit in fp.GetOnBits())[:12])
            if on_bits not in buckets:
                buckets[on_bits] = len(buckets)
            cluster_ids[local_idx] = buckets[on_bits]
        logger.info(
            "Similarity-balanced sampler: fallback fingerprint buckets=%d "
            "(exact clustering skipped for %d samples)",
            len(buckets),
            len(fps),
        )

    cluster_sizes = Counter(cluster_ids)
    strength = max(float(strength), 0.0)
    weights = [
        float(cluster_sizes[cluster_id]) ** (-strength)
        for cluster_id in cluster_ids
    ]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    logger.info(
        "Similarity-balanced sampler active: min_cluster=%d max_cluster=%d",
        min(cluster_sizes.values()),
        max(cluster_sizes.values()),
    )
    return sampler


def topk_recall_batch(
    pred_spectrum: torch.Tensor,
    target_spectrum: torch.Tensor,
    k: int,
) -> torch.Tensor:
    pred_topk = torch.topk(pred_spectrum, k=k, dim=-1).indices
    target_topk = torch.topk(target_spectrum, k=k, dim=-1).indices
    hits = (pred_topk.unsqueeze(-1) == target_topk.unsqueeze(1)).any(dim=1)
    return hits.float().mean(dim=-1)


def weighted_cosine_similarity_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mz_power: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Cosine loss after up-weighting high m/z diagnostic bins."""
    output_dim = predicted.size(-1)
    mz = torch.arange(
        1,
        output_dim + 1,
        dtype=predicted.dtype,
        device=predicted.device,
    )
    weights = (mz / float(output_dim)).pow(float(mz_power)).unsqueeze(0)
    pred_norm = F.normalize(predicted * weights, p=2, dim=-1, eps=eps)
    target_norm = F.normalize(target * weights, p=2, dim=-1, eps=eps)
    return 1.0 - (pred_norm * target_norm).sum(dim=-1).mean()


def build_neutral_loss_target(
    target_spectrum: torch.Tensor,
    molecular_ion_mz: torch.Tensor,
) -> torch.Tensor:
    """
    Convert m/z target bins to neutral-loss bins.

    Neutral-loss bin 0 means M - m/z = 0, bin 60 means loss of 60 Da, etc.
    """
    output_dim = target_spectrum.size(-1)
    device = target_spectrum.device
    mz = torch.arange(1, output_dim + 1, dtype=torch.long, device=device).view(1, -1)
    parent_mz = molecular_ion_mz.view(-1, 1).to(device=device, dtype=torch.long)
    loss_idx = parent_mz - mz
    valid = (loss_idx >= 0) & (loss_idx < output_dim)
    safe_idx = loss_idx.clamp(min=0, max=output_dim - 1)
    neutral_target = torch.zeros_like(target_spectrum)
    neutral_target.scatter_add_(1, safe_idx, target_spectrum * valid.to(target_spectrum.dtype))
    max_intensity = neutral_target.max(dim=-1, keepdim=True).values
    neutral_target = torch.where(
        max_intensity > 0,
        neutral_target / max_intensity.clamp_min(1e-8),
        neutral_target,
    )
    return neutral_target


def neutral_loss_to_mz_spectrum(
    neutral_loss_spectrum: torch.Tensor,
    molecular_ion_mz: torch.Tensor,
) -> torch.Tensor:
    """Map a neutral-loss prediction back to m/z bins for final spectrum scoring."""
    output_dim = neutral_loss_spectrum.size(-1)
    device = neutral_loss_spectrum.device
    loss_idx = torch.arange(0, output_dim, dtype=torch.long, device=device).view(1, -1)
    parent_mz = molecular_ion_mz.view(-1, 1).to(device=device, dtype=torch.long)
    mz_idx = parent_mz - loss_idx - 1
    valid = (mz_idx >= 0) & (mz_idx < output_dim)
    safe_idx = mz_idx.clamp(min=0, max=output_dim - 1)
    mz_spectrum = torch.zeros_like(neutral_loss_spectrum)
    mz_spectrum.scatter_add_(
        1,
        safe_idx,
        neutral_loss_spectrum * valid.to(neutral_loss_spectrum.dtype),
    )
    return mz_spectrum


def fragment_candidate_loss(
    pred_spectrum: torch.Tensor,
    fragment_candidate_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Penalize probability mass placed outside structure-derived candidate bins."""
    if fragment_candidate_mask is None:
        return torch.zeros((), device=pred_spectrum.device)
    mask = fragment_candidate_mask.to(device=pred_spectrum.device, dtype=pred_spectrum.dtype)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    if mask.size(0) != pred_spectrum.size(0):
        mask = mask.view(pred_spectrum.size(0), -1)
    mask = mask.clamp(0.0, 1.0)
    return (pred_spectrum * (1.0 - mask)).sum(dim=-1).mean()


def parse_model_outputs(
    outputs,
    return_intermediate: bool,
    return_act_stats: bool,
    return_aux_outputs: bool,
):
    """Unpack model outputs while preserving the legacy forward API."""
    aux_outputs = {}
    if return_aux_outputs:
        if return_intermediate and return_act_stats:
            pred_spectrum, intermediate_spectra, act_stats, aux_outputs = outputs
        elif return_intermediate:
            pred_spectrum, intermediate_spectra, aux_outputs = outputs
            act_stats = None
        elif return_act_stats:
            pred_spectrum, act_stats, aux_outputs = outputs
            intermediate_spectra = []
        else:
            pred_spectrum, aux_outputs = outputs
            intermediate_spectra = []
            act_stats = None
    else:
        if return_intermediate and return_act_stats:
            pred_spectrum, intermediate_spectra, act_stats = outputs
        elif return_intermediate:
            pred_spectrum, intermediate_spectra = outputs
            act_stats = None
        elif return_act_stats:
            pred_spectrum, act_stats = outputs
            intermediate_spectra = []
        else:
            pred_spectrum = outputs
            intermediate_spectra = []
            act_stats = None
    return pred_spectrum, intermediate_spectra, act_stats, aux_outputs


def apply_auxiliary_prediction_losses(
    pred_spectrum: torch.Tensor,
    aux_outputs: Dict[str, torch.Tensor],
    batch,
    args,
):
    """
    Apply optional prediction-space constraints and return the spectrum used for
    main scoring plus the additional loss term.
    """
    extra_loss = torch.zeros((), device=pred_spectrum.device)
    scoring_spectrum = pred_spectrum

    neutral_loss_spectrum = aux_outputs.get("neutral_loss_spectrum")
    if neutral_loss_spectrum is not None:
        molecular_ion_mz = getattr(batch, "molecular_ion_mz", None)
        if molecular_ion_mz is not None:
            neutral_target = build_neutral_loss_target(batch.y, molecular_ion_mz)
            extra_loss = extra_loss + args.neutral_loss_weight * cosine_similarity_loss(
                neutral_loss_spectrum,
                neutral_target,
            )
            if args.neutral_loss_fusion_weight > 0:
                neutral_mz_spectrum = neutral_loss_to_mz_spectrum(
                    neutral_loss_spectrum,
                    molecular_ion_mz,
                )
                fusion = min(max(float(args.neutral_loss_fusion_weight), 0.0), 1.0)
                scoring_spectrum = (1.0 - fusion) * scoring_spectrum + fusion * neutral_mz_spectrum

    if args.weighted_cosine_loss_weight > 0:
        extra_loss = extra_loss + args.weighted_cosine_loss_weight * weighted_cosine_similarity_loss(
            scoring_spectrum,
            batch.y,
            mz_power=args.weighted_cosine_mz_power,
        )

    if args.fragment_candidate_loss_weight > 0:
        extra_loss = extra_loss + args.fragment_candidate_loss_weight * fragment_candidate_loss(
            scoring_spectrum,
            getattr(batch, "fragment_candidate_mask", None),
        )

    return scoring_spectrum, extra_loss
def _forward_and_loss(
    model: nn.Module,
    batch,
    args,
    training: bool = True,
):
    """
    Shared forward/loss helper for train and evaluation.
    Returns:
        loss:
            Total loss used for optimization/evaluation.
        main_loss:
            Main cosine similarity loss.
        act_loss:
            ACT ponder cost.
        aux_loss:
            Intermediate spectra auxiliary loss.
        pred_spectrum:
            Predicted spectrum.
        act_stats:
            ACT statistics or None.
    """
    use_act_stats = bool(args.use_recurrent_refinement and args.use_act)
    use_intermediate = bool(
        args.use_recurrent_refinement and args.use_intermediate_loss and training
    )
    return_aux_outputs = bool(args.use_neutral_loss_head)
    graph_features = getattr(batch, "graph_features", None)
    if use_intermediate:
        # Intermediate supervision uses fixed-loop mode.
        if use_act_stats:
            outputs = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
                graph_features=graph_features,
                refinement_steps=args.refinement_steps,
                return_intermediate=True,
                return_act_stats=True,
                return_aux_outputs=return_aux_outputs,
                force_no_act=True,
            )
        else:
            outputs = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
                graph_features=graph_features,
                refinement_steps=args.refinement_steps,
                return_intermediate=True,
                return_act_stats=False,
                return_aux_outputs=return_aux_outputs,
                force_no_act=True,
            )
        pred_spectrum, intermediate_spectra, act_stats, aux_outputs = parse_model_outputs(
            outputs,
            return_intermediate=True,
            return_act_stats=use_act_stats,
            return_aux_outputs=return_aux_outputs,
        )
        pred_spectrum, extra_loss = apply_auxiliary_prediction_losses(
            pred_spectrum,
            aux_outputs,
            batch,
            args,
        )
        main_loss = cosine_similarity_loss(pred_spectrum, batch.y)
        if len(intermediate_spectra) > 0:
            aux_loss = torch.zeros((), device=batch.y.device)
            for spectrum_t in intermediate_spectra:
                aux_loss = aux_loss + cosine_similarity_loss(spectrum_t, batch.y)
            aux_loss = aux_loss / len(intermediate_spectra)
        else:
            aux_loss = torch.zeros((), device=batch.y.device)
        act_loss = (
            act_stats["act_loss"]
            if act_stats is not None and "act_loss" in act_stats
            else torch.zeros((), device=batch.y.device)
        )
        loss = (
            main_loss
            + args.intermediate_loss_weight * aux_loss
            + args.act_loss_weight * act_loss
            + extra_loss
        )
        return loss, main_loss, act_loss, aux_loss, pred_spectrum, act_stats
    if use_act_stats:
        outputs = model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch,
            graph_features=graph_features,
            refinement_steps=args.refinement_steps,
            return_act_stats=True,
            return_aux_outputs=return_aux_outputs,
            force_no_act=args.force_no_act_eval and not training,
        )
        pred_spectrum, _, act_stats, aux_outputs = parse_model_outputs(
            outputs,
            return_intermediate=False,
            return_act_stats=True,
            return_aux_outputs=return_aux_outputs,
        )
        pred_spectrum, extra_loss = apply_auxiliary_prediction_losses(
            pred_spectrum,
            aux_outputs,
            batch,
            args,
        )
        main_loss = cosine_similarity_loss(pred_spectrum, batch.y)
        act_loss = act_stats["act_loss"]
        aux_loss = torch.zeros((), device=batch.y.device)
        loss = (
            main_loss
            + args.act_loss_weight * act_loss
            + extra_loss
        )
        return loss, main_loss, act_loss, aux_loss, pred_spectrum, act_stats
    outputs = model(
        x=batch.x,
        edge_index=batch.edge_index,
        edge_attr=batch.edge_attr,
        batch=batch.batch,
        graph_features=graph_features,
        refinement_steps=args.refinement_steps if args.use_recurrent_refinement else None,
        return_aux_outputs=return_aux_outputs,
        force_no_act=args.force_no_act_eval and not training,
    )
    pred_spectrum, _, _, aux_outputs = parse_model_outputs(
        outputs,
        return_intermediate=False,
        return_act_stats=False,
        return_aux_outputs=return_aux_outputs,
    )
    pred_spectrum, extra_loss = apply_auxiliary_prediction_losses(
        pred_spectrum,
        aux_outputs,
        batch,
        args,
    )
    main_loss = cosine_similarity_loss(pred_spectrum, batch.y)
    act_loss = torch.zeros((), device=batch.y.device)
    aux_loss = torch.zeros((), device=batch.y.device)
    loss = main_loss + extra_loss
    return loss, main_loss, act_loss, aux_loss, pred_spectrum, None
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    args,
) -> Dict:
    """
    Train for one epoch.
    """
    model.train()
    total_loss = 0.0
    total_main_loss = 0.0
    total_act_loss = 0.0
    total_aux_loss = 0.0
    total_cosine_sim = 0.0
    total_ponder_steps = 0.0
    total_top5 = 0.0
    total_top10 = 0.0
    total_top20 = 0.0
    total_samples = 0
    num_batches = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        if batch is None:
            continue
        batch = batch.to(device)
        optimizer.zero_grad()
        loss, main_loss, act_loss, aux_loss, pred_spectrum, act_stats = _forward_and_loss(
            model=model,
            batch=batch,
            args=args,
            training=True,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()
        with torch.no_grad():
            cosine_sim = 1.0 - main_loss.item()
            total_loss += loss.item()
            total_main_loss += main_loss.item()
            total_act_loss += act_loss.item()
            total_aux_loss += aux_loss.item()
            total_cosine_sim += cosine_sim
            if act_stats is not None:
                total_ponder_steps += act_stats["mean_ponder_steps"].item()
            batch_size = batch.y.size(0)
            total_top5 += topk_recall_batch(pred_spectrum, batch.y, 5).sum().item()
            total_top10 += topk_recall_batch(pred_spectrum, batch.y, 10).sum().item()
            total_top20 += topk_recall_batch(pred_spectrum, batch.y, 20).sum().item()
            total_samples += batch_size
            num_batches += 1
            avg_loss = total_loss / num_batches
            avg_main_loss = total_main_loss / num_batches
            avg_cos_sim = total_cosine_sim / num_batches
            avg_ponder = total_ponder_steps / num_batches if total_ponder_steps > 0 else 0.0
        pbar.set_postfix(
            {
                "loss": f"{avg_loss:.4f}",
                "main": f"{avg_main_loss:.4f}",
                "cos": f"{avg_cos_sim:.4f}",
                "ponder": f"{avg_ponder:.2f}",
            }
        )
    if num_batches == 0:
        raise RuntimeError("No valid batches were produced during training.")
    metrics = {
        "loss": total_loss / num_batches,
        "main_loss": total_main_loss / num_batches,
        "act_loss": total_act_loss / num_batches,
        "aux_loss": total_aux_loss / num_batches,
        "cosine_sim": total_cosine_sim / num_batches,
        "top5": total_top5 / total_samples,
        "top10": total_top10 / total_samples,
        "top20": total_top20 / total_samples,
        "mean_ponder_steps": total_ponder_steps / num_batches if total_ponder_steps > 0 else 0.0,
    }
    return metrics
@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    args,
) -> Dict:
    """
    Evaluate model.
    """
    model.eval()
    total_loss = 0.0
    total_main_loss = 0.0
    total_act_loss = 0.0
    total_aux_loss = 0.0
    total_cosine_sim = 0.0
    total_ponder_steps = 0.0
    total_top5 = 0.0
    total_top10 = 0.0
    total_top20 = 0.0
    total_samples = 0
    num_batches = 0
    bucket_stats = defaultdict(
        lambda: {"count": 0, "cosine": 0.0, "top5": 0.0, "top10": 0.0, "top20": 0.0}
    )
    pbar = tqdm(dataloader, desc="Evaluating")
    for batch in pbar:
        if batch is None:
            continue
        batch = batch.to(device)
        loss, main_loss, act_loss, aux_loss, pred_spectrum, act_stats = _forward_and_loss(
            model=model,
            batch=batch,
            args=args,
            training=False,
        )
        cosine_sim = 1.0 - main_loss.item()
        total_loss += loss.item()
        total_main_loss += main_loss.item()
        total_act_loss += act_loss.item()
        total_aux_loss += aux_loss.item()
        total_cosine_sim += cosine_sim
        if act_stats is not None:
            total_ponder_steps += act_stats["mean_ponder_steps"].item()
        batch_size = batch.y.size(0)
        sample_cosine = F.cosine_similarity(pred_spectrum, batch.y, dim=-1)
        sample_top5 = topk_recall_batch(pred_spectrum, batch.y, 5)
        sample_top10 = topk_recall_batch(pred_spectrum, batch.y, 10)
        sample_top20 = topk_recall_batch(pred_spectrum, batch.y, 20)
        total_top5 += sample_top5.sum().item()
        total_top10 += sample_top10.sum().item()
        total_top20 += sample_top20.sum().item()
        max_train_similarity = getattr(batch, "max_train_similarity", None)
        if max_train_similarity is not None:
            sims = max_train_similarity.view(-1).to(device)
            bucket_masks = {
                "near": sims >= 0.8,
                "mid": (sims >= 0.6) & (sims < 0.8),
                "far": sims < 0.6,
            }
            for bucket, mask in bucket_masks.items():
                count = int(mask.sum().item())
                if count == 0:
                    continue
                bucket_stats[bucket]["count"] += count
                bucket_stats[bucket]["cosine"] += sample_cosine[mask].sum().item()
                bucket_stats[bucket]["top5"] += sample_top5[mask].sum().item()
                bucket_stats[bucket]["top10"] += sample_top10[mask].sum().item()
                bucket_stats[bucket]["top20"] += sample_top20[mask].sum().item()
        total_samples += batch_size
        num_batches += 1
        avg_loss = total_loss / num_batches
        avg_cos_sim = total_cosine_sim / num_batches
        avg_ponder = total_ponder_steps / num_batches if total_ponder_steps > 0 else 0.0
        pbar.set_postfix(
            {
                "loss": f"{avg_loss:.4f}",
                "cos": f"{avg_cos_sim:.4f}",
                "ponder": f"{avg_ponder:.2f}",
            }
        )
    if num_batches == 0:
        raise RuntimeError("No valid batches were produced during evaluation.")
    metrics = {
        "loss": total_loss / num_batches,
        "main_loss": total_main_loss / num_batches,
        "act_loss": total_act_loss / num_batches,
        "aux_loss": total_aux_loss / num_batches,
        "cosine_sim": total_cosine_sim / num_batches,
        "top5": total_top5 / total_samples,
        "top10": total_top10 / total_samples,
        "top20": total_top20 / total_samples,
        "mean_ponder_steps": total_ponder_steps / num_batches if total_ponder_steps > 0 else 0.0,
    }
    if bucket_stats:
        metrics["similarity_buckets"] = {
            bucket: {
                "count": values["count"],
                "cosine_sim": values["cosine"] / values["count"],
                "top5": values["top5"] / values["count"],
                "top10": values["top10"] / values["count"],
                "top20": values["top20"] / values["count"],
            }
            for bucket, values in bucket_stats.items()
            if values["count"] > 0
        }
    return metrics
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train QCGN2oEI_Minimal on NIST17"
    )
    parser.add_argument("--nist-msp", type=str, required=True, help="Path to NIST MSP file")
    parser.add_argument("--mol-files-dir", type=str, default="data/mol_files", help="Directory containing NIST MOL files")
    parser.add_argument("--bde-cache", type=str, default=None, help="Path to BDE HDF5 cache")
    parser.add_argument("--bondnet-model", type=str, default=None, help="Path to BonDNet model for on-the-fly BDE calculation")
    parser.add_argument("--output", type=str, default="models/qcgn2oei_minimal.pth", help="Output model path")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=768, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=14, help="Number of GATv2 layers")
    parser.add_argument("--num-heads", type=int, default=24, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples to use. 0 means all.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split and training")
    parser.add_argument(
        "--annotate-similarity-buckets",
        action="store_true",
        help="Annotate validation samples with max train Morgan/Tanimoto similarity.",
    )
    parser.add_argument(
        "--similarity-balanced-sampler",
        action="store_true",
        help="Use Morgan fingerprint cluster-balanced sampling for the training subset.",
    )
    parser.add_argument(
        "--similarity-fp-radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius for similarity diagnostics/sampler.",
    )
    parser.add_argument(
        "--similarity-fp-bits",
        type=int,
        default=2048,
        help="Morgan fingerprint bit size for similarity diagnostics/sampler.",
    )
    parser.add_argument(
        "--similarity-cluster-threshold",
        type=float,
        default=0.8,
        help="Tanimoto threshold used by exact Butina clustering.",
    )
    parser.add_argument(
        "--similarity-cluster-max-exact",
        type=int,
        default=3000,
        help="Maximum training samples for exact O(n^2) Butina clustering.",
    )
    parser.add_argument(
        "--similarity-sampler-strength",
        type=float,
        default=1.0,
        help="Cluster balancing strength. 1.0 gives inverse cluster-size weights.",
    )
    parser.add_argument(
        "--similarity-max-train-refs",
        type=int,
        default=10000,
        help="Maximum train fingerprints used when annotating validation similarity.",
    )
    parser.add_argument(
        "--spectrum-augment-intensity-jitter",
        type=float,
        default=0.0,
        help="Training-only multiplicative intensity jitter stddev.",
    )
    parser.add_argument(
        "--spectrum-augment-peak-dropout",
        type=float,
        default=0.0,
        help="Training-only dropout probability for minor target peaks.",
    )
    parser.add_argument(
        "--spectrum-augment-mz-spread",
        type=float,
        default=0.0,
        help="Training-only fraction of target intensity spread to +/-1 m/z bins.",
    )
    neutral_loss_group = parser.add_mutually_exclusive_group()
    neutral_loss_group.add_argument(
        "--use-neutral-loss-head",
        dest="use_neutral_loss_head",
        action="store_true",
        default=True,
        help="Enable the neutral-loss prediction head (default).",
    )
    neutral_loss_group.add_argument(
        "--no-neutral-loss-head",
        dest="use_neutral_loss_head",
        action="store_false",
        help="Disable the neutral-loss prediction head.",
    )
    parser.add_argument(
        "--neutral-loss-weight",
        type=float,
        default=0.05,
        help="Loss weight for the neutral-loss auxiliary head.",
    )
    parser.add_argument(
        "--neutral-loss-fusion-weight",
        type=float,
        default=0.30,
        help="Blend neutral-loss head remapped to m/z into scored prediction.",
    )
    parser.add_argument(
        "--weighted-cosine-loss-weight",
        type=float,
        default=0.0,
        help="Additional mass-weighted cosine loss weight.",
    )
    parser.add_argument(
        "--weighted-cosine-mz-power",
        type=float,
        default=2.0,
        help="m/z exponent for mass-weighted cosine loss.",
    )
    parser.add_argument(
        "--use-fragment-candidate-mask",
        action="store_true",
        help="Attach structure-derived fragment candidate masks to graphs.",
    )
    parser.add_argument(
        "--fragment-candidate-loss-weight",
        type=float,
        default=0.0,
        help="Penalty weight for predicted probability outside fragment candidate masks.",
    )
    parser.add_argument(
        "--fragment-mask-window",
        type=int,
        default=2,
        help="+/- m/z window marked around fragment candidate masses.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--save-interval", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument(
        "--pooling-mode",
        type=str,
        default="mean_add_max",
        choices=["mean", "mean_add_max"],
        help="Graph pooling mode",
    )
    parser.add_argument(
        "--no-graph-features",
        action="store_true",
        help="Disable graph-level molecular mass/formula features",
    )
    parser.add_argument(
        "--no-bde",
        action="store_true",
        help="Disable BDE edge feature values",
    )
    parser.add_argument(
        "--no-mass-features",
        action="store_true",
        help="Zero exact/molecular mass graph features while keeping formula counts",
    )
    parser.add_argument(
        "--no-formula-features",
        action="store_true",
        help="Zero formula-count graph features while keeping molecular mass",
    )
    # Recurrent refinement options
    parser.add_argument(
        "--use-recurrent-refinement",
        action="store_true",
        help="Enable stable recurrent spectral refinement block",
    )
    parser.add_argument(
        "--refinement-steps",
        type=int,
        default=4,
        help="Default number of recurrent refinement steps",
    )
    parser.add_argument(
        "--refinement-max-steps",
        type=int,
        default=8,
        help="Maximum number of recurrent refinement steps",
    )
    parser.add_argument(
        "--refinement-dropout",
        type=float,
        default=0.1,
        help="Dropout rate inside recurrent refinement block",
    )
    parser.add_argument(
        "--use-lti-stabilization",
        action="store_true",
        help="Use LTI-style alpha/beta stable recurrent update",
    )
    parser.add_argument(
        "--use-loop-embedding",
        action="store_true",
        help="Use loop-step embedding in recurrent refinement",
    )
    parser.add_argument(
        "--use-act",
        action="store_true",
        help="Use Adaptive Computation Time early halting",
    )
    parser.add_argument(
        "--act-epsilon",
        type=float,
        default=0.01,
        help="ACT halting epsilon. Threshold is 1 - epsilon.",
    )
    parser.add_argument(
        "--act-loss-weight",
        type=float,
        default=0.01,
        help="Weight for ACT ponder cost",
    )
    parser.add_argument(
        "--use-intermediate-loss",
        action="store_true",
        help="Use auxiliary loss on intermediate spectra. Uses fixed-loop refinement during training.",
    )
    parser.add_argument(
        "--intermediate-loss-weight",
        type=float,
        default=0.3,
        help="Weight for intermediate spectra auxiliary loss",
    )
    parser.add_argument(
        "--force-no-act-eval",
        action="store_true",
        help="During evaluation, force fixed-loop refinement even if ACT is enabled.",
    )
    return parser
def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.fragment_candidate_loss_weight > 0 and not args.use_fragment_candidate_mask:
        args.use_fragment_candidate_mask = True
    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA was requested but is not available. Falling back to CPU.")
        args.device = "cpu"
    if args.hidden_dim % args.num_heads != 0:
        raise ValueError(
            f"--hidden-dim must be divisible by --num-heads. "
            f"Got hidden_dim={args.hidden_dim}, num_heads={args.num_heads}"
        )
    if args.refinement_steps > args.refinement_max_steps:
        logger.warning(
            "--refinement-steps is larger than --refinement-max-steps. "
            "It will be clipped inside the model."
        )
    logger.info("=" * 80)
    logger.info("NExtIMS v1.0: Minimal GNN Training")
    logger.info("=" * 80)
    logger.info(f"NIST MSP: {args.nist_msp}")
    logger.info(f"MOL files dir: {args.mol_files_dir}")
    logger.info(f"BDE cache: {args.bde_cache}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Use recurrent refinement: {args.use_recurrent_refinement}")
    logger.info(f"Refinement steps: {args.refinement_steps}")
    logger.info(f"Refinement max steps: {args.refinement_max_steps}")
    logger.info(f"Use LTI stabilization: {args.use_lti_stabilization}")
    logger.info(f"Use loop embedding: {args.use_loop_embedding}")
    logger.info(f"Use ACT: {args.use_act}")
    logger.info(f"ACT loss weight: {args.act_loss_weight}")
    logger.info(f"Use intermediate loss: {args.use_intermediate_loss}")
    logger.info(f"Pooling mode: {args.pooling_mode}")
    logger.info(f"Use graph features: {not args.no_graph_features}")
    logger.info(f"Use BDE: {not args.no_bde}")
    logger.info(f"Use mass features: {not args.no_mass_features}")
    logger.info(f"Use formula features: {not args.no_formula_features}")
    logger.info(f"Annotate similarity buckets: {args.annotate_similarity_buckets}")
    logger.info(f"Similarity-balanced sampler: {args.similarity_balanced_sampler}")
    logger.info(
        "Spectrum augmentation: "
        f"jitter={args.spectrum_augment_intensity_jitter}, "
        f"dropout={args.spectrum_augment_peak_dropout}, "
        f"mz_spread={args.spectrum_augment_mz_spread}"
    )
    logger.info(f"Use neutral loss head: {args.use_neutral_loss_head}")
    logger.info(f"Neutral loss weight: {args.neutral_loss_weight}")
    logger.info(f"Neutral loss fusion weight: {args.neutral_loss_fusion_weight}")
    logger.info(f"Weighted cosine loss weight: {args.weighted_cosine_loss_weight}")
    logger.info(f"Weighted cosine m/z power: {args.weighted_cosine_mz_power}")
    logger.info(f"Use fragment candidate mask: {args.use_fragment_candidate_mask}")
    logger.info(f"Fragment candidate loss weight: {args.fragment_candidate_loss_weight}")
    logger.info(f"Fragment mask window: {args.fragment_mask_window}")
    logger.info("")
    dataset = NISTGraphDataset(
        msp_path=args.nist_msp,
        bde_cache_path=args.bde_cache,
        bondnet_model=args.bondnet_model,
        mol_files_dir=args.mol_files_dir,
        use_bde=not args.no_bde,
        use_mass_features=not args.no_mass_features,
        use_formula_features=not args.no_formula_features,
        use_fragment_candidates=args.use_fragment_candidate_mask,
        fragment_mask_window=args.fragment_mask_window,
        max_samples=args.max_samples,
    )
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError(
            f"Invalid train/val split. train_size={train_size}, val_size={val_size}"
        )
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_indices = list(train_dataset.indices)
    val_indices = list(val_dataset.indices)
    if args.annotate_similarity_buckets:
        bucket_counts = annotate_validation_similarity(
            dataset=dataset,
            train_indices=train_indices,
            val_indices=val_indices,
            radius=args.similarity_fp_radius,
            n_bits=args.similarity_fp_bits,
            max_train_refs=args.similarity_max_train_refs,
            seed=args.seed,
        )
        if bucket_counts:
            logger.info(
                "Validation similarity buckets: %s",
                ", ".join(f"{key}={value}" for key, value in sorted(bucket_counts.items())),
            )
    train_sampler = None
    if args.similarity_balanced_sampler:
        train_sampler = build_similarity_balanced_sampler(
            dataset=dataset,
            train_indices=train_indices,
            radius=args.similarity_fp_radius,
            n_bits=args.similarity_fp_bits,
            threshold=args.similarity_cluster_threshold,
            exact_max_samples=args.similarity_cluster_max_exact,
            strength=args.similarity_sampler_strength,
            seed=args.seed,
        )
    if (
        args.spectrum_augment_intensity_jitter > 0
        or args.spectrum_augment_peak_dropout > 0
        or args.spectrum_augment_mz_spread > 0
    ):
        train_dataset = SpectrumAugmentedDataset(
            train_dataset,
            intensity_jitter=args.spectrum_augment_intensity_jitter,
            peak_dropout=args.spectrum_augment_peak_dropout,
            mz_spread=args.spectrum_augment_mz_spread,
        )
    logger.info(f"Train size: {train_size:,}")
    logger.info(f"Val size: {val_size:,}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.num_workers,
        collate_fn=collate_pyg_graphs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pyg_graphs,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = QCGN2oEI_Minimal(
        node_dim=34,
        edge_dim=10,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        output_dim=1000,
        dropout=args.dropout,
        use_recurrent_refinement=args.use_recurrent_refinement,
        refinement_steps=args.refinement_steps,
        refinement_max_steps=args.refinement_max_steps,
        refinement_dropout=args.refinement_dropout,
        use_lti_stabilization=args.use_lti_stabilization,
        use_loop_embedding=args.use_loop_embedding,
        use_act=args.use_act,
        act_epsilon=args.act_epsilon,
        act_loss_weight=args.act_loss_weight,
        pooling_mode=args.pooling_mode,
        graph_feature_dim=GRAPH_FEATURE_DIM,
        use_graph_features=not args.no_graph_features,
        use_neutral_loss_head=args.use_neutral_loss_head,
    ).to(args.device)
    model_info = model.get_model_info()
    logger.info("Model info:")
    for key, value in model_info.items():
        logger.info(f"  {key}: {value}")
    try:
        from torch.optim import RAdam
        optimizer = RAdam(model.parameters(), lr=args.lr)
        logger.info("Using RAdam optimizer")
    except ImportError:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        logger.info("RAdam not available, using Adam")
    best_val_cosine = 0.0
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=args.device,
            epoch=epoch,
            args=args,
        )
        logger.info(
            "Train - "
            f"Loss: {train_metrics['loss']:.4f}, "
            f"Main Loss: {train_metrics['main_loss']:.4f}, "
            f"ACT Loss: {train_metrics['act_loss']:.4f}, "
            f"Aux Loss: {train_metrics['aux_loss']:.4f}, "
            f"Cosine Sim: {train_metrics['cosine_sim']:.4f}, "
            f"Top5: {train_metrics['top5']:.4f}, "
            f"Top10: {train_metrics['top10']:.4f}, "
            f"Top20: {train_metrics['top20']:.4f}, "
            f"Ponder: {train_metrics['mean_ponder_steps']:.2f}"
        )
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=args.device,
            args=args,
        )
        logger.info(
            "Val - "
            f"Loss: {val_metrics['loss']:.4f}, "
            f"Main Loss: {val_metrics['main_loss']:.4f}, "
            f"ACT Loss: {val_metrics['act_loss']:.4f}, "
            f"Aux Loss: {val_metrics['aux_loss']:.4f}, "
            f"Cosine Sim: {val_metrics['cosine_sim']:.4f}, "
            f"Top5: {val_metrics['top5']:.4f}, "
            f"Top10: {val_metrics['top10']:.4f}, "
            f"Top20: {val_metrics['top20']:.4f}, "
            f"Ponder: {val_metrics['mean_ponder_steps']:.2f}"
        )
        logger.info(
            "Gap - "
            f"Cosine(train-val): {train_metrics['cosine_sim'] - val_metrics['cosine_sim']:.4f}, "
            f"Top5(train-val): {train_metrics['top5'] - val_metrics['top5']:.4f}, "
            f"Top10(train-val): {train_metrics['top10'] - val_metrics['top10']:.4f}, "
            f"Top20(train-val): {train_metrics['top20'] - val_metrics['top20']:.4f}"
        )
        for bucket, bucket_metrics in sorted(
            val_metrics.get("similarity_buckets", {}).items()
        ):
            logger.info(
                "Val[%s] - Count: %d, Cosine Sim: %.4f, Top5: %.4f, "
                "Top10: %.4f, Top20: %.4f",
                bucket,
                bucket_metrics["count"],
                bucket_metrics["cosine_sim"],
                bucket_metrics["top5"],
                bucket_metrics["top10"],
                bucket_metrics["top20"],
            )
        if val_metrics["cosine_sim"] > best_val_cosine:
            best_val_cosine = val_metrics["cosine_sim"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_cosine_sim": best_val_cosine,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                    "model_info": model.get_model_info(),
                },
                args.output,
            )
            logger.info(f"✓ Saved best model (cosine sim: {best_val_cosine:.4f})")
        if epoch % args.save_interval == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch{epoch}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_cosine_sim": val_metrics["cosine_sim"],
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                    "model_info": model.get_model_info(),
                },
                checkpoint_path,
            )
            logger.info(f"✓ Saved checkpoint: {checkpoint_path}")
    logger.info("")
    logger.info("=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)
    logger.info(f"Best validation cosine similarity: {best_val_cosine:.4f}")
    logger.info(f"Model saved to: {args.output}")
if __name__ == "__main__":
    main()
