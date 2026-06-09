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
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
from rdkit import Chem
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
            default_bde=85.0,
        )
        logger.info(f"Loading NIST data from {msp_path}")
        mol_files_dir = "data/mol_files"
        entries = parse_msp_file(msp_path, mol_files_dir=mol_files_dir)
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
            graph = self.graph_gen.mol_to_graph(
                mol=item["mol"],
                smiles=item["smiles"],
                spectrum=item["spectrum"],
                compound_name=item["name"],
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
def molecular_ion_auxiliary_loss(
    pred_spectrum: torch.Tensor,
    target_spectrum: torch.Tensor,
    molecular_ion_index: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Penalize incorrect relative intensity at the nominal molecular-ion bin.
    Prediction is normalized by its base peak because the main head uses Softmax.
    """
    if molecular_ion_index is None:
        return torch.zeros((), device=pred_spectrum.device)
    indices = molecular_ion_index.long().view(-1).to(pred_spectrum.device)
    output_dim = pred_spectrum.size(-1)
    indices = indices.clamp(min=0, max=output_dim - 1)
    pred_relative = pred_spectrum / pred_spectrum.max(
        dim=-1,
        keepdim=True,
    ).values.clamp_min(1e-8)
    pred_m = pred_relative.gather(1, indices.unsqueeze(-1)).squeeze(-1)
    target_m = target_spectrum.gather(1, indices.unsqueeze(-1)).squeeze(-1)
    return torch.mean((pred_m - target_m) ** 2)
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
        molecular_ion_loss:
            Molecular-ion relative-intensity auxiliary loss.
        pred_spectrum:
            Predicted spectrum.
        act_stats:
            ACT statistics or None.
    """
    use_act_stats = bool(args.use_recurrent_refinement and args.use_act)
    use_intermediate = bool(
        args.use_recurrent_refinement and args.use_intermediate_loss and training
    )
    graph_features = getattr(batch, "graph_features", None)
    molecular_ion_index = getattr(batch, "molecular_ion_index", None)
    if use_intermediate:
        # Intermediate supervision uses fixed-loop mode.
        if use_act_stats:
            pred_spectrum, intermediate_spectra, act_stats = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
                graph_features=graph_features,
                refinement_steps=args.refinement_steps,
                return_intermediate=True,
                return_act_stats=True,
                force_no_act=True,
            )
        else:
            pred_spectrum, intermediate_spectra = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
                graph_features=graph_features,
                refinement_steps=args.refinement_steps,
                return_intermediate=True,
                return_act_stats=False,
                force_no_act=True,
            )
            act_stats = None
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
        molecular_ion_loss = molecular_ion_auxiliary_loss(
            pred_spectrum=pred_spectrum,
            target_spectrum=batch.y,
            molecular_ion_index=molecular_ion_index,
        )
        loss = (
            main_loss
            + args.intermediate_loss_weight * aux_loss
            + args.act_loss_weight * act_loss
            + args.molecular_ion_loss_weight * molecular_ion_loss
        )
        return loss, main_loss, act_loss, aux_loss, molecular_ion_loss, pred_spectrum, act_stats
    if use_act_stats:
        pred_spectrum, act_stats = model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch,
            graph_features=graph_features,
            refinement_steps=args.refinement_steps,
            return_act_stats=True,
            force_no_act=args.force_no_act_eval and not training,
        )
        main_loss = cosine_similarity_loss(pred_spectrum, batch.y)
        act_loss = act_stats["act_loss"]
        aux_loss = torch.zeros((), device=batch.y.device)
        molecular_ion_loss = molecular_ion_auxiliary_loss(
            pred_spectrum=pred_spectrum,
            target_spectrum=batch.y,
            molecular_ion_index=molecular_ion_index,
        )
        loss = (
            main_loss
            + args.act_loss_weight * act_loss
            + args.molecular_ion_loss_weight * molecular_ion_loss
        )
        return loss, main_loss, act_loss, aux_loss, molecular_ion_loss, pred_spectrum, act_stats
    pred_spectrum = model(
        x=batch.x,
        edge_index=batch.edge_index,
        edge_attr=batch.edge_attr,
        batch=batch.batch,
        graph_features=graph_features,
        refinement_steps=args.refinement_steps if args.use_recurrent_refinement else None,
        force_no_act=args.force_no_act_eval and not training,
    )
    main_loss = cosine_similarity_loss(pred_spectrum, batch.y)
    act_loss = torch.zeros((), device=batch.y.device)
    aux_loss = torch.zeros((), device=batch.y.device)
    molecular_ion_loss = molecular_ion_auxiliary_loss(
        pred_spectrum=pred_spectrum,
        target_spectrum=batch.y,
        molecular_ion_index=molecular_ion_index,
    )
    loss = main_loss + args.molecular_ion_loss_weight * molecular_ion_loss
    return loss, main_loss, act_loss, aux_loss, molecular_ion_loss, pred_spectrum, None
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
    total_molecular_ion_loss = 0.0
    total_cosine_sim = 0.0
    total_ponder_steps = 0.0
    num_batches = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        if batch is None:
            continue
        batch = batch.to(device)
        optimizer.zero_grad()
        loss, main_loss, act_loss, aux_loss, molecular_ion_loss, pred_spectrum, act_stats = _forward_and_loss(
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
            total_molecular_ion_loss += molecular_ion_loss.item()
            total_cosine_sim += cosine_sim
            if act_stats is not None:
                total_ponder_steps += act_stats["mean_ponder_steps"].item()
            num_batches += 1
            avg_loss = total_loss / num_batches
            avg_main_loss = total_main_loss / num_batches
            avg_cos_sim = total_cosine_sim / num_batches
            avg_ponder = total_ponder_steps / num_batches if total_ponder_steps > 0 else 0.0
        pbar.set_postfix(
            {
                "loss": f"{avg_loss:.4f}",
                "main": f"{avg_main_loss:.4f}",
                "mi": f"{total_molecular_ion_loss / num_batches:.4f}",
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
        "molecular_ion_loss": total_molecular_ion_loss / num_batches,
        "cosine_sim": total_cosine_sim / num_batches,
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
    total_molecular_ion_loss = 0.0
    total_cosine_sim = 0.0
    total_ponder_steps = 0.0
    num_batches = 0
    pbar = tqdm(dataloader, desc="Evaluating")
    for batch in pbar:
        if batch is None:
            continue
        batch = batch.to(device)
        loss, main_loss, act_loss, aux_loss, molecular_ion_loss, pred_spectrum, act_stats = _forward_and_loss(
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
        total_molecular_ion_loss += molecular_ion_loss.item()
        total_cosine_sim += cosine_sim
        if act_stats is not None:
            total_ponder_steps += act_stats["mean_ponder_steps"].item()
        num_batches += 1
        avg_loss = total_loss / num_batches
        avg_cos_sim = total_cosine_sim / num_batches
        avg_ponder = total_ponder_steps / num_batches if total_ponder_steps > 0 else 0.0
        pbar.set_postfix(
            {
                "loss": f"{avg_loss:.4f}",
                "mi": f"{total_molecular_ion_loss / num_batches:.4f}",
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
        "molecular_ion_loss": total_molecular_ion_loss / num_batches,
        "cosine_sim": total_cosine_sim / num_batches,
        "mean_ponder_steps": total_ponder_steps / num_batches if total_ponder_steps > 0 else 0.0,
    }
    return metrics
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train QCGN2oEI_Minimal on NIST17"
    )
    parser.add_argument("--nist-msp", type=str, required=True, help="Path to NIST MSP file")
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
        "--molecular-ion-loss-weight",
        type=float,
        default=0.1,
        help="Weight for molecular-ion relative-intensity auxiliary loss",
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
    logger.info(f"Molecular ion loss weight: {args.molecular_ion_loss_weight}")
    logger.info("")
    dataset = NISTGraphDataset(
        msp_path=args.nist_msp,
        bde_cache_path=args.bde_cache,
        bondnet_model=args.bondnet_model,
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
    )
    logger.info(f"Train size: {train_size:,}")
    logger.info(f"Val size: {val_size:,}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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
            f"MI Loss: {train_metrics['molecular_ion_loss']:.4f}, "
            f"Cosine Sim: {train_metrics['cosine_sim']:.4f}, "
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
            f"MI Loss: {val_metrics['molecular_ion_loss']:.4f}, "
            f"Cosine Sim: {val_metrics['cosine_sim']:.4f}, "
            f"Ponder: {val_metrics['mean_ponder_steps']:.2f}"
        )
        if val_metrics["cosine_sim"] > best_val_cosine:
            best_val_cosine = val_metrics["cosine_sim"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_cosine_sim": best_val_cosine,
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
