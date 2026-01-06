#!/usr/bin/env python3
"""
NExtIMS v4.2: Minimal GNN Training Script

Trains QCGN2oEI_Minimal model on NIST17 EI-MS dataset with:
- RAdam optimizer (lr=5e-5)
- Cosine similarity loss
- Batch size optimization for RTX 5070 Ti
- BDE-enriched graph inputs

Usage:
    python scripts/train_gnn_minimal.py \\
        --nist-msp data/NIST17.MSP \\
        --bde-cache data/processed/bde_cache/nist17_bde_cache.h5 \\
        --output models/qcgn2oei_minimal.pth \\
        --epochs 200 \\
        --batch-size 32
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import time
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.qcgn2oei_minimal import QCGN2oEI_Minimal
from src.training.losses import cosine_similarity_loss
from src.data.graph_generator_minimal import MinimalGraphGenerator
from src.data.nist_dataset import parse_msp_file, peaks_to_spectrum
from rdkit import Chem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NISTGraphDataset(Dataset):
    """
    NIST EI-MS Dataset for PyG graphs

    Loads NIST MSP data and converts to PyG graphs on-the-fly.
    """

    def __init__(
        self,
        msp_path: str,
        bde_cache_path: str = None,
        bondnet_model: str = None,
        max_samples: int = 0,
        min_mz: int = 1,
        max_mz: int = 1000
    ):
        """
        Args:
            msp_path: Path to NIST MSP file
            bde_cache_path: Path to BDE HDF5 cache
            bondnet_model: Path to BonDNet model (optional, for on-the-fly calculation)
            max_samples: Maximum samples to load (0 = all)
            min_mz: Minimum m/z (default: 1)
            max_mz: Maximum m/z (default: 1000)
        """
        self.min_mz = min_mz
        self.max_mz = max_mz

        # Initialize graph generator
        self.graph_gen = MinimalGraphGenerator(
            bde_cache_path=bde_cache_path,
            use_bde_calculator=(bondnet_model is not None),
            bondnet_model=bondnet_model,
            default_bde=85.0
        )

        # Parse MSP file and load chemical structures from MOL files
        logger.info(f"Loading NIST data from {msp_path}")
        mol_files_dir = "data/mol_files"
        entries = parse_msp_file(msp_path, mol_files_dir=mol_files_dir)

        if max_samples > 0:
            entries = entries[:max_samples]

        logger.info(f"Loaded {len(entries):,} entries")

        # Filter valid entries
        self.data = []
        invalid_count = 0

        for entry in tqdm(entries, desc="Processing entries"):
            if 'smiles' not in entry or 'peaks' not in entry:
                invalid_count += 1
                continue

            smiles = entry['smiles']
            peaks = entry['peaks']

            # Validate SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid_count += 1
                continue

            # Convert peaks to spectrum
            spectrum = peaks_to_spectrum(peaks, min_mz=min_mz, max_mz=max_mz)

            self.data.append({
                'smiles': smiles,
                'spectrum': spectrum,
                'name': entry.get('name', 'Unknown'),
                'mol': mol
            })

        logger.info(f"Valid entries: {len(self.data):,}")
        logger.info(f"Invalid entries: {invalid_count:,}")

        # Print BDE statistics
        self.graph_gen.print_stats()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Generate graph
        try:
            graph = self.graph_gen.mol_to_graph(
                mol=item['mol'],
                smiles=item['smiles'],
                spectrum=item['spectrum'],
                compound_name=item['name']
            )
            return graph
        except ValueError:
            return None


def collate_pyg_graphs(batch_list):
    """Collate PyG graphs into a batch"""
    # Filter out None items (invalid graphs)
    batch_list = [item for item in batch_list if item is not None]
    if len(batch_list) == 0:
        return None
    return Batch.from_data_list(batch_list)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int
) -> Dict:
    """Train for one epoch"""
    model.train()

    total_loss = 0
    total_cosine_sim = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        if batch is None:
            continue

        batch = batch.to(device)

        # Forward pass
        optimizer.zero_grad()
        pred_spectrum = model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch
        )

        # Cosine similarity loss
        loss = cosine_similarity_loss(pred_spectrum, batch.y)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Compute cosine similarity for monitoring
        with torch.no_grad():
            cosine_sim = 1.0 - loss.item()

        total_loss += loss.item()
        total_cosine_sim += cosine_sim
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'cos_sim': f"{cosine_sim:.4f}"
        })

    metrics = {
        'loss': total_loss / num_batches,
        'cosine_sim': total_cosine_sim / num_batches
    }

    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str
) -> Dict:
    """Evaluate model"""
    model.eval()

    total_loss = 0
    total_cosine_sim = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        if batch is None:
            continue

        batch = batch.to(device)

        # Forward pass
        pred_spectrum = model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch
        )

        # Loss
        loss = cosine_similarity_loss(pred_spectrum, batch.y)
        cosine_sim = 1.0 - loss.item()

        total_loss += loss.item()
        total_cosine_sim += cosine_sim
        num_batches += 1

    metrics = {
        'loss': total_loss / num_batches,
        'cosine_sim': total_cosine_sim / num_batches
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train QCGN2oEI_Minimal on NIST17"
    )
    parser.add_argument('--nist-msp', type=str, required=True,
                        help='Path to NIST MSP file')
    parser.add_argument('--bde-cache', type=str, default=None,
                        help='Path to BDE HDF5 cache')
    parser.add_argument('--bondnet-model', type=str, default=None,
                        help='Path to BonDNet model for on-the-fly BDE calculation')
    parser.add_argument('--output', type=str, default='models/qcgn2oei_minimal.pth',
                        help='Output model path')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs (default: 200)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32 for RTX 5070 Ti)')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate (default: 5e-5)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension (default: 256)')
    parser.add_argument('--num-layers', type=int, default=10,
                        help='Number of GATv2 layers (default: 10)')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of attention heads (default: 8)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')
    parser.add_argument('--max-samples', type=int, default=0,
                        help='Max samples to use (0 = all)')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split (default: 0.1)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers (default: 4)')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save checkpoint every N epochs')

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("NExtIMS v4.2: Minimal GNN Training")
    logger.info("="*80)
    logger.info(f"NIST MSP: {args.nist_msp}")
    logger.info(f"BDE cache: {args.bde_cache}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Device: {args.device}")
    logger.info("")

    # Load dataset
    dataset = NISTGraphDataset(
        msp_path=args.nist_msp,
        bde_cache_path=args.bde_cache,
        bondnet_model=args.bondnet_model,
        max_samples=args.max_samples
    )

    # Train/val split
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    logger.info(f"Train size: {train_size:,}")
    logger.info(f"Val size: {val_size:,}")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_pyg_graphs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pyg_graphs
    )

    # Create model
    model = QCGN2oEI_Minimal(
        node_dim=16,
        edge_dim=3,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        output_dim=1000,
        dropout=args.dropout
    ).to(args.device)

    # Optimizer: RAdam
    try:
        from torch.optim import RAdam
        optimizer = RAdam(model.parameters(), lr=args.lr)
        logger.info("Using RAdam optimizer")
    except ImportError:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        logger.info("RAdam not available, using Adam")

    # Training loop
    best_val_cosine = 0.0
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, args.device, epoch)
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                   f"Cosine Sim: {train_metrics['cosine_sim']:.4f}")

        # Validate
        val_metrics = evaluate(model, val_loader, args.device)
        logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                   f"Cosine Sim: {val_metrics['cosine_sim']:.4f}")

        # Save best model
        if val_metrics['cosine_sim'] > best_val_cosine:
            best_val_cosine = val_metrics['cosine_sim']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_cosine_sim': best_val_cosine,
                'args': vars(args)
            }, args.output)
            logger.info(f"✓ Saved best model (cosine sim: {best_val_cosine:.4f})")

        # Periodic checkpoint
        if epoch % args.save_interval == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_cosine_sim': val_metrics['cosine_sim'],
                'args': vars(args)
            }, checkpoint_path)
            logger.info(f"✓ Saved checkpoint: {checkpoint_path}")

    logger.info("")
    logger.info("="*80)
    logger.info("Training Complete!")
    logger.info("="*80)
    logger.info(f"Best validation cosine similarity: {best_val_cosine:.4f}")
    logger.info(f"Model saved to: {args.output}")


if __name__ == '__main__':
    main()
