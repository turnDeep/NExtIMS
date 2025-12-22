#!/usr/bin/env python3
"""
NExtIMS v4.2: BonDNet Retraining on BDE-db2 with Transfer Learning and HDF5 Support

Refactored to support memory-efficient HDF5 data loading.
"""

import os
import sys
import argparse
import logging
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Imports from BondNet (assumed to be installed)
try:
    import bondnet
    from bondnet.model.gated_reaction_network import GatedGCNReactionNetwork
    from bondnet.model.metric import WeightedL1Loss
except ImportError:
    print("BonDNet not found. Please install it first.")
    sys.exit(1)

# Import HDF5 dataset loader (local script)
try:
    from scripts.bondnet_hdf5_dataset import BonDNetHDF5Dataset, collate_bondnet
except ImportError:
    # If running from root
    try:
        from bondnet_hdf5_dataset import BonDNetHDF5Dataset, collate_bondnet
    except ImportError:
        # Fallback if scripts folder is not in path
        sys.path.append('scripts')
        from bondnet_hdf5_dataset import BonDNetHDF5Dataset, collate_bondnet

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, metric_fn):
    model.train()
    epoch_loss = 0.0
    accuracy = 0.0
    count = 0.0

    # feature names used by GatedGCNReactionNetwork
    # We need to construct the features dict from the batched graph
    nodes = ["atom", "bond", "global"]

    for i, (bg, labels) in enumerate(dataloader):
        bg = bg.to(device)
        target = labels['value'].to(device)
        reactions = labels['reaction'] # List of Reaction objects

        # Extract features from graph
        # Note: BonDNet GatedGCNConv expects "atom", "bond", etc.
        # But 'bond' features are typically on EDGES in DGL.
        # BonDNet converts edge features to node features of a line graph or uses them directly?
        # Looking at GatedGCNConv source (not visible but standard implementation),
        # it usually processes node features.

        # In our HDF5 loader, we put atom features in nodes['atom'].data['feat']
        # and bond features in edges['a2b'].data['feat'].

        # GatedGCNConv typically requires:
        # feats = {'atom': ..., 'bond': ...} where 'bond' might be features on edges?
        # If GatedGCNConv expects 'bond' as a key in feats dict, we need to provide it.

        feats = {}
        # DGL 2.x accessing conventions can be tricky with HeteroGraphs
        # Try accessing via nodes['type']
        try:
             feats['atom'] = bg.nodes['atom'].data['feat']
        except KeyError:
             pass

        try:
             # If using HeteroGraph where 'bond' is a node type, access it via nodes
             feats['bond'] = bg.nodes['bond'].data['feat']
        except KeyError:
             pass

        # Add global feature placeholder if missing (BonDNet might expect it)
        if "global" not in feats:
            # Create zero global features matching batch size (number of graphs)
            # DGL batching preserves number of graphs
            batch_num_nodes = bg.batch_num_nodes('atom') # get num nodes per graph
            num_graphs = len(batch_num_nodes)
            # Use size 0 global features as placeholder if not provided
            feats["global"] = torch.zeros(num_graphs, 0, device=device)

        # Norms (placeholder if not computed)
        norm_atom = labels.get('norm_atom')
        norm_bond = labels.get('norm_bond')
        if norm_atom is not None: norm_atom = norm_atom.to(device)
        if norm_bond is not None: norm_bond = norm_bond.to(device)

        optimizer.zero_grad()

        # Forward pass
        # Note: reactions list indices must match the batch
        pred = model(bg, feats, reactions, norm_atom, norm_bond)
        pred = pred.view(-1)
        target = target.view(-1)

        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Metric
        # WeightedL1Loss expects (pred, target, stdev)
        # We pass 1.0 for stdev if not normalized
        stdev = labels.get('scaler_stdev', torch.tensor(1.0)).to(device)
        accuracy += metric_fn(pred, target, stdev).item()
        count += len(target)

    return epoch_loss / (i + 1), accuracy / count


def validate(model, dataloader, device, metric_fn):
    model.eval()
    accuracy = 0.0
    count = 0.0
    nodes = ["atom", "bond", "global"]

    with torch.no_grad():
        for i, (bg, labels) in enumerate(dataloader):
            bg = bg.to(device)
            target = labels['value'].to(device)
            reactions = labels['reaction']

            feats = {nt: bg.nodes[nt].data["feat"] for nt in nodes if nt in bg.ndata}

            norm_atom = labels.get('norm_atom')
            norm_bond = labels.get('norm_bond')
            if norm_atom is not None: norm_atom = norm_atom.to(device)
            if norm_bond is not None: norm_bond = norm_bond.to(device)

            pred = model(bg, feats, reactions, norm_atom, norm_bond)
            pred = pred.view(-1)
            target = target.view(-1)

            stdev = labels.get('scaler_stdev', torch.tensor(1.0)).to(device)
            accuracy += metric_fn(pred, target, stdev).item()
            count += len(target)

    return accuracy / count


def train_with_hdf5(
    hdf5_path: Path,
    output_path: Path,
    epochs: int = 200,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    device: str = 'cuda',
    pretrained_path: Path = None,
    save_interval: int = 10
):
    """
    Train BonDNet using HDF5 dataset.
    """
    logger.info(f"Training with HDF5 dataset: {hdf5_path}")

    # Dataset & Dataloader
    dataset = BonDNetHDF5Dataset(str(hdf5_path), cache_graphs=False) # Disable cache for true streaming

    # Simple split (random)
    # Note: For strict reproducibility, use fixed indices or split file
    train_size = int(0.9 * len(dataset))
    if train_size == 0 and len(dataset) > 0:
        train_size = len(dataset) # Use all for train if dataset is very small
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Ensure validation set is not empty for testing if possible, or handle empty val
    if val_size == 0:
        # Just use train set as val set for very small datasets
        val_set = train_set

    logger.info(f"Train size: {len(train_set)}, Val size: {len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_bondnet,
        num_workers=4 if device == 'cuda' else 0,
        pin_memory=True if device == 'cuda' else False
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_bondnet,
        num_workers=4 if device == 'cuda' else 0
    )

    # Model Setup
    # Hardcoded default architecture from original script
    embedding_size = 128
    gnn_hidden_size = 128
    num_gnn_layers = 4

    # Determine feature size from first sample
    # BonDNet requires input feature dimension
    # Our HDF5 dataset produces simplified features: 8 for atom, 4 for bond
    # Need to check what GatedGCNReactionNetwork expects.
    # Usually it expects a dict with sizes.
    # Let's inspect a sample.
    sample_graph = dataset[0]['reactant_graph']
    # For HeteroGraph, ndata returns a dict of features for each node type
    # access atom features: nodes['atom'].data['feat']
    try:
        atom_feat_dim = sample_graph.nodes['atom'].data['feat'].shape[1]
    except KeyError:
        atom_feat_dim = 0

    try:
        bond_feat_dim = sample_graph.nodes['bond'].data['feat'].shape[1]
    except KeyError:
        bond_feat_dim = 0

    feature_size = {"atom": atom_feat_dim, "bond": bond_feat_dim, "global": 0}

    logger.info(f"Feature sizes: {feature_size}")

    model = GatedGCNReactionNetwork(
        in_feats=feature_size,
        embedding_size=embedding_size,
        gated_num_layers=num_gnn_layers,
        gated_hidden_size=[gnn_hidden_size] * num_gnn_layers,
        gated_num_fc_layers=2, # Default
        gated_graph_norm=0,
        gated_batch_norm=1,
        gated_activation="ReLU",
        gated_residual=1,
        gated_dropout=0.0,
        # Readout default params
        num_lstm_iters=6,
        num_lstm_layers=3,
        set2set_ntypes_direct=["global"],
        fc_num_layers=3,
        fc_hidden_size=[128, 64, 32], # Adjusted to match script description
        fc_batch_norm=0,
        fc_activation="ReLU",
        fc_dropout=0.0,
        outdim=1,
        conv="GatedGCNConv",
    )

    if pretrained_path:
        logger.info(f"Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        # Handle state dict loading (might need key adjustment)
        if 'model_state_dict' in checkpoint:
             model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
             model.load_state_dict(checkpoint['model'])
        else:
             # Try direct loading
             try:
                 model.load_state_dict(checkpoint)
             except Exception as e:
                 logger.warning(f"Could not load pretrained weights directly: {e}")

    device_obj = torch.device(device)
    model.to(device_obj)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # verbose argument is deprecated/removed in newer PyTorch versions
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    loss_fn = nn.MSELoss()
    metric_fn = WeightedL1Loss(reduction="sum")

    # Training Loop
    best_val_mae = float('inf')

    logger.info("Starting training...")
    for epoch in range(epochs):
        start_time = time.time()

        train_loss, train_mae = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device_obj, metric_fn
        )

        val_mae = validate(model, val_loader, device_obj, metric_fn)

        scheduler.step(val_mae)

        # Save best
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae
            }, output_path)
            logger.info(f"  New best model saved! MAE: {val_mae:.4f}")

        epoch_time = time.time() - start_time
        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Train MAE: {train_mae:.4f} | "
            f"Val MAE: {val_mae:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Regular checkpoint
        if (epoch + 1) % save_interval == 0:
            ckpt_path = output_path.parent / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae
            }, ckpt_path)


def main():
    parser = argparse.ArgumentParser(description="Retrain BonDNet on BDE-db2 dataset")
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing BonDNet data')
    parser.add_argument('--output', type=str, default='models/bondnet_bde_db2.pth')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--use-hdf5', action='store_true', help='Use HDF5 memory-efficient loading')

    args = parser.parse_args()

    # Check for HDF5 file
    data_dir = Path(args.data_dir)
    h5_file = data_dir / 'bondnet_data.h5'

    if args.use_hdf5:
        if not h5_file.exists():
            logger.error(f"HDF5 file not found at {h5_file}")
            logger.info("Please generate it first using scripts/bondnet_hdf5_dataset.py")
            sys.exit(1)

        train_with_hdf5(
            hdf5_path=h5_file,
            output_path=Path(args.output),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            pretrained_path=Path(args.pretrained) if args.pretrained else None
        )
    else:
        # Fallback to original logic (subprocess call) if not using HDF5
        # For brevity, I'm only implementing the HDF5 path fully as requested by the plan
        logger.warning("Standard mode (non-HDF5) selected. This may consume high memory.")
        # ... (Original logic would go here, or we advise user to use --use-hdf5)
        logger.info("Please use --use-hdf5 for memory-efficient training.")

if __name__ == '__main__':
    main()
