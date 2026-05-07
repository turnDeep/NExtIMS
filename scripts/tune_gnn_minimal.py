#!/usr/bin/env python3
"""
NExtIMS v1.0: Hyperparameter Tuning Script
Automated grid search for optimal GNN architecture (hidden_dim, num_layers, num_heads).
Uses Early Stopping to evaluate configurations quickly without running 200 epochs.
"""
import os
import sys
import argparse
import logging
import json
import itertools
from pathlib import Path
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.qcgn2oei_minimal import QCGN2oEI_Minimal
from scripts.train_gnn_minimal import NISTGraphDataset, collate_pyg_graphs, train_epoch, evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def run_trial(config: Dict[str, Any], train_loader: DataLoader, val_loader: DataLoader, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run a single training trial with the given configuration and Early Stopping.
    """
    device = args.device
    logger.info("=" * 60)
    logger.info(f"Starting Trial: {config['name']}")
    logger.info(f"Config: hidden_dim={config['hidden_dim']}, layers={config['num_layers']}, heads={config['num_heads']}")
    logger.info("=" * 60)

    # Convert args to a namespace-like object for the train/eval functions
    trial_args = argparse.Namespace(**vars(args))
    trial_args.hidden_dim = config['hidden_dim']
    trial_args.num_layers = config['num_layers']
    trial_args.num_heads = config['num_heads']

    model = QCGN2oEI_Minimal(
        node_dim=34,
        edge_dim=10,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        output_dim=1000,
        dropout=args.dropout,
        use_recurrent_refinement=args.use_recurrent_refinement,
        refinement_steps=args.refinement_steps,
        refinement_max_steps=args.refinement_max_steps,
        use_lti_stabilization=args.use_lti_stabilization,
        use_loop_embedding=args.use_loop_embedding,
        use_act=args.use_act,
        act_epsilon=args.act_epsilon,
        act_loss_weight=args.act_loss_weight,
    ).to(device)

    try:
        from torch.optim import RAdam
        optimizer = RAdam(model.parameters(), lr=args.lr)
    except ImportError:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_cosine = 0.0
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, args.max_epochs + 1):
        # We suppress the inner progress bars slightly to avoid terminal spam
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, trial_args)
        val_metrics = evaluate(model, val_loader, device, trial_args)

        val_cos = val_metrics['cosine_sim']

        logger.info(
            f"Epoch {epoch:03d} | "
            f"Train Cos: {train_metrics['cosine_sim']:.4f} | "
            f"Val Cos: {val_cos:.4f} | "
            f"Ponder: {val_metrics['mean_ponder_steps']:.2f}"
        )

        if val_cos > best_val_cosine:
            best_val_cosine = val_cos
            best_epoch = epoch
            patience_counter = 0
            # Optional: save the best model for this trial
            if args.save_models:
                torch.save(
                    model.state_dict(),
                    os.path.join(args.output_dir, f"model_{config['name']}_best.pth")
                )
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            logger.info(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} with CosSim {best_val_cosine:.4f}")
            break

    result = {
        "config": config,
        "best_val_cosine": best_val_cosine,
        "best_epoch": best_epoch,
        "stopped_epoch": epoch
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Tune GNN Architecture Hyperparameters")
    parser.add_argument("--nist-msp", type=str, required=True, help="Path to NIST MSP file")
    parser.add_argument("--bde-cache", type=str, default=None, help="Path to BDE cache")
    parser.add_argument("--bondnet-model", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/tuning", help="Output directory")

    # Tuning specific args
    parser.add_argument("--max-epochs", type=int, default=80, help="Max epochs per trial (default: 80)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (default: 15)")
    parser.add_argument("--save-models", action="store_true", help="Save the best model for each trial")

    # Static training args
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples to use. 0 means all.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # ACT parameters
    parser.add_argument("--use-recurrent-refinement", action="store_true", default=True)
    parser.add_argument("--refinement-steps", type=int, default=4)
    parser.add_argument("--refinement-max-steps", type=int, default=8)
    parser.add_argument("--use-lti-stabilization", action="store_true", default=True)
    parser.add_argument("--use-loop-embedding", action="store_true", default=True)
    parser.add_argument("--use-act", action="store_true", default=True)
    parser.add_argument("--act-epsilon", type=float, default=0.01)
    parser.add_argument("--act-loss-weight", type=float, default=0.001)
    parser.add_argument("--use-intermediate-loss", action="store_true")
    parser.add_argument("--intermediate-loss-weight", type=float, default=0.3)
    parser.add_argument("--force-no-act-eval", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data
    dataset = NISTGraphDataset(
        msp_path=args.nist_msp,
        bde_cache_path=args.bde_cache,
        bondnet_model=args.bondnet_model,
        max_samples=args.max_samples,
    )
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_pyg_graphs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_pyg_graphs)

    # 2. Define Search Space
    hidden_dims = [512, 768, 1024]
    num_layers_list = [10, 12, 14]
    num_heads_list = [16, 24, 32]

    # Generate valid configurations
    configs = []
    trial_idx = 1
    for h, l, hd in itertools.product(hidden_dims, num_layers_list, num_heads_list):
        if h % hd == 0:  # Hidden dim must be divisible by heads
            configs.append({
                "name": f"trial_{trial_idx}_h{h}_l{l}_hd{hd}",
                "hidden_dim": h,
                "num_layers": l,
                "num_heads": hd
            })
            trial_idx += 1

    logger.info(f"Total valid configurations to test: {len(configs)}")

    # 3. Run Tuning
    results = []
    for config in configs:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        res = run_trial(config, train_loader, val_loader, args)
        results.append(res)

        # Save intermediate results in case script is stopped
        with open(os.path.join(args.output_dir, "tuning_results_partial.json"), "w") as f:
            json.dump(results, f, indent=4)

    # 4. Final Report
    logger.info("=" * 80)
    logger.info("Tuning Complete!")
    logger.info("=" * 80)

    # Sort results by best validation cosine similarity descending
    results.sort(key=lambda x: x["best_val_cosine"], reverse=True)

    for i, res in enumerate(results):
        c = res['config']
        logger.info(f"Rank {i+1}: {c['name']} (h={c['hidden_dim']}, l={c['num_layers']}, hd={c['num_heads']}) -> Val CosSim: {res['best_val_cosine']:.4f} (at epoch {res['best_epoch']})")

    # Save final results
    final_path = os.path.join(args.output_dir, "tuning_results_final.json")
    with open(final_path, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Saved full report to {final_path}")

if __name__ == "__main__":
    main()
