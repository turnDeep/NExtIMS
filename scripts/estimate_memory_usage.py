import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.qcgn2oei_minimal import QCGN2oEI_Minimal

def run_regression_and_predict():
    """
    Refined estimation based on initial results.
    Baseline (768, 14, 24) = 14GB (Assumed safe upper bound of usage on 16GB card).
    """

    configs = [
        # (hidden, layers, heads, label)
        (768, 14, 24, "Baseline (16GB OK)"),
        (1024, 16, 32, "User Attempt (16GB OOM)"),

        # 32GB Targets
        (1024, 20, 32, "Target: Deep 1K"),
        (1280, 14, 32, "Target: Wide 1.2K (Low Depth)"),
        (1280, 16, 32, "Target: Wide 1.2K (Med Depth)"),
        (1280, 18, 32, "Target: Wide 1.2K (High Depth)"),
        (1280, 20, 32, "Target: Wide 1.2K (Max Depth)"),
    ]

    print(f"{'Config':<30} | {'Params':<12} | {'Static Mem (GB)':<15} | {'Est. Peak (GB)':<15}")
    print("-" * 80)

    results = []

    # 1. Collect Stats
    for (h, l, heads, label) in configs:
        with torch.device('meta'):
            model = QCGN2oEI_Minimal(
                node_dim=34,
                edge_dim=10,
                hidden_dim=h,
                num_layers=l,
                num_heads=heads,
                output_dim=1000
            )
            total_params = sum(p.numel() for p in model.parameters())

        static_mem_gb = (total_params * 16) / (1024**3)

        results.append({
            'h': h, 'l': l, 'heads': heads,
            'params': total_params,
            'static': static_mem_gb,
            'label': label
        })

    # 2. Calibrate based on Baseline
    baseline = results[0] # 768

    # Assume Baseline used 14GB total (conservative estimate for 16GB card)
    est_baseline_total = 14.0
    activ_baseline = est_baseline_total - baseline['static'] # ~13.46 GB

    # 3. Predict
    for res in results:
        # Scaling factor for activations: L * H
        # (Assuming linear scaling with depth and width)

        metric_activs = res['l'] * res['h']
        base_metric_activs = baseline['l'] * baseline['h']

        ratio_activs = metric_activs / base_metric_activs

        curr_activ = activ_baseline * ratio_activs
        est_total = res['static'] + curr_activ

        # Check against 32GB
        status = "OK" if est_total < 30 else ("TIGHT" if est_total < 32 else "OOM")

        print(f"{str((res['h'], res['l'], res['heads'])):<30} | {res['params']:<12,} | {res['static']:<15.2f} | {est_total:<6.2f} ({status})")

if __name__ == "__main__":
    run_regression_and_predict()
