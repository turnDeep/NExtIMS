#!/usr/bin/env python3
"""
NExtIMS v4.4: Feature Ablation Study Driver

Runs a series of training jobs with different feature sets disabled to quantify
feature importance. Generates a summary plot and CSV.

Usage:
    python scripts/run_ablation_study.py --nist-msp data/NIST17.MSP --bde-cache data/processed/bde_cache/nist17_bde_cache.h5
"""

import os
import sys
import argparse
import subprocess
import re
import matplotlib.pyplot as plt
import pandas as pd
import logging
from pathlib import Path
from typing import List, Tuple, Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define ablation configurations: (Name, List of flags)
CONFIGURATIONS = [
    ("Full", []),
    ("-BDE", ["--no-bde"]),
    ("-BondOrder", ["--no-bond-order"]),
    ("-Chirality", ["--no-chirality"]),
    ("-Stereo", ["--no-stereo"]),
    ("-Hybridization", ["--no-hybridization"]),
    ("-Aromaticity", ["--no-aromaticity"]),
    ("-Conjugation", ["--no-conjugation"]),
    ("-Ring", ["--no-ring"]),
    ("-NumHs", ["--no-num-hs"]),
    ("-Charge", ["--no-formal-charge"]),
    ("-Radical", ["--no-radical"]),
    ("-AtomType", ["--no-atom-type"]),
]

def parse_best_cosine(log_output: str) -> float:
    """Extract best validation cosine similarity from log output"""
    match = re.search(r"Best validation cosine similarity: ([0-9.]+)", log_output)
    if match:
        return float(match.group(1))
    return 0.0

def run_experiment(
    config_name: str,
    flags: List[str],
    base_args: List[str],
    log_dir: Path
) -> float:
    """Run a single training experiment with output streaming"""
    logger.info(f"Running experiment: {config_name}")

    output_model = log_dir / f"model_{config_name}.pth"
    log_file = log_dir / f"log_{config_name}.txt"

    cmd = [
        "python3", "scripts/train_gnn_minimal.py",
        "--output", str(output_model),
        "--epochs", "100",  # Fixed to 100 epochs as requested
        # Use a small batch size/workers for stability if needed,
        # or rely on defaults. Here we rely on base_args.
    ] + base_args + flags

    logger.info(f"Command: {' '.join(cmd)}")

    full_log_output = ""

    try:
        # Run command and capture output continuously
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Merge stderr into stdout
            text=True,
            bufsize=1 # Line buffered
        )

        with open(log_file, "w") as f:
            for line in process.stdout:
                sys.stdout.write(line) # Print to console
                f.write(line) # Write to log file
                full_log_output += line

        process.wait()

        if process.returncode != 0:
             raise subprocess.CalledProcessError(process.returncode, cmd)

        # Parse result
        best_cos = parse_best_cosine(full_log_output)
        logger.info(f"  Result ({config_name}): {best_cos:.4f}")
        return best_cos

    except subprocess.CalledProcessError as e:
        logger.error(f"Experiment {config_name} failed!")
        # Log is already written to file and stdout
        return 0.0
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return 0.0

def plot_results(results: List[Tuple[str, float]], output_dir: Path):
    """Generate bar chart of ablation results"""
    df = pd.DataFrame(results, columns=["Configuration", "CosineSimilarity"])

    # Calculate relative drop
    baseline = df.loc[df["Configuration"] == "Full", "CosineSimilarity"].values[0]
    df["RelativeDrop"] = (baseline - df["CosineSimilarity"]) / baseline * 100

    # Save CSV
    df.to_csv(output_dir / "ablation_results.csv", index=False)

    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(df["Configuration"], df["CosineSimilarity"], color='skyblue')

    # Highlight baseline
    bars[0].set_color('steelblue')

    plt.xlabel("Feature Removed")
    plt.ylabel("Cosine Similarity")
    plt.title("Feature Ablation Study (100 Epochs)")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add labels
    for bar, val, drop in zip(bars, df["CosineSimilarity"], df["RelativeDrop"]):
        height = bar.get_height()
        label = f"{val:.3f}"
        if drop > 0: # Don't show 0% drop for baseline
             label += f"\n(-{drop:.1f}%)"

        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            label,
            ha='center', va='bottom', fontsize=9
        )

    plt.tight_layout()
    plt.savefig(output_dir / "ablation_results.png", dpi=300)
    logger.info(f"Plot saved to {output_dir / 'ablation_results.png'}")

def main():
    parser = argparse.ArgumentParser(description="Run Feature Ablation Study")
    parser.add_argument('--nist-msp', type=str, required=True)
    parser.add_argument('--bde-cache', type=str, required=True)
    parser.add_argument('--work-dir', type=str, default="ablation_study")
    parser.add_argument('--device', type=str, default="cuda")

    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    base_args = [
        "--nist-msp", args.nist_msp,
        "--bde-cache", args.bde_cache,
        "--device", args.device,
        "--save-interval", "50" # Reduce checkpoint overhead
    ]

    results = []

    for config_name, flags in CONFIGURATIONS:
        score = run_experiment(config_name, flags, base_args, work_dir)
        results.append((config_name, score))

    logger.info("="*60)
    logger.info("Ablation Study Complete")
    logger.info("="*60)
    for name, score in results:
        logger.info(f"{name:<15}: {score:.4f}")

    plot_results(results, work_dir)

if __name__ == '__main__':
    main()
