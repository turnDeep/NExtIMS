#!/usr/bin/env python3
"""
NExtIMS v4.2: Comprehensive Model Evaluation Script

Evaluates QCGN2oEI_Minimal model on NIST17 test set with:
- Cosine similarity metric
- Top-K recall (K=5,10,20,50)
- MSE/RMSE metrics
- Spectrum visualization
- Performance benchmarking (inference time, memory usage)
- Decision logic for iterative refinement

Usage:
    # Basic evaluation
    python scripts/evaluate_minimal.py \\
        --model models/qcgn2oei_minimal_best.pth \\
        --nist-msp data/NIST17.MSP \\
        --bde-cache data/processed/bde_cache/nist17_bde_cache.h5

    # With visualization
    python scripts/evaluate_minimal.py \\
        --model models/qcgn2oei_minimal_best.pth \\
        --nist-msp data/NIST17.MSP \\
        --visualize \\
        --num-visualize 10 \\
        --output-dir results/evaluation

    # Performance benchmarking
    python scripts/evaluate_minimal.py \\
        --model models/qcgn2oei_minimal_best.pth \\
        --nist-msp data/NIST17.MSP \\
        --benchmark \\
        --batch-size 64
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import time
import json
from typing import Dict, List, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from rdkit import RDLogger

# Suppress RDKit errors
RDLogger.DisableLog('rdApp.*')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.qcgn2oei_minimal import QCGN2oEI_Minimal
from src.data.graph_generator_minimal import MinimalGraphGenerator
from src.data.nist_dataset import parse_msp_file, peaks_to_spectrum

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive evaluator for QCGN2oEI_Minimal model
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        node_dim: int = 34,
        edge_dim: int = 10,
        hidden_dim: int = 1024,
        num_layers: int = 16,
        num_heads: int = 32,
        output_dim: int = 1000
    ):
        """
        Initialize evaluator

        Args:
            model_path: Path to trained model checkpoint
            device: Device to use (cuda/cpu)
            node_dim: Node feature dimension
            edge_dim: Edge feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of GATv2 layers
            num_heads: Number of attention heads
            output_dim: Output spectrum dimension
        """
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load model
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        # Load args from checkpoint if available
        ckpt_args = checkpoint.get('args', {})
        if 'node_dim' in ckpt_args:
             node_dim = ckpt_args['node_dim']
        if 'edge_dim' in ckpt_args:
             edge_dim = ckpt_args['edge_dim']
        if 'hidden_dim' in ckpt_args:
             hidden_dim = ckpt_args['hidden_dim']
        if 'num_layers' in ckpt_args:
             num_layers = ckpt_args['num_layers']
        if 'num_heads' in ckpt_args:
             num_heads = ckpt_args['num_heads']

        self.model = QCGN2oEI_Minimal(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            output_dim=output_dim,
            dropout=0.1
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded successfully on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Store model config
        self.config = {
            'node_dim': node_dim,
            'edge_dim': edge_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'output_dim': output_dim
        }

    @staticmethod
    def cosine_similarity_metric(pred: np.ndarray, target: np.ndarray) -> float:
        """
        Calculate mean cosine similarity between predictions and targets

        Args:
            pred: Predicted spectra [batch_size, spectrum_dim]
            target: Target spectra [batch_size, spectrum_dim]

        Returns:
            Mean cosine similarity
        """
        pred_norm = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
        target_norm = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-8)
        cosine_sim = (pred_norm * target_norm).sum(axis=1)
        return cosine_sim.mean()

    @staticmethod
    def top_k_recall(pred: np.ndarray, target: np.ndarray, k: int = 10) -> float:
        """
        Calculate Top-K Recall

        Args:
            pred: Predicted spectra [batch_size, spectrum_dim]
            target: Target spectra [batch_size, spectrum_dim]
            k: Number of top peaks to consider

        Returns:
            Mean Top-K Recall
        """
        recalls = []
        for p, t in zip(pred, target):
            true_top_k = set(np.argsort(t)[-k:])
            pred_top_k = set(np.argsort(p)[-k:])
            recall = len(true_top_k & pred_top_k) / k
            recalls.append(recall)
        return np.mean(recalls)

    @staticmethod
    def spectral_angle(pred: np.ndarray, target: np.ndarray) -> float:
        """
        Calculate spectral angle (alternative similarity metric)

        Args:
            pred: Predicted spectra [batch_size, spectrum_dim]
            target: Target spectra [batch_size, spectrum_dim]

        Returns:
            Mean spectral angle in degrees
        """
        cosine_sim = ModelEvaluator.cosine_similarity_metric(pred, target)
        # Clamp to avoid numerical issues with arccos
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        angle_rad = np.arccos(cosine_sim)
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def evaluate(
        self,
        test_data: List[Dict],
        graph_generator: MinimalGraphGenerator,
        batch_size: int = 32,
        max_samples: int = 0
    ) -> Dict:
        """
        Evaluate model on test dataset

        Args:
            test_data: List of test samples (NIST entries)
            graph_generator: Graph generator instance
            batch_size: Batch size for evaluation
            max_samples: Maximum samples to evaluate (0 = all)

        Returns:
            Dictionary with evaluation metrics
        """
        if max_samples > 0:
            test_data = test_data[:max_samples]

        logger.info(f"Evaluating on {len(test_data):,} samples")

        all_predictions = []
        all_targets = []
        all_smiles = []
        inference_times = []

        # Process in batches
        for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating"):
            batch_entries = test_data[i:i + batch_size]

            # Convert to graphs
            graphs = []
            targets = []
            smiles_list = []

            for entry in batch_entries:
                if 'smiles' not in entry or 'peaks' not in entry:
                    continue

                smiles = entry['smiles']
                peaks = entry['peaks']

                # Generate spectrum
                spectrum = peaks_to_spectrum(
                    peaks,
                    min_mz=1,
                    max_mz=1000
                )

                # Generate graph
                try:
                    graph = graph_generator.smiles_to_graph(
                        smiles=smiles,
                        molecule_idx=i + len(graphs)  # Dummy index
                    )

                    if graph is not None:
                        graphs.append(graph)
                        targets.append(spectrum)
                        smiles_list.append(smiles)
                except ValueError:
                    continue

            if len(graphs) == 0:
                continue

            # Batch graphs
            from torch_geometric.data import Batch
            batch_graph = Batch.from_data_list(graphs)
            batch_graph = batch_graph.to(self.device)

            # Inference
            start_time = time.time()
            with torch.no_grad():
                pred = self.model(
                    x=batch_graph.x,
                    edge_index=batch_graph.edge_index,
                    edge_attr=batch_graph.edge_attr,
                    batch=batch_graph.batch
                )
            inference_time = time.time() - start_time
            inference_times.append(inference_time / len(graphs))  # Per-sample time

            # Collect results
            pred_np = pred.cpu().numpy()
            all_predictions.append(pred_np)
            all_targets.append(np.array(targets))
            all_smiles.extend(smiles_list)

        # Concatenate results
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        logger.info(f"Evaluated {len(predictions):,} samples")

        # Calculate metrics
        metrics = {}

        # Cosine similarity
        metrics['cosine_similarity'] = float(self.cosine_similarity_metric(predictions, targets))

        # Top-K Recall
        for k in [5, 10, 20, 50]:
            metrics[f'top{k}_recall'] = float(self.top_k_recall(predictions, targets, k=k))

        # MSE/RMSE/MAE
        metrics['mse'] = float(mean_squared_error(targets.flatten(), predictions.flatten()))
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
        metrics['mae'] = float(mean_absolute_error(targets.flatten(), predictions.flatten()))

        # Spectral angle
        metrics['spectral_angle_deg'] = float(self.spectral_angle(predictions, targets))

        # Performance metrics
        metrics['avg_inference_time_ms'] = float(np.mean(inference_times) * 1000)
        metrics['total_samples'] = len(predictions)

        # Store predictions for visualization
        self.last_predictions = predictions
        self.last_targets = targets
        self.last_smiles = all_smiles

        return metrics

    def benchmark_performance(
        self,
        test_data: List[Dict],
        graph_generator: MinimalGraphGenerator,
        batch_sizes: List[int] = [1, 8, 16, 32, 64, 128]
    ) -> Dict:
        """
        Benchmark inference performance with different batch sizes

        Args:
            test_data: Test samples
            graph_generator: Graph generator
            batch_sizes: List of batch sizes to test

        Returns:
            Benchmark results
        """
        logger.info("Running performance benchmark...")

        # Use subset for benchmarking
        test_subset = test_data[:min(1000, len(test_data))]

        # Prepare graphs
        graphs = []
        for i, entry in enumerate(tqdm(test_subset, desc="Preparing graphs")):
            if 'smiles' not in entry:
                continue

            try:
                graph = graph_generator.smiles_to_graph(
                    smiles=entry['smiles'],
                    molecule_idx=i
                )

                if graph is not None:
                    graphs.append(graph)
            except ValueError:
                continue

        if len(graphs) < 10:
            logger.warning("Not enough valid graphs for benchmarking")
            return {}

        logger.info(f"Benchmarking with {len(graphs)} graphs")

        results = {}

        for batch_size in batch_sizes:
            if batch_size > len(graphs):
                continue

            # Measure inference time
            times = []
            memory_usages = []

            num_iterations = min(10, len(graphs) // batch_size)

            for i in range(num_iterations):
                batch_graphs = graphs[i * batch_size:(i + 1) * batch_size]

                from torch_geometric.data import Batch
                batch = Batch.from_data_list(batch_graphs)
                batch = batch.to(self.device)

                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # Measure memory before
                if torch.cuda.is_available():
                    mem_before = torch.cuda.memory_allocated() / 1024 / 1024  # MB

                # Inference
                start = time.time()
                with torch.no_grad():
                    _ = self.model(
                        x=batch.x,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                        batch=batch.batch
                    )

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                elapsed = time.time() - start

                # Measure memory after
                if torch.cuda.is_available():
                    mem_after = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                    memory_usages.append(mem_after - mem_before)

                times.append(elapsed)

            avg_time = np.mean(times)
            avg_memory = np.mean(memory_usages) if memory_usages else 0

            results[f'batch_{batch_size}'] = {
                'batch_size': batch_size,
                'avg_time_sec': float(avg_time),
                'time_per_sample_ms': float(avg_time / batch_size * 1000),
                'throughput_samples_per_sec': float(batch_size / avg_time),
                'avg_memory_mb': float(avg_memory) if avg_memory > 0 else None
            }

            logger.info(
                f"Batch size {batch_size:3d}: "
                f"{avg_time / batch_size * 1000:.2f} ms/sample, "
                f"{batch_size / avg_time:.1f} samples/sec"
            )

        return results

    def visualize_predictions(
        self,
        output_dir: str = "results/evaluation",
        num_samples: int = 10,
        figsize: Tuple[int, int] = (12, 4)
    ):
        """
        Visualize prediction vs ground truth for random samples

        Args:
            output_dir: Directory to save plots
            num_samples: Number of samples to visualize
            figsize: Figure size for each plot
        """
        if not hasattr(self, 'last_predictions'):
            logger.warning("No predictions available for visualization. Run evaluate() first.")
            return

        os.makedirs(output_dir, exist_ok=True)

        predictions = self.last_predictions
        targets = self.last_targets
        smiles = self.last_smiles

        # Random sample indices
        indices = np.random.choice(len(predictions), size=min(num_samples, len(predictions)), replace=False)

        for idx in tqdm(indices, desc="Generating plots"):
            pred = predictions[idx]
            target = targets[idx]
            mol_smiles = smiles[idx] if idx < len(smiles) else "Unknown"

            # Calculate metrics for this sample
            cosine_sim = self.cosine_similarity_metric(
                pred[np.newaxis, :],
                target[np.newaxis, :]
            )
            top10_recall = self.top_k_recall(
                pred[np.newaxis, :],
                target[np.newaxis, :],
                k=10
            )

            # Create plot
            fig, ax = plt.subplots(figsize=figsize)

            mz_values = np.arange(1, len(pred) + 1)

            # Plot target (ground truth) in blue
            ax.stem(
                mz_values,
                target,
                linefmt='b-',
                markerfmt='bo',
                basefmt=' ',
                label='Ground Truth'
            )

            # Plot prediction in red (offset slightly for visibility)
            ax.stem(
                mz_values + 0.3,
                pred,
                linefmt='r-',
                markerfmt='ro',
                basefmt=' ',
                label='Prediction'
            )

            ax.set_xlabel('m/z', fontsize=12)
            ax.set_ylabel('Relative Intensity', fontsize=12)
            ax.set_title(
                f'Sample {idx}: Cosine Sim={cosine_sim:.4f}, Top-10 Recall={top10_recall:.4f}\n{mol_smiles[:60]}',
                fontsize=10
            )
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1000)

            plt.tight_layout()
            plt.savefig(f"{output_dir}/prediction_sample_{idx:04d}.png", dpi=150, bbox_inches='tight')
            plt.close()

        logger.info(f"Saved {len(indices)} visualization plots to {output_dir}")

    def generate_report(
        self,
        metrics: Dict,
        benchmark: Dict = None,
        output_path: str = "results/evaluation_report.json"
    ):
        """
        Generate comprehensive evaluation report

        Args:
            metrics: Evaluation metrics
            benchmark: Benchmark results
            output_path: Path to save report
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        report = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'model_config': self.config,
            'device': str(self.device),
            'metrics': metrics,
        }

        if benchmark:
            report['benchmark'] = benchmark

        # Add performance assessment
        cosine_sim = metrics['cosine_similarity']
        if cosine_sim >= 0.85:
            assessment = "EXCELLENT"
            recommendation = "Adopt v4.2 minimal configuration. No feature expansion needed."
        elif cosine_sim >= 0.80:
            assessment = "GOOD"
            recommendation = "Consider minor feature additions. Proceed to Phase 4 for targeted expansion."
        elif cosine_sim >= 0.75:
            assessment = "MODERATE"
            recommendation = "Feature expansion recommended. Proceed to Phase 4."
        else:
            assessment = "INSUFFICIENT"
            recommendation = "Significant feature expansion required. Consider intermediate configuration (64/32 dims)."

        report['assessment'] = {
            'level': assessment,
            'recommendation': recommendation,
            'cosine_similarity': cosine_sim
        }

        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Evaluation report saved to {output_path}")

        return report

    def print_summary(self, metrics: Dict):
        """
        Print evaluation summary to console

        Args:
            metrics: Evaluation metrics
        """
        print("\n" + "=" * 70)
        print("QC-GN2oEI Minimal Configuration Evaluation")
        print("=" * 70)
        print(f"Model: {self.model_path}")
        print(f"Device: {self.device}")
        print(f"Node features: {self.config['node_dim']} dims")
        print(f"Edge features: {self.config['edge_dim']} dims")
        print(f"Hidden dimension: {self.config['hidden_dim']}")
        print(f"Number of layers: {self.config['num_layers']}")
        print("-" * 70)
        print("\nSpectral Similarity Metrics:")
        print(f"  Cosine Similarity:  {metrics['cosine_similarity']:.4f}")
        print(f"  Spectral Angle:     {metrics['spectral_angle_deg']:.2f}°")
        print("\nTop-K Recall:")
        print(f"  Top-5 Recall:       {metrics['top5_recall']:.4f}")
        print(f"  Top-10 Recall:      {metrics['top10_recall']:.4f}")
        print(f"  Top-20 Recall:      {metrics['top20_recall']:.4f}")
        print(f"  Top-50 Recall:      {metrics['top50_recall']:.4f}")
        print("\nError Metrics:")
        print(f"  MSE:                {metrics['mse']:.6f}")
        print(f"  RMSE:               {metrics['rmse']:.6f}")
        print(f"  MAE:                {metrics['mae']:.6f}")
        print("\nPerformance:")
        print(f"  Avg Inference Time: {metrics['avg_inference_time_ms']:.2f} ms/sample")
        print(f"  Total Samples:      {metrics['total_samples']:,}")
        print("=" * 70)

        # Performance assessment
        cosine_sim = metrics['cosine_similarity']
        print("\n" + "=" * 70)
        print("Performance Assessment")
        print("=" * 70)

        if cosine_sim >= 0.85:
            print("✅ EXCELLENT: Cosine Similarity >= 0.85")
            print("   Recommendation: Adopt v4.2 minimal configuration!")
            print("   No feature expansion needed.")
        elif cosine_sim >= 0.80:
            print("⚠️  GOOD: Cosine Similarity 0.80-0.85")
            print("   Recommendation: Consider minor feature additions")
            print("   Proceed to Phase 4 for targeted feature expansion")
        elif cosine_sim >= 0.75:
            print("⚠️  MODERATE: Cosine Similarity 0.75-0.80")
            print("   Recommendation: Feature expansion recommended")
            print("   Proceed to Phase 4 for systematic feature addition")
        else:
            print("❌ INSUFFICIENT: Cosine Similarity < 0.75")
            print("   Recommendation: Significant expansion required")
            print("   Consider intermediate configuration (64/32 dims)")

        print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate QCGN2oEI_Minimal model"
    )

    # Required arguments
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--nist-msp',
        type=str,
        required=True,
        help='Path to NIST MSP file'
    )

    # Optional arguments
    parser.add_argument(
        '--bde-cache',
        type=str,
        default=None,
        help='Path to BDE cache HDF5 file'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation (default: 32)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=0,
        help='Maximum samples to evaluate (0 = all, default: 0)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/evaluation',
        help='Output directory for results (default: results/evaluation)'
    )

    # Visualization
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate prediction visualizations'
    )
    parser.add_argument(
        '--num-visualize',
        type=int,
        default=10,
        help='Number of samples to visualize (default: 10)'
    )

    # Benchmarking
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run performance benchmarking'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize evaluator
    logger.info("Initializing evaluator...")
    evaluator = ModelEvaluator(
        model_path=args.model,
        device=args.device
    )

    # Load test data
    logger.info(f"Loading NIST data from {args.nist_msp}")
    mol_files_dir = "data/mol_files"
    all_entries = parse_msp_file(args.nist_msp, mol_files_dir=mol_files_dir)

    # Use last 10% as test set (or specify test split)
    test_size = int(len(all_entries) * 0.1)
    test_data = all_entries[-test_size:]
    logger.info(f"Using {len(test_data):,} samples as test set")

    # Initialize graph generator
    graph_gen = MinimalGraphGenerator(
        bde_cache_path=args.bde_cache,
        use_bde_calculator=False,
        default_bde=85.0
    )

    # Run evaluation
    logger.info("Running evaluation...")
    metrics = evaluator.evaluate(
        test_data=test_data,
        graph_generator=graph_gen,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )

    # Print summary
    evaluator.print_summary(metrics)

    # Benchmark (optional)
    benchmark_results = None
    if args.benchmark:
        benchmark_results = evaluator.benchmark_performance(
            test_data=test_data,
            graph_generator=graph_gen
        )

    # Generate report
    report_path = os.path.join(args.output_dir, 'evaluation_report.json')
    evaluator.generate_report(
        metrics=metrics,
        benchmark=benchmark_results,
        output_path=report_path
    )

    # Visualize (optional)
    if args.visualize:
        logger.info("Generating visualizations...")
        evaluator.visualize_predictions(
            output_dir=args.output_dir,
            num_samples=args.num_visualize
        )

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
