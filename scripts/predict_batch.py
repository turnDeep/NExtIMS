#!/usr/bin/env python3
"""
NExtIMS v4.2: Batch EI-MS Prediction

Predicts EI-MS spectra for multiple molecules from CSV file.

Input CSV format:
    smiles,id,name
    CCO,mol_001,ethanol
    CC(C)O,mol_002,isopropanol
    ...

Output CSV format:
    id,smiles,name,prediction_status,base_peak_mz,base_peak_intensity,num_peaks,inference_time_ms,top_10_mz,top_10_intensity
    mol_001,CCO,ethanol,success,46,0.9234,15,45.2,"[46,31,45,27,29]","[0.923,0.782,0.543,0.321,0.299]"
    ...

Usage:
    # Basic batch prediction
    python scripts/predict_batch.py \\
        --input molecules.csv \\
        --output predictions.csv \\
        --model models/qcgn2oei_minimal_best.pth

    # With BDE cache (faster)
    python scripts/predict_batch.py \\
        --input molecules.csv \\
        --output predictions.csv \\
        --model models/qcgn2oei_minimal_best.pth \\
        --bde-cache data/processed/bde_cache/nist17_bde_cache.h5 \\
        --batch-size 64

    # Save spectra to NPY file
    python scripts/predict_batch.py \\
        --input molecules.csv \\
        --output predictions.csv \\
        --model models/qcgn2oei_minimal_best.pth \\
        --save-spectra predictions_spectra.npy
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import time
import json

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.qcgn2oei_minimal import QCGN2oEI_Minimal
from src.data.graph_generator_minimal import MinimalGraphGenerator
from src.data.filters import SUPPORTED_ELEMENTS
from rdkit import Chem
from rdkit.Chem import Descriptors

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchPredictor:
    """
    Batch predictor for EI-MS spectra
    """

    def __init__(
        self,
        model_path: str,
        bde_cache_path: str = None,
        device: str = "cuda",
        min_mz: int = 1,
        max_mz: int = 1000
    ):
        """
        Initialize batch predictor

        Args:
            model_path: Path to trained model checkpoint
            bde_cache_path: Path to BDE cache (optional)
            device: Device for inference
            min_mz: Minimum m/z
            max_mz: Maximum m/z
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.output_dim = max_mz - min_mz + 1

        # Initialize graph generator
        logger.info("Initializing graph generator...")
        self.graph_gen = MinimalGraphGenerator(
            bde_cache_path=bde_cache_path,
            use_bde_calculator=bde_cache_path is None,
            default_bde=85.0
        )

        # Load model
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        # Load model args from checkpoint if available
        ckpt_args = checkpoint.get('args', {})
        node_dim = ckpt_args.get('node_dim', 34)
        edge_dim = ckpt_args.get('edge_dim', 10)
        hidden_dim = ckpt_args.get('hidden_dim', 768)
        num_layers = ckpt_args.get('num_layers', 14)
        num_heads = ckpt_args.get('num_heads', 24)

        self.model = QCGN2oEI_Minimal(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            output_dim=self.output_dim,
            dropout=0.1
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded successfully on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def validate_smiles(self, smiles: str) -> tuple[bool, str]:
        """
        Validate SMILES string

        Args:
            smiles: SMILES string

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False, "Invalid SMILES syntax"

            # Check elements
            for atom in mol.GetAtoms():
                if atom.GetSymbol() not in SUPPORTED_ELEMENTS:
                    return False, f"Unsupported element: {atom.GetSymbol()}"

            # Check molecular weight
            mw = Descriptors.MolWt(mol)
            if mw > 1000.0:
                return True, f"Warning: MW={mw:.1f} > 1000 Da"

            return True, ""

        except Exception as e:
            return False, str(e)

    def predict_single(self, smiles: str, molecule_idx: int) -> dict:
        """
        Predict spectrum for a single molecule

        Args:
            smiles: SMILES string
            molecule_idx: Molecule index (for BDE cache)

        Returns:
            Dictionary with prediction results
        """
        result = {
            'smiles': smiles,
            'status': 'success',
            'error': None,
            'spectrum': None,
            'inference_time_ms': 0,
            'base_peak_mz': None,
            'base_peak_intensity': None,
            'num_peaks': 0,
            'top_peaks': []
        }

        try:
            # Validate
            is_valid, error_msg = self.validate_smiles(smiles)
            if not is_valid:
                result['status'] = 'failed'
                result['error'] = error_msg
                return result

            if error_msg:  # Warning
                result['warning'] = error_msg

            # Generate graph
            graph = self.graph_gen.smiles_to_graph(
                smiles=smiles,
                molecule_idx=molecule_idx
            )

            if graph is None:
                result['status'] = 'failed'
                result['error'] = "Graph generation failed"
                return result

            # Predict
            graph = graph.to(self.device)

            start_time = time.time()
            with torch.no_grad():
                spectrum_pred = self.model(
                    x=graph.x,
                    edge_index=graph.edge_index,
                    edge_attr=graph.edge_attr,
                    batch=graph.batch
                )
            inference_time = time.time() - start_time

            spectrum = spectrum_pred.cpu().numpy().squeeze()

            # Extract peaks
            top_k = 10
            top_indices = np.argsort(spectrum)[-top_k:][::-1]
            top_peaks = [
                (int(idx + self.min_mz), float(spectrum[idx]))
                for idx in top_indices
                if spectrum[idx] > 0.001
            ]

            result['spectrum'] = spectrum
            result['inference_time_ms'] = inference_time * 1000
            result['base_peak_mz'] = top_peaks[0][0] if top_peaks else None
            result['base_peak_intensity'] = top_peaks[0][1] if top_peaks else None
            result['num_peaks'] = int((spectrum > 0.01).sum())
            result['top_peaks'] = top_peaks

        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)

        return result

    def predict_batch(
        self,
        smiles_list: list[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> list[dict]:
        """
        Predict spectra for a batch of molecules

        Args:
            smiles_list: List of SMILES strings
            batch_size: Batch size for processing
            show_progress: Show progress bar

        Returns:
            List of prediction results
        """
        logger.info(f"Predicting spectra for {len(smiles_list):,} molecules")

        results = []
        iterator = tqdm(smiles_list, desc="Predicting") if show_progress else smiles_list

        for i, smiles in enumerate(iterator):
            result = self.predict_single(smiles, molecule_idx=i)
            results.append(result)

        # Summary
        num_success = sum(1 for r in results if r['status'] == 'success')
        num_failed = len(results) - num_success

        logger.info(f"Prediction complete: {num_success}/{len(results)} succeeded, {num_failed} failed")

        return results


def load_input_csv(input_path: str) -> pd.DataFrame:
    """
    Load input CSV file

    Args:
        input_path: Path to input CSV

    Returns:
        DataFrame with columns: smiles, id (optional), name (optional)
    """
    df = pd.read_csv(input_path)

    # Check required columns
    if 'smiles' not in df.columns:
        raise ValueError("Input CSV must have 'smiles' column")

    # Add optional columns
    if 'id' not in df.columns:
        df['id'] = [f"mol_{i:06d}" for i in range(len(df))]

    if 'name' not in df.columns:
        df['name'] = df['smiles']

    logger.info(f"Loaded {len(df):,} molecules from {input_path}")

    return df


def save_results_csv(
    results: list[dict],
    input_df: pd.DataFrame,
    output_path: str
):
    """
    Save prediction results to CSV

    Args:
        results: List of prediction results
        input_df: Input DataFrame
        output_path: Output CSV path
    """
    rows = []

    for i, result in enumerate(results):
        row = {
            'id': input_df.iloc[i]['id'],
            'smiles': result['smiles'],
            'name': input_df.iloc[i]['name'],
            'prediction_status': result['status'],
            'error': result.get('error', ''),
            'inference_time_ms': result['inference_time_ms'],
            'base_peak_mz': result['base_peak_mz'],
            'base_peak_intensity': result['base_peak_intensity'],
            'num_peaks': result['num_peaks'],
        }

        # Top peaks as JSON
        if result['top_peaks']:
            top_mz = [p[0] for p in result['top_peaks']]
            top_intensity = [p[1] for p in result['top_peaks']]
            row['top_10_mz'] = json.dumps(top_mz)
            row['top_10_intensity'] = json.dumps(top_intensity)
        else:
            row['top_10_mz'] = '[]'
            row['top_10_intensity'] = '[]'

        rows.append(row)

    output_df = pd.DataFrame(rows)
    output_df.to_csv(output_path, index=False)

    logger.info(f"Results saved to {output_path}")


def save_spectra_npy(
    results: list[dict],
    output_path: str
):
    """
    Save all spectra to NPY file

    Args:
        results: List of prediction results
        output_path: Output NPY path
    """
    spectra = []
    for result in results:
        if result['spectrum'] is not None:
            spectra.append(result['spectrum'])
        else:
            # Placeholder for failed predictions
            spectra.append(np.zeros(1000))

    spectra_array = np.array(spectra)
    np.save(output_path, spectra_array)

    logger.info(f"Spectra saved to {output_path} (shape: {spectra_array.shape})")


def print_summary(results: list[dict], total_time: float):
    """
    Print prediction summary

    Args:
        results: List of prediction results
        total_time: Total processing time (seconds)
    """
    num_total = len(results)
    num_success = sum(1 for r in results if r['status'] == 'success')
    num_failed = num_total - num_success

    avg_time = np.mean([r['inference_time_ms'] for r in results if r['status'] == 'success'])

    print("\n" + "=" * 70)
    print("NExtIMS v4.2: Batch Prediction Summary")
    print("=" * 70)
    print(f"Total molecules: {num_total:,}")
    print(f"Successful predictions: {num_success:,} ({num_success/num_total*100:.1f}%)")
    print(f"Failed predictions: {num_failed:,} ({num_failed/num_total*100:.1f}%)")
    print("-" * 70)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per molecule: {avg_time:.2f} ms")
    print(f"Throughput: {num_total/total_time:.1f} molecules/second")
    print("=" * 70 + "\n")

    # Show failed samples
    if num_failed > 0:
        print("Failed samples:")
        print("-" * 70)
        failed_samples = [(i, r) for i, r in enumerate(results) if r['status'] == 'failed']
        for i, (idx, result) in enumerate(failed_samples[:10]):  # Show first 10
            print(f"{i+1}. Index {idx}: {result['smiles'][:50]} - {result['error']}")
        if num_failed > 10:
            print(f"... and {num_failed - 10} more")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Batch EI-MS spectrum prediction"
    )

    # Required arguments
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input CSV file with "smiles" column'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )

    # Optional arguments
    parser.add_argument(
        '--bde-cache',
        type=str,
        default=None,
        help='Path to BDE cache HDF5 file (optional, for faster prediction)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for inference (default: cuda)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for processing (default: 32)'
    )
    parser.add_argument(
        '--save-spectra',
        type=str,
        default=None,
        help='Save all spectra to NPY file (optional)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=0,
        help='Maximum samples to process (0 = all, default: 0)'
    )

    args = parser.parse_args()

    try:
        # Load input
        input_df = load_input_csv(args.input)

        # Limit samples
        if args.max_samples > 0:
            input_df = input_df.head(args.max_samples)
            logger.info(f"Limited to {len(input_df):,} samples")

        # Initialize predictor
        predictor = BatchPredictor(
            model_path=args.model,
            bde_cache_path=args.bde_cache,
            device=args.device
        )

        # Predict
        start_time = time.time()
        results = predictor.predict_batch(
            smiles_list=input_df['smiles'].tolist(),
            batch_size=args.batch_size,
            show_progress=True
        )
        total_time = time.time() - start_time

        # Save results
        save_results_csv(results, input_df, args.output)

        # Save spectra (optional)
        if args.save_spectra:
            save_spectra_npy(results, args.save_spectra)

        # Print summary
        print_summary(results, total_time)

        logger.info("Batch prediction complete!")

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
