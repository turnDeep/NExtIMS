#!/usr/bin/env python3
"""
NExtIMS v4.2: Single Molecule EI-MS Prediction

Predicts EI-MS spectrum for a single molecule using QCGN2oEI_Minimal model.

Usage:
    # Basic prediction
    python scripts/predict_single.py "CCO" \\
        --model models/qcgn2oei_minimal_best.pth

    # With visualization
    python scripts/predict_single.py "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" \\
        --model models/qcgn2oei_minimal_best.pth \\
        --output caffeine_spectrum.png \\
        --visualize

    # With BDE cache (faster)
    python scripts/predict_single.py "CCO" \\
        --model models/qcgn2oei_minimal_best.pth \\
        --bde-cache data/processed/bde_cache/nist17_bde_cache.h5
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.qcgn2oei_minimal import QCGN2oEI_Minimal
from src.data.graph_generator_minimal import MinimalGraphGenerator
from src.data.filters import SUPPORTED_ELEMENTS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_molecule(smiles: str) -> tuple[Chem.Mol, dict]:
    """
    Validate SMILES and extract molecular properties

    Args:
        smiles: SMILES string

    Returns:
        Tuple of (RDKit Mol object, properties dict)

    Raises:
        ValueError: If SMILES is invalid or contains unsupported elements
    """
    # Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Check supported elements
    unsupported = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in SUPPORTED_ELEMENTS:
            unsupported.append(symbol)

    if unsupported:
        raise ValueError(
            f"Unsupported elements: {', '.join(set(unsupported))}. "
            f"Supported: {', '.join(sorted(SUPPORTED_ELEMENTS))}"
        )

    # Calculate properties
    mw = Descriptors.MolWt(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)

    if mw > 1000.0:
        logger.warning(f"MW={mw:.1f} > 1000 Da. Prediction may be less accurate.")

    properties = {
        'molecular_weight': mw,
        'formula': formula,
        'num_atoms': mol.GetNumAtoms(),
        'num_bonds': mol.GetNumBonds(),
        'num_heavy_atoms': mol.GetNumHeavyAtoms()
    }

    return mol, properties


def predict_spectrum(
    smiles: str,
    model_path: str,
    bde_cache_path: str = None,
    device: str = "cuda",
    min_mz: int = 1,
    max_mz: int = 1000
) -> tuple[np.ndarray, dict]:
    """
    Predict EI-MS spectrum for a single molecule

    Args:
        smiles: SMILES string
        model_path: Path to trained QCGN2oEI_Minimal model
        bde_cache_path: Path to BDE cache HDF5 (optional)
        device: Device for inference (cuda/cpu)
        min_mz: Minimum m/z (default: 1)
        max_mz: Maximum m/z (default: 1000)

    Returns:
        Tuple of (spectrum array, metadata dict)
    """
    import torch

    logger.info(f"Predicting spectrum for: {smiles}")

    # Validate molecule
    mol, properties = validate_molecule(smiles)
    logger.info(f"  Formula: {properties['formula']}")
    logger.info(f"  Molecular weight: {properties['molecular_weight']:.2f} Da")
    logger.info(f"  Atoms: {properties['num_atoms']} ({properties['num_heavy_atoms']} heavy)")

    # Initialize graph generator
    logger.info("Generating molecular graph...")
    graph_gen = MinimalGraphGenerator(
        bde_cache_path=bde_cache_path,
        use_bde_calculator=bde_cache_path is None,  # Use calculator if no cache
        default_bde=85.0
    )

    # Generate graph (use dummy index since we don't have cache)
    graph = graph_gen.smiles_to_graph(
        smiles=smiles,
        molecule_idx=0  # Dummy index
    )

    if graph is None:
        raise RuntimeError("Failed to generate molecular graph")

    logger.info(f"  Graph: {graph.x.shape[0]} nodes, {graph.edge_index.shape[1]} edges")
    logger.info(f"  Node features: {graph.x.shape[1]} dims")
    logger.info(f"  Edge features: {graph.edge_attr.shape[1]} dims")

    # Load model
    logger.info("Loading trained model...")
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)

    # Load model args from checkpoint if available
    ckpt_args = checkpoint.get('args', {})
    node_dim = ckpt_args.get('node_dim', 34)
    edge_dim = ckpt_args.get('edge_dim', 10)
    hidden_dim = ckpt_args.get('hidden_dim', 512)
    num_layers = ckpt_args.get('num_layers', 12)
    num_heads = ckpt_args.get('num_heads', 16)

    model = QCGN2oEI_Minimal(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        output_dim=max_mz - min_mz + 1,
        dropout=0.1
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logger.info(f"  Model loaded from: {model_path}")
    logger.info(f"  Using device: {device}")

    # Inference
    logger.info("Predicting spectrum...")
    graph = graph.to(device)

    start_time = time.time()
    with torch.no_grad():
        spectrum_pred = model(
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            graph.batch
        )
    inference_time = time.time() - start_time

    spectrum = spectrum_pred.cpu().numpy().squeeze()

    logger.info(f"  Prediction complete ({inference_time*1000:.2f} ms)")
    logger.info(f"  Spectrum shape: {spectrum.shape}")
    logger.info(f"  Max intensity: {spectrum.max():.4f}")
    logger.info(f"  Number of peaks (intensity > 0.01): {(spectrum > 0.01).sum()}")

    # Metadata
    metadata = {
        'smiles': smiles,
        'properties': properties,
        'inference_time_ms': inference_time * 1000,
        'spectrum_shape': spectrum.shape,
        'max_intensity': float(spectrum.max()),
        'num_peaks': int((spectrum > 0.01).sum()),
        'mz_range': (min_mz, max_mz)
    }

    return spectrum, metadata


def get_top_peaks(
    spectrum: np.ndarray,
    top_k: int = 10,
    mz_offset: int = 1,
    min_intensity: float = 0.001
) -> list[tuple[int, float]]:
    """
    Extract top-K peaks from spectrum

    Args:
        spectrum: Intensity array
        top_k: Number of top peaks
        mz_offset: m/z offset (default: 1 for m/z 1-1000)
        min_intensity: Minimum intensity threshold

    Returns:
        List of (m/z, intensity) tuples
    """
    # Find top-K indices
    top_indices = np.argsort(spectrum)[-top_k:][::-1]

    # Convert to (m/z, intensity) pairs
    peaks = []
    for idx in top_indices:
        mz = idx + mz_offset
        intensity = spectrum[idx]
        if intensity > min_intensity:
            peaks.append((mz, intensity))

    return peaks


def visualize_spectrum(
    spectrum: np.ndarray,
    smiles: str,
    output_path: str = "predicted_spectrum.png",
    mz_range: tuple[int, int] = (1, 1000),
    figsize: tuple[int, int] = (12, 6),
    dpi: int = 300
):
    """
    Visualize predicted spectrum

    Args:
        spectrum: Predicted intensity array
        smiles: SMILES string (for title)
        output_path: Path to save plot
        mz_range: m/z range (min, max)
        figsize: Figure size
        dpi: DPI for output
    """
    mz_values = np.arange(mz_range[0], mz_range[0] + len(spectrum))

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Stem plot for spectrum
    markerline, stemlines, baseline = ax.stem(
        mz_values,
        spectrum,
        linefmt='b-',
        markerfmt='bo',
        basefmt=' '
    )
    stemlines.set_linewidth(0.5)
    markerline.set_markersize(2)

    ax.set_xlabel("m/z", fontsize=12, fontweight='bold')
    ax.set_ylabel("Relative Intensity", fontsize=12, fontweight='bold')
    ax.set_title(f"Predicted EI-MS Spectrum\n{smiles}", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(mz_range)
    ax.set_ylim(0, spectrum.max() * 1.1)

    # Add annotation for base peak
    base_peak_idx = np.argmax(spectrum)
    base_peak_mz = base_peak_idx + mz_range[0]
    base_peak_intensity = spectrum[base_peak_idx]

    ax.annotate(
        f'Base peak\nm/z {base_peak_mz}\n({base_peak_intensity:.3f})',
        xy=(base_peak_mz, base_peak_intensity),
        xytext=(base_peak_mz + 50, base_peak_intensity * 0.8),
        arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"\n✅ Spectrum saved to: {output_path}")


def print_results(
    spectrum: np.ndarray,
    metadata: dict,
    top_k: int = 10
):
    """
    Print prediction results to console

    Args:
        spectrum: Predicted spectrum
        metadata: Metadata dictionary
        top_k: Number of top peaks to show
    """
    print("\n" + "=" * 70)
    print("NExtIMS v4.2: Single Molecule Prediction Results")
    print("=" * 70)
    print(f"SMILES: {metadata['smiles']}")
    print(f"Formula: {metadata['properties']['formula']}")
    print(f"Molecular Weight: {metadata['properties']['molecular_weight']:.2f} Da")
    print(f"Atoms: {metadata['properties']['num_atoms']} ({metadata['properties']['num_heavy_atoms']} heavy)")
    print("-" * 70)
    print(f"Inference Time: {metadata['inference_time_ms']:.2f} ms")
    print(f"Max Intensity: {metadata['max_intensity']:.4f}")
    print(f"Number of Peaks (>1%): {metadata['num_peaks']}")
    print("-" * 70)

    # Top peaks
    top_peaks = get_top_peaks(spectrum, top_k=top_k, mz_offset=metadata['mz_range'][0])

    print(f"\nTop {len(top_peaks)} Predicted Peaks:")
    print("-" * 70)
    print(f"{'Rank':<6} {'m/z':<8} {'Intensity':<12} {'Relative %'}")
    print("-" * 70)

    if len(top_peaks) > 0:
        base_peak_intensity = top_peaks[0][1]
        for i, (mz, intensity) in enumerate(top_peaks, 1):
            relative_pct = (intensity / base_peak_intensity) * 100
            print(f"{i:<6} {mz:<8} {intensity:<12.4f} {relative_pct:>6.1f}%")
    else:
        print("No significant peaks found.")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Predict EI-MS spectrum for a single molecule"
    )

    # Required arguments
    parser.add_argument(
        'smiles',
        type=str,
        help='SMILES string of the molecule'
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
        '--output',
        type=str,
        default='predicted_spectrum.png',
        help='Output plot path (default: predicted_spectrum.png)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plot'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of top peaks to display (default: 10)'
    )
    parser.add_argument(
        '--min-mz',
        type=int,
        default=1,
        help='Minimum m/z (default: 1)'
    )
    parser.add_argument(
        '--max-mz',
        type=int,
        default=1000,
        help='Maximum m/z (default: 1000)'
    )

    args = parser.parse_args()

    try:
        # Predict spectrum
        spectrum, metadata = predict_spectrum(
            smiles=args.smiles,
            model_path=args.model,
            bde_cache_path=args.bde_cache,
            device=args.device,
            min_mz=args.min_mz,
            max_mz=args.max_mz
        )

        # Print results
        print_results(spectrum, metadata, top_k=args.top_k)

        # Visualize (optional)
        if args.visualize:
            visualize_spectrum(
                spectrum=spectrum,
                smiles=args.smiles,
                output_path=args.output,
                mz_range=(args.min_mz, args.max_mz)
            )

        logger.info("Prediction complete!")

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
