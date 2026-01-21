#!/usr/bin/env python3
"""
NExtIMS v1.0: BDE Pre-computation Script

Pre-computes Bond Dissociation Energy (BDE) values for NIST17 EI-MS dataset
using BonDNet, storing results in HDF5 cache for efficient training.

This replaces the v2.0 PCQM4Mv2 pretraining approach with a direct NIST17
BDE precomputation strategy suitable for the minimal v1.0 architecture.

Migration from v2.0:
    - Dataset: PCQM4Mv2 (3.74M molecules) → NIST17 (~300K compounds)
    - Model: ALFABET (4 elements) → BonDNet (10 elements)
    - Coverage: C,H,O,N only → C,H,N,O,S,Cl,F,P,Br,I
    - Accuracy: MAE 0.60 → 0.51 kcal/mol
    - Ring bonds: Estimated → Predicted

Usage:
    # Use pre-trained BonDNet model
    python scripts/precompute_bde.py \\
        --nist-msp data/NIST17.MSP \\
        --output data/processed/bde_cache/nist17_bde_cache.h5

    # Use custom retrained model (after train_bondnet_bde_db2.py)
    python scripts/precompute_bde.py \\
        --nist-msp data/NIST17.MSP \\
        --model models/bondnet_bde_db2.pth \\
        --output data/processed/bde_cache/nist17_bde_cache_custom.h5

Estimated Time:
    - Pre-trained model: ~30-60 min for 300K molecules (GPU)
    - Custom model: Similar performance
    - Storage: ~2-5 GB HDF5 file

Requirements:
    - PyTorch >= 1.10.0
    - BonDNet (pip install git+https://github.com/mjwen/bondnet.git)
    - RDKit, h5py
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
import warnings

import h5py
import numpy as np
from tqdm import tqdm
from rdkit import Chem

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.bde_calculator import BDECalculator, BDECache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_nist_msp(msp_path: str, max_compounds: int = 0) -> List[str]:
    """
    Extract SMILES from NIST MSP file

    Args:
        msp_path: Path to NIST17.MSP file
        max_compounds: Maximum compounds to process (0 = all)

    Returns:
        smiles_list: List of SMILES strings
    """
    logger.info(f"Parsing NIST MSP file: {msp_path}")

    smiles_list = []
    current_smiles = None

    with open(msp_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()

            if line.startswith('SMILES:'):
                current_smiles = line.split(':', 1)[1].strip()
                if current_smiles:
                    smiles_list.append(current_smiles)

                # Limit if requested
                if max_compounds > 0 and len(smiles_list) >= max_compounds:
                    break

    logger.info(f"Extracted {len(smiles_list):,} SMILES strings")
    return smiles_list


def validate_smiles_list(
    smiles_list: List[str],
    calculator: BDECalculator
) -> Dict:
    """
    Validate SMILES and report statistics

    Args:
        smiles_list: List of SMILES
        calculator: BDECalculator instance

    Returns:
        stats: {valid_count, invalid_count, unsupported_elements, ...}
    """
    logger.info("Validating SMILES...")

    valid_count = 0
    invalid_count = 0
    unsupported_elements = set()
    error_types = {}

    for smiles in tqdm(smiles_list, desc="Validating"):
        is_valid, error_msg = calculator.validate_molecule(smiles)

        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1

            # Track error types
            error_types[error_msg] = error_types.get(error_msg, 0) + 1

            # Extract unsupported elements
            if "Unsupported element" in error_msg:
                element = error_msg.split(":")[-1].strip()
                unsupported_elements.add(element)

    stats = {
        'total': len(smiles_list),
        'valid': valid_count,
        'invalid': invalid_count,
        'unsupported_elements': list(unsupported_elements),
        'error_types': error_types
    }

    logger.info(f"Validation results:")
    logger.info(f"  Valid: {valid_count:,} / {len(smiles_list):,} ({valid_count/len(smiles_list)*100:.1f}%)")
    logger.info(f"  Invalid: {invalid_count:,}")

    if unsupported_elements:
        logger.warning(f"  Unsupported elements: {', '.join(sorted(unsupported_elements))}")

    # Show top error types
    if error_types:
        logger.info("  Top errors:")
        for error, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"    - {error}: {count}")

    return stats


def precompute_bde_hdf5(
    smiles_list: List[str],
    calculator: BDECalculator,
    output_path: str,
    skip_validation: bool = False,
    batch_size: int = 100,
    checkpoint_interval: int = 10000
):
    """
    Pre-compute BDE values and save to HDF5

    Args:
        smiles_list: List of SMILES strings
        calculator: BDECalculator instance
        output_path: Output HDF5 file path
        skip_validation: Skip validation (faster but risky)
        batch_size: Processing batch size (for progress tracking)
        checkpoint_interval: Save checkpoint every N molecules
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate SMILES (optional)
    if not skip_validation:
        validation_stats = validate_smiles_list(smiles_list, calculator)
        logger.info(f"Proceeding with {validation_stats['valid']:,} valid molecules")
    else:
        logger.warning("Skipping validation (--skip-validation)")

    # Open HDF5 file for streaming writes
    logger.info(f"Creating HDF5 cache: {output_path}")

    success_count = 0
    failed_count = 0
    total_bde_values = 0
    bde_values_sample = []  # For statistics

    start_time = time.time()

    with h5py.File(output_path, 'w') as h5_file:
        # Store metadata
        h5_file.attrs['creation_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        h5_file.attrs['status'] = 'in_progress'
        h5_file.attrs['model'] = calculator.model_name or str(calculator.model_path)
        h5_file.attrs['total_molecules'] = len(smiles_list)

        # Process molecules
        for i, smiles in enumerate(tqdm(smiles_list, desc="Computing BDE")):
            try:
                # Calculate BDE
                bde_dict = calculator.calculate_bde(smiles)

                if bde_dict:
                    # Create group for this molecule
                    grp = h5_file.create_group(smiles)

                    # Store BDE values
                    for bond_idx, bde_value in bde_dict.items():
                        grp.create_dataset(
                            str(bond_idx),
                            data=bde_value,
                            dtype='float32'
                        )

                    success_count += 1
                    total_bde_values += len(bde_dict)

                    # Sample for statistics (first 100K bonds)
                    if len(bde_values_sample) < 100000:
                        bde_values_sample.extend(bde_dict.values())

                else:
                    failed_count += 1

            except Exception as e:
                logger.debug(f"Failed to process {smiles}: {e}")
                failed_count += 1

            # Periodic progress update
            if (i + 1) % checkpoint_interval == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(smiles_list) - i - 1) / rate if rate > 0 else 0

                logger.info(
                    f"Progress: {success_count:,} / {i+1:,} molecules "
                    f"({elapsed/60:.1f} min elapsed, ~{remaining/60:.1f} min remaining)"
                )

                # Update metadata checkpoint
                h5_file.attrs['checkpoint_index'] = i + 1
                h5_file.attrs['checkpoint_success'] = success_count
                h5_file.flush()

        # Final metadata
        h5_file.attrs['num_molecules'] = success_count
        h5_file.attrs['num_bde_values'] = total_bde_values
        h5_file.attrs['num_failed'] = failed_count
        h5_file.attrs['status'] = 'completed'

    elapsed_time = time.time() - start_time

    logger.info("")
    logger.info("="*80)
    logger.info("BDE Pre-computation Complete!")
    logger.info("="*80)
    logger.info(f"Time elapsed: {elapsed_time/60:.1f} minutes ({elapsed_time/3600:.2f} hours)")
    logger.info(f"Success: {success_count:,} / {len(smiles_list):,} molecules")
    logger.info(f"Failed: {failed_count:,}")
    logger.info(f"Total BDE values: {total_bde_values:,}")

    if success_count > 0:
        logger.info(f"Average bonds/molecule: {total_bde_values / success_count:.1f}")

    # BDE statistics
    if bde_values_sample:
        logger.info("")
        logger.info("BDE Statistics (from sample):")
        logger.info(f"  Mean: {np.mean(bde_values_sample):.2f} kcal/mol")
        logger.info(f"  Std:  {np.std(bde_values_sample):.2f} kcal/mol")
        logger.info(f"  Min:  {np.min(bde_values_sample):.2f} kcal/mol")
        logger.info(f"  Max:  {np.max(bde_values_sample):.2f} kcal/mol")

    # File size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Cache file size: {file_size_mb:.2f} MB")

    logger.info("")
    logger.info("Next steps:")
    logger.info(f"  1. Verify cache: python -c \"import h5py; f=h5py.File('{output_path}', 'r'); print(dict(f.attrs))\"")
    logger.info("  2. Use in training: Set config.yaml -> data.bde_cache_path")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute BDE values for NIST17 dataset using BonDNet"
    )
    parser.add_argument(
        '--nist-msp',
        type=str,
        required=True,
        help='Path to NIST17.MSP file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/bde_cache/nist17_bde_cache.h5',
        help='Output HDF5 file path'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to custom BonDNet model (optional, uses pre-trained if not specified)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='bdncm/20200808',
        help='Pre-trained model name (default: bdncm/20200808)'
    )
    parser.add_argument(
        '--max-compounds',
        type=int,
        default=0,
        help='Maximum compounds to process (0 = all, default: 0)'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip SMILES validation (faster but risky)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for progress tracking (default: 100)'
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=10000,
        help='Progress update interval (default: 10000)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing cache file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device: cuda or cpu (default: cuda)'
    )

    args = parser.parse_args()

    # Check if output exists
    if Path(args.output).exists() and not args.force:
        logger.error(f"Output file already exists: {args.output}")
        logger.error("Use --force to overwrite")
        sys.exit(1)

    # Check if NIST MSP exists
    if not Path(args.nist_msp).exists():
        logger.error(f"NIST MSP file not found: {args.nist_msp}")
        sys.exit(1)

    logger.info("="*80)
    logger.info("NExtIMS v1.0: BDE Pre-computation for NIST17")
    logger.info("="*80)
    logger.info(f"Input:  {args.nist_msp}")
    logger.info(f"Output: {args.output}")
    logger.info("")

    # Initialize BDE calculator
    logger.info("Initializing BDE calculator...")

    try:
        if args.model:
            # Custom model
            calculator = BDECalculator(
                model_path=args.model,
                device=args.device
            )
        else:
            # Pre-trained model
            calculator = BDECalculator(
                model_name=args.model_name,
                device=args.device
            )
    except Exception as e:
        logger.error(f"Failed to initialize BDE calculator: {e}")
        logger.error("Make sure BonDNet is installed:")
        logger.error("  pip install git+https://github.com/mjwen/bondnet.git")
        sys.exit(1)

    logger.info("✓ BDE calculator initialized")
    logger.info("")

    # Load SMILES from NIST MSP
    smiles_list = parse_nist_msp(args.nist_msp, args.max_compounds)

    if not smiles_list:
        logger.error("No SMILES found in MSP file")
        sys.exit(1)

    # Pre-compute BDE values
    precompute_bde_hdf5(
        smiles_list=smiles_list,
        calculator=calculator,
        output_path=args.output,
        skip_validation=args.skip_validation,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval
    )


if __name__ == '__main__':
    main()
