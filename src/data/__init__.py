# src/data/__init__.py
"""
NEIMS v2.0 Data Module

Datasets and utilities for NEIMS v2.0 training and evaluation.
"""

from .nist_dataset import (
    NISTDataset,
    parse_msp_file,
    peaks_to_spectrum,
    mol_to_graph,
    mol_to_ecfp,
    mol_to_count_fp
)

from .preprocessing import (
    validate_smiles,
    canonicalize_smiles,
    filter_by_molecular_weight,
    filter_by_num_atoms,
    normalize_spectrum,
    remove_noise_peaks,
    peaks_to_spectrum_array,
    spectrum_to_peaks,
    compute_spectrum_statistics,
    compute_molecular_descriptors,
    split_dataset,
    batch_to_device
)

__all__ = [
    # NIST Dataset
    'NISTDataset',
    'parse_msp_file',
    'peaks_to_spectrum',
    'mol_to_graph',
    'mol_to_ecfp',
    'mol_to_count_fp',

    # Preprocessing
    'validate_smiles',
    'canonicalize_smiles',
    'filter_by_molecular_weight',
    'filter_by_num_atoms',
    'normalize_spectrum',
    'remove_noise_peaks',
    'peaks_to_spectrum_array',
    'spectrum_to_peaks',
    'compute_spectrum_statistics',
    'compute_molecular_descriptors',
    'split_dataset',
    'batch_to_device',
]
