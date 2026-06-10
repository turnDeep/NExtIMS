#!/usr/bin/env python3
# src/data/bde_calculator.py
"""
NExtIMS v4.2 BDE (Bond Dissociation Energy) Calculator

Uses BonDNet for accurate BDE prediction with support for:
- Pre-trained BonDNet models (bdncm/20200808)
- Custom models retrained on BDE-db2 dataset
- HDF5 caching for efficient reuse
- 10 elements: C, H, N, O, S, Cl, F, P, Br, I

Migration from ALFABET (v2.0) to BonDNet (v4.2):
- Higher accuracy: MAE 0.51 vs 0.60 kcal/mol
- Broader coverage: 10 elements vs 4 (C, H, O, N)
- Halogen support: F, Cl, Br, I (critical for NIST17)
- Ring bonds: Properly predicted (ALFABET limitation removed)
- PyTorch-only: TensorFlow dependencies eliminated

BonDNet: https://github.com/mjwen/bondnet
Paper: https://www.nature.com/articles/s41467-021-25639-8
BDE-db2: 531,244 BDEs from 65,540 molecules
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
from rdkit import Chem

logger = logging.getLogger(__name__)
EV_TO_KCAL_MOL = 23.0605

# Try to import BonDNet
try:
    from bondnet.prediction.predictor import predict_single_molecule, get_prediction
    from bondnet.prediction.load_model import get_model_path, get_model_info, load_model
    from bondnet.prediction.io import PredictionOneReactant
    BONDNET_AVAILABLE = True
    logger.info("BonDNet successfully imported")
except ImportError as e:
    BONDNET_AVAILABLE = False
    logger.warning(f"BonDNet not available: {e}")
    logger.warning("Install with: pip install git+https://github.com/mjwen/bondnet.git")


class BDECalculator:
    """
    BDE Calculator using BonDNet

    Predicts bond dissociation energies for all bonds in a molecule.
    Supports both pre-trained and custom-trained BonDNet models.

    Key Features:
    - Multi-charge prediction: Evaluates neutral, cation, anion
    - Minimum BDE selection: Most favorable fragmentation path
    - Ring bond support: All bonds predicted (vs ALFABET limitation)
    - Unit conversion: eV → kcal/mol (×23.06)

    Usage:
        # Use pre-trained model
        calc = BDECalculator(model_name="bdncm/20200808")
        bde_dict = calc.calculate_bde("c1ccccc1")

        # Use custom retrained model
        calc = BDECalculator(model_path="models/bondnet_bde_db2.pth")
        bde_dict = calc.calculate_bde("CC(=O)O")
    """

    def __init__(
        self,
        model_name: Optional[str] = "bdncm/20200808",
        model_path: Optional[str] = None,
        allowed_charges: List[int] = [0, -1, 1],
        ring_bond: bool = True,
        device: str = 'cuda'
    ):
        """
        Initialize BDE Calculator

        Args:
            model_name: BonDNet pre-trained model name (default: "bdncm/20200808")
            model_path: Path to custom trained model (overrides model_name)
            allowed_charges: Charge states to evaluate ([0, -1, 1] default)
            ring_bond: Whether to predict ring bonds (True recommended)
            device: 'cuda' or 'cpu'
        """
        if not BONDNET_AVAILABLE:
            raise ImportError(
                "BonDNet is required. Install with:\n"
                "  pip install git+https://github.com/mjwen/bondnet.git"
            )

        self.allowed_charges = allowed_charges
        self.ring_bond = ring_bond
        self.device = device

        # Load model
        if model_path:
            # Custom model
            self.model_path = Path(model_path)
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            logger.info(f"Using custom BonDNet model: {model_path}")
            self.model_name = None
        else:
            # Pre-trained model
            self.model_name = model_name
            self.model_path = get_model_path(model_name)
            logger.info(f"Using pre-trained BonDNet model: {model_name}")

        # Get model info (includes unit conversion factor)
        model_info = get_model_info(self.model_path)
        unit_converter = float(model_info.get("unit_conversion", EV_TO_KCAL_MOL))
        # BonDNet's official BDNCM model reports unit_conversion: 1.0 and returns
        # BDE-like values in eV. NExtIMS edge features are normalized as kcal/mol.
        self.unit_converter = EV_TO_KCAL_MOL if unit_converter <= 1.0 else unit_converter

        logger.info(f"Unit conversion to kcal/mol: ×{self.unit_converter}")
        logger.info(f"Charge states: {self.allowed_charges}")
        logger.info(f"Ring bonds: {self.ring_bond}")

    @staticmethod
    def _bond_key_to_index(mol: Chem.Mol, bond_key) -> Optional[int]:
        """
        Convert BonDNet bond identifiers to RDKit bond indices.

        Recent BonDNet pretrained prediction paths return atom-index tuples
        such as (0, 1), while older NExtIMS code expected integer bond ids.
        The GNN graph only has explicit RDKit bonds, so implicit-H reactions are
        ignored by returning None.
        """
        if isinstance(bond_key, (int, np.integer)):
            return int(bond_key)
        if isinstance(bond_key, (tuple, list)) and len(bond_key) >= 2:
            begin = int(bond_key[0])
            end = int(bond_key[1])
            if begin < 0 or end < 0:
                return None
            if begin >= mol.GetNumAtoms() or end >= mol.GetNumAtoms():
                return None
            bond = mol.GetBondBetweenAtoms(begin, end)
            if bond is not None:
                return int(bond.GetIdx())
        return None

    def calculate_bde(
        self,
        smiles: str,
        charge: int = 0,
        return_all_charges: bool = False
    ) -> Dict[int, float]:
        """
        Calculate BDE for all bonds in a molecule

        Evaluates multiple charge states and returns minimum BDE per bond
        (most favorable fragmentation pathway).

        Args:
            smiles: SMILES string
            charge: Initial molecule charge (0 for neutral)
            return_all_charges: If True, return all charge states

        Returns:
            bde_dict: {bond_idx: BDE (kcal/mol)}

        Example:
            >>> calc = BDECalculator()
            >>> bde = calc.calculate_bde("c1ccccc1")  # Benzene
            >>> print(bde)
            {0: 124.3, 1: 124.3, 2: 124.3, 3: 124.3, 4: 124.3, 5: 124.3}
        """
        try:
            # Create predictor for this molecule
            predictor = PredictionOneReactant(
                smiles,
                charge=charge,
                format='smiles',
                allowed_product_charges=self.allowed_charges,
                ring_bond=self.ring_bond,
                one_per_iso_bond_group=True  # One reaction per bond group
            )

            # Prepare data for BonDNet
            molecules, labels, extra_features = predictor.prepare_data()

            # Get predictions
            predictions = get_prediction(
                self.model_path,
                self.unit_converter,
                molecules,
                labels,
                extra_features
            )

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}

            # Map predictions to bond indices
            # BonDNet returns one prediction per reaction (bond × charge state)
            # We need to group by bond and select minimum BDE
            predictions_by_bond = defaultdict(list)

            for rxn_idx, pred_value in enumerate(predictions):
                if pred_value is not None and not np.isnan(pred_value):
                    bond_key = predictor.rxn_idx_to_bond_map[rxn_idx]
                    bond_idx = self._bond_key_to_index(mol, bond_key)
                    if bond_idx is not None:
                        predictions_by_bond[bond_idx].append(float(pred_value))

            # Select minimum BDE per bond (most favorable fragmentation)
            bde_dict = {}
            for bond_idx, bde_values in predictions_by_bond.items():
                if bde_values:
                    bde_dict[bond_idx] = min(bde_values)  # Most favorable

            return bde_dict

        except Exception as e:
            logger.warning(f"BDE calculation failed for {smiles}: {e}")
            return {}

    def calculate_bde_batch(
        self,
        smiles_list: List[str],
        show_progress: bool = True
    ) -> Dict[str, Dict[int, float]]:
        """
        Calculate BDE for multiple molecules

        Args:
            smiles_list: List of SMILES strings
            show_progress: Show progress bar

        Returns:
            results: {smiles: {bond_idx: BDE (kcal/mol)}}
        """
        results = {}

        iterator = smiles_list
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(smiles_list, desc="Calculating BDE")
            except ImportError:
                pass

        for smiles in iterator:
            bde_dict = self.calculate_bde(smiles)
            if bde_dict:
                results[smiles] = bde_dict

        return results

    def get_bde_for_bond(
        self,
        smiles: str,
        bond_idx: int
    ) -> Optional[float]:
        """
        Get BDE for a specific bond

        Args:
            smiles: SMILES string
            bond_idx: Bond index

        Returns:
            BDE value (kcal/mol) or None if failed
        """
        bde_dict = self.calculate_bde(smiles)
        return bde_dict.get(bond_idx)

    def validate_molecule(self, smiles: str) -> Tuple[bool, str]:
        """
        Validate if molecule can be processed by BonDNet

        Args:
            smiles: SMILES string

        Returns:
            (is_valid, error_message)
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False, "Invalid SMILES"

            # Check elements
            allowed_elements = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}  # H, C, N, O, F, P, S, Cl, Br, I
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() not in allowed_elements:
                    element = atom.GetSymbol()
                    return False, f"Unsupported element: {element}"

            # Check if molecule has bonds
            if mol.GetNumBonds() == 0:
                return False, "No bonds in molecule"

            return True, "OK"

        except Exception as e:
            return False, str(e)


class BDECache:
    """
    HDF5 cache for BDE values

    Stores pre-computed BDE values for efficient reuse.
    Format: {smiles: {bond_idx: BDE_value}}
    """

    def __init__(self, cache_path: str):
        """
        Initialize BDE cache

        Args:
            cache_path: Path to HDF5 cache file
        """
        try:
            import h5py
            self.h5py = h5py
        except ImportError:
            raise ImportError("h5py is required. Install with: pip install h5py")

        self.cache_path = Path(cache_path)
        self.cache_file = None

        # Create directory if needed
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    def open(self, mode='r'):
        """Open cache file"""
        self.cache_file = self.h5py.File(str(self.cache_path), mode)
        return self

    def close(self):
        """Close cache file"""
        if self.cache_file:
            self.cache_file.close()
            self.cache_file = None

    def __enter__(self):
        return self.open('r')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get(self, smiles: str) -> Optional[Dict[int, float]]:
        """
        Get BDE values from cache

        Args:
            smiles: SMILES string

        Returns:
            bde_dict or None if not cached
        """
        if not self.cache_file:
            raise RuntimeError("Cache file not opened")

        if smiles not in self.cache_file:
            return None

        try:
            bde_group = self.cache_file[smiles]
            bde_dict = {}

            for bond_idx_str in bde_group.keys():
                bond_idx = int(bond_idx_str)
                bde_value = float(bde_group[bond_idx_str][()])
                bde_dict[bond_idx] = bde_value

            return bde_dict

        except Exception as e:
            logger.warning(f"Failed to read from cache: {e}")
            return None

    def put(self, smiles: str, bde_dict: Dict[int, float]):
        """
        Store BDE values in cache

        Args:
            smiles: SMILES string
            bde_dict: {bond_idx: BDE_value}
        """
        if not self.cache_file:
            raise RuntimeError("Cache file not opened")

        # Create group for this molecule
        if smiles in self.cache_file:
            del self.cache_file[smiles]

        grp = self.cache_file.create_group(smiles)

        # Store BDE values for each bond
        for bond_idx, bde_value in bde_dict.items():
            grp.create_dataset(
                str(bond_idx),
                data=bde_value,
                dtype='float32'
            )

    def exists(self) -> bool:
        """Check if cache file exists"""
        return self.cache_path.exists()

    def get_stats(self) -> Dict:
        """
        Get cache statistics

        Returns:
            stats: {num_molecules, creation_time, model, ...}
        """
        if not self.cache_file:
            with self.open('r') as cache:
                return cache.get_stats()

        stats = {
            'num_molecules': len(self.cache_file.keys()),
            'file_size_mb': self.cache_path.stat().st_size / (1024 * 1024)
        }

        # Add metadata attributes
        for key in self.cache_file.attrs.keys():
            stats[key] = self.cache_file.attrs[key]

        return stats


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Test BDE calculation
    print("=" * 60)
    print("BDE Calculator Test")
    print("=" * 60)

    # Create calculator
    calc = BDECalculator(model_name="bdncm/20200808")

    # Test molecules
    test_molecules = [
        ("c1ccccc1", "Benzene"),
        ("CC(=O)O", "Acetic acid"),
        ("CCO", "Ethanol"),
        ("c1cc(O)ccc1C(=O)O", "Aspirin (simplified)")
    ]

    for smiles, name in test_molecules:
        print(f"\n{name}: {smiles}")

        # Validate
        is_valid, msg = calc.validate_molecule(smiles)
        if not is_valid:
            print(f"  ✗ Invalid: {msg}")
            continue

        # Calculate BDE
        bde_dict = calc.calculate_bde(smiles)

        if bde_dict:
            print(f"  ✓ {len(bde_dict)} bonds predicted")
            print(f"  BDE range: {min(bde_dict.values()):.1f} - {max(bde_dict.values()):.1f} kcal/mol")

            # Show first 5 bonds
            for bond_idx, bde in list(bde_dict.items())[:5]:
                print(f"    Bond {bond_idx}: {bde:.2f} kcal/mol")
            if len(bde_dict) > 5:
                print(f"    ... and {len(bde_dict) - 5} more bonds")
        else:
            print(f"  ✗ Prediction failed")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
