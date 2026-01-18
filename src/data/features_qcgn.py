#!/usr/bin/env python3
# src/data/features_qcgn.py
"""
NExtIMS v4.4: Stereochemistry & Scaled Architecture

Implements enhanced node and edge features with Stereochemistry:
- Node features (34 dims):
    - 15 atom types (one-hot)
    - 6 hybridization types (one-hot)
    - 1 aromaticity (bool)
    - 5 total Num Hs (one-hot)
    - 1 formal charge (integer)
    - 1 radical electrons (integer)
    - 1 in ring (bool)
    - 4 chiral tag (one-hot) [NEW]
- Edge features (10 dims):
    - 1 bond_order (float)
    - 1 BDE (normalized)
    - 1 in_ring (bool)
    - 1 is_conjugated (bool)
    - 6 bond stereo (one-hot) [NEW]

Design Philosophy:
- Incorporate chirality and cis/trans isomerism (critical for mass spec differentiation)
- Prepare features for scaled-up model architecture
"""

import numpy as np
import torch
from rdkit import Chem
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class QCGNFeaturizer:
    """
    Enhanced molecular featurizer for v4.4 (Stereochemistry)
    Supports ablation studies by toggling specific features.
    """

    # Atom masses for one-hot encoding
    ATOM_MASSES = np.array([
        12.011,  # C  (0)
        15.999,  # O  (1)
        1.008,   # H  (2)
        14.007,  # N  (3)
        32.067,  # S  (4)
        28.086,  # Si (5)
        30.974,  # P  (6)
        35.453,  # Cl (7)
        18.998,  # F  (8)
        126.904, # I  (9)
        78.96,   # Se (10)
        74.922,  # As (11)
        10.812,  # B  (12)
        79.904,  # Br (13)
        118.711  # Sn (14)
    ], dtype=np.float32)

    # Hybridization types
    HYBRIDIZATIONS = [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]

    # Chiral Tags
    CHIRAL_TAGS = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ]

    # Bond Stereo
    BOND_STEREO = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS
    ]

    # BDE normalization range (kcal/mol)
    BDE_MIN = 50.0
    BDE_MAX = 200.0

    def __init__(
        self,
        use_bde: bool = True,
        bde_min: float = 50.0,
        bde_max: float = 200.0,
        include_atom_type: bool = True,
        include_hybridization: bool = True,
        include_aromaticity: bool = True,
        include_num_hs: bool = True,
        include_formal_charge: bool = True,
        include_radical_electrons: bool = True,
        include_ring_info: bool = True,
        include_chirality: bool = True,
        include_bond_order: bool = True,
        include_conjugation: bool = True,
        include_stereo: bool = True
    ):
        """
        Initialize Enhanced QCGN featurizer with ablation support

        Args:
            use_bde: Whether to include BDE in edge features
            bde_min: Minimum BDE for normalization (kcal/mol)
            bde_max: Maximum BDE for normalization (kcal/mol)
            include_*: Flags to include specific features
        """
        self.use_bde = use_bde
        self.bde_min = bde_min
        self.bde_max = bde_max

        # Node feature flags
        self.include_atom_type = include_atom_type
        self.include_hybridization = include_hybridization
        self.include_aromaticity = include_aromaticity
        self.include_num_hs = include_num_hs
        self.include_formal_charge = include_formal_charge
        self.include_radical_electrons = include_radical_electrons
        self.include_ring_info = include_ring_info
        self.include_chirality = include_chirality

        # Edge feature flags
        self.include_bond_order = include_bond_order
        self.include_conjugation = include_conjugation
        self.include_stereo = include_stereo

        # Calculate dimensions dynamically
        self.node_dim = 0
        if self.include_atom_type: self.node_dim += 15
        if self.include_hybridization: self.node_dim += 6
        if self.include_aromaticity: self.node_dim += 1
        if self.include_num_hs: self.node_dim += 5
        if self.include_formal_charge: self.node_dim += 1
        if self.include_radical_electrons: self.node_dim += 1
        if self.include_ring_info: self.node_dim += 1
        if self.include_chirality: self.node_dim += 4

        self.edge_dim = 0
        if self.include_bond_order: self.edge_dim += 1
        if self.use_bde: self.edge_dim += 1
        if self.include_ring_info: self.edge_dim += 1
        if self.include_conjugation: self.edge_dim += 1
        if self.include_stereo: self.edge_dim += 6

        logger.info("QCGNFeaturizer (v4.4 Ablation Ready) initialized:")
        logger.info(f"  Node features: {self.node_dim}-dim")
        logger.info(f"  Edge features: {self.edge_dim}-dim")

    def get_atom_features(self, atom: Chem.Atom) -> np.ndarray:
        """
        Get enhanced node features

        Args:
            atom: RDKit Atom object

        Returns:
            features: numpy array
        """
        features_list = []

        # 1. Atom type via mass matching (15-dim one-hot)
        if self.include_atom_type:
            feat = np.zeros(15, dtype=np.float32)
            atom_mass = atom.GetMass()
            mass_diffs = np.abs(self.ATOM_MASSES - atom_mass)
            mass_idx = np.argmin(mass_diffs)
            if mass_diffs[mass_idx] < 0.1:
                feat[mass_idx] = 1.0
            features_list.append(feat)

        # 2. Hybridization (6-dim one-hot)
        if self.include_hybridization:
            feat = np.zeros(6, dtype=np.float32)
            hyb = atom.GetHybridization()
            try:
                hyb_idx = self.HYBRIDIZATIONS.index(hyb)
                feat[hyb_idx] = 1.0
            except ValueError:
                pass
            features_list.append(feat)

        # 3. Is Aromatic (1-dim bool)
        if self.include_aromaticity:
            feat = np.array([1.0 if atom.GetIsAromatic() else 0.0], dtype=np.float32)
            features_list.append(feat)

        # 4. Total Num Hs (5-dim one-hot: 0, 1, 2, 3, 4+)
        if self.include_num_hs:
            feat = np.zeros(5, dtype=np.float32)
            num_hs = atom.GetTotalNumHs()
            hs_idx = min(num_hs, 4)
            feat[hs_idx] = 1.0
            features_list.append(feat)

        # 5. Formal Charge (1-dim integer)
        if self.include_formal_charge:
            feat = np.array([float(atom.GetFormalCharge())], dtype=np.float32)
            features_list.append(feat)

        # 6. Radical Electrons (1-dim integer)
        if self.include_radical_electrons:
            feat = np.array([float(atom.GetNumRadicalElectrons())], dtype=np.float32)
            features_list.append(feat)

        # 7. In Ring (1-dim bool)
        if self.include_ring_info:
            feat = np.array([1.0 if atom.IsInRing() else 0.0], dtype=np.float32)
            features_list.append(feat)

        # 8. Chiral Tag (4-dim one-hot) [NEW]
        if self.include_chirality:
            feat = np.zeros(4, dtype=np.float32)
            chi = atom.GetChiralTag()
            try:
                chi_idx = self.CHIRAL_TAGS.index(chi)
                feat[chi_idx] = 1.0
            except ValueError:
                pass
            features_list.append(feat)

        if not features_list:
            return np.array([], dtype=np.float32)
        return np.concatenate(features_list)

    def get_edge_features(
        self,
        bond: Chem.Bond,
        bde_value: Optional[float] = None
    ) -> np.ndarray:
        """
        Get enhanced edge features

        Args:
            bond: RDKit Bond object
            bde_value: BDE value in kcal/mol (optional)

        Returns:
            features: numpy array
        """
        features_list = []

        # 1. Bond order
        if self.include_bond_order:
            bond_type_map = {
                Chem.BondType.SINGLE: 1.0,
                Chem.BondType.DOUBLE: 2.0,
                Chem.BondType.TRIPLE: 3.0,
                Chem.BondType.AROMATIC: 1.5
            }
            feat = np.array([bond_type_map.get(bond.GetBondType(), 1.0)], dtype=np.float32)
            features_list.append(feat)

        # 2. BDE normalized
        if self.use_bde:
            if bde_value is not None:
                bde_normalized = (bde_value - self.bde_min) / (self.bde_max - self.bde_min)
                val = np.clip(bde_normalized, 0.0, 1.0)
            else:
                val = 0.5
            features_list.append(np.array([val], dtype=np.float32))

        # 3. In ring
        if self.include_ring_info:
            feat = np.array([float(bond.IsInRing())], dtype=np.float32)
            features_list.append(feat)

        # 4. Is Conjugated
        if self.include_conjugation:
            feat = np.array([float(bond.GetIsConjugated())], dtype=np.float32)
            features_list.append(feat)

        # 5. Bond Stereo (6-dim one-hot) [NEW]
        if self.include_stereo:
            feat = np.zeros(6, dtype=np.float32)
            stereo = bond.GetStereo()
            try:
                stereo_idx = self.BOND_STEREO.index(stereo)
                feat[stereo_idx] = 1.0
            except ValueError:
                pass
            features_list.append(feat)

        if not features_list:
            return np.array([], dtype=np.float32)
        return np.concatenate(features_list)

    def normalize_bde(self, bde: float) -> float:
        bde_norm = (bde - self.bde_min) / (self.bde_max - self.bde_min)
        return np.clip(bde_norm, 0.0, 1.0)

    def denormalize_bde(self, bde_normalized: float) -> float:
        return bde_normalized * (self.bde_max - self.bde_min) + self.bde_min

    def get_node_dim(self) -> int:
        return self.node_dim

    def get_edge_dim(self) -> int:
        return self.edge_dim

    def validate_molecule(self, mol: Chem.Mol) -> Tuple[bool, str]:
        if mol is None:
            return False, "Invalid molecule (None)"
        if mol.GetNumAtoms() == 0:
            return False, "No atoms in molecule"

        unsupported_elements = []
        for atom in mol.GetAtoms():
            atom_mass = atom.GetMass()
            mass_diffs = np.abs(self.ATOM_MASSES - atom_mass)
            if mass_diffs.min() >= 0.1:
                unsupported_elements.append(atom.GetSymbol())

        if unsupported_elements:
            unique_elements = sorted(set(unsupported_elements))
            return False, f"Unsupported elements: {', '.join(unique_elements)}"

        return True, "OK"

    def get_feature_info(self) -> Dict:
        return {
            'node_dim': self.node_dim,
            'edge_dim': self.edge_dim,
            'atom_types': [
                'C', 'O', 'H', 'N', 'S', 'Si', 'P', 'Cl',
                'F', 'I', 'Se', 'As', 'B', 'Br', 'Sn'
            ],
            'use_bde': self.use_bde,
            'bde_range_kcal_mol': (self.bde_min, self.bde_max),
            'edge_features': ['bond_order', 'BDE_normalized', 'in_ring', 'is_conjugated', 'bond_stereo']
        }


# Example usage and testing
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Initialize featurizer with default (Full)
    print("\n" + "="*60)
    print("Testing Full Features")
    featurizer = QCGNFeaturizer()
    info = featurizer.get_feature_info()
    print(f"Node dim: {info['node_dim']} (Expected 34)")
    print(f"Edge dim: {info['edge_dim']} (Expected 10)")

    # Test Ablation: No BDE
    print("\n" + "="*60)
    print("Testing Ablation: No BDE")
    featurizer_no_bde = QCGNFeaturizer(use_bde=False)
    info = featurizer_no_bde.get_feature_info()
    print(f"Node dim: {info['node_dim']} (Expected 34)")
    print(f"Edge dim: {info['edge_dim']} (Expected 9)")

    # Test Ablation: No Bond Order
    print("\n" + "="*60)
    print("Testing Ablation: No Bond Order")
    featurizer_no_bo = QCGNFeaturizer(include_bond_order=False)
    info = featurizer_no_bo.get_feature_info()
    print(f"Node dim: {info['node_dim']} (Expected 34)")
    print(f"Edge dim: {info['edge_dim']} (Expected 9)")

    # Test Ablation: No Chirality
    print("\n" + "="*60)
    print("Testing Ablation: No Chirality")
    featurizer_no_chi = QCGNFeaturizer(include_chirality=False)
    info = featurizer_no_chi.get_feature_info()
    print(f"Node dim: {info['node_dim']} (Expected 30)")
    print(f"Edge dim: {info['edge_dim']} (Expected 10)")

    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
