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

    Node Features (34-dim):
        - [0-14] 15 atom types (one-hot via mass matching)
        - [15-20] 6 hybridization types (S, SP, SP2, SP3, SP3D, SP3D2)
        - [21] Is Aromatic (bool)
        - [22-26] Total Num Hs (one-hot: 0, 1, 2, 3, 4+)
        - [27] Formal Charge (integer)
        - [28] Radical Electrons (integer)
        - [29] In Ring (bool)
        - [30-33] Chiral Tag (one-hot: Unspecified, CW, CCW, Other)

    Edge Features (10-dim):
        - [0] bond_order: 1.0 (single), 2.0 (double), 3.0 (triple), 1.5 (aromatic)
        - [1] BDE: Bond Dissociation Energy (kcal/mol), normalized [0, 1]
        - [2] in_ring: Binary indicator
        - [3] is_conjugated: Binary indicator
        - [4-9] Bond Stereo (one-hot: None, Any, Z, E, Cis, Trans)

    Key Differences from v4.3:
        - Node: 30-dim → 34-dim (Adds Chirality)
        - Edge: 4-dim → 10-dim (Adds Bond Stereo)
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
        bde_max: float = 200.0
    ):
        """
        Initialize Enhanced QCGN featurizer

        Args:
            use_bde: Whether to include BDE in edge features
            bde_min: Minimum BDE for normalization (kcal/mol)
            bde_max: Maximum BDE for normalization (kcal/mol)
        """
        self.use_bde = use_bde
        self.bde_min = bde_min
        self.bde_max = bde_max

        # Calculate dimensions
        self.node_dim = 34
        self.edge_dim = 10

        logger.info("QCGNFeaturizer (v4.4 Stereochemistry) initialized:")
        logger.info(f"  Node features: {self.node_dim}-dim (+Chirality)")
        logger.info(f"  Edge features: {self.edge_dim}-dim (+Stereo)")

    def get_atom_features(self, atom: Chem.Atom) -> np.ndarray:
        """
        Get enhanced node features (34-dim)

        Args:
            atom: RDKit Atom object

        Returns:
            features: [34] numpy array
        """
        features = np.zeros(self.node_dim, dtype=np.float32)
        offset = 0

        # 1. Atom type via mass matching (15-dim one-hot)
        atom_mass = atom.GetMass()
        mass_diffs = np.abs(self.ATOM_MASSES - atom_mass)
        mass_idx = np.argmin(mass_diffs)

        if mass_diffs[mass_idx] < 0.1:
            features[offset + mass_idx] = 1.0
        offset += 15

        # 2. Hybridization (6-dim one-hot)
        hyb = atom.GetHybridization()
        try:
            hyb_idx = self.HYBRIDIZATIONS.index(hyb)
            features[offset + hyb_idx] = 1.0
        except ValueError:
            pass
        offset += 6

        # 3. Is Aromatic (1-dim bool)
        features[offset] = 1.0 if atom.GetIsAromatic() else 0.0
        offset += 1

        # 4. Total Num Hs (5-dim one-hot: 0, 1, 2, 3, 4+)
        num_hs = atom.GetTotalNumHs()
        hs_idx = min(num_hs, 4)
        features[offset + hs_idx] = 1.0
        offset += 5

        # 5. Formal Charge (1-dim integer)
        features[offset] = float(atom.GetFormalCharge())
        offset += 1

        # 6. Radical Electrons (1-dim integer)
        features[offset] = float(atom.GetNumRadicalElectrons())
        offset += 1

        # 7. In Ring (1-dim bool)
        features[offset] = 1.0 if atom.IsInRing() else 0.0
        offset += 1

        # 8. Chiral Tag (4-dim one-hot) [NEW]
        chi = atom.GetChiralTag()
        try:
            chi_idx = self.CHIRAL_TAGS.index(chi)
            features[offset + chi_idx] = 1.0
        except ValueError:
            pass # Should not happen if list is complete
        offset += 4

        return features

    def get_edge_features(
        self,
        bond: Chem.Bond,
        bde_value: Optional[float] = None
    ) -> np.ndarray:
        """
        Get enhanced edge features (10-dim)

        Args:
            bond: RDKit Bond object
            bde_value: BDE value in kcal/mol (optional)

        Returns:
            features: [10] numpy array
        """
        features = np.zeros(self.edge_dim, dtype=np.float32)

        # 1. Bond order
        bond_type_map = {
            Chem.BondType.SINGLE: 1.0,
            Chem.BondType.DOUBLE: 2.0,
            Chem.BondType.TRIPLE: 3.0,
            Chem.BondType.AROMATIC: 1.5
        }
        features[0] = bond_type_map.get(bond.GetBondType(), 1.0)

        # 2. BDE normalized
        if self.use_bde and bde_value is not None:
            bde_normalized = (bde_value - self.bde_min) / (self.bde_max - self.bde_min)
            features[1] = np.clip(bde_normalized, 0.0, 1.0)
        else:
            features[1] = 0.5

        # 3. In ring
        features[2] = float(bond.IsInRing())

        # 4. Is Conjugated
        features[3] = float(bond.GetIsConjugated())

        # 5. Bond Stereo (6-dim one-hot) [NEW]
        stereo = bond.GetStereo()
        try:
            stereo_idx = self.BOND_STEREO.index(stereo)
            features[4 + stereo_idx] = 1.0
        except ValueError:
            pass

        return features

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

    # Initialize featurizer
    featurizer = QCGNFeaturizer()

    # Print feature info
    info = featurizer.get_feature_info()
    print("\n" + "="*60)
    print("QC-GN Enhanced Featurizer (v4.4) Information")
    print("="*60)
    print(f"Node dimension: {info['node_dim']}")
    print(f"Edge dimension: {info['edge_dim']}")
    print(f"Atom types: {', '.join(info['atom_types'])}")
    print(f"Edge features: {', '.join(info['edge_features'])}")

    # Test molecules
    test_smiles = [
        ("C[C@H](O)C(=O)O", "L-Lactic Acid (Chiral)"),
        ("C/C=C/C", "Trans-2-Butene"),
        ("C/C=C\C", "Cis-2-Butene"),
    ]

    print("\n" + "="*60)
    print("Testing Molecules")
    print("="*60)

    for smiles, name in test_smiles:
        mol = Chem.MolFromSmiles(smiles)

        is_valid, msg = featurizer.validate_molecule(mol)
        print(f"\n{name}: {smiles}")
        print(f"  Valid: {is_valid}")

        # Node features
        print(f"  Atoms: {mol.GetNumAtoms()}")
        for i, atom in enumerate(mol.GetAtoms()):
            features = featurizer.get_atom_features(atom)
            # Check chiral feature (last 4 dims)
            chiral_vec = features[-4:]
            if chiral_vec.sum() > 0:
                print(f"    Atom {i} ({atom.GetSymbol()}): Chiral={chiral_vec}")

        # Edge features
        print(f"  Bonds: {mol.GetNumBonds()}")
        for i, bond in enumerate(mol.GetBonds()):
            features = featurizer.get_edge_features(bond, bde_value=85.0)
            # Check stereo feature (last 6 dims)
            stereo_vec = features[-6:]
            if stereo_vec[0] == 0: # If not STEREONONE
                 print(f"    Bond {i}: Stereo={stereo_vec}")

    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
