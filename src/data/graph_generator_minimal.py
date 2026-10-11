#!/usr/bin/env python3
# src/data/graph_generator_minimal.py
"""Legacy graph generator without BDE dependencies.

The 10-column edge layout is preserved; the unused BDE column is constant 0.5.
Current training and prediction use scripts/nextims.py and its frozen support.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from torch_geometric.data import Data

from src.data.features_qcgn import QCGNFeaturizer

logger = logging.getLogger(__name__)

GRAPH_FEATURE_ELEMENTS = [
    "C",
    "O",
    "H",
    "N",
    "S",
    "Si",
    "P",
    "Cl",
    "F",
    "I",
    "Se",
    "As",
    "B",
    "Br",
    "Sn",
]
GRAPH_FEATURE_DIM = 2 + len(GRAPH_FEATURE_ELEMENTS)
MASS_SCALE = 1000.0
COUNT_SCALE = 100.0
COMMON_NEUTRAL_LOSSES = (15, 18, 28, 29, 31, 32, 42, 43, 44, 45, 46, 60, 73)


class MinimalGraphGenerator:
    """Legacy graph conversion; BDE execution was retired in v1.1.0."""

    def __init__(
        self,
        bde_cache_path: Optional[str] = None,
        use_bde_calculator: bool = False,
        bondnet_model: Optional[str] = None,
        use_bde: bool = False,
        use_mass_features: bool = True,
        use_formula_features: bool = True,
        use_fragment_candidates: bool = False,
        fragment_output_dim: int = 1000,
        fragment_mask_window: int = 2,
        bde_min: float = 50.0,
        bde_max: float = 200.0,
        default_bde: float = 85.0
    ):
        """
        Initialize minimal graph generator

        Args:
            bde_cache_path: Path to pre-computed BDE HDF5 cache
            use_bde_calculator: Enable on-the-fly BDE calculation
            bondnet_model: BonDNet model name or path
            use_bde: Enable BDE edge feature values
            use_mass_features: Enable exact/molecular mass graph features
            use_formula_features: Enable formula-count graph features
            use_fragment_candidates: Attach a structure-derived candidate m/z mask
            fragment_output_dim: Number of m/z bins in the candidate mask
            fragment_mask_window: +/- m/z bins marked around each candidate
            bde_min: Minimum BDE for normalization (kcal/mol)
            bde_max: Maximum BDE for normalization (kcal/mol)
            default_bde: Default BDE if cache miss and no calculator
        """
        if use_bde or bde_cache_path or use_bde_calculator or bondnet_model:
            raise ValueError("BDE execution was removed. Use no-BDE checkpoints or the historical Git revision.")
        self.use_bde = False
        self.use_mass_features = use_mass_features
        self.use_formula_features = use_formula_features
        self.use_fragment_candidates = use_fragment_candidates
        self.fragment_output_dim = int(fragment_output_dim)
        self.fragment_mask_window = max(int(fragment_mask_window), 0)
        self.bde_min = bde_min
        self.bde_max = bde_max
        self.default_bde = default_bde

        # Initialize featurizer
        self.featurizer = QCGNFeaturizer(
            use_bde=use_bde,
            bde_min=bde_min,
            bde_max=bde_max
        )

        self.stats = {"molecules": 0}

    def get_graph_features(self, mol: Chem.Mol) -> Tuple[np.ndarray, float, int]:
        """
        Build graph-level mass and formula-count features.

        Returns:
            features:
                [exact_mass, molecular_weight, element_counts...] normalized
                to compact numeric ranges.
            exact_mass:
                Unnormalized exact molecular mass.
            molecular_ion_mz:
                Nominal molecular-ion m/z bin for singly-charged EI-MS.
        """
        exact_mass = float(rdMolDescriptors.CalcExactMolWt(mol))
        molecular_weight = float(Descriptors.MolWt(mol))
        counts = {element: 0.0 for element in GRAPH_FEATURE_ELEMENTS}
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol in counts:
                counts[symbol] += 1.0
            hydrogen_count = float(atom.GetTotalNumHs())
            if symbol != "H" and "H" in counts:
                counts["H"] += hydrogen_count
        feature_values = [
            exact_mass / MASS_SCALE,
            molecular_weight / MASS_SCALE,
        ]
        feature_values.extend(
            counts[element] / COUNT_SCALE for element in GRAPH_FEATURE_ELEMENTS
        )
        if not self.use_mass_features:
            feature_values[0] = 0.0
            feature_values[1] = 0.0
        if not self.use_formula_features:
            for idx in range(2, len(feature_values)):
                feature_values[idx] = 0.0
        molecular_ion_mz = max(1, int(round(exact_mass)))
        return np.array(feature_values, dtype=np.float32), exact_mass, molecular_ion_mz

    def get_fragment_candidate_mask(
        self,
        mol: Chem.Mol,
        molecular_ion_mz: int,
    ) -> np.ndarray:
        """
        Build a soft candidate m/z mask from structure only.

        The mask intentionally stays broad: molecular ion, single-bond cleavage
        fragment masses, and a short list of common EI neutral losses are marked
        with a small +/- window. It is used only as a weak regularizer.
        """
        mask = np.zeros(self.fragment_output_dim, dtype=np.float32)

        def mark_mz(mz_value: float):
            mz = int(round(float(mz_value)))
            for candidate_mz in range(
                mz - self.fragment_mask_window,
                mz + self.fragment_mask_window + 1,
            ):
                if 1 <= candidate_mz <= self.fragment_output_dim:
                    mask[candidate_mz - 1] = 1.0

        mark_mz(molecular_ion_mz)
        for neutral_loss in COMMON_NEUTRAL_LOSSES:
            mark_mz(molecular_ion_mz - neutral_loss)

        for bond in mol.GetBonds():
            if bond.IsInRing():
                continue
            try:
                fragmented = Chem.FragmentOnBonds(
                    mol,
                    [bond.GetIdx()],
                    addDummies=False,
                )
                fragments = Chem.GetMolFrags(
                    fragmented,
                    asMols=True,
                    sanitizeFrags=True,
                )
            except Exception:
                continue
            for fragment in fragments:
                try:
                    mark_mz(rdMolDescriptors.CalcExactMolWt(fragment))
                except Exception:
                    continue
        return mask

    def get_bde_for_molecule(self, mol: Chem.Mol, smiles: str) -> Dict[int, float]:
        """Compatibility argument only; the disabled BDE column is constant."""
        self.stats["molecules"] += 1
        return {}

    def mol_to_graph(
        self,
        mol: Chem.Mol,
        smiles: Optional[str] = None,
        spectrum: Optional[np.ndarray] = None,
        **kwargs
    ) -> Data:
        """
        Convert RDKit molecule to PyTorch Geometric graph

        Args:
            mol: RDKit Mol object
            smiles: SMILES string (for BDE cache lookup)
            spectrum: Target spectrum array [1000] (m/z 1-1000)
            **kwargs: Additional metadata to store in graph

        Returns:
            data: PyG Data object with node/edge features and spectrum
        """
        if mol is None:
            raise ValueError("Invalid molecule (None)")

        # Validate molecule
        is_valid, error_msg = self.featurizer.validate_molecule(mol)
        if not is_valid:
            raise ValueError(f"Molecule validation failed: {error_msg}")

        # Generate SMILES if not provided
        if smiles is None:
            smiles = Chem.MolToSmiles(mol)

        # Get BDE values
        bde_dict = self.get_bde_for_molecule(mol, smiles)

        # Node features (16-dim)
        node_features = []
        for atom in mol.GetAtoms():
            features = self.featurizer.get_atom_features(atom)
            node_features.append(features)

        x = torch.tensor(np.array(node_features), dtype=torch.float32)

        # Edge features (3-dim) with BDE
        edge_indices = []
        edge_features = []

        for bond_idx, bond in enumerate(mol.GetBonds()):
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # Get BDE for this bond
            bde_value = bde_dict.get(bond_idx, self.default_bde)

            # Get edge features
            edge_feat = self.featurizer.get_edge_features(bond, bde_value)

            # Bidirectional edges
            edge_indices.append([i, j])
            edge_indices.append([j, i])
            edge_features.append(edge_feat)
            edge_features.append(edge_feat)

        # Create edge tensors
        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float32)
        else:
            # Isolated atom (no bonds)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, self.featurizer.get_edge_dim()), dtype=torch.float32)

        graph_features, exact_mass, molecular_ion_mz = self.get_graph_features(mol)

        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.graph_features = torch.tensor(graph_features, dtype=torch.float32).unsqueeze(0)
        data.molecular_mass = torch.tensor([exact_mass], dtype=torch.float32)
        data.molecular_ion_mz = torch.tensor([molecular_ion_mz], dtype=torch.long)
        data.molecular_ion_index = torch.tensor([molecular_ion_mz - 1], dtype=torch.long)
        if self.use_fragment_candidates:
            fragment_candidate_mask = self.get_fragment_candidate_mask(
                mol=mol,
                molecular_ion_mz=molecular_ion_mz,
            )
            data.fragment_candidate_mask = torch.tensor(
                fragment_candidate_mask,
                dtype=torch.float32,
            ).unsqueeze(0)

        # Add target spectrum
        if spectrum is not None:
            data.y = torch.tensor(spectrum, dtype=torch.float32).unsqueeze(0)

        # Add metadata
        data.smiles = smiles
        data.num_atoms = mol.GetNumAtoms()
        data.num_bonds = mol.GetNumBonds()

        # Add any additional kwargs
        for key, value in kwargs.items():
            setattr(data, key, value)

        return data

    def smiles_to_graph(
        self,
        smiles: str,
        spectrum: Optional[np.ndarray] = None,
        **kwargs
    ) -> Data:
        """
        Convert SMILES to PyTorch Geometric graph

        Args:
            smiles: SMILES string
            spectrum: Target spectrum array
            **kwargs: Additional metadata

        Returns:
            data: PyG Data object
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        return self.mol_to_graph(mol, smiles=smiles, spectrum=spectrum, **kwargs)

    def get_stats(self) -> Dict:
        return dict(total_molecules=self.stats["molecules"], bde_enabled=False)

    def print_stats(self):
        logger.info("Graph records: %s; BDE disabled", self.stats["molecules"])


# Example usage and testing
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("="*60)
    print("Minimal Graph Generator Test")
    print("="*60)

    # Initialize generator (without BDE cache for testing)
    generator = MinimalGraphGenerator(
        use_bde_calculator=False,  # Don't use calculator for quick test
        default_bde=85.0
    )

    # Test molecules
    test_molecules = [
        ("c1ccccc1", "Benzene"),
        ("CC(=O)O", "Acetic acid"),
        ("CCO", "Ethanol"),
    ]

    for smiles, name in test_molecules:
        print(f"\n{name}: {smiles}")

        try:
            # Create dummy spectrum
            spectrum = np.random.rand(1000).astype(np.float32)

            # Generate graph
            graph = generator.smiles_to_graph(
                smiles,
                spectrum=spectrum,
                compound_name=name
            )

            print(f"  ✓ Graph generated successfully")
            print(f"    Nodes: {graph.num_nodes}")
            print(f"    Edges: {graph.num_edges}")
            print(f"    Node features: {graph.x.shape}")
            print(f"    Edge features: {graph.edge_attr.shape}")
            print(f"    Graph features: {graph.graph_features.shape}")
            print(f"    Molecular ion m/z: {graph.molecular_ion_mz.item()}")
            print(f"    Spectrum: {graph.y.shape}")
            print(f"    Metadata: smiles={graph.smiles}, name={graph.compound_name}")

        except Exception as e:
            print(f"  ✗ Failed: {e}")

    # Print statistics
    print("\n" + "="*60)
    generator.print_stats()
    print("="*60)
