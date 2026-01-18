import unittest
import sys
import numpy as np
from rdkit import Chem
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.features_qcgn import QCGNFeaturizer

class TestFeaturizerAblation(unittest.TestCase):
    def setUp(self):
        self.mol = Chem.MolFromSmiles("C[C@H](O)C(=O)O") # L-Lactic Acid

    def test_full_features(self):
        f = QCGNFeaturizer()
        self.assertEqual(f.get_node_dim(), 34)
        self.assertEqual(f.get_edge_dim(), 10)

        node_feat = f.get_atom_features(self.mol.GetAtomWithIdx(0))
        self.assertEqual(len(node_feat), 34)

        edge_feat = f.get_edge_features(self.mol.GetBondWithIdx(0))
        self.assertEqual(len(edge_feat), 10)

    def test_no_bde(self):
        f = QCGNFeaturizer(use_bde=False)
        self.assertEqual(f.get_edge_dim(), 9)
        edge_feat = f.get_edge_features(self.mol.GetBondWithIdx(0))
        self.assertEqual(len(edge_feat), 9)

    def test_no_chirality(self):
        f = QCGNFeaturizer(include_chirality=False)
        self.assertEqual(f.get_node_dim(), 30) # 34 - 4
        node_feat = f.get_atom_features(self.mol.GetAtomWithIdx(0))
        self.assertEqual(len(node_feat), 30)

    def test_no_stereo(self):
        f = QCGNFeaturizer(include_stereo=False)
        self.assertEqual(f.get_edge_dim(), 4) # 10 - 6
        edge_feat = f.get_edge_features(self.mol.GetBondWithIdx(0))
        self.assertEqual(len(edge_feat), 4)

    def test_complex_ablation(self):
        # Disable BDE, Stereo, and Chirality
        f = QCGNFeaturizer(
            use_bde=False,
            include_stereo=False,
            include_chirality=False
        )
        self.assertEqual(f.get_node_dim(), 30)
        self.assertEqual(f.get_edge_dim(), 3) # 10 - 1(BDE) - 6(Stereo)

        node_feat = f.get_atom_features(self.mol.GetAtomWithIdx(0))
        edge_feat = f.get_edge_features(self.mol.GetBondWithIdx(0))
        self.assertEqual(len(node_feat), 30)
        self.assertEqual(len(edge_feat), 3)

if __name__ == '__main__':
    unittest.main()
