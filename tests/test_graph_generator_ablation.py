import unittest
import sys
import numpy as np
from rdkit import Chem
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.graph_generator_minimal import MinimalGraphGenerator

class TestGraphGeneratorAblation(unittest.TestCase):
    def setUp(self):
        self.smiles = "C[C@H](O)C(=O)O" # L-Lactic Acid

    def test_pass_through_kwargs(self):
        # Test that kwargs are correctly passed to featurizer
        gen = MinimalGraphGenerator(
            use_bde=False,
            include_chirality=False,
            use_bde_calculator=False
        )

        # Check underlying featurizer properties
        self.assertFalse(gen.featurizer.use_bde)
        self.assertFalse(gen.featurizer.include_chirality)
        self.assertTrue(gen.featurizer.include_atom_type) # Default should remain true

        # Check graph dimensions
        graph = gen.smiles_to_graph(self.smiles)

        # Node dim: 34 - 4 (chirality) = 30
        self.assertEqual(graph.x.shape[1], 30)

        # Edge dim: 10 - 1 (BDE) = 9
        self.assertEqual(graph.edge_attr.shape[1], 9)

if __name__ == '__main__':
    unittest.main()
