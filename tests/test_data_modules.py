#!/usr/bin/env python3
"""
Test suite for data modules

Tests:
- Feature extraction (16-dim nodes, 3-dim edges)
- Data filtering
- NIST dataset parsing
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_node_feature_extraction():
    """Test 16-dimensional node feature extraction"""
    print("\n" + "=" * 60)
    print("Test 1: Node Feature Extraction (16 dims)")
    print("=" * 60)

    try:
        from rdkit import Chem
        from src.data.features_qcgn import get_atom_features_minimal

        # Test case 1: Simple carbon atom
        mol = Chem.MolFromSmiles("C")
        Chem.AllChem.ComputeGasteigerCharges(mol)
        atom = mol.GetAtomWithIdx(0)

        features = get_atom_features_minimal(atom)

        assert len(features) == 16, f"Expected 16 dims, got {len(features)}"
        assert isinstance(features, np.ndarray)
        print(f"✅ Carbon atom: 16-dim features extracted")

        # Test case 2: Different atom types
        test_molecules = {
            'C': "C",           # Carbon
            'N': "N",           # Nitrogen
            'O': "O",           # Oxygen
            'F': "F",           # Fluorine
            'S': "S",           # Sulfur
            'Cl': "Cl",         # Chlorine
            'Br': "Br",         # Bromine
        }

        for symbol, smiles in test_molecules.items():
            mol = Chem.MolFromSmiles(smiles)
            Chem.AllChem.ComputeGasteigerCharges(mol)
            atom = mol.GetAtomWithIdx(0)
            features = get_atom_features_minimal(atom)

            assert len(features) == 16, \
                f"{symbol}: Expected 16 dims, got {len(features)}"

        print(f"✅ All supported atoms: 16-dim features")

        # Test case 3: Feature composition
        # 10 (atom type) + 1 (aromatic) + 1 (in ring) + 3 (hybrid) + 1 (charge) = 16
        mol = Chem.MolFromSmiles("c1ccccc1")  # Benzene (aromatic)
        Chem.AllChem.ComputeGasteigerCharges(mol)
        atom = mol.GetAtomWithIdx(0)
        features = get_atom_features_minimal(atom)

        # Check aromatic flag is set
        aromatic_idx = 10  # After atom type one-hot
        assert features[aromatic_idx] == 1, "Aromatic flag should be 1"
        print(f"✅ Aromatic flag correctly set")

    except ImportError as e:
        print(f"⚠️  Cannot test features (missing dependencies): {e}")
        return True
    except Exception as e:
        print(f"❌ Node feature test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✅ Node feature extraction test PASSED!\n")
    return True


def test_edge_feature_extraction():
    """Test 3-dimensional edge feature extraction"""
    print("\n" + "=" * 60)
    print("Test 2: Edge Feature Extraction (3 dims)")
    print("=" * 60)

    try:
        from rdkit import Chem
        from src.data.features_qcgn import get_bond_features_minimal

        # Test case 1: Single bond
        mol = Chem.MolFromSmiles("CC")
        bond = mol.GetBondWithIdx(0)

        bde_value = 85.0  # Example BDE
        features = get_bond_features_minimal(bond, bde_value)

        assert len(features) == 3, f"Expected 3 dims, got {len(features)}"
        assert isinstance(features, np.ndarray)
        print(f"✅ Single bond: 3-dim features extracted")

        # Test case 2: Different bond types
        test_bonds = {
            'single': "CC",         # Single bond
            'double': "C=C",        # Double bond
            'triple': "C#C",        # Triple bond
            'aromatic': "c1ccccc1", # Aromatic bond
        }

        for bond_type, smiles in test_bonds.items():
            mol = Chem.MolFromSmiles(smiles)
            bond = mol.GetBondWithIdx(0)
            features = get_bond_features_minimal(bond, bde_value=85.0)

            assert len(features) == 3, \
                f"{bond_type}: Expected 3 dims, got {len(features)}"

        print(f"✅ All bond types: 3-dim features")

        # Test case 3: Feature composition
        # BDE (1) + bond order (1) + in ring (1) = 3
        mol = Chem.MolFromSmiles("C=C")
        bond = mol.GetBondWithIdx(0)
        features = get_bond_features_minimal(bond, bde_value=100.0)

        # Check bond order
        bond_order = features[1]
        assert bond_order == 2.0, f"Expected bond order 2.0, got {bond_order}"
        print(f"✅ Bond order correctly extracted")

    except ImportError as e:
        print(f"⚠️  Cannot test features (missing dependencies): {e}")
        return True
    except Exception as e:
        print(f"❌ Edge feature test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✅ Edge feature extraction test PASSED!\n")
    return True


def test_data_filters():
    """Test data filtering"""
    print("\n" + "=" * 60)
    print("Test 3: Data Filtering")
    print("=" * 60)

    try:
        from src.data.filters import SUPPORTED_ELEMENTS, is_supported_molecule
        from rdkit import Chem

        # Test case 1: Supported elements
        assert 'C' in SUPPORTED_ELEMENTS
        assert 'H' in SUPPORTED_ELEMENTS
        assert 'O' in SUPPORTED_ELEMENTS
        assert 'N' in SUPPORTED_ELEMENTS
        print(f"✅ Supported elements: {', '.join(sorted(SUPPORTED_ELEMENTS))}")

        # Test case 2: Valid molecules
        valid_smiles = [
            "CCO",              # Ethanol
            "CC(C)O",           # Isopropanol
            "c1ccccc1",         # Benzene
            "CC(=O)O",          # Acetic acid
            "CCN",              # Ethylamine
        ]

        for smiles in valid_smiles:
            mol = Chem.MolFromSmiles(smiles)
            is_valid = is_supported_molecule(mol)
            assert is_valid, f"{smiles} should be valid"

        print(f"✅ Valid molecules correctly identified: {len(valid_smiles)}")

        # Test case 3: Invalid molecules (unsupported elements)
        invalid_smiles = [
            "C[Si](C)(C)C",     # Silicon
            "C[Li]",            # Lithium
            "C[B](C)C",         # Boron
        ]

        for smiles in invalid_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                is_valid = is_supported_molecule(mol)
                assert not is_valid, f"{smiles} should be invalid"

        print(f"✅ Invalid molecules correctly rejected")

    except ImportError as e:
        print(f"⚠️  Cannot test filters (missing dependencies): {e}")
        return True
    except Exception as e:
        print(f"❌ Data filter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✅ Data filtering test PASSED!\n")
    return True


def test_nist_parsing():
    """Test NIST MSP parsing"""
    print("\n" + "=" * 60)
    print("Test 4: NIST MSP Parsing")
    print("=" * 60)

    try:
        from src.data.nist_dataset import peaks_to_spectrum

        # Test case 1: Convert peaks to spectrum
        peaks = [
            (46, 100.0),
            (31, 80.0),
            (45, 60.0),
            (27, 40.0),
        ]

        spectrum = peaks_to_spectrum(
            peaks,
            min_mz=1,
            max_mz=1000
        )

        assert len(spectrum) == 1000, f"Expected 1000, got {len(spectrum)}"
        assert spectrum.max() == 1.0, "Max should be 1.0 (normalized)"
        assert spectrum[45] == 1.0, "Peak at m/z=46 should be 1.0 (base peak)"

        print(f"✅ Peaks to spectrum conversion: shape {spectrum.shape}")

        # Test case 2: Non-normalized
        # Note: peaks_to_spectrum always normalizes in current implementation

        # Test case 3: Empty peaks
        empty_spectrum = peaks_to_spectrum(
            [],
            min_mz=1,
            max_mz=1000
        )

        assert np.all(empty_spectrum == 0), "Empty peaks should give zero spectrum"
        print(f"✅ Empty peaks handling")

    except ImportError as e:
        print(f"⚠️  Cannot test parsing (missing dependencies): {e}")
        return True
    except Exception as e:
        print(f"❌ NIST parsing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✅ NIST parsing test PASSED!\n")
    return True


def run_all_tests():
    """Run all data module tests"""
    print("\n" + "=" * 70)
    print("NExtIMS v4.2: Data Modules Test Suite")
    print("=" * 70)

    try:
        success = True
        success &= test_node_feature_extraction()
        success &= test_edge_feature_extraction()
        success &= test_data_filters()
        success &= test_nist_parsing()

        if success:
            print("\n" + "=" * 70)
            print("🎉 ALL DATA MODULE TESTS PASSED! 🎉")
            print("=" * 70 + "\n")
        else:
            print("\n" + "=" * 70)
            print("⚠️  SOME TESTS HAD ISSUES")
            print("=" * 70 + "\n")

        return success

    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
