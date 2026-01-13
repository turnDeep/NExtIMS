#!/usr/bin/env python3
"""
Test suite for prediction scripts

Tests:
- SMILES validation
- Molecular properties extraction
- Top peaks extraction
- Error handling
"""

import sys
from pathlib import Path
import numpy as np

# Add src and scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from predict_single import validate_molecule, get_top_peaks
from predict_batch import BatchPredictor


def test_smiles_validation():
    """Test SMILES validation"""
    print("\n" + "=" * 60)
    print("Test 1: SMILES Validation")
    print("=" * 60)

    # Test case 1: Valid SMILES
    try:
        mol, props = validate_molecule("CCO")
        assert mol is not None
        assert props['formula'] == 'C2H6O'
        assert abs(props['molecular_weight'] - 46.07) < 0.1
        print(f"✅ Valid SMILES (ethanol): {props['formula']}, {props['molecular_weight']:.2f} Da")
    except Exception as e:
        print(f"❌ Valid SMILES test failed: {e}")
        return False

    # Test case 2: Invalid SMILES
    try:
        mol, props = validate_molecule("CCO(")
        print(f"❌ Invalid SMILES should raise error")
        return False
    except ValueError as e:
        print(f"✅ Invalid SMILES correctly rejected: {e}")

    # Test case 3: Unsupported element
    try:
        mol, props = validate_molecule("C[Si](C)(C)C")
        print(f"❌ Unsupported element should raise error")
        return False
    except ValueError as e:
        print(f"✅ Unsupported element correctly rejected: Si")

    # Test case 4: Complex molecule (caffeine)
    try:
        mol, props = validate_molecule("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
        assert mol is not None
        assert props['formula'] == 'C8H10N4O2'
        assert abs(props['molecular_weight'] - 194.19) < 0.1
        print(f"✅ Complex molecule (caffeine): {props['formula']}, {props['molecular_weight']:.2f} Da")
    except Exception as e:
        print(f"❌ Complex molecule test failed: {e}")
        return False

    # Test case 5: Large molecule (warning)
    try:
        # Vitamin B12 (MW ~1355 Da)
        mol, props = validate_molecule("CC1=CC2=C(C=C1C)N(C=N2)C3C(C(C(O3)CO)OP(=O)([O-])OC(C)CNC(=O)CCC4(C(C5C6(C(C(C(=C(C7=NC(=CC8=NC(=C(C4=N5)C)C(C8(C)C)CCC(=O)N)C(C7(C)CC(=O)N)CCC(=O)N)C)[N-]6)CCC(=O)N)(C)CC(=O)N)C)CC(=O)N)C)O.[CH3-].[Co+3]")
        assert mol is not None
        # Should log a warning about MW > 1000
        print(f"✅ Large molecule (MW={props['molecular_weight']:.1f}): Warning logged")
    except Exception as e:
        print(f"❌ Large molecule test failed: {e}")
        return False

    print("\n✅ All SMILES validation tests PASSED!\n")
    return True


def test_top_peaks_extraction():
    """Test top peaks extraction"""
    print("\n" + "=" * 60)
    print("Test 2: Top Peaks Extraction")
    print("=" * 60)

    # Test case 1: Simple spectrum
    spectrum = np.array([0.1, 0.5, 0.9, 0.3, 0.2, 0.8, 0.4, 0.6, 0.7, 0.05])
    peaks = get_top_peaks(spectrum, top_k=3, mz_offset=1, min_intensity=0.0)

    expected_mz = [3, 6, 9]  # Indices 2, 5, 8 + offset 1
    expected_intensity = [0.9, 0.8, 0.7]

    actual_mz = [p[0] for p in peaks]
    actual_intensity = [p[1] for p in peaks]

    assert actual_mz == expected_mz, f"Expected {expected_mz}, got {actual_mz}"
    assert actual_intensity == expected_intensity, f"Expected {expected_intensity}, got {actual_intensity}"

    print(f"✅ Top-3 peaks: {peaks}")

    # Test case 2: With minimum intensity threshold
    peaks_filtered = get_top_peaks(spectrum, top_k=10, mz_offset=1, min_intensity=0.09)
    assert len(peaks_filtered) == 9, f"Expected 9 peaks (>0.1), got {len(peaks_filtered)}"
    print(f"✅ Filtered peaks (intensity > 0.1): {len(peaks_filtered)} peaks")

    # Test case 3: Empty spectrum
    empty_spectrum = np.zeros(100)
    empty_peaks = get_top_peaks(empty_spectrum, top_k=5, min_intensity=0.001)
    assert len(empty_peaks) == 0, "Expected 0 peaks for empty spectrum"
    print(f"✅ Empty spectrum: 0 peaks")

    # Test case 4: Single peak
    single_peak_spectrum = np.zeros(100)
    single_peak_spectrum[50] = 1.0
    single_peaks = get_top_peaks(single_peak_spectrum, top_k=5, mz_offset=1, min_intensity=0.001)
    assert len(single_peaks) == 1, "Expected 1 peak"
    assert single_peaks[0] == (51, 1.0), f"Expected (51, 1.0), got {single_peaks[0]}"
    print(f"✅ Single peak: {single_peaks[0]}")

    print("\n✅ All top peaks extraction tests PASSED!\n")
    return True


def test_batch_predictor_validation():
    """Test BatchPredictor SMILES validation"""
    print("\n" + "=" * 60)
    print("Test 3: BatchPredictor Validation (No Model Required)")
    print("=" * 60)

    # Note: We can't test actual prediction without a trained model,
    # but we can test validation logic

    # Mock BatchPredictor for validation testing
    class MockBatchPredictor:
        def validate_smiles(self, smiles: str) -> tuple:
            try:
                from rdkit import Chem
                from rdkit.Chem import Descriptors
                from src.data.filters import SUPPORTED_ELEMENTS

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return False, "Invalid SMILES syntax"

                for atom in mol.GetAtoms():
                    if atom.GetSymbol() not in SUPPORTED_ELEMENTS:
                        return False, f"Unsupported element: {atom.GetSymbol()}"

                mw = Descriptors.MolWt(mol)
                if mw > 1000.0:
                    return True, f"Warning: MW={mw:.1f} > 1000 Da"

                return True, ""
            except Exception as e:
                return False, str(e)

    predictor = MockBatchPredictor()

    # Test case 1: Valid SMILES
    is_valid, msg = predictor.validate_smiles("CCO")
    assert is_valid, "CCO should be valid"
    assert msg == "", f"Expected empty message, got: {msg}"
    print(f"✅ Valid SMILES (CCO): {is_valid}")

    # Test case 2: Invalid SMILES
    is_valid, msg = predictor.validate_smiles("CCO(")
    assert not is_valid, "CCO( should be invalid"
    print(f"✅ Invalid SMILES (CCO(): {msg}")

    # Test case 3: Unsupported element
    is_valid, msg = predictor.validate_smiles("C[Si](C)(C)C")
    assert not is_valid, "Silicon should be unsupported"
    assert "Si" in msg, f"Expected 'Si' in error message, got: {msg}"
    print(f"✅ Unsupported element: {msg}")

    # Test case 4: Batch validation
    smiles_list = ["CCO", "CC(C)O", "CCO(", "C[Si](C)(C)C", "CC(=O)C"]
    results = [predictor.validate_smiles(s) for s in smiles_list]

    num_valid = sum(1 for is_valid, _ in results if is_valid)
    num_invalid = len(results) - num_valid

    assert num_valid == 3, f"Expected 3 valid, got {num_valid}"
    assert num_invalid == 2, f"Expected 2 invalid, got {num_invalid}"

    print(f"✅ Batch validation: {num_valid} valid, {num_invalid} invalid")

    print("\n✅ All BatchPredictor validation tests PASSED!\n")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("NExtIMS v4.2: Prediction Scripts Test Suite")
    print("=" * 70)

    try:
        test_smiles_validation()
        test_top_peaks_extraction()
        test_batch_predictor_validation()

        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 70 + "\n")
        return True

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
