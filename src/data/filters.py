# src/data/filters.py
"""
Data filtering utilities
"""

# Supported elements in NIST17/BonDNet
SUPPORTED_ELEMENTS = {'C', 'H', 'O', 'N', 'F', 'S', 'P', 'Cl', 'Br', 'I'}

def is_supported_molecule(mol):
    """Check if molecule contains only supported elements"""
    if mol is None:
        return False

    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in SUPPORTED_ELEMENTS:
            return False

    return True
