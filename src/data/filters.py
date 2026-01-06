# src/data/filters.py
"""
Data filtering utilities for NExtIMS
"""

SUPPORTED_ELEMENTS = {
    'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'
}

def is_supported_molecule(mol, supported_elements=None) -> bool:
    """
    Check if molecule contains only supported elements
    """
    if supported_elements is None:
        supported_elements = SUPPORTED_ELEMENTS

    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in supported_elements:
            return False
    return True

def apply_all_filters(mol, filters_config=None) -> bool:
    """
    Apply all filters to a molecule
    """
    # Check elements
    if not is_supported_molecule(mol):
        return False

    # Add more filters as needed based on config

    return True
