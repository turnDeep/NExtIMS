#!/usr/bin/env python3
"""
BDE-db2 to BonDNet Format Converter

Converts the BDE-db2 dataset (CSV format) to BonDNet training format
(SDF + YAML files).

Input:  bde-db2.csv (531,244 BDEs from 65,540 molecules)
Output: molecules.sdf, molecule_attributes.yaml, reactions.yaml

Usage:
    python scripts/convert_bde_db2_to_bondnet.py \
        --input data/external/bde-db2/bde-db2.csv \
        --output data/processed/bondnet_training/ \
        --max-molecules 0  # 0 = all molecules
"""

import os
import sys
import argparse
import logging
import pandas as pd
import yaml
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_bde_db2_csv(csv_file, max_molecules=0):
    """
    Parse BDE-db2 CSV file

    Expected CSV format:
        - smiles: SMILES string of parent molecule
        - bond_index: Index of bond to break
        - fragment1_smiles: SMILES of fragment 1
        - fragment2_smiles: SMILES of fragment 2
        - bde_enthalpy: BDE enthalpy (kcal/mol)
        - bde_gibbs: BDE Gibbs free energy (kcal/mol)
    """
    logger.info(f"Reading BDE-db2 dataset from {csv_file}")

    # Read CSV
    df = pd.read_csv(csv_file)
    logger.info(f"Loaded {len(df):,} BDE entries")

    # Display column names
    logger.info(f"Columns: {list(df.columns)}")

    # Group by parent molecule
    molecules = defaultdict(list)
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing BDEs"):
        smiles = row.get('smiles') or row.get('parent_smiles') or row.get('molecule')

        if pd.isna(smiles):
            continue

        bde_entry = {
            'bond_index': int(row.get('bond_index', row.get('bond_idx', -1))),
            'fragment1': row.get('fragment1_smiles', row.get('fragment1', '')),
            'fragment2': row.get('fragment2_smiles', row.get('fragment2', '')),
            'bde_enthalpy': float(row.get('bde_enthalpy', row.get('BDE_enth', 0.0))),
            'bde_gibbs': float(row.get('bde_gibbs', row.get('BDE_gibbs', 0.0)))
        }

        molecules[smiles].append(bde_entry)

    logger.info(f"Found {len(molecules):,} unique molecules")

    # Limit molecules if requested
    if max_molecules > 0 and len(molecules) > max_molecules:
        logger.info(f"Limiting to {max_molecules:,} molecules")
        molecules = dict(list(molecules.items())[:max_molecules])

    return molecules


def collect_all_molecules(molecules_dict):
    """
    Collect all unique molecules (parents + fragments) from the BDE data

    Args:
        molecules_dict: Dictionary of parent molecule SMILES and their BDE data

    Returns:
        tuple: (all_molecules_set, smiles_to_index) where:
            - all_molecules_set: Set of all unique SMILES (parents + fragments)
            - smiles_to_index: Dictionary mapping SMILES to index for BonDNet
    """
    logger.info("Collecting all molecules (parents + fragments)...")

    all_molecules = set()

    # Add parent molecules
    for smiles in molecules_dict.keys():
        all_molecules.add(smiles)

    # Add fragments
    for smiles, bde_list in tqdm(molecules_dict.items(), desc="Collecting fragments"):
        for bde_entry in bde_list:
            frag1 = bde_entry['fragment1']
            frag2 = bde_entry['fragment2']

            if frag1 and not pd.isna(frag1):
                all_molecules.add(frag1)
            if frag2 and not pd.isna(frag2):
                all_molecules.add(frag2)

    logger.info(f"Total unique molecules (parents + fragments): {len(all_molecules):,}")

    # Create SMILES to index mapping
    # Sort for reproducibility
    all_molecules_list = sorted(list(all_molecules))
    smiles_to_index = {smiles: idx for idx, smiles in enumerate(all_molecules_list)}

    logger.info(f"Created SMILES to index mapping for {len(smiles_to_index):,} molecules")

    return all_molecules_list, smiles_to_index


def create_sdf_file(all_molecules_list, output_path):
    """Create molecules.sdf file with 3D conformers for all molecules (parents + fragments)

    Args:
        all_molecules_list: List of all unique SMILES (parents + fragments)
        output_path: Path to output SDF file

    Returns:
        tuple: (mol_count, successful_smiles_list) - number of molecules written and list of successful SMILES in order
    """
    logger.info("Generating molecules.sdf with 3D coordinates...")
    logger.info(f"Processing {len(all_molecules_list):,} unique molecules (parents + fragments)")

    writer = Chem.SDWriter(str(output_path))

    mol_count = 0
    failed_count = 0
    successful_smiles = []  # Track successful molecules IN ORDER

    for smiles in tqdm(all_molecules_list, desc="Generating SDF"):
        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                failed_count += 1
                continue

            # Add hydrogens
            mol = Chem.AddHs(mol)

            # Generate 3D conformer
            result = AllChem.EmbedMolecule(mol, randomSeed=42)
            if result == -1:
                # Fallback: try without constraints
                result = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)

            if result == -1:
                # Still failed, skip
                failed_count += 1
                continue

            # Optimize geometry (MMFF94)
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except:
                # Optimization failed, but keep the structure
                pass

            # Set properties
            mol.SetProp("_Name", smiles)
            mol.SetProp("SMILES", smiles)

            writer.write(mol)
            mol_count += 1
            successful_smiles.append(smiles)  # Track success IN ORDER

        except Exception as e:
            logger.debug(f"Failed to process {smiles}: {e}")
            failed_count += 1

    writer.close()

    logger.info(f"✓ Created {output_path}")
    logger.info(f"  Molecules written: {mol_count:,}")
    logger.info(f"  Failed: {failed_count:,}")

    return mol_count, successful_smiles


def create_molecule_attributes(successful_smiles_list, output_path):
    """Create molecule_attributes.yaml

    Args:
        successful_smiles_list: List of SMILES that were successfully written to SDF (in order)
        output_path: Path to output YAML file
    """
    logger.info("Creating molecule_attributes.yaml...")

    # BonDNet expects a LIST of dictionaries, in the same order as molecules.sdf
    attributes = []

    for smiles in tqdm(successful_smiles_list, desc="Processing attributes"):

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            # Calculate molecular properties
            # BonDNet expects list format, in same order as molecules.sdf
            attributes.append({
                'smiles': smiles,
                'charge': 0,  # Neutral molecules (fragments might have radicals, but charge=0)
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
                'molecular_weight': Descriptors.MolWt(mol)
            })

        except Exception as e:
            logger.debug(f"Failed to process attributes for {smiles}: {e}")

    # Write YAML as a list
    with open(output_path, 'w') as f:
        yaml.dump(attributes, f, default_flow_style=False, sort_keys=False)

    logger.info(f"✓ Created {output_path}")
    logger.info(f"  Molecules: {len(attributes):,}")

    return len(attributes)


def create_reactions_file(molecules_dict, output_path, smiles_to_index):
    """Create reactions.yaml with BDE data using molecule indices

    Args:
        molecules_dict: Dictionary of parent molecule SMILES and their BDE data
        output_path: Path to output YAML file
        smiles_to_index: Dictionary mapping SMILES to index in molecules.sdf
    """
    logger.info("Creating reactions.yaml...")

    reactions = []
    reaction_count = 0
    skipped_count = 0

    for parent_smiles, bde_list in tqdm(molecules_dict.items(), desc="Processing reactions"):
        # Skip parent molecule if not in index (failed SDF generation)
        if parent_smiles not in smiles_to_index:
            skipped_count += len(bde_list)
            continue

        parent_idx = smiles_to_index[parent_smiles]

        for bde_entry in bde_list:
            frag1_smiles = bde_entry['fragment1']
            frag2_smiles = bde_entry['fragment2']

            # Skip if fragments not in index (failed SDF generation)
            if frag1_smiles not in smiles_to_index or frag2_smiles not in smiles_to_index:
                skipped_count += 1
                continue

            frag1_idx = smiles_to_index[frag1_smiles]
            frag2_idx = smiles_to_index[frag2_smiles]

            # BonDNet expects reactants and products as INTEGER INDICES
            reaction = {
                'reactants': [parent_idx],  # Index, not SMILES
                'products': [frag1_idx, frag2_idx],  # Indices, not SMILES
                'energy': bde_entry['bde_enthalpy'],  # BonDNet expects this key
                'bond_index': bde_entry['bond_index'],
                'bde_enthalpy': bde_entry['bde_enthalpy'],  # Keep for reference
                'bde_gibbs': bde_entry['bde_gibbs'],  # Keep for reference
                'reaction_type': 'bond_dissociation',
                'homolytic': True
            }
            reactions.append(reaction)
            reaction_count += 1

    # Write YAML
    with open(output_path, 'w') as f:
        yaml.dump(reactions, f, default_flow_style=False, sort_keys=False)

    logger.info(f"✓ Created {output_path}")
    logger.info(f"  Reactions: {reaction_count:,}")
    if skipped_count > 0:
        logger.info(f"  Skipped (missing fragments): {skipped_count:,}")

    return reaction_count


def create_training_script(output_dir):
    """Create example training script"""
    script_content = """#!/bin/bash
# BonDNet Retraining Script
# Auto-generated by convert_bde_db2_to_bondnet.py

set -e

# Paths
DATA_DIR="$(pwd)"
MODEL_DIR="$(pwd)/../../models"
BONDNET_REPO="$HOME/bondnet"  # Clone from https://github.com/mjwen/bondnet

# Check if BonDNet is installed
if [ ! -d "$BONDNET_REPO" ]; then
    echo "Error: BonDNet repository not found at $BONDNET_REPO"
    echo "Please clone it:"
    echo "  git clone https://github.com/mjwen/bondnet.git $HOME/bondnet"
    echo "  cd $HOME/bondnet"
    echo "  pip install -e ."
    exit 1
fi

# Training parameters
EPOCHS=200
BATCH_SIZE=128
LEARNING_RATE=0.001
DEVICE="cuda"  # or "cpu"

echo "========================================="
echo "BonDNet Retraining on BDE-db2"
echo "========================================="
echo "Data directory: $DATA_DIR"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo ""

# Run training
python $BONDNET_REPO/bondnet/scripts/train_bde_distributed.py \\
    $DATA_DIR/molecules.sdf \\
    $DATA_DIR/molecule_attributes.yaml \\
    $DATA_DIR/reactions.yaml \\
    --device $DEVICE \\
    --epochs $EPOCHS \\
    --batch-size $BATCH_SIZE \\
    --lr $LEARNING_RATE \\
    --output-dir $MODEL_DIR \\
    --model-name bondnet_bde_db2

echo ""
echo "========================================="
echo "Training complete!"
echo "Model saved to: $MODEL_DIR/bondnet_bde_db2.pth"
echo "========================================="
"""

    script_path = output_dir / "train_bondnet.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)

    # Make executable
    os.chmod(script_path, 0o755)

    logger.info(f"✓ Created training script: {script_path}")


def create_readme(output_dir, stats):
    """Create README with conversion statistics"""
    readme_content = f"""# BonDNet Training Data (from BDE-db2)

Converted from BDE-db2 dataset (531,244 BDEs, 65,540 molecules).

## Conversion Statistics

- **Molecules processed:** {stats['molecules']:,}
- **BDE reactions:** {stats['reactions']:,}
- **Success rate:** {stats['molecules'] / 65540 * 100:.1f}%

## Files

- `molecules.sdf` - Molecular structures with 3D coordinates
- `molecule_attributes.yaml` - Molecular properties (charge, MW, etc.)
- `reactions.yaml` - Bond dissociation reactions with BDE values
- `train_bondnet.sh` - Training script (executable)

## Training

### Prerequisites

1. Install BonDNet:
```bash
git clone https://github.com/mjwen/bondnet.git
cd bondnet
pip install -e .
```

2. Install dependencies:
```bash
pip install torch>=1.10.0
pip install dgl>=0.5.0
pip install pymatgen>=2022.01.08
pip install rdkit>=2020.03.5
pip install openbabel>=3.1.1
```

### Run Training

```bash
cd {output_dir.absolute()}
./train_bondnet.sh
```

Or manually:
```bash
python $BONDNET_REPO/bondnet/scripts/train_bde_distributed.py \\
    molecules.sdf \\
    molecule_attributes.yaml \\
    reactions.yaml \\
    --device cuda \\
    --epochs 200 \\
    --batch-size 128
```

### Estimated Training Time

- **GPU (RTX 5070 Ti 16GB):** ~2-3 days
- **GPU (A100 40GB):** ~1-2 days
- **CPU (not recommended):** ~2-3 weeks

## Expected Results

After training on BDE-db2:
- **MAE:** 0.5-0.55 kcal/mol (vs 0.51 kcal/mol on original 64K dataset)
- **Element coverage:** C, H, N, O, S, Cl, F, P, Br, I (10 elements)
- **NIST17 coverage:** 99%+ (vs 95% with original BonDNet)

## Next Steps

After training, use the model for BDE precomputation:
```bash
python scripts/precompute_bde.py \\
    --model models/bondnet_bde_db2.pth \\
    --dataset nist17 \\
    --output data/processed/bde_cache/nist17_bde_cache.h5
```
"""

    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    logger.info(f"✓ Created README: {readme_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert BDE-db2 dataset to BonDNet training format"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input BDE-db2 CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for BonDNet files'
    )
    parser.add_argument(
        '--max-molecules',
        type=int,
        default=0,
        help='Maximum number of molecules to convert (0 = all)'
    )
    parser.add_argument(
        '--skip-sdf',
        action='store_true',
        help='Skip SDF generation (use existing molecules.sdf)'
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    output_dir = Path(args.output)

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BDE-db2 to BonDNet Format Converter")
    logger.info("=" * 60)
    logger.info(f"Input:  {input_file.absolute()}")
    logger.info(f"Output: {output_dir.absolute()}")
    logger.info("")

    # Step 1: Parse CSV
    molecules_dict = parse_bde_db2_csv(input_file, args.max_molecules)

    # Step 2: Collect all molecules (parents + fragments) and create index mapping
    all_molecules_list, smiles_to_index = collect_all_molecules(molecules_dict)

    stats = {}

    # Step 3: Create SDF file with all molecules (parents + fragments)
    if not args.skip_sdf:
        sdf_path = output_dir / "molecules.sdf"
        mol_count, successful_smiles_list = create_sdf_file(all_molecules_list, sdf_path)
        stats['molecules'] = mol_count

        # Update smiles_to_index to only include successful molecules
        smiles_to_index = {smiles: idx for idx, smiles in enumerate(successful_smiles_list)}
    else:
        logger.warning("--skip-sdf option is deprecated with fragment support")
        logger.warning("Regenerating SDF file anyway to ensure consistency")
        sdf_path = output_dir / "molecules.sdf"
        mol_count, successful_smiles_list = create_sdf_file(all_molecules_list, sdf_path)
        stats['molecules'] = mol_count
        smiles_to_index = {smiles: idx for idx, smiles in enumerate(successful_smiles_list)}

    # Step 4: Create molecule attributes (only for successful molecules, in order)
    attr_path = output_dir / "molecule_attributes.yaml"
    create_molecule_attributes(successful_smiles_list, attr_path)

    # Step 5: Create reactions file using indices
    rxn_path = output_dir / "reactions.yaml"
    reaction_count = create_reactions_file(molecules_dict, rxn_path, smiles_to_index)
    stats['reactions'] = reaction_count

    # Step 6: Create training script
    create_training_script(output_dir)

    # Step 7: Create README
    create_readme(output_dir, stats)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Conversion Complete!")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info(f"Molecules: {stats['molecules']:,}")
    logger.info(f"Reactions: {stats['reactions']:,}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Review the generated files:")
    logger.info(f"   cd {output_dir.absolute()}")
    logger.info("   ls -lh")
    logger.info("")
    logger.info("2. Install BonDNet (if not already installed):")
    logger.info("   git clone https://github.com/mjwen/bondnet.git")
    logger.info("   cd bondnet && pip install -e .")
    logger.info("")
    logger.info("3. Start training:")
    logger.info(f"   cd {output_dir.absolute()}")
    logger.info("   ./train_bondnet.sh")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
