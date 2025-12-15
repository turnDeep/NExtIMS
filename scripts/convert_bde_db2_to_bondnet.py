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


def create_sdf_file(molecules, output_path):
    """Create molecules.sdf file with 3D conformers

    Returns:
        tuple: (mol_count, successful_smiles_set) - number of molecules written and set of successful SMILES
    """
    logger.info("Generating molecules.sdf with 3D coordinates...")

    writer = Chem.SDWriter(str(output_path))

    mol_count = 0
    failed_count = 0
    successful_smiles = set()  # Track successful molecules

    for smiles, bde_list in tqdm(molecules.items(), desc="Generating SDF"):
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
            mol.SetProp("NumBDEs", str(len(bde_list)))

            writer.write(mol)
            mol_count += 1
            successful_smiles.add(smiles)  # Track success

        except Exception as e:
            logger.debug(f"Failed to process {smiles}: {e}")
            failed_count += 1

    writer.close()

    logger.info(f"✓ Created {output_path}")
    logger.info(f"  Molecules written: {mol_count:,}")
    logger.info(f"  Failed: {failed_count:,}")

    return mol_count, successful_smiles


def create_molecule_attributes(molecules, output_path, successful_smiles=None):
    """Create molecule_attributes.yaml

    Args:
        molecules: Dictionary of molecule SMILES and BDE data
        output_path: Path to output YAML file
        successful_smiles: Set of SMILES that were successfully written to SDF (optional filter)
    """
    logger.info("Creating molecule_attributes.yaml...")

    # BonDNet expects a LIST of dictionaries, not a dict of dicts
    attributes = []

    for smiles, bde_list in tqdm(molecules.items(), desc="Processing attributes"):
        # Skip molecules that failed SDF generation
        if successful_smiles is not None and smiles not in successful_smiles:
            continue

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            # Calculate molecular properties
            # BonDNet expects list format, not dict
            attributes.append({
                'smiles': smiles,
                'charge': 0,  # BDE-db2 uses neutral molecules
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
                'molecular_weight': Descriptors.MolWt(mol),
                'num_bdes': len(bde_list)
            })

        except Exception as e:
            logger.debug(f"Failed to process attributes for {smiles}: {e}")

    # Write YAML as a list
    with open(output_path, 'w') as f:
        yaml.dump(attributes, f, default_flow_style=False, sort_keys=False)

    logger.info(f"✓ Created {output_path}")
    logger.info(f"  Molecules: {len(attributes):,}")

    return len(attributes)


def create_reactions_file(molecules, output_path, successful_smiles=None):
    """Create reactions.yaml with BDE data

    Args:
        molecules: Dictionary of molecule SMILES and BDE data
        output_path: Path to output YAML file
        successful_smiles: Set of SMILES that were successfully written to SDF (optional filter)
    """
    logger.info("Creating reactions.yaml...")

    reactions = []
    reaction_count = 0

    for smiles, bde_list in tqdm(molecules.items(), desc="Processing reactions"):
        # Skip molecules that failed SDF generation
        if successful_smiles is not None and smiles not in successful_smiles:
            continue

        for bde_entry in bde_list:
            reaction = {
                'reactants': [smiles],
                'products': [bde_entry['fragment1'], bde_entry['fragment2']],
                'bond_index': bde_entry['bond_index'],
                'bde_enthalpy': bde_entry['bde_enthalpy'],
                'bde_gibbs': bde_entry['bde_gibbs'],
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
    molecules = parse_bde_db2_csv(input_file, args.max_molecules)

    stats = {}
    successful_smiles = None

    # Step 2: Create SDF file
    if not args.skip_sdf:
        sdf_path = output_dir / "molecules.sdf"
        mol_count, successful_smiles = create_sdf_file(molecules, sdf_path)
        stats['molecules'] = mol_count
    else:
        logger.info("Skipping SDF generation (--skip-sdf)")
        stats['molecules'] = len(molecules)
        # If skipping SDF, use all molecules
        successful_smiles = None

    # Step 3: Create molecule attributes (only for molecules in SDF)
    attr_path = output_dir / "molecule_attributes.yaml"
    create_molecule_attributes(molecules, attr_path, successful_smiles)

    # Step 4: Create reactions file (only for molecules in SDF)
    rxn_path = output_dir / "reactions.yaml"
    reaction_count = create_reactions_file(molecules, rxn_path, successful_smiles)
    stats['reactions'] = reaction_count

    # Step 5: Create training script
    create_training_script(output_dir)

    # Step 6: Create README
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
