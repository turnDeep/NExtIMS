#!/usr/bin/env python3
"""
Download BonDNet Pretrained Model Weights

Downloads pretrained BonDNet model weights from Zenodo for transfer learning.
The model was trained on the BDNCM dataset (60,000+ bond dissociations).

Dataset/Model Information:
    - BDNCM Dataset: https://zenodo.org/records/15117901
    - BonDNet GitHub: https://github.com/mjwen/bondnet
    - Model trained on: Homolytic and heterolytic BDEs for neutral and charged molecules
    - Elements: C, H, O, N (primary coverage)
    - MAE: ~0.022 eV (~0.51 kcal/mol)

Usage:
    python scripts/download_bondnet_pretrained.py --output models/bondnet_pretrained.pth

For transfer learning with BDE-db2:
    1. Download pretrained weights (this script)
    2. Fine-tune on BDE-db2:
       python scripts/train_bondnet_bde_db2.py \\
           --data-dir data/processed/bondnet_training/ \\
           --pretrained models/bondnet_pretrained.pth \\
           --output models/bondnet_bde_db2_finetuned.pth \\
           --epochs 50 \\
           --lr 0.0001
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path
import requests
from tqdm import tqdm
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Zenodo record for BDNCM dataset and pretrained models
ZENODO_RECORD_ID = "15117901"
ZENODO_BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

# Alternative: Clone pretrained branch from GitHub
BONDNET_REPO_URL = "https://github.com/mjwen/bondnet.git"
PRETRAINED_BRANCH = "pretrained"


def download_file(url: str, output_path: Path, description: str = "Downloading"):
    """Download file with progress bar"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f, tqdm(
            desc=description,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)

        logger.info(f"✓ Downloaded: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to download from {url}: {e}")
        return False


def get_zenodo_files():
    """Get file list from Zenodo record"""
    try:
        response = requests.get(ZENODO_API_URL, timeout=30)
        response.raise_for_status()
        data = response.json()

        files = data.get('files', [])
        logger.info(f"Found {len(files)} files in Zenodo record {ZENODO_RECORD_ID}")

        for file_info in files:
            logger.info(f"  - {file_info['key']} ({file_info['size'] / (1024**2):.1f} MB)")

        return files

    except Exception as e:
        logger.error(f"Failed to access Zenodo record: {e}")
        return None


def download_from_zenodo(output_path: Path):
    """Download pretrained model from Zenodo"""
    logger.info("Attempting to download from Zenodo...")
    logger.info(f"Zenodo record: {ZENODO_BASE_URL}")

    files = get_zenodo_files()

    if not files:
        logger.warning("Could not retrieve file list from Zenodo")
        return False

    # Look for model weight files (.pth, .pt, .pkl, .ckpt)
    model_extensions = ['.pth', '.pt', '.pkl', '.ckpt']
    model_files = [f for f in files if any(f['key'].endswith(ext) for ext in model_extensions)]

    if not model_files:
        logger.warning("No model weight files found in Zenodo record")
        logger.info("Available files:")
        for f in files:
            logger.info(f"  - {f['key']}")
        return False

    # Download the model file - prefer checkpoint.pkl or files with 'checkpoint', 'bondnet', or 'model' in name
    target_file = None

    # First priority: checkpoint.pkl
    for f in model_files:
        if 'checkpoint' in f['key'].lower():
            target_file = f
            break

    # Second priority: bondnet or model in filename
    if not target_file:
        for f in model_files:
            if 'bondnet' in f['key'].lower() or 'model' in f['key'].lower():
                target_file = f
                break

    # Fallback: first model file
    if not target_file:
        target_file = model_files[0]

    logger.info(f"Downloading: {target_file['key']}")
    download_url = target_file['links']['self']

    return download_file(download_url, output_path, f"Downloading {target_file['key']}")


def clone_pretrained_branch(output_dir: Path):
    """Clone BonDNet pretrained branch from GitHub"""
    logger.info("Attempting to clone pretrained branch from GitHub...")
    logger.info(f"Repository: {BONDNET_REPO_URL}")
    logger.info(f"Branch: {PRETRAINED_BRANCH}")

    repo_dir = output_dir / "bondnet_pretrained_repo"

    # Check if repository already exists
    if repo_dir.exists():
        logger.info(f"Repository already exists at: {repo_dir}")
        logger.info("Skipping clone and using existing repository")

        # Look for checkpoint.pkl files in the existing repository
        pretrained_dir = repo_dir / 'bondnet' / 'prediction' / 'pretrained'

        if pretrained_dir.exists():
            checkpoint_files = list(pretrained_dir.glob('**/checkpoint.pkl'))
            if checkpoint_files:
                logger.info(f"Found {len(checkpoint_files)} checkpoint file(s) in existing repo:")
                for cf in checkpoint_files:
                    logger.info(f"  - {cf.relative_to(repo_dir)}")
                return repo_dir

        logger.warning("Existing repository does not contain checkpoint files")
        logger.info("Removing existing directory and re-cloning...")
        import shutil
        shutil.rmtree(repo_dir)

    try:
        # Clone with specific branch
        cmd = [
            'git', 'clone',
            '--branch', PRETRAINED_BRANCH,
            '--depth', '1',  # Shallow clone
            BONDNET_REPO_URL,
            str(repo_dir)
        ]

        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"✓ Cloned pretrained branch to: {repo_dir}")

        # Look for checkpoint.pkl files in the pretrained directory
        # BonDNet uses .pkl format, not .pth/.pt
        pretrained_dir = repo_dir / 'bondnet' / 'prediction' / 'pretrained'

        if not pretrained_dir.exists():
            logger.warning(f"Pretrained directory not found: {pretrained_dir}")
            return repo_dir

        # Search for checkpoint.pkl files
        checkpoint_files = list(pretrained_dir.glob('**/checkpoint.pkl'))

        if checkpoint_files:
            logger.info(f"Found {len(checkpoint_files)} checkpoint file(s):")
            for cf in checkpoint_files:
                logger.info(f"  - {cf.relative_to(repo_dir)}")
            return repo_dir
        else:
            logger.warning("No checkpoint.pkl files found in pretrained branch")
            # Fall back to searching for any .pkl, .pth, .pt files
            model_files = (list(repo_dir.glob('**/*.pkl')) +
                          list(repo_dir.glob('**/*.pth')) +
                          list(repo_dir.glob('**/*.pt')))
            if model_files:
                logger.info(f"Found {len(model_files)} model file(s):")
                for mf in model_files[:5]:  # Show first 5
                    logger.info(f"  - {mf.relative_to(repo_dir)}")
            return repo_dir

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone pretrained branch: {e}")
        logger.error(e.stderr.decode() if e.stderr else "")
        return None
    except Exception as e:
        logger.error(f"Error cloning repository: {e}")
        return None


def create_readme(output_path: Path):
    """Create README with pretrained model information"""
    readme_content = f"""# BonDNet Pretrained Model

Downloaded from: {ZENODO_BASE_URL}

## Model Information

**Dataset:** BDNCM (Bond Dissociation for Neutral and Charged Molecules)
- Training data: 60,000+ unique homolytic and heterolytic bond dissociations
- Molecules: Neutral and charged species
- Elements: C, H, O, N (primary coverage)
- BDE types: Homolytic and heterolytic
- Performance: MAE ~0.022 eV (~0.51 kcal/mol)

**Paper:** BonDNet: a graph neural network for the prediction of bond dissociation
energies for charged molecules
- Published: Chemical Science, 2021
- DOI: 10.1039/D0SC05251E
- GitHub: https://github.com/mjwen/bondnet

## Usage for Transfer Learning

This pretrained model can be used as a starting point for fine-tuning on BDE-db2:

```bash
# Fine-tune on BDE-db2 dataset
python scripts/train_bondnet_bde_db2.py \\
    --data-dir data/processed/bondnet_training/ \\
    --pretrained {output_path} \\
    --output models/bondnet_bde_db2_finetuned.pth \\
    --epochs 50 \\
    --lr 0.0001
```

## Benefits of Transfer Learning

1. **Faster training**: 50-100 epochs instead of 200+
2. **Better generalization**: Pretrained on diverse molecules
3. **Lower learning rate**: 0.0001 (1/10 of from-scratch training)
4. **Retained knowledge**: C, H, O, N atom features already learned

## Note on Element Coverage

The pretrained model was trained primarily on C, H, O, N molecules.
For BDE-db2's additional elements (S, Cl, F, P, Br, I), fine-tuning will:
- Retain learned features for C, H, O, N
- Learn new features for halogens and other elements
- Achieve better performance than training from scratch

## Download Date

Model downloaded: {Path(__file__).stat().st_mtime if Path(__file__).exists() else 'N/A'}
"""

    readme_path = output_path.parent / "PRETRAINED_MODEL_README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    logger.info(f"Created README: {readme_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download BonDNet pretrained model weights for transfer learning"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/bondnet_pretrained.pth',
        help='Output path for pretrained model (default: models/bondnet_pretrained.pth)'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['zenodo', 'github', 'auto'],
        default='auto',
        help='Download method: zenodo (from Zenodo), github (clone pretrained branch), auto (try both)'
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("BonDNet Pretrained Model Download")
    logger.info("=" * 80)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output path: {output_path.absolute()}")
    logger.info(f"Download method: {args.method}")
    logger.info("")

    success = False

    # Method 1: Try Zenodo
    if args.method in ['zenodo', 'auto']:
        logger.info("Trying Zenodo download...")
        if download_from_zenodo(output_path):
            success = True
            logger.info("✓ Successfully downloaded from Zenodo")
        else:
            logger.warning("Zenodo download failed")

    # Method 2: Try GitHub pretrained branch
    if not success and args.method in ['github', 'auto']:
        logger.info("\nTrying GitHub pretrained branch...")
        repo_dir = clone_pretrained_branch(output_path.parent)

        if repo_dir:
            # Look for checkpoint.pkl files (BonDNet format)
            # Prefer BDNCM model over PubChem model
            bdncm_checkpoint = repo_dir / 'bondnet' / 'prediction' / 'pretrained' / 'bdncm' / '20200808' / 'checkpoint.pkl'
            pubchem_checkpoint = repo_dir / 'bondnet' / 'prediction' / 'pretrained' / 'pubchem' / '20200810' / 'checkpoint.pkl'

            checkpoint_to_use = None
            if bdncm_checkpoint.exists():
                checkpoint_to_use = bdncm_checkpoint
                logger.info("Found BDNCM pretrained model (preferred for transfer learning)")
            elif pubchem_checkpoint.exists():
                checkpoint_to_use = pubchem_checkpoint
                logger.info("Found PubChem pretrained model")
            else:
                # Fall back to searching for any checkpoint.pkl
                checkpoint_files = list(repo_dir.glob('**/checkpoint.pkl'))
                if checkpoint_files:
                    checkpoint_to_use = checkpoint_files[0]
                    logger.info(f"Found checkpoint at: {checkpoint_to_use.relative_to(repo_dir)}")

            if checkpoint_to_use:
                # Copy the checkpoint file to output path
                import shutil
                shutil.copy2(checkpoint_to_use, output_path)
                logger.info(f"✓ Copied model: {checkpoint_to_use.relative_to(repo_dir)} -> {output_path.name}")
                success = True
            else:
                logger.warning("No checkpoint.pkl files found in pretrained branch")
                logger.info(f"Repository cloned to: {repo_dir}")
                logger.info("You may need to manually locate the checkpoint file")

    # Create README
    if success:
        create_readme(output_path)

        logger.info("")
        logger.info("=" * 80)
        logger.info("Download Complete!")
        logger.info("=" * 80)
        logger.info(f"Pretrained model: {output_path.absolute()}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Ensure BDE-db2 data conversion is complete")
        logger.info("2. Fine-tune BonDNet with transfer learning:")
        logger.info(f"   python scripts/train_bondnet_bde_db2.py \\")
        logger.info(f"       --data-dir data/processed/bondnet_training/ \\")
        logger.info(f"       --pretrained {output_path} \\")
        logger.info(f"       --output models/bondnet_bde_db2_finetuned.pth \\")
        logger.info(f"       --epochs 50 \\")
        logger.info(f"       --lr 0.0001")
        logger.info("=" * 80)
    else:
        logger.error("")
        logger.error("=" * 80)
        logger.error("Download Failed!")
        logger.error("=" * 80)
        logger.error("Could not download pretrained model from Zenodo or GitHub")
        logger.error("")
        logger.error("Manual download instructions:")
        logger.error("1. Visit Zenodo: https://zenodo.org/records/15117901")
        logger.error("2. Download the model weight file (.pth, .pt, .pkl)")
        logger.error(f"3. Place it at: {output_path.absolute()}")
        logger.error("")
        logger.error("Or clone the pretrained branch manually:")
        logger.error(f"  git clone -b {PRETRAINED_BRANCH} {BONDNET_REPO_URL}")
        logger.error("  # Then locate the .pth file in the cloned repository")
        logger.error("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
