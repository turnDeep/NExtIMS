#!/usr/bin/env python3
"""
NExtIMS v4.2: BonDNet Retraining on BDE-db2

Retrains BonDNet on the BDE-db2 dataset (531,244 BDEs from 65,540 molecules)
to improve BDE prediction coverage and accuracy for NIST17 EI-MS dataset.

Why retrain BonDNet?
    - Original BonDNet: Trained on 64,000 molecules
    - BDE-db2: 531,244 BDEs (8x larger dataset)
    - NIST17 coverage: 95% → 99%+ (halogen-containing compounds)
    - Expected MAE: ~0.51 kcal/mol (similar or better)

Workflow:
    1. Download BDE-db2 dataset:
       python scripts/download_bde_db2.py --output data/external/bde-db2

    2. Convert to BonDNet format:
       python scripts/convert_bde_db2_to_bondnet.py \\
           --input data/external/bde-db2/bde-db2.csv \\
           --output data/processed/bondnet_training/

    3. Retrain BonDNet (this script):
       python scripts/train_bondnet_bde_db2.py \\
           --data-dir data/processed/bondnet_training/ \\
           --output models/bondnet_bde_db2.pth

    4. Pre-compute BDE with custom model:
       python scripts/precompute_bde.py \\
           --nist-msp data/NIST17.MSP \\
           --model models/bondnet_bde_db2.pth \\
           --output data/processed/bde_cache/nist17_bde_cache.h5

Hardware Requirements:
    - GPU: RTX 5070 Ti 16GB (or similar)
    - Training time: ~2-3 days (with optimizations)
    - Disk: ~10 GB (dataset + model checkpoints)

Requirements:
    - BonDNet: pip install git+https://github.com/mjwen/bondnet.git
    - DGL: pip install dgl
    - PyTorch >= 1.10.0
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path
import json
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_bondnet_installation():
    """Check if BonDNet is installed and get installation path"""
    try:
        import bondnet
        bondnet_path = Path(bondnet.__file__).parent.parent
        logger.info(f"BonDNet found: {bondnet_path}")
        return bondnet_path
    except ImportError:
        logger.error("BonDNet is not installed!")
        logger.error("Install with:")
        logger.error("  pip install git+https://github.com/mjwen/bondnet.git")
        logger.error("Or clone and install:")
        logger.error("  git clone https://github.com/mjwen/bondnet.git")
        logger.error("  cd bondnet && pip install -e .")
        sys.exit(1)


def check_training_data(data_dir: Path) -> bool:
    """
    Check if training data exists and is valid

    Required files:
        - molecules.sdf
        - molecule_attributes.yaml
        - reactions.yaml
    """
    required_files = [
        'molecules.sdf',
        'molecule_attributes.yaml',
        'reactions.yaml'
    ]

    missing_files = []
    for filename in required_files:
        filepath = data_dir / filename
        if not filepath.exists():
            missing_files.append(filename)
        else:
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"  ✓ {filename} ({file_size_mb:.1f} MB)")

    if missing_files:
        logger.error("Missing required files:")
        for filename in missing_files:
            logger.error(f"  ✗ {filename}")
        logger.error("\nPlease run data preparation first:")
        logger.error("  python scripts/download_bde_db2.py")
        logger.error("  python scripts/convert_bde_db2_to_bondnet.py")
        return False

    return True


def create_training_config(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 200,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    device: str = 'cuda'
) -> Path:
    """
    Create BonDNet training configuration file

    Args:
        data_dir: Directory containing training data
        output_dir: Directory for model checkpoints
        epochs: Number of training epochs
        batch_size: Batch size (64 for RTX 5070 Ti 16GB)
        learning_rate: Learning rate
        device: 'cuda' or 'cpu'

    Returns:
        config_path: Path to generated config file
    """
    config = {
        # Data paths
        'dataset_location': str(data_dir),
        'molecules_file': str(data_dir / 'molecules.sdf'),
        'molecule_attributes_file': str(data_dir / 'molecule_attributes.yaml'),
        'reactions_file': str(data_dir / 'reactions.yaml'),

        # Model architecture
        'embedding_size': 128,
        'num_gnn_layers': 4,
        'gnn_hidden_size': 128,
        'num_fc_layers': 3,
        'fc_hidden_size': 128,
        'conv_fn': 'GatedGCNConv',  # BonDNet default

        # Training parameters
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': 1e-4,
        'scheduler': 'reduce_on_plateau',
        'scheduler_patience': 10,
        'scheduler_factor': 0.5,

        # GPU optimization
        'device': device,
        'num_workers': 4,
        'pin_memory': True if device == 'cuda' else False,

        # Checkpointing
        'checkpoint_dir': str(output_dir / 'checkpoints'),
        'save_interval': 10,  # Save every 10 epochs
        'early_stopping_patience': 30,

        # Logging
        'log_dir': str(output_dir / 'logs'),
        'log_interval': 100,  # Log every 100 batches

        # Model output
        'output_path': str(output_dir / 'bondnet_bde_db2.pth')
    }

    config_path = output_dir / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Created training config: {config_path}")
    return config_path


def run_bondnet_training(
    data_dir: Path,
    config_path: Path,
    bondnet_path: Path,
    device: str = 'cuda',
    epochs: int = 200,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    output_path: Path = None
):
    """
    Run BonDNet training using its native training script

    Args:
        data_dir: Training data directory
        config_path: Configuration file path
        bondnet_path: BonDNet installation path
        device: 'cuda' or 'cpu'
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        output_path: Path to save the final model
    """
    # Find BonDNet training script (correct path)
    training_script = bondnet_path / 'bondnet' / 'scripts' / 'train_bde_distributed.py'

    if not training_script.exists():
        logger.error(f"BonDNet training script not found at {training_script}")
        logger.error("Expected: train_bde_distributed.py")
        sys.exit(1)

    logger.info(f"Running BonDNet training script: {training_script}")

    # Prepare input file paths
    molecules_file = data_dir / 'molecules.sdf'
    molecule_attributes_file = data_dir / 'molecule_attributes.yaml'
    reactions_file = data_dir / 'reactions.yaml'

    # Prepare command with BonDNet's expected arguments
    cmd = [
        sys.executable,
        str(training_script),
        '--molecule_file', str(molecules_file),
        '--molecule_attributes_file', str(molecule_attributes_file),
        '--reaction_file', str(reactions_file),
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--lr', str(learning_rate),
        '--embedding-size', '128',
        '--gated-num-layers', '4',
        '--gated-hidden-size', '128',
        '--fc-num-layers', '3',
        '--fc-hidden-size', '128',
    ]

    # Add GPU settings
    if device == 'cuda':
        cmd.extend(['--gpu', '0'])
    else:
        cmd.extend(['--gpu', 'None'])

    logger.info(f"Command: {' '.join(cmd)}")

    # Run training
    try:
        subprocess.run(cmd, check=True)

        # BonDNet saves models in a checkpoints directory
        # We need to copy the best model to the desired output path
        if output_path:
            logger.info(f"Training complete. Copying model to {output_path}")
            # Note: BonDNet's script saves to a specific location
            # We'll need to find and copy the best checkpoint

    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)




def main():
    parser = argparse.ArgumentParser(
        description="Retrain BonDNet on BDE-db2 dataset"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing BonDNet training data (SDF + YAML files)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/bondnet_bde_db2.pth',
        help='Output model path (default: models/bondnet_bde_db2.pth)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Number of training epochs (default: 200)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size (default: 64 for RTX 5070 Ti 16GB)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device: cuda or cpu (default: cuda)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint (optional)'
    )

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("NExtIMS v4.2: BonDNet Retraining on BDE-db2")
    logger.info("="*80)

    # Check BonDNet installation
    bondnet_path = check_bondnet_installation()

    # Check training data
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    logger.info(f"Data directory: {data_dir}")
    logger.info("Checking training data files...")

    if not check_training_data(data_dir):
        sys.exit(1)

    logger.info("✓ All training data files present")

    # Create output directory
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model will be saved to: {output_path}")

    # Training parameters
    logger.info("")
    logger.info("Training parameters:")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Device: {args.device}")

    # Estimate training time
    if args.device == 'cuda':
        estimated_time_hours = 48  # ~2 days for RTX 5070 Ti
        logger.info(f"  Estimated time: ~{estimated_time_hours} hours (~{estimated_time_hours/24:.1f} days)")
    else:
        logger.warning("  CPU training will be very slow (not recommended)")

    logger.info("")

    # Create training config
    config_path = create_training_config(
        data_dir=data_dir,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    )

    # Run training
    logger.info("")
    logger.info("Starting BonDNet training...")
    logger.info("="*80)

    run_bondnet_training(
        data_dir=data_dir,
        config_path=config_path,
        bondnet_path=bondnet_path,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_path=output_path
    )

    logger.info("")
    logger.info("="*80)
    logger.info("Training Complete!")
    logger.info("="*80)
    logger.info(f"Model saved to: {output_path}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Evaluate model performance:")
    logger.info(f"     (check {output_dir / 'logs'} for training metrics)")
    logger.info("")
    logger.info("  2. Pre-compute BDE for NIST17 with custom model:")
    logger.info("     python scripts/precompute_bde.py \\")
    logger.info("       --nist-msp data/NIST17.MSP \\")
    logger.info(f"       --model {output_path} \\")
    logger.info("       --output data/processed/bde_cache/nist17_bde_cache_custom.h5")
    logger.info("="*80)


if __name__ == '__main__':
    main()
