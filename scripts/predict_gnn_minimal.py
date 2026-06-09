#!/usr/bin/env python3
"""
NExtIMS v1.0: Prediction Script for QCGN2oEI_Minimal
Supports:
- Baseline QCGN2oEI_Minimal
- Stable ACT Recurrent Spectral Refinement
- LTI-style stabilization
- Loop embedding
- ACT inference
- Fixed-loop inference override
"""
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.qcgn2oei_minimal import QCGN2oEI_Minimal
from src.data.graph_generator_minimal import MinimalGraphGenerator
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
def load_checkpoint(checkpoint_path: str, device: str) -> Dict[str, Any]:
    """
    Load checkpoint safely.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary.")
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")
    return checkpoint
def get_model_config_from_checkpoint(
    checkpoint: Dict[str, Any],
    args,
) -> Dict[str, Any]:
    """
    Restore model configuration from checkpoint.
    Priority:
        1. CLI override
        2. checkpoint["model_info"]
        3. checkpoint["args"]
        4. safe defaults
    """
    model_info = checkpoint.get("model_info", {})
    train_args = checkpoint.get("args", {})
    def pick(name: str, default):
        """
        Select config value.
        CLI explicit override is handled separately for some fields.
        """
        if name in model_info:
            return model_info[name]
        if name in train_args:
            return train_args[name]
        return default
    config = {
        "node_dim": int(pick("node_dim", 34)),
        "edge_dim": int(pick("edge_dim", 10)),
        "hidden_dim": int(pick("hidden_dim", 768)),
        "num_layers": int(pick("num_layers", 14)),
        "num_heads": int(pick("num_heads", 24)),
        "output_dim": int(pick("output_dim", 1000)),
        "dropout": float(pick("dropout", 0.0)),
        "use_edge_attr": bool(pick("use_edge_attr", True)),
        "use_recurrent_refinement": bool(
            pick("use_recurrent_refinement", False)
        ),
        "refinement_steps": int(
            pick("refinement_steps", 4)
        ),
        "refinement_max_steps": int(
            pick("refinement_max_steps", 8)
        ),
        "refinement_dropout": float(
            pick("refinement_dropout", 0.0)
        ),
        "use_lti_stabilization": bool(
            pick("use_lti_stabilization", True)
        ),
        "use_loop_embedding": bool(
            pick("use_loop_embedding", True)
        ),
        "use_act": bool(
            pick("use_act", False)
        ),
        "act_epsilon": float(
            pick("act_epsilon", 0.01)
        ),
        "act_loss_weight": float(
            pick("act_loss_weight", 0.0)
        ),
    }
    # CLI overrides
    if args.refinement_steps is not None:
        config["refinement_steps"] = args.refinement_steps
    if args.refinement_max_steps is not None:
        config["refinement_max_steps"] = args.refinement_max_steps
    if args.force_no_act:
        # Model still has ACT weights, but forward can bypass ACT.
        # Do not set config["use_act"] = False here,
        # otherwise state_dict loading may fail for halt_head or related weights.
        pass
    return config
def build_model_from_checkpoint(
    checkpoint: Dict[str, Any],
    args,
    device: str,
) -> QCGN2oEI_Minimal:
    """
    Build model using the same architecture as training.
    """
    config = get_model_config_from_checkpoint(
        checkpoint=checkpoint,
        args=args,
    )
    logger.info("Model config restored from checkpoint:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    model = QCGN2oEI_Minimal(**config).to(device)
    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False,
    )
    if missing:
        logger.warning("Missing keys when loading state_dict:")
        for key in missing:
            logger.warning(f"  missing: {key}")
    if unexpected:
        logger.warning("Unexpected keys when loading state_dict:")
        for key in unexpected:
            logger.warning(f"  unexpected: {key}")
    # ここで refinement 関連の key が大量に missing なら、学習時と推論時の構成が不一致
    critical_missing = [
        key for key in missing
        if key.startswith("refinement_block")
    ]
    if critical_missing and config["use_recurrent_refinement"]:
        raise RuntimeError(
            "Refinement block keys are missing. "
            "The checkpoint and prediction model config are inconsistent."
        )
    model.eval()
    return model
def smiles_to_graph(
    smiles: str,
    graph_generator: MinimalGraphGenerator,
):
    """
    Convert SMILES to PyG graph.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    graph = graph_generator.mol_to_graph(
        mol=mol,
        smiles=smiles,
        spectrum=None,
        compound_name="prediction_target",
    )
    return graph
def predict_spectrum(
    model: QCGN2oEI_Minimal,
    graph,
    device: str,
    refinement_steps: Optional[int] = None,
    force_no_act: bool = False,
    return_act_stats: bool = True,
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """
    Predict EI-MS spectrum for one graph.
    """
    graph = graph.to(device)
    with torch.no_grad():
        if return_act_stats:
            spectrum, act_stats = model(
                x=graph.x,
                edge_index=graph.edge_index,
                edge_attr=graph.edge_attr,
                batch=None,
                refinement_steps=refinement_steps,
                return_act_stats=True,
                force_no_act=force_no_act,
            )
        else:
            spectrum = model(
                x=graph.x,
                edge_index=graph.edge_index,
                edge_attr=graph.edge_attr,
                batch=None,
                refinement_steps=refinement_steps,
                return_act_stats=False,
                force_no_act=force_no_act,
            )
            act_stats = None
    spectrum_np = spectrum.squeeze(0).detach().cpu().numpy()

    # Normalize to relative intensity (base peak = 100)
    max_intensity = spectrum_np.max()
    if max_intensity > 0:
        spectrum_np = (spectrum_np / max_intensity) * 100.0

    act_stats_out = None
    if act_stats is not None:
        act_stats_out = {}
        for key, value in act_stats.items():
            if torch.is_tensor(value):
                act_stats_out[key] = value.detach().cpu()
            else:
                act_stats_out[key] = value
    return spectrum_np, act_stats_out
def visualize_spectrum(
    spectrum: np.ndarray,
    smiles: str,
    output_path: str,
    mz_range: tuple[int, int] = (1, 1000),
    figsize: tuple[int, int] = (12, 6),
    dpi: int = 300
):
    """
    Visualize predicted spectrum (same format as legacy script).
    """
    mz_values = np.arange(mz_range[0], mz_range[0] + len(spectrum))

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Stem plot for spectrum
    markerline, stemlines, baseline = ax.stem(
        mz_values,
        spectrum,
        linefmt='b-',
        markerfmt=' ',  # No markers
        basefmt=' '
    )
    stemlines.set_linewidth(0.5)

    ax.set_xlabel("m/z", fontsize=12, fontweight='bold')
    ax.set_ylabel("Relative Intensity (%)", fontsize=12, fontweight='bold')
    ax.set_title(f"Predicted EI-MS Spectrum\n{smiles}", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Try to set x-axis limit based on molecular weight
    try:
        mol = Chem.MolFromSmiles(smiles)
        mw = Descriptors.MolWt(mol)
        ax.set_xlim(0, mw + 50)
    except Exception:
        ax.set_xlim(mz_range)

    ax.set_ylim(0, 110)

    # Annotate Top 10 peaks
    top_indices = np.argsort(spectrum)[::-1]
    count = 0
    for idx in top_indices:
        if count >= 10:
            break

        intensity = spectrum[idx]
        if intensity < 0.1:  # Skip noise (0.1%)
            continue

        mz = idx + mz_range[0]

        ax.text(
            mz,
            intensity + 1,
            f"{int(mz)}",
            ha='center',
            va='bottom',
            fontsize=9
        )
        count += 1

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def get_top_peaks(
    spectrum: np.ndarray,
    top_k: int = 20,
    min_intensity: float = 0.0,
) -> List[Tuple[int, float]]:
    """
    Get top-k m/z peaks from predicted spectrum.
    spectrum index 0 corresponds to m/z 1.
    """
    if spectrum.ndim != 1:
        raise ValueError("Spectrum must be 1D.")
    indices = np.argsort(spectrum)[::-1]
    peaks = []
    for idx in indices:
        intensity = float(spectrum[idx])
        mz = int(idx + 1)
        if intensity < min_intensity:
            continue
        peaks.append((mz, intensity))
        if len(peaks) >= top_k:
            break
    return peaks
def save_spectrum_csv(
    output_path: str,
    spectrum: np.ndarray,
):
    """
    Save full predicted spectrum to CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("mz,intensity\n")
        for idx, intensity in enumerate(spectrum):
            mz = idx + 1
            f.write(f"{mz},{float(intensity):.10f}\n")
def save_top_peaks_csv(
    output_path: str,
    peaks: List[Tuple[int, float]],
):
    """
    Save top peaks to CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("rank,mz,intensity\n")
        for rank, (mz, intensity) in enumerate(peaks, start=1):
            f.write(f"{rank},{mz},{intensity:.10f}\n")
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict EI-MS spectrum using QCGN2oEI_Minimal"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint .pth",
    )
    parser.add_argument(
        "--smiles",
        type=str,
        required=True,
        help="Input SMILES string",
    )
    parser.add_argument(
        "--bde-cache",
        type=str,
        default=None,
        help="Path to BDE HDF5 cache",
    )
    parser.add_argument(
        "--bondnet-model",
        type=str,
        default=None,
        help="Optional BonDNet model path for BDE calculation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda or cpu",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top peaks to print",
    )
    parser.add_argument(
        "--min-intensity",
        type=float,
        default=0.0,
        help="Minimum intensity threshold for top peak output",
    )
    parser.add_argument(
        "--output-spectrum",
        type=str,
        default=None,
        help="Optional path to save full predicted spectrum CSV",
    )
    parser.add_argument(
        "--output-peaks",
        type=str,
        default=None,
        help="Optional path to save top peaks CSV",
    )
    parser.add_argument(
        "--output-image",
        type=str,
        default=None,
        help="Optional path to save predicted spectrum plot image (e.g., predicted.png)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="If set, saves spectrum image. Will save to 'predicted_spectrum.png' if --output-image is not provided.",
    )
    parser.add_argument(
        "--refinement-steps",
        type=int,
        default=None,
        help=(
            "Override refinement steps at inference. "
            "If omitted, use checkpoint setting."
        ),
    )
    parser.add_argument(
        "--refinement-max-steps",
        type=int,
        default=None,
        help=(
            "Override refinement max steps at model construction. "
            "Usually keep this equal to checkpoint setting."
        ),
    )
    parser.add_argument(
        "--force-no-act",
        action="store_true",
        help=(
            "Force fixed-loop inference even if the checkpoint was trained with ACT. "
            "Useful for comparing ACT vs fixed-loop."
        ),
    )
    parser.add_argument(
        "--no-act-stats",
        action="store_true",
        help="Do not return/print ACT statistics",
    )
    return parser
def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA was requested but is not available. Falling back to CPU.")
        args.device = "cpu"
    checkpoint = load_checkpoint(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    model = build_model_from_checkpoint(
        checkpoint=checkpoint,
        args=args,
        device=args.device,
    )
    graph_generator = MinimalGraphGenerator(
        bde_cache_path=args.bde_cache,
        use_bde_calculator=(args.bondnet_model is not None),
        bondnet_model=args.bondnet_model,
        default_bde=85.0,
    )
    graph = smiles_to_graph(
        smiles=args.smiles,
        graph_generator=graph_generator,
    )
    spectrum, act_stats = predict_spectrum(
        model=model,
        graph=graph,
        device=args.device,
        refinement_steps=args.refinement_steps,
        force_no_act=args.force_no_act,
        return_act_stats=not args.no_act_stats,
    )
    peaks = get_top_peaks(
        spectrum=spectrum,
        top_k=args.top_k,
        min_intensity=args.min_intensity,
    )
    print("")
    print("=" * 80)
    print("Prediction Result")
    print("=" * 80)
    print(f"SMILES: {args.smiles}")
    print(f"Checkpoint: {args.checkpoint}")
    if args.refinement_steps is not None:
        print(f"Refinement steps override: {args.refinement_steps}")
    print(f"Force no ACT: {args.force_no_act}")
    if act_stats is not None:
        print("")
        print("ACT stats:")
        if "mean_ponder_steps" in act_stats:
            print(f"  mean_ponder_steps: {float(act_stats['mean_ponder_steps']):.4f}")
        if "max_ponder_steps" in act_stats:
            print(f"  max_ponder_steps: {float(act_stats['max_ponder_steps']):.4f}")
        if "ponder_steps" in act_stats:
            ponder = act_stats["ponder_steps"].numpy()
            print(f"  ponder_steps: {ponder}")
        if "halt_probs" in act_stats:
            halt_probs = act_stats["halt_probs"].numpy()
            print(f"  halt_probs shape: {halt_probs.shape}")
    print("")
    print(f"Top {args.top_k} predicted peaks:")
    print("-" * 40)
    print("rank\tm/z\tintensity")
    print("-" * 40)
    for rank, (mz, intensity) in enumerate(peaks, start=1):
        print(f"{rank}\t{mz}\t{intensity:.4f}")
    print("-" * 40)
    print(f"Spectrum sum: {float(spectrum.sum()):.4f}")
    print(f"Spectrum max: {float(spectrum.max()):.4f}")
    print(f"Spectrum argmax m/z: {int(np.argmax(spectrum) + 1)}")
    if args.output_spectrum is not None:
        save_spectrum_csv(args.output_spectrum, spectrum)
        print(f"Saved full spectrum: {args.output_spectrum}")
    if args.output_peaks is not None:
        save_top_peaks_csv(args.output_peaks, peaks)
        print(f"Saved top peaks: {args.output_peaks}")

    if args.visualize or args.output_image is not None:
        image_path = args.output_image if args.output_image is not None else "predicted_spectrum.png"
        visualize_spectrum(
            spectrum=spectrum,
            smiles=args.smiles,
            output_path=image_path
        )
        print(f"Saved spectrum image: {image_path}")

    print("=" * 80)
if __name__ == "__main__":
    main()
