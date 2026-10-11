#!/usr/bin/env python3
"""EXP-V020 component-removal study; preserves V007/V019 source files."""

import copy
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import run_generalization_algorithms as GEN
from spectral_feedback_support.model import fuse_neutral_loss_prediction

BASE, COMMON = GEN.BASE, GEN.COMMON
ORIGINAL_CONFIG = BASE.model_config
ORIGINAL_MODEL = COMMON.QCGN2oEI_Minimal
ORIGINAL_HASHES = GEN.source_hashes
ORIGINAL_EVALUATE = BASE.evaluate
VERSION = "EXP-V020/r01"
DEFAULT_COMPONENTS = dict(recurrence=True, act=True, steps=4, residual=True,
                          feedback_input=True, feedback_diagnostics=True, stage_teacher=True)
CHANGES = {
    "full_sam": {}, "no_sam": {}, "fixed4": {"act": False}, "act2": {"steps": 2},
    "no_residual": {"residual": False}, "no_feedback_input": {"feedback_input": False},
    "no_feedback_all": {"feedback_input": False, "feedback_diagnostics": False},
    "no_stage_teacher": {"stage_teacher": False},
    "fusion_only": {"recurrence": False, "stage_teacher": False},
}


def source_hashes():
    return {**ORIGINAL_HASHES(), Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def zero_gate(module, inputs, output):
    return torch.zeros_like(output)


class ComponentModel(ORIGINAL_MODEL):
    """Same initialization, optionally inactive components, explicit checkpoint recipe."""

    def __init__(self, component_config=None, **kwargs):
        self.component_config = {**DEFAULT_COMPONENTS, **(component_config or {})}
        super().__init__(**kwargs)
        config = self.component_config
        self.use_act = config["act"]
        self.refinement_steps = config["steps"]
        self.spectral_residual_enabled = config["residual"]
        self.spectral_feedback_enabled = config["feedback_diagnostics"]
        if config["feedback_input"] and not config["feedback_diagnostics"]:
            raise ValueError("This study does not test diagnostics OFF with feedback input ON")
        if not config["feedback_input"] and config["feedback_diagnostics"]:
            self.spectral_feedback_gate.register_forward_hook(zero_gate)

    def forward(self, *args, **kwargs):
        if self.component_config["recurrence"]:
            return super().forward(*args, **kwargs)
        if kwargs.get("return_intermediate"):
            raise ValueError("Use the native fused study outputs")
        want_stats, want_aux = kwargs.get("return_act_stats", False), kwargs.get("return_aux_outputs", False)
        inputs = {**kwargs, "return_act_stats": False, "return_aux_outputs": True}
        original = self.use_recurrent_refinement, self.spectral_feedback_pilot
        try:
            self.use_recurrent_refinement, self.spectral_feedback_pilot = False, False
            direct, aux = super().forward(*args, **inputs)
        finally:
            self.use_recurrent_refinement, self.spectral_feedback_pilot = original
        neutral = aux["neutral_loss_spectrum"]
        prediction = fuse_neutral_loss_prediction(direct, neutral, kwargs["molecular_ion_mz"],
                                                  self.spectral_feedback_fusion_weight)
        zero = prediction.new_zeros(prediction.size(0))
        stats = dict(halt_probs=zero[:, None], ponder_steps=zero, act_loss=zero.mean(),
                     mean_ponder_steps=zero.mean(), max_ponder_steps=zero.max())
        aux.update(spectrum_already_fused=torch.tensor(True, device=prediction.device),
                   act_exit_spectra=prediction[:, None, :],
                   act_exit_weights=torch.ones_like(zero[:, None]))
        if want_stats and want_aux:
            return prediction, stats, aux
        if want_stats:
            return prediction, stats
        if want_aux:
            return prediction, aux
        return prediction


def loss_terms(model, batch, args, condition, training):
    prediction, stats, aux = BASE.forward(model, batch)
    main = (1 - F.cosine_similarity(prediction, batch.y, dim=-1)).mean()
    converted = BASE.neutral_target(batch.y, batch.molecular_ion_mz)
    neutral = (1 - F.cosine_similarity(aux["neutral_loss_spectrum"], converted, dim=-1)).mean()
    cosine, ranking = main.new_zeros(()), main.new_zeros(())
    components = model.component_config
    if training and condition == "teacher" and components["recurrence"] and components["stage_teacher"]:
        # Fixed-depth runs supervise all computed stages, not only the final output weight.
        weights = aux["act_exit_weights"] if components["act"] else torch.ones_like(aux["act_exit_weights"])
        cosine, ranking = BASE.stage_teacher_losses(aux["act_exit_spectra"], weights, batch.y, args.stage_rank_margin)
    loss = (main + args.neutral_loss_weight * neutral + args.act_loss_weight * stats["act_loss"]
            + args.stage_cosine_weight * cosine + args.stage_topk_weight * ranking)
    return loss, prediction, stats, dict(main_loss=main, neutral_loss=neutral,
                                        stage_cosine_loss=cosine, stage_topk_loss=ranking)


def rewrite_columns(path, changes):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        row.update(changes)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(model, loader, directory, condition, seed, split, epoch, device):
    result = ORIGINAL_EVALUATE(model, loader, directory, condition, seed, split, epoch, device)
    if not model.component_config["recurrence"]:
        folder = directory / "evaluation"
        rewrite_columns(folder / f"{split}_per_sample.csv", {"stop_depth": 0})
        rewrite_columns(folder / f"{split}_per_step.csv", {"step": 0})
        rewrite_columns(folder / f"{split}_per_step_summary.csv", {"step": 0})
        result["stop_depth"] = 0.
        BASE.write_json(folder / f"{split}_summary.json", result)
    return result


@contextmanager
def installed(recipe):
    old = BASE.model_config, BASE.loss_terms, BASE.evaluate, COMMON.QCGN2oEI_Minimal, GEN.source_hashes, GEN.VERSION
    try:
        BASE.model_config = lambda args: {**ORIGINAL_CONFIG(args), "component_config": recipe}
        BASE.loss_terms, BASE.evaluate = loss_terms, evaluate
        COMMON.QCGN2oEI_Minimal = ComponentModel
        GEN.source_hashes, GEN.VERSION = source_hashes, VERSION
        yield
    finally:
        BASE.model_config, BASE.loss_terms, BASE.evaluate, COMMON.QCGN2oEI_Minimal, GEN.source_hashes, GEN.VERSION = old


def component_recipe(name, override=None):
    return {**DEFAULT_COMPONENTS, **CHANGES.get(name, {}), **(override or {})}


def run_component(args, path, preparation, panel, name, seed, device, override=None):
    adjusted = copy.deepcopy(args)
    recipe = component_recipe(name, override)
    adjusted.output = args.output / name
    adjusted.component_name, adjusted.component_config = name, recipe
    if not recipe["act"] or not recipe["recurrence"]:
        adjusted.act_loss_weight = 0.
    if not recipe["stage_teacher"] or not recipe["recurrence"]:
        adjusted.stage_cosine_weight = adjusted.stage_topk_weight = 0.
    algorithm = "baseline" if name == "no_sam" else "sam"
    with installed(recipe):
        return GEN.run(adjusted, path, preparation, panel, algorithm, seed, device)


def main():
    parser = GEN.parser()
    parser.description = __doc__
    parser.add_argument("--components", choices=list(CHANGES), nargs="+", default=["full_sam"])
    args = parser.parse_args()
    if args.mode != "train" or not args.prepared_db or not args.panel_json:
        raise ValueError("Requires prepared index and fixed panel")
    if args.grad_accumulation != 1 or args.patience != 0 or args.fusion_weight != .3:
        raise ValueError("This study fixes accumulation1, no early stopping and fusion0.30")
    BASE.validate_update_schedule(args)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    path = args.prepared_db.resolve()
    preparation = json.loads((path.parent / "dataset_manifest.json").read_text())
    panel = json.loads(args.panel_json.read_text())["ids"]
    for seed in args.seeds:
        for component in args.components:
            run_component(args, path, preparation, panel, component, seed, device)


if __name__ == "__main__":
    main()
