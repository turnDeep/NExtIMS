"""Weak supervision of native fused ACT exits, without feeding targets back."""

import torch
import torch.nn.functional as F


def topk_boundary_ranking_loss(prediction, target, ks=(5, 10, 20), margin=0.2):
    """Rank positive target top-k bins above hard negatives outside each set."""
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("Expected equal [samples, mz_bins] tensors")
    if prediction.size(1) <= max(ks):
        raise ValueError("Spectrum must have more bins than the largest k")
    log_prediction = prediction.clamp_min(1e-8).log()
    losses = []
    for k in ks:
        positive_indices = target.topk(k, dim=-1).indices
        positive_valid = target.gather(1, positive_indices) > 0
        positive_set = torch.zeros_like(target, dtype=torch.bool)
        positive_set.scatter_(1, positive_indices, positive_valid)
        negative_scores = log_prediction.masked_fill(positive_set, -torch.inf)
        negative_indices = negative_scores.detach().topk(k, dim=-1).indices
        positive_log = log_prediction.gather(1, positive_indices)
        negative_log = log_prediction.gather(1, negative_indices)
        pair_loss = F.softplus(
            negative_log.unsqueeze(1) - positive_log.unsqueeze(2) + margin
        )
        valid_pairs = positive_valid.unsqueeze(2).expand_as(pair_loss)
        sample_loss = (pair_loss * valid_pairs).sum(dim=(1, 2)) / (
            valid_pairs.sum(dim=(1, 2)).clamp_min(1)
        )
        losses.append(sample_loss)
    return torch.stack(losses, dim=1).mean(dim=1)


def stage_teacher_losses(exit_spectra, exit_weights, target, margin=0.2):
    """Equal-per-active-exit teachers; detached masks cannot reward halting."""
    if exit_spectra.ndim != 3 or target.ndim != 2:
        raise ValueError("Expected [samples, steps, bins] and [samples, bins]")
    if exit_spectra.shape[:2] != exit_weights.shape:
        raise ValueError("ACT weights do not match exit spectra")
    expanded_target = target.unsqueeze(1).expand_as(exit_spectra)
    active = (exit_weights.detach() > 0).to(exit_spectra.dtype)
    cosine_per_exit = 1.0 - F.cosine_similarity(
        exit_spectra, expanded_target, dim=-1
    )
    ranking_per_exit = topk_boundary_ranking_loss(
        exit_spectra.reshape(-1, exit_spectra.size(-1)),
        expanded_target.reshape(-1, target.size(-1)),
        margin=margin,
    ).view_as(active)
    denominator = active.sum(dim=1).clamp_min(1)
    cosine_loss = ((cosine_per_exit * active).sum(dim=1) / denominator).mean()
    ranking_loss = ((ranking_per_exit * active).sum(dim=1) / denominator).mean()
    return cosine_loss, ranking_loss
