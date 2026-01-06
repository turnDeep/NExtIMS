#!/usr/bin/env python3
# src/training/losses.py
"""
NEIMS v2.0 Loss Functions

Implements all loss functions for Teacher and Student training,
including knowledge distillation, load balancing, and entropy regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def compute_spectral_angle_loss(predicted, target):
    """
    Spectral Angle Mapper (SAM) Loss

    Measures the angle between predicted and target spectra.
    Useful for spectrum similarity.
    """
    # Normalize vectors
    pred_norm = F.normalize(predicted, p=2, dim=-1)
    target_norm = F.normalize(target, p=2, dim=-1)

    # Compute cosine similarity
    cos_sim = (pred_norm * target_norm).sum(dim=-1)

    # Clamp to avoid numerical issues
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

    # Convert to angle (in radians)
    angle = torch.acos(cos_sim)

    # Average angle
    loss = angle.mean()

    return loss


def compute_peak_loss(predicted, target, threshold=0.05):
    """
    Peak-Focused Loss

    Emphasizes correct prediction of significant peaks.
    """
    # Identify significant peaks in target
    peak_mask = target > (target.max(dim=-1, keepdim=True)[0] * threshold)

    # Compute weighted MSE (higher weight on peaks)
    weights = torch.where(peak_mask, torch.tensor(10.0), torch.tensor(1.0)).to(predicted.device)

    loss = (weights * (predicted - target) ** 2).mean()

    return loss


def cosine_similarity_loss(predicted: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Cosine Similarity Loss for MS/MS Prediction

    Used in QC-GN2oMS2 and adapted for NExtIMS v4.2 EI-MS prediction.
    Measures spectral similarity using cosine similarity metric.

    Loss = 1 - mean(cosine_similarity(pred, target))

    This loss function:
    - Focuses on spectral pattern matching rather than absolute intensities
    - Robust to intensity scale differences
    - Proven effective in QC-GN2oMS2 (achieved 0.88 cosine similarity)
    - Well-suited for MS spectrum prediction tasks

    Args:
        predicted: Predicted spectrum [batch_size, spectrum_dim]
        target: Target spectrum [batch_size, spectrum_dim]
        eps: Small constant for numerical stability (default: 1e-8)

    Returns:
        loss: Scalar loss value (range: [0, 2], optimal: 0)

    Example:
        >>> pred = torch.randn(32, 1000)  # Batch of 32 spectra, m/z 1-1000
        >>> target = torch.randn(32, 1000)
        >>> loss = cosine_similarity_loss(pred, target)
        >>> loss.backward()

    References:
        - QC-GN2oMS2: https://github.com/PNNL-m-q/QC-GN2oMS2
        - Code: qcgnoms/train.py (lines 156-158)
    """
    # Normalize to unit vectors
    predicted_norm = F.normalize(predicted, p=2, dim=-1, eps=eps)
    target_norm = F.normalize(target, p=2, dim=-1, eps=eps)

    # Compute cosine similarity per sample
    cosine_sim = (predicted_norm * target_norm).sum(dim=-1)

    # Cosine similarity loss: 1 - cosine_similarity
    # Range: [0, 2] where 0 is perfect match, 2 is opposite
    loss = 1.0 - cosine_sim.mean()

    return loss


class CosineSimilarityLoss(nn.Module):
    """
    Cosine Similarity Loss Module (nn.Module wrapper)

    Convenience wrapper for cosine_similarity_loss as a PyTorch module.
    Useful for integration with training frameworks that expect nn.Module losses.

    Example:
        >>> criterion = CosineSimilarityLoss()
        >>> loss = criterion(predicted, target)
    """

    def __init__(self, eps: float = 1e-8):
        """
        Args:
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity loss

        Args:
            predicted: Predicted spectrum [batch_size, spectrum_dim]
            target: Target spectrum [batch_size, spectrum_dim]

        Returns:
            loss: Scalar loss value
        """
        return cosine_similarity_loss(predicted, target, eps=self.eps)
