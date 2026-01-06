#!/usr/bin/env python3
# src/models/modules.py
"""
Shared modules for NEIMS v2.0

Common components used by both Teacher and Student models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Attention-based Graph Pooling
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, batch):
        """
        Args:
            x: Node features [N, hidden_dim]
            batch: Batch assignment [N]

        Returns:
            pooled: Graph-level representation [batch_size, hidden_dim]
        """
        from torch_geometric.nn import global_add_pool

        # Compute attention scores
        scores = self.attention(x)  # [N, 1]

        # Softmax per graph
        max_scores = global_add_pool(scores, batch)[batch]
        attention_weights = torch.exp(scores - max_scores)

        # Normalize per graph
        sum_weights = global_add_pool(attention_weights, batch)[batch]
        attention_weights = attention_weights / (sum_weights + 1e-8)

        # Weighted pooling
        weighted_x = x * attention_weights
        pooled = global_add_pool(weighted_x, batch)

        return pooled


class SpectrumNormalizer(nn.Module):
    """
    Spectrum Normalization Module

    Ensures predicted spectra are in valid range [0, 999] with proper peak distribution.
    """
    def __init__(self, mode='softmax'):
        super().__init__()
        self.mode = mode

    def forward(self, raw_spectrum):
        """
        Args:
            raw_spectrum: Raw prediction [batch_size, 501]

        Returns:
            normalized: Normalized spectrum [batch_size, 501] in range [0, 999]
        """
        if self.mode == 'softmax':
            # Softmax + scale to 999
            spectrum = F.softmax(raw_spectrum, dim=-1) * 999.0

        elif self.mode == 'sigmoid':
            # Sigmoid + scale to 999
            spectrum = torch.sigmoid(raw_spectrum) * 999.0

        elif self.mode == 'relu':
            # ReLU + normalize + scale
            spectrum = F.relu(raw_spectrum)
            max_val = spectrum.max(dim=-1, keepdim=True)[0]
            spectrum = (spectrum / (max_val + 1e-8)) * 999.0

        else:
            raise ValueError(f"Unknown normalization mode: {self.mode}")

        return spectrum


class GaussianSmoothing(nn.Module):
    """
    Gaussian Smoothing for Label Distribution Smoothing (LDS)

    Applies 1D Gaussian convolution to smooth spectra.
    """
    def __init__(self, sigma: float = 1.5, kernel_size: int = 7):
        super().__init__()
        self.sigma = sigma
        self.kernel_size = kernel_size

        # Create Gaussian kernel
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        self.register_buffer('kernel', kernel)

    def _create_gaussian_kernel(self, kernel_size, sigma):
        """Create 1D Gaussian kernel"""
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        return gauss.view(1, 1, -1)

    def forward(self, spectrum):
        """
        Args:
            spectrum: Input spectrum [batch_size, 501]

        Returns:
            smoothed: Smoothed spectrum [batch_size, 501]
        """
        # Add channel dimension
        spectrum = spectrum.unsqueeze(1)  # [batch_size, 1, 501]

        # Ensure kernel has the same dtype and device as input
        kernel = self.kernel.to(dtype=spectrum.dtype, device=spectrum.device)

        # Apply convolution
        padding = self.kernel_size // 2
        smoothed = F.conv1d(spectrum, kernel, padding=padding)

        # Remove channel dimension
        smoothed = smoothed.squeeze(1)  # [batch_size, 501]

        return smoothed


class ExponentialMovingAverage:
    """
    Exponential Moving Average (EMA) for model weights

    Useful for stabilizing training and improving generalization.
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        self.register()

    def register(self):
        """Register model parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA weights to model (for evaluation)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def init_weights(module):
    """
    Initialize model weights

    Uses Xavier/Glorot initialization for Linear layers.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def count_parameters(model):
    """
    Count total and trainable parameters

    Returns:
        total: Total parameters
        trainable: Trainable parameters
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total, trainable


def get_model_size_mb(model):
    """
    Estimate model size in MB

    Returns:
        size_mb: Model size in megabytes
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())

    size_bytes = param_size + buffer_size
    size_mb = size_bytes / (1024 ** 2)

    return size_mb
