#!/usr/bin/env python3
# src/models/qcgn2oei_minimal.py
"""
NExtIMS v5.0: QCGN2oEI_Minimal Large-Scale Model

Quantum Chemistry-augmented Graph Neural Network for EI-MS prediction.
Large-scale configuration for maximum performance.

Architecture:
- 14-layer GATv2Conv with residual connections
- 24 attention heads per layer
- Hidden dimension: 768
- Global mean pooling
- Output: m/z 1-1000 spectrum (1000 dimensions)

Design Philosophy:
- v5.0 Large-Scale: ~36M parameters (vs 14M in v4.4)
- Deep GATv2 architecture (14 layers)
- High capacity (768 dim, 24 heads)
- Gradient Checkpointing support for memory efficiency
- BDE-enriched edges for fragmentation modeling

References:
- QC-GN2oMS2: https://github.com/PNNL-m-q/QC-GN2oMS2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch.utils.checkpoint import checkpoint
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class QCGN2oEI_Minimal(nn.Module):
    """
    Minimal Graph Neural Network for EI-MS Prediction (Large Scale)

    Configuration (v5.0):
    - Input: 34-dim node features, 10-dim edge features
    - GNN: 14-layer GATv2Conv with residual connections
    - Hidden dim: 768
    - Heads: 24
    - Params: ~36M
    """

    def __init__(
        self,
        node_dim: int = 34,
        edge_dim: int = 10,
        hidden_dim: int = 768,
        num_layers: int = 14,
        num_heads: int = 24,
        output_dim: int = 1000,
        dropout: float = 0.1,
        use_edge_attr: bool = True,
        gradient_checkpointing: bool = False
    ):
        """
        Initialize QCGN2oEI_Minimal model (v5.0 Large-Scale)

        Args:
            node_dim: Node feature dimension (default: 34)
            edge_dim: Edge feature dimension (default: 10)
            hidden_dim: Hidden dimension (default: 768)
            num_layers: Number of GATv2 layers (default: 14)
            num_heads: Number of attention heads (default: 24)
            output_dim: Output spectrum dimension (default: 1000)
            dropout: Dropout rate (default: 0.1)
            use_edge_attr: Whether to use edge attributes (default: True)
            gradient_checkpointing: Enable gradient checkpointing to save memory (default: False)
        """
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.dropout = dropout
        self.use_edge_attr = use_edge_attr
        self.gradient_checkpointing = gradient_checkpointing

        # Node encoder: node_dim → hidden_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )

        # Edge encoder: edge_dim → hidden_dim
        if use_edge_attr:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout)
            )
        else:
            self.edge_encoder = None

        # GATv2Conv layers
        self.gat_layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()

        for i in range(num_layers):
            # GATv2Conv layer
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    edge_dim=hidden_dim if use_edge_attr else None,
                    dropout=dropout,
                    concat=True,  # Concatenate heads -> returns [N, heads * (dim/heads)] = [N, dim]
                    residual=False  # We implement custom residual
                )
            )

            # Residual projection layer
            self.residual_layers.append(
                nn.Linear(hidden_dim, hidden_dim)
            )

        # Spectrum prediction head
        self.spectrum_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

        logger.info(f"QCGN2oEI_Minimal initialized (v5.0 Large-Scale):")
        logger.info(f"  Node dim: {node_dim}")
        logger.info(f"  Edge dim: {edge_dim}")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  Num layers: {num_layers}")
        logger.info(f"  Num heads: {num_heads}")
        logger.info(f"  Gradient Checkpointing: {gradient_checkpointing}")

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"  Total parameters: {total_params:,}")

    def _run_layer(self, i: int, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Helper method for a single layer block (used for checkpointing)
        Includes: GAT -> Residual -> ELU -> Dropout
        """
        # Store input for residual
        residual = x

        # GATv2Conv layer
        x = self.gat_layers[i](
            x,
            edge_index,
            edge_attr=edge_attr
        )

        # Residual connection with projection
        x = F.elu(x + self.residual_layers[i](residual))

        # Dropout
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        """
        # Encode node features
        x = self.node_encoder(x)

        # Encode edge features
        if self.use_edge_attr and edge_attr is not None:
            edge_attr_encoded = self.edge_encoder(edge_attr)
        else:
            edge_attr_encoded = None

        # GAT Layers
        for i in range(self.num_layers):
            if self.gradient_checkpointing and self.training:
                # Use checkpointing
                # Note: `i` is not a tensor, but checkpoint can handle non-tensor args if they don't require grad.
                # However, checkpoint only computes gradients for tensor inputs that have requires_grad=True.
                # `x` has requires_grad=True.
                x = checkpoint(
                    self._run_layer,
                    i,              # Arg 1
                    x,              # Arg 2 (Tensor, requires_grad)
                    edge_index,     # Arg 3 (Tensor, no grad)
                    edge_attr_encoded, # Arg 4 (Tensor, maybe grad)
                    use_reentrant=False # Recommended for newer PyTorch
                )
            else:
                x = self._run_layer(i, x, edge_index, edge_attr_encoded)

        # Global mean pooling
        if batch is None:
            graph_repr = x.mean(dim=0, keepdim=True)
        else:
            graph_repr = global_mean_pool(x, batch)

        # Spectrum prediction
        spectrum = self.spectrum_head(graph_repr)

        return spectrum

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'QCGN2oEI_Minimal',
            'version': 'v5.0',
            'node_dim': self.node_dim,
            'edge_dim': self.edge_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'gradient_checkpointing': self.gradient_checkpointing,
            'total_params': total_params,
            'trainable_params': trainable_params
        }


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("="*60)
    print("QCGN2oEI_Minimal v5.0 Test")
    print("="*60)

    # Create model
    model = QCGN2oEI_Minimal(
        node_dim=34,
        edge_dim=10,
        hidden_dim=768,
        num_layers=14,
        num_heads=24,
        gradient_checkpointing=True
    )

    # Print model info
    info = model.get_model_info()
    print(f"Total Parameters: {info['total_params']:,}")
    print(f"Gradient Checkpointing: {info['gradient_checkpointing']}")

    # Dummy Forward Pass
    x = torch.randn(10, 34)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.randn(2, 10)

    # Enable training mode for checkpointing check
    model.train()

    # Forward needs to allow grads for checkpointing to trigger properly (conceptually)
    x.requires_grad = True

    out = model(x, edge_index, edge_attr)
    print(f"Output shape: {out.shape}")

    # Backward check
    loss = out.sum()
    loss.backward()
    print("Backward pass successful.")
