#!/usr/bin/env python3
# src/models/qcgn2oei_minimal.py
"""
NExtIMS v4.2: QCGN2oEI_Minimal Model

Quantum Chemistry-augmented Graph Neural Network for EI-MS prediction.
Minimal configuration based on QC-GN2oMS2 architecture, adapted for EI-MS.

Architecture:
- 14-layer GATv2Conv with residual connections
- 24 attention heads per layer
- Hidden dimension: 768
- Global mean pooling
- Output: m/z 1-1000 spectrum (1000 dimensions)

Design Philosophy:
- Minimal feature dimensions (34-dim nodes, 10-dim edges)
- Deep GATv2 architecture for representation learning
- BDE-enriched edges for fragmentation modeling
- ELU activation for smooth gradients
- Residual connections for gradient flow

References:
- QC-GN2oMS2: https://github.com/PNNL-m-q/QC-GN2oMS2
- Architecture: qcgnoms/qc2.py (lines 120-180)
- Achieved: 0.88 cosine similarity on MS/MS prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class QCGN2oEI_Minimal(nn.Module):
    """
    Minimal Graph Neural Network for EI-MS Prediction

    QC-GN2oMS2-inspired architecture adapted for EI-MS:
    - Input: 34-dim node features, 10-dim edge features
    - GNN: 14-layer GATv2Conv with residual connections
    - Pooling: Global mean pooling
    - Output: 1000-dim spectrum (m/z 1-1000)

    Key Differences from QC-GN2oMS2:
    - Task: MS/MS → EI-MS
    - Node features: 16-dim (QC-GN2oMS2) → 34-dim (v4.4)
    - Edge features: 2-dim (QC-GN2oMS2) → 10-dim (v4.4, adds BDE + stereo)
    - Output: Variable m/z → Fixed m/z 1-1000
    - Hidden dim: 128 (QC-GN2oMS2) → 768 (v5.1 Scaled)
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
        use_edge_attr: bool = True
    ):
        """
        Initialize QCGN2oEI_Minimal model (v5.1 Scaled)

        Args:
            node_dim: Node feature dimension (default: 34 for v4.4)
            edge_dim: Edge feature dimension (default: 10 for v4.4)
            hidden_dim: Hidden dimension (default: 768)
            num_layers: Number of GATv2 layers (default: 14)
            num_heads: Number of attention heads (default: 24)
            output_dim: Output spectrum dimension (default: 1000 for m/z 1-1000)
            dropout: Dropout rate (default: 0.1)
            use_edge_attr: Whether to use edge attributes (default: True)
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

        # Node encoder: node_dim → hidden_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )

        # Edge encoder: edge_dim → hidden_dim (if using edge attributes)
        if use_edge_attr:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout)
            )
        else:
            self.edge_encoder = None

        # GATv2Conv layers with residual connections
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
                    concat=True,  # Concatenate heads
                    residual=False  # We implement custom residual
                )
            )

            # Residual projection layer
            self.residual_layers.append(
                nn.Linear(hidden_dim, hidden_dim)
            )

        # Global pooling
        # Using global_mean_pool (matches QC-GN2oMS2)
        # Alternative: global_add_pool or AttentionPooling

        # Spectrum prediction head
        self.spectrum_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)  # Normalize to probability distribution
        )

        logger.info(f"QCGN2oEI_Minimal initialized:")
        logger.info(f"  Node dim: {node_dim}")
        logger.info(f"  Edge dim: {edge_dim}")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  Num layers: {num_layers}")
        logger.info(f"  Num heads: {num_heads}")
        logger.info(f"  Output dim: {output_dim} (m/z 1-{output_dim})")
        logger.info(f"  Dropout: {dropout}")
        logger.info(f"  Use edge attr: {use_edge_attr}")

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim] (optional)
            batch: Batch assignment [num_nodes] (optional, default: single graph)

        Returns:
            spectrum: Predicted spectrum [batch_size, output_dim]
        """
        # Encode node features
        x = self.node_encoder(x)  # [num_nodes, hidden_dim]

        # Encode edge features
        if self.use_edge_attr and edge_attr is not None:
            edge_attr_encoded = self.edge_encoder(edge_attr)  # [num_edges, hidden_dim]
        else:
            edge_attr_encoded = None

        # GATv2Conv layers with residual connections
        for i in range(self.num_layers):
            # Store input for residual
            residual = x

            # GATv2Conv layer
            x = self.gat_layers[i](
                x,
                edge_index,
                edge_attr=edge_attr_encoded
            )

            # Residual connection with projection
            # Following QC-GN2oMS2: output = ELU(GATv2(x) + Linear(residual))
            x = F.elu(x + self.residual_layers[i](residual))

            # Dropout
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global mean pooling
        if batch is None:
            # Single graph: manual mean pooling
            graph_repr = x.mean(dim=0, keepdim=True)  # [1, hidden_dim]
        else:
            # Batch of graphs: PyG global pooling
            graph_repr = global_mean_pool(x, batch)  # [batch_size, hidden_dim]

        # Spectrum prediction
        spectrum = self.spectrum_head(graph_repr)  # [batch_size, output_dim]

        return spectrum

    def get_model_info(self):
        """
        Get model information

        Returns:
            info: Dictionary with model configuration and statistics
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'QCGN2oEI_Minimal',
            'node_dim': self.node_dim,
            'edge_dim': self.edge_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'output_dim': self.output_dim,
            'dropout': self.dropout,
            'use_edge_attr': self.use_edge_attr,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'memory_estimate_mb': total_params * 4 / (1024 * 1024)  # FP32
        }


# Example usage and testing
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("="*60)
    print("QCGN2oEI_Minimal Model Test")
    print("="*60)

    # Create model with defaults (v5.1 Scaled)
    model = QCGN2oEI_Minimal()

    # Print model info
    info = model.get_model_info()
    print("\nModel Information:")
    for key, value in info.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Test forward pass with dummy data
    print("\n" + "="*60)
    print("Testing Forward Pass")
    print("="*60)

    # Create dummy graph (benzene-like)
    num_nodes = 6
    num_edges = 12  # Bidirectional

    x = torch.randn(num_nodes, info['node_dim'])  # Node features
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]
    ], dtype=torch.long)
    edge_attr = torch.randn(num_edges, info['edge_dim'])  # Edge features

    # Single graph
    print("\nSingle graph:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {num_edges}")

    with torch.no_grad():
        spectrum = model(x, edge_index, edge_attr)

    print(f"  Output spectrum shape: {spectrum.shape}")
    print(f"  Spectrum sum: {spectrum.sum().item():.4f} (should be ~1.0)")
    print(f"  Spectrum range: [{spectrum.min().item():.6f}, {spectrum.max().item():.6f}]")

    # Batch of graphs
    print("\nBatch of 3 graphs:")
    batch = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)

    with torch.no_grad():
        spectrum_batch = model(x, edge_index, edge_attr, batch)

    print(f"  Output spectrum shape: {spectrum_batch.shape}")
    print(f"  Spectrum sums: {spectrum_batch.sum(dim=-1)}")

    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
