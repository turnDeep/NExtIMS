#!/usr/bin/env python3
# src/models/qcgn2oei_minimal.py
"""
NExtIMS v1.0: QCGN2oEI_Minimal Model
Quantum Chemistry-augmented Graph Neural Network for EI-MS prediction.
This version adds:
- Stable ACT Recurrent Spectral Refinement Block
- LTI-style stable recurrent update
- Loop embedding
- Adaptive Computation Time (ACT)
- Optional fixed-loop inference
- Optional intermediate spectra for auxiliary loss
Base architecture:
- GATv2Conv encoder
- Graph pooling with mean/add/max aggregation
- Optional graph-level molecular features
- Spectrum prediction head
"""
import logging
from typing import Optional, Tuple, Dict, Any, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool, global_max_pool, global_mean_pool
logger = logging.getLogger(__name__)


def neutral_loss_to_mz_spectrum(
    neutral_loss_spectrum: torch.Tensor,
    molecular_ion_mz: torch.Tensor,
) -> torch.Tensor:
    """
    Map neutral-loss bins back to absolute m/z bins.

    loss = molecular_ion_mz - fragment_mz, so
    fragment_mz = molecular_ion_mz - loss.
    """
    output_dim = neutral_loss_spectrum.size(-1)
    device = neutral_loss_spectrum.device
    loss_idx = torch.arange(0, output_dim, dtype=torch.long, device=device).view(1, -1)
    parent_mz = molecular_ion_mz.view(-1, 1).to(device=device, dtype=torch.long)
    mz_idx = parent_mz - loss_idx - 1
    valid = (mz_idx >= 0) & (mz_idx < output_dim)
    safe_idx = mz_idx.clamp(min=0, max=output_dim - 1)
    mz_spectrum = torch.zeros_like(neutral_loss_spectrum)
    mz_spectrum.scatter_add_(
        1,
        safe_idx,
        neutral_loss_spectrum * valid.to(neutral_loss_spectrum.dtype),
    )
    return mz_spectrum


def fuse_neutral_loss_prediction(
    mz_spectrum: torch.Tensor,
    neutral_loss_spectrum: Optional[torch.Tensor],
    molecular_ion_mz: Optional[torch.Tensor],
    fusion_weight: float,
) -> torch.Tensor:
    """Blend direct m/z prediction with neutral-loss-derived m/z prediction."""
    if neutral_loss_spectrum is None or molecular_ion_mz is None:
        return mz_spectrum
    fusion = min(max(float(fusion_weight), 0.0), 1.0)
    if fusion <= 0:
        return mz_spectrum
    neutral_mz_spectrum = neutral_loss_to_mz_spectrum(
        neutral_loss_spectrum,
        molecular_ion_mz,
    )
    return (1.0 - fusion) * mz_spectrum + fusion * neutral_mz_spectrum


class StableACTRecurrentSpectralRefinementBlock(nn.Module):
    """
    Stable ACT Recurrent Spectral Refinement Block.
    This block refines graph-level molecular representation after GNN pooling.
    Features:
        1. LTI-style stable update:
            z_{t+1} = alpha * z_t + beta * candidate + (1 - alpha - beta) * context
        2. Loop embedding:
            Adds step-specific embedding so each loop can learn a different role.
        3. ACT:
            Adaptive Computation Time for sample-wise early halting.
    """
    def __init__(
        self,
        hidden_dim: int,
        max_steps: int = 8,
        dropout: float = 0.1,
        act_epsilon: float = 0.01,
        use_lti_stabilization: bool = True,
        use_loop_embedding: bool = True,
        use_act: bool = True,
    ):
        super().__init__()
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        self.act_epsilon = act_epsilon
        self.use_lti_stabilization = use_lti_stabilization
        self.use_loop_embedding = use_loop_embedding
        self.use_act = use_act
        self.norm_z = nn.LayerNorm(hidden_dim)
        self.norm_context = nn.LayerNorm(hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)
        if use_loop_embedding:
            self.loop_embedding = nn.Embedding(max_steps, hidden_dim)
        else:
            self.loop_embedding = None
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.halt_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        # LTI-style coefficients.
        # Initialized so alpha is relatively high and beta is modest.
        self.raw_alpha = nn.Parameter(torch.tensor(1.0))
        self.raw_beta = nn.Parameter(torch.tensor(0.0))
    def _stable_mix_coefficients(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Constrain alpha and beta to a stable region.
        alpha = sigmoid(raw_alpha)
        beta = (1 - alpha) * sigmoid(raw_beta)
        Therefore:
            0 < alpha < 1
            0 < beta < 1 - alpha
            alpha + beta < 1
        """
        alpha = torch.sigmoid(self.raw_alpha)
        beta = (1.0 - alpha) * torch.sigmoid(self.raw_beta)
        return alpha, beta
    def single_step(
        self,
        z: torch.Tensor,
        context: torch.Tensor,
        step_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run one recurrent refinement step.
        Args:
            z:
                Current latent representation [batch_size, hidden_dim]
            context:
                Original graph representation [batch_size, hidden_dim]
            step_idx:
                Current loop index
        Returns:
            z_next:
                Refined latent representation [batch_size, hidden_dim]
            halt_prob:
                ACT halting probability [batch_size]
        """
        z_norm = self.norm_z(z)
        context_norm = self.norm_context(context)
        if self.loop_embedding is not None:
            safe_step_idx = min(step_idx, self.max_steps - 1)
            step_tensor = torch.full(
                size=(z.size(0),),
                fill_value=safe_step_idx,
                dtype=torch.long,
                device=z.device,
            )
            z_norm = z_norm + self.loop_embedding(step_tensor)
        inp = torch.cat([z_norm, context_norm], dim=-1)
        delta = self.update(inp)
        gate = self.gate(inp)
        candidate = z + gate * delta
        candidate = self.out_norm(candidate)
        if self.use_lti_stabilization:
            alpha, beta = self._stable_mix_coefficients()
            z_next = alpha * z + beta * candidate + (1.0 - alpha - beta) * context
        else:
            z_next = z + 0.5 * gate * delta
        z_next = self.out_norm(z_next)
        halt_prob = self.halt_head(z_next).squeeze(-1)
        return z_next, halt_prob
    def forward(
        self,
        z: torch.Tensor,
        context: torch.Tensor,
        steps: Optional[int] = None,
        return_act_stats: bool = False,
        force_no_act: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Run recurrent refinement with optional ACT.
        Args:
            z:
                Initial latent representation [batch_size, hidden_dim]
            context:
                Original graph representation [batch_size, hidden_dim]
            steps:
                Max number of recurrent steps. If None, use self.max_steps.
            return_act_stats:
                Whether to return ACT statistics.
            force_no_act:
                If True, use fixed-loop refinement even when self.use_act is True.
        Returns:
            refined_z
            or
            refined_z, act_stats
        """
        max_steps = self.max_steps if steps is None else min(int(steps), self.max_steps)
        max_steps = max(max_steps, 1)
        batch_size = z.size(0)
        device = z.device
        use_act_now = self.use_act and not force_no_act
        if not use_act_now:
            halt_probs_per_step: List[torch.Tensor] = []
            for step_idx in range(max_steps):
                z, halt_prob = self.single_step(z, context, step_idx)
                halt_probs_per_step.append(halt_prob)
            if return_act_stats:
                halt_probs = torch.stack(halt_probs_per_step, dim=1)
                ponder_steps = torch.full(
                    size=(batch_size,),
                    fill_value=float(max_steps),
                    dtype=torch.float32,
                    device=device,
                )
                act_stats = {
                    "halt_probs": halt_probs,
                    "ponder_steps": ponder_steps,
                    "act_loss": torch.tensor(0.0, device=device),
                    "mean_ponder_steps": ponder_steps.mean().detach(),
                    "max_ponder_steps": ponder_steps.max().detach(),
                }
                return z, act_stats
            return z
        threshold = 1.0 - self.act_epsilon
        halting_probability = torch.zeros(batch_size, device=device)
        remainders = torch.zeros(batch_size, device=device)
        n_updates = torch.zeros(batch_size, device=device)
        still_running = torch.ones(batch_size, dtype=torch.bool, device=device)
        weighted_z = torch.zeros_like(z)
        halt_probs_per_step: List[torch.Tensor] = []
        for step_idx in range(max_steps):
            z_next, halt_prob = self.single_step(z, context, step_idx)
            halt_probs_per_step.append(halt_prob)
            running_float = still_running.float()
            new_halted = (
                halting_probability + halt_prob * running_float > threshold
            ) & still_running
            still_running_next = (
                halting_probability + halt_prob * running_float <= threshold
            ) & still_running
            # Force all remaining samples to halt on the final step.
            if step_idx == max_steps - 1:
                new_halted = still_running
                still_running_next = torch.zeros_like(still_running)
            new_halted_float = new_halted.float()
            still_running_next_float = still_running_next.float()
            remainder = 1.0 - halting_probability
            remainders = remainders + new_halted_float * remainder
            update_weight = (
                new_halted_float * remainder
                + still_running_next_float * halt_prob
            )
            weighted_z = weighted_z + update_weight.unsqueeze(-1) * z_next
            halting_probability = (
                halting_probability
                + still_running_next_float * halt_prob
                + new_halted_float * remainder
            )
            n_updates = n_updates + running_float
            z = z_next
            still_running = still_running_next
            if not still_running.any():
                break
        refined_z = weighted_z
        ponder_steps = n_updates + remainders
        act_loss = ponder_steps.mean()
        if return_act_stats:
            halt_probs = torch.stack(halt_probs_per_step, dim=1)
            act_stats = {
                "halt_probs": halt_probs,
                "ponder_steps": ponder_steps,
                "act_loss": act_loss,
                "mean_ponder_steps": ponder_steps.mean().detach(),
                "max_ponder_steps": ponder_steps.max().detach(),
            }
            return refined_z, act_stats
        return refined_z
class QCGN2oEI_Minimal(nn.Module):
    """
    Minimal Graph Neural Network for EI-MS Prediction.
    QC-GN2oMS2-inspired architecture adapted for EI-MS.
    Input:
        34-dim node features
        10-dim edge features
    Core:
        GATv2Conv encoder with residual connections
    Optional post-encoder refinement:
        StableACTRecurrentSpectralRefinementBlock
    Output:
        Fixed m/z 1-1000 spectrum.
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
        use_recurrent_refinement: bool = False,
        refinement_steps: int = 4,
        refinement_max_steps: int = 8,
        refinement_dropout: float = 0.1,
        use_lti_stabilization: bool = True,
        use_loop_embedding: bool = True,
        use_act: bool = False,
        act_epsilon: float = 0.01,
        act_loss_weight: float = 0.01,
        pooling_mode: str = "mean_add_max",
        graph_feature_dim: int = 17,
        use_graph_features: bool = True,
        use_neutral_loss_head: bool = False,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim must be divisible by num_heads. "
                f"Got hidden_dim={hidden_dim}, num_heads={num_heads}"
            )
        if pooling_mode not in {"mean", "mean_add_max"}:
            raise ValueError(
                f"pooling_mode must be 'mean' or 'mean_add_max'. Got {pooling_mode}"
            )
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.dropout = dropout
        self.use_edge_attr = use_edge_attr
        self.use_recurrent_refinement = use_recurrent_refinement
        self.refinement_steps = refinement_steps
        self.refinement_max_steps = refinement_max_steps
        self.refinement_dropout = refinement_dropout
        self.use_lti_stabilization = use_lti_stabilization
        self.use_loop_embedding = use_loop_embedding
        self.use_act = use_act
        self.act_epsilon = act_epsilon
        self.act_loss_weight = act_loss_weight
        self.pooling_mode = pooling_mode
        self.graph_feature_dim = graph_feature_dim
        self.use_graph_features = use_graph_features and graph_feature_dim > 0
        self.use_neutral_loss_head = use_neutral_loss_head
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        if use_edge_attr:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout),
            )
        else:
            self.edge_encoder = None
        self.gat_layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    edge_dim=hidden_dim if use_edge_attr else None,
                    dropout=dropout,
                    concat=True,
                    residual=False,
                )
            )
            self.residual_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        if self.use_graph_features:
            self.graph_feature_encoder = nn.Sequential(
                nn.LayerNorm(graph_feature_dim),
                nn.Linear(graph_feature_dim, hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout),
            )
        else:
            self.graph_feature_encoder = None
        pooled_dim = hidden_dim if pooling_mode == "mean" else hidden_dim * 3
        if self.use_graph_features:
            pooled_dim += hidden_dim
        if pooled_dim == hidden_dim:
            self.pool_projection = nn.Identity()
        else:
            self.pool_projection = nn.Sequential(
                nn.Linear(pooled_dim, hidden_dim),
                nn.ELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            )
        if use_recurrent_refinement:
            self.refinement_block = StableACTRecurrentSpectralRefinementBlock(
                hidden_dim=hidden_dim,
                max_steps=refinement_max_steps,
                dropout=refinement_dropout,
                act_epsilon=act_epsilon,
                use_lti_stabilization=use_lti_stabilization,
                use_loop_embedding=use_loop_embedding,
                use_act=use_act,
            )
        else:
            self.refinement_block = None
        self.spectrum_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1),
        )
        if self.use_neutral_loss_head:
            self.neutral_loss_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.Softmax(dim=-1),
            )
        else:
            self.neutral_loss_head = None
        logger.info("QCGN2oEI_Minimal initialized:")
        logger.info(f"  Node dim: {node_dim}")
        logger.info(f"  Edge dim: {edge_dim}")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  Num layers: {num_layers}")
        logger.info(f"  Num heads: {num_heads}")
        logger.info(f"  Output dim: {output_dim} (m/z 1-{output_dim})")
        logger.info(f"  Dropout: {dropout}")
        logger.info(f"  Use edge attr: {use_edge_attr}")
        logger.info(f"  Use recurrent refinement: {use_recurrent_refinement}")
        logger.info(f"  Refinement steps: {refinement_steps}")
        logger.info(f"  Refinement max steps: {refinement_max_steps}")
        logger.info(f"  Use LTI stabilization: {use_lti_stabilization}")
        logger.info(f"  Use loop embedding: {use_loop_embedding}")
        logger.info(f"  Use ACT: {use_act}")
        logger.info(f"  Pooling mode: {pooling_mode}")
        logger.info(f"  Use graph features: {self.use_graph_features}")
        logger.info(f"  Graph feature dim: {graph_feature_dim}")
        logger.info(f"  Use neutral loss head: {self.use_neutral_loss_head}")
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
    def _pool_graph(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor],
        graph_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if batch is None:
            graph_mean = x.mean(dim=0, keepdim=True)
            if self.pooling_mode == "mean":
                pooled_parts = [graph_mean]
            else:
                pooled_parts = [
                    graph_mean,
                    x.sum(dim=0, keepdim=True),
                    x.max(dim=0, keepdim=True).values,
                ]
        else:
            graph_mean = global_mean_pool(x, batch)
            if self.pooling_mode == "mean":
                pooled_parts = [graph_mean]
            else:
                pooled_parts = [
                    graph_mean,
                    global_add_pool(x, batch),
                    global_max_pool(x, batch),
                ]
        batch_size = pooled_parts[0].size(0)
        if self.use_graph_features:
            if graph_features is None:
                graph_features = torch.zeros(
                    batch_size,
                    self.graph_feature_dim,
                    dtype=x.dtype,
                    device=x.device,
                )
            else:
                graph_features = graph_features.to(device=x.device, dtype=x.dtype)
                if graph_features.dim() == 1:
                    graph_features = graph_features.unsqueeze(0)
                if graph_features.size(0) != batch_size:
                    graph_features = graph_features.view(batch_size, -1)
            pooled_parts.append(self.graph_feature_encoder(graph_features))
        return self.pool_projection(torch.cat(pooled_parts, dim=-1))
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        graph_features: Optional[torch.Tensor] = None,
        refinement_steps: Optional[int] = None,
        return_intermediate: bool = False,
        return_act_stats: bool = False,
        return_aux_outputs: bool = False,
        force_no_act: bool = False,
    ):
        """
        Forward pass.
        Args:
            x:
                Node features [num_nodes, node_dim]
            edge_index:
                Edge indices [2, num_edges]
            edge_attr:
                Edge attributes [num_edges, edge_dim]
            batch:
                Batch assignment [num_nodes]
            graph_features:
                Graph-level molecular features [batch_size, graph_feature_dim].
            refinement_steps:
                Optional override of recurrent refinement loop count.
            return_intermediate:
                If True, return intermediate spectra for auxiliary loss.
                This mode uses fixed-loop refinement.
            return_act_stats:
                If True, return ACT stats.
            return_aux_outputs:
                If True, append optional auxiliary predictions.
            force_no_act:
                If True, use fixed-loop refinement even if ACT is enabled.
        Returns:
            spectrum
            or
            spectrum, intermediate_spectra
            or
            spectrum, act_stats
            or
            spectrum, intermediate_spectra, act_stats
        """
        x = self.node_encoder(x)
        if self.use_edge_attr and edge_attr is not None:
            edge_attr_encoded = self.edge_encoder(edge_attr)
        else:
            edge_attr_encoded = None
        for i in range(self.num_layers):
            residual = x
            x = self.gat_layers[i](
                x,
                edge_index,
                edge_attr=edge_attr_encoded,
            )
            x = F.elu(x + self.residual_layers[i](residual))
            x = self.layer_norms[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        graph_repr = self._pool_graph(x, batch, graph_features)
        act_stats = None
        intermediate_spectra: List[torch.Tensor] = []
        if self.use_recurrent_refinement and self.refinement_block is not None:
            z = graph_repr
            context = graph_repr
            steps = self.refinement_steps if refinement_steps is None else refinement_steps
            steps = min(int(steps), self.refinement_max_steps)
            steps = max(steps, 1)
            if return_intermediate:
                # Fixed-loop mode for intermediate supervision.
                # ACT is intentionally bypassed here to keep auxiliary loss simple.
                halt_probs_per_step: List[torch.Tensor] = []
                for step_idx in range(steps):
                    z, halt_prob = self.refinement_block.single_step(
                        z=z,
                        context=context,
                        step_idx=step_idx,
                    )
                    halt_probs_per_step.append(halt_prob)
                    intermediate_spectra.append(self.spectrum_head(z))
                graph_repr = z
                if return_act_stats:
                    halt_probs = torch.stack(halt_probs_per_step, dim=1)
                    ponder_steps = torch.full(
                        size=(z.size(0),),
                        fill_value=float(steps),
                        dtype=torch.float32,
                        device=z.device,
                    )
                    act_stats = {
                        "halt_probs": halt_probs,
                        "ponder_steps": ponder_steps,
                        "act_loss": torch.tensor(0.0, device=z.device),
                        "mean_ponder_steps": ponder_steps.mean().detach(),
                        "max_ponder_steps": ponder_steps.max().detach(),
                    }
            else:
                if return_act_stats:
                    graph_repr, act_stats = self.refinement_block(
                        z=z,
                        context=context,
                        steps=steps,
                        return_act_stats=True,
                        force_no_act=force_no_act,
                    )
                else:
                    graph_repr = self.refinement_block(
                        z=z,
                        context=context,
                        steps=steps,
                        return_act_stats=False,
                        force_no_act=force_no_act,
                    )
        spectrum = self.spectrum_head(graph_repr)
        aux_outputs: Dict[str, torch.Tensor] = {}
        if return_aux_outputs and self.neutral_loss_head is not None:
            aux_outputs["neutral_loss_spectrum"] = self.neutral_loss_head(graph_repr)
        if return_intermediate and return_act_stats:
            if return_aux_outputs:
                return spectrum, intermediate_spectra, act_stats, aux_outputs
            return spectrum, intermediate_spectra, act_stats
        if return_intermediate:
            if return_aux_outputs:
                return spectrum, intermediate_spectra, aux_outputs
            return spectrum, intermediate_spectra
        if return_act_stats:
            if return_aux_outputs:
                return spectrum, act_stats, aux_outputs
            return spectrum, act_stats
        if return_aux_outputs:
            return spectrum, aux_outputs
        return spectrum
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_name": "QCGN2oEI_Minimal",
            "node_dim": self.node_dim,
            "edge_dim": self.edge_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "output_dim": self.output_dim,
            "dropout": self.dropout,
            "use_edge_attr": self.use_edge_attr,
            "use_recurrent_refinement": self.use_recurrent_refinement,
            "refinement_steps": self.refinement_steps,
            "refinement_max_steps": self.refinement_max_steps,
            "refinement_dropout": self.refinement_dropout,
            "use_lti_stabilization": self.use_lti_stabilization,
            "use_loop_embedding": self.use_loop_embedding,
            "use_act": self.use_act,
            "act_epsilon": self.act_epsilon,
            "act_loss_weight": self.act_loss_weight,
            "pooling_mode": self.pooling_mode,
            "graph_feature_dim": self.graph_feature_dim,
            "use_graph_features": self.use_graph_features,
            "use_neutral_loss_head": self.use_neutral_loss_head,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "memory_estimate_mb": total_params * 4 / (1024 * 1024),
        }
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("QCGN2oEI_Minimal Model Test")
    print("=" * 60)
    model = QCGN2oEI_Minimal(
        use_recurrent_refinement=True,
        refinement_steps=4,
        refinement_max_steps=8,
        use_lti_stabilization=True,
        use_loop_embedding=True,
        use_act=True,
    )
    info = model.get_model_info()
    print("\nModel Information:")
    for key, value in info.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    num_nodes = 6
    num_edges = 12
    x = torch.randn(num_nodes, info["node_dim"])
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5],
        ],
        dtype=torch.long,
    )
    edge_attr = torch.randn(num_edges, info["edge_dim"])
    graph_features = torch.randn(1, info["graph_feature_dim"])
    print("\nSingle graph:")
    with torch.no_grad():
        spectrum, act_stats = model(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            graph_features=graph_features,
            return_act_stats=True,
        )
    print(f"  Output spectrum shape: {spectrum.shape}")
    print(f"  Spectrum sum: {spectrum.sum().item():.4f}")
    print(f"  Spectrum range: [{spectrum.min().item():.6f}, {spectrum.max().item():.6f}]")
    print(f"  Mean ponder steps: {act_stats['mean_ponder_steps'].item():.4f}")
    print("\nBatch of 3 graphs:")
    batch = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    graph_features_batch = torch.randn(3, info["graph_feature_dim"])
    with torch.no_grad():
        spectrum_batch, act_stats_batch = model(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            graph_features=graph_features_batch,
            return_act_stats=True,
        )
    print(f"  Output spectrum shape: {spectrum_batch.shape}")
    print(f"  Spectrum sums: {spectrum_batch.sum(dim=-1)}")
    print(f"  Mean ponder steps: {act_stats_batch['mean_ponder_steps'].item():.4f}")
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
