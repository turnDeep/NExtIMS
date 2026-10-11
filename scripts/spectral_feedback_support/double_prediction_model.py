"""Portable EXP-V013(3) predictor used by EXP-V014 double_control."""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import scatter, softmax

from .model import QCGN2oEI_Minimal, fuse_neutral_loss_prediction


class DoublePredictionModel(QCGN2oEI_Minimal):
    def __init__(self, **config):
        super().__init__(**config)
        h = self.hidden_dim
        # Retain the screen's allocation order and state keys for exact weight
        # transfer. These unused control modules do not receive gradients.
        self.neutral_residual_head = nn.Sequential(nn.Linear(h, h // 2), nn.ELU(),
                                                   nn.Dropout(self.dropout), nn.Linear(h // 2, self.output_dim))
        nn.init.zeros_(self.neutral_residual_head[-1].weight)
        nn.init.zeros_(self.neutral_residual_head[-1].bias)
        self.charge_site_head = nn.Linear(h, 1)
        self.fragment_encoder = nn.Sequential(nn.Linear(3 * h + 24, h), nn.ELU(), nn.Linear(h, h), nn.ELU())
        self.event_head = nn.Sequential(nn.Linear(2 * h, h), nn.ELU(), nn.Linear(h, 1))
        self.charge_side_head = nn.Linear(h, 1)
        self.survival_head = nn.Linear(h, 1)
        self.transition_head = nn.Sequential(nn.Linear(2 * h, h), nn.ELU(), nn.Linear(h, 1))
        self.chemical_gain_logit = nn.Parameter(torch.zeros(()))
        self.use_act = False
        self.refinement_block.use_act = False
        self.spectral_feedback_enabled = False

    def encode_nodes(self, batch):
        x, edge = self.node_encoder(batch.x), self.edge_encoder(batch.edge_attr)
        for i in range(self.num_layers):
            x = F.elu(self.gat_layers[i](x, batch.edge_index, edge_attr=edge) + self.residual_layers[i](x))
            x = F.dropout(self.layer_norms[i](x), p=self.dropout, training=self.training)
        return x, self._pool_graph(x, batch.batch, batch.graph_features)

    def chemical_spectrum(self, batch, nodes, context):
        count = batch.frag_mz.numel()
        f, a = batch.frag_members
        mean = scatter(nodes[a], f, dim=0, dim_size=count, reduce="mean")
        attention = softmax(self.charge_site_head(nodes[a]).squeeze(-1), f, num_nodes=count)
        charge_context = scatter(nodes[a] * attention[:, None], f, dim=0, dim_size=count, reduce="sum")
        features = self.fragment_encoder(torch.cat([mean, charge_context, context[batch.frag_graph],
                                                     batch.frag_basic, batch.frag_charged], -1))
        first, second = batch.event_index
        graph = batch.frag_graph[first]
        event_logits = self.event_head(torch.cat([(features[first] + features[second]) / 2,
                                                  (features[first] - features[second]).abs()], -1)).squeeze(-1)
        event_prob = softmax(event_logits, graph, num_nodes=batch.num_graphs)
        side_logits = torch.stack([self.charge_side_head(features[first]).squeeze(-1),
                                   self.charge_side_head(features[second]).squeeze(-1)], -1)
        side_logits[:, 1] = side_logits[:, 1].masked_fill(~batch.event_paired, -1e4)
        sides = torch.softmax(side_logits, -1)
        state = scatter(torch.cat([event_prob * sides[:, 0], event_prob * sides[:, 1]]),
                        torch.cat([first, second]), dim=0, dim_size=count, reduce="sum")
        mass = scatter(state, batch.frag_graph, dim=0, dim_size=batch.num_graphs, reduce="sum")
        bins = batch.frag_graph * self.output_dim + batch.frag_mz - 1
        spectrum = scatter(state, bins, dim=0, dim_size=batch.num_graphs * self.output_dim,
                           reduce="sum").reshape(batch.num_graphs, self.output_dim)
        return spectrum, (mass - 1).abs().max(), state

    def predict(self, batch):
        nodes, context = self.encode_nodes(batch)
        direct, neutral = self.spectrum_head(context), self.neutral_loss_head(context)
        weight = self.spectral_feedback_fusion_weight
        anchors = [fuse_neutral_loss_prediction(direct, neutral, batch.molecular_ion_mz, weight)]
        z = context
        for step in range(self.refinement_steps):
            z, _ = self.refinement_block.single_step(z, context, step)
            residual = .25 * torch.tanh(self.spectrum_residual_head(z))
            direct = torch.softmax(torch.log(direct.clamp_min(1e-8)) + residual, -1)
            neutral = self.neutral_loss_head(z)
            anchors.append(fuse_neutral_loss_prediction(direct, neutral, batch.molecular_ion_mz, weight))
        chemical, flow, state = self.chemical_spectrum(batch, nodes, context)
        gain = torch.sigmoid(self.chemical_gain_logit)
        evidence = torch.tanh(10 * (chemical - chemical.mean(-1, keepdim=True)))
        exits = [torch.where(batch.ion_eligible[:, None].bool(),
                  torch.softmax(torch.log(a.clamp_min(1e-8)) + gain * evidence, -1), a) for a in anchors]
        return dict(pred=exits[-1], anchor=anchors[-1], neutral=neutral, chemical=chemical,
                    ion_weights=state, flow_error=flow, chemical_gain=gain,
                    exits=torch.stack(exits, 1), anchors=torch.stack(anchors, 1),
                    computed_steps=len(anchors) - 1)

    def forward(self, batch):
        return self.predict(batch)["pred"]
