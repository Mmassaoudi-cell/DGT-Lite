"""
FlexibleScheduler: one configurable module that instantiates the DCLD-net
reproduction, every architectural ablation of it, and the simple deep baselines
(MLP / LSTM / GCN-only / Transformer-over-VMs), by toggling which of the
temporal / attention / graph branches are active. This directly implements the
component ablation the source paper never ran (SOURCE_PAPER_AUDIT.md, section 11).

    use_temporal in {'none', 'gru', 'bigru'}
    use_attention: bool          (attention pooling over the temporal axis, Eqs. 8-10)
    use_graph     in {'none', 'gcn', 'gat_complete'}   ('gat_complete' = Transformer-over-VMs)

DCLD-net reproduction = FlexibleScheduler(temporal='bigru', attention=True, graph='gcn')
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import AttentionPool, GCNLayer, GATLayer, normalize_adj

FEAT_DIM = 10  # see EdgeSimulator.vm_snapshot_features


class FlexibleScheduler(nn.Module):
    def __init__(self, use_temporal="bigru", use_attention=True, use_graph="gcn",
                 hidden=32, dropout=0.2, feat_dim=FEAT_DIM):
        super().__init__()
        self.use_temporal = use_temporal
        self.use_attention = use_attention
        self.use_graph = use_graph
        self.hidden = hidden

        if use_temporal == "none":
            self.temporal = None
            temporal_out = 0
        else:
            bidir = use_temporal == "bigru"
            self.temporal = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True,
                                    bidirectional=bidir)
            temporal_out = hidden * (2 if bidir else 1)
            if use_attention:
                self.attn_pool = AttentionPool(temporal_out)

        node_in_dim = feat_dim + temporal_out
        self.input_proj = nn.Linear(node_in_dim, hidden)
        self.dropout = nn.Dropout(dropout)

        if use_graph == "gcn":
            self.g1 = GCNLayer(hidden, hidden)
            self.g2 = GCNLayer(hidden, hidden)
        elif use_graph == "gat_complete":
            self.g1 = GATLayer(hidden, hidden, n_heads=4)
            self.g2 = GATLayer(hidden, hidden, n_heads=4)
        else:
            self.g1 = self.g2 = None
            self.mlp2 = nn.Linear(hidden, hidden)

        self.out = nn.Linear(hidden, 1)

    def forward(self, feat: torch.Tensor, hist: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """feat: (N, F), hist: (N, T), adj: (N, N) -> logits (N,)"""
        n = feat.shape[0]
        parts = [feat]
        if self.temporal is not None:
            h_seq, _ = self.temporal(hist.unsqueeze(-1))          # (N, T, H*)
            if self.use_attention:
                ctx, _ = self.attn_pool(h_seq)
            else:
                ctx = h_seq[:, -1, :]
            parts.append(ctx)
        node_in = torch.cat(parts, dim=-1)
        x = torch.relu(self.input_proj(node_in))
        x = self.dropout(x)

        if self.use_graph == "gcn":
            a_norm = normalize_adj(adj)
            x = torch.relu(self.g1(x, a_norm))
            x = self.dropout(x)
            x = torch.relu(self.g2(x, a_norm))
        elif self.use_graph == "gat_complete":
            complete = torch.ones(n, n, device=feat.device)
            x = torch.relu(self.g1(x, complete))
            x = self.dropout(x)
            x = torch.relu(self.g2(x, complete))
        else:
            x = torch.relu(self.mlp2(x))

        logits = self.out(x).squeeze(-1)
        return logits


def build(name: str, feat_dim=FEAT_DIM, hidden=32, dropout=0.2) -> FlexibleScheduler:
    presets = {
        "MLP": dict(use_temporal="none", use_attention=False, use_graph="none"),
        "LSTM": dict(use_temporal="gru", use_attention=False, use_graph="none"),
        "Transformer": dict(use_temporal="none", use_attention=False, use_graph="gat_complete"),
        "GCN-only": dict(use_temporal="none", use_attention=False, use_graph="gcn"),
        "DCLD-net-repro": dict(use_temporal="bigru", use_attention=True, use_graph="gcn"),
        "DCLD-net-CE": dict(use_temporal="bigru", use_attention=True, use_graph="gcn"),
        "DCLD-net-noattn": dict(use_temporal="bigru", use_attention=False, use_graph="gcn"),
        "DCLD-net-nogcn": dict(use_temporal="bigru", use_attention=True, use_graph="none"),
        "DCLD-net-notemporal": dict(use_temporal="none", use_attention=False, use_graph="gcn"),
    }
    cfg = presets[name]
    return FlexibleScheduler(hidden=hidden, dropout=dropout, feat_dim=feat_dim, **cfg)
