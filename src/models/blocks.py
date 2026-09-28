"""Shared low-level building blocks used by the DCLD-net reproduction, its
architectural ablations, the simple deep baselines (MLP/LSTM/Transformer/GCN-only),
and the new hybrid candidate models (C1-C4)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    """Eqs. (8)-(10) of the source paper: e_t = tanh(W_a h_t + b_a), softmax, weighted sum."""

    def __init__(self, dim):
        super().__init__()
        self.w = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, 1, bias=False)

    def forward(self, h):  # h: (N, T, D)
        e = torch.tanh(self.w(h))
        scores = self.v(e).squeeze(-1)          # (N, T)
        alpha = torch.softmax(scores, dim=-1)    # (N, T)
        ctx = torch.einsum("nt,ntd->nd", alpha, h)
        return ctx, alpha


class GCNLayer(nn.Module):
    """Kipf-Welling propagation: H' = sigma(D'^-1/2 A' D'^-1/2 H W), Eq. (5)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj_norm):  # x: (N, Fin), adj_norm: (N, N)
        return self.lin(adj_norm @ x)


def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    n = adj.shape[0]
    a_hat = adj + torch.eye(n, device=adj.device, dtype=adj.dtype)
    deg = a_hat.sum(-1)
    d_inv_sqrt = torch.pow(deg.clamp(min=1e-6), -0.5)
    d_mat = torch.diag(d_inv_sqrt)
    return d_mat @ a_hat @ d_mat


class GATLayer(nn.Module):
    """Compact single-head-then-concat graph attention (Velickovic-style additive attention),
    used both as the "Transformer over VMs" baseline (complete-graph adjacency) and inside the
    new hybrid candidates (sparse task/host-topology adjacency)."""

    def __init__(self, in_dim, out_dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.W = nn.Linear(in_dim, out_dim)
        self.a_src = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.1)
        self.out_dim = out_dim

    def forward(self, x, adj_mask, extra_bias=None):
        # x: (N, Fin), adj_mask: (N, N) in {0,1}; extra_bias: optional (N, N) additive logit bias
        n = x.shape[0]
        h = self.W(x).view(n, self.n_heads, self.head_dim)               # (N, H, d)
        src_score = torch.einsum("nhd,hd->nh", h, self.a_src)             # (N, H)
        dst_score = torch.einsum("nhd,hd->nh", h, self.a_dst)             # (N, H)
        logits = src_score.unsqueeze(1) + dst_score.unsqueeze(0)          # (N, N, H)
        logits = F.leaky_relu(logits, 0.2).permute(2, 0, 1)               # (H, N, N)
        mask = (adj_mask > 0).unsqueeze(0)
        if extra_bias is not None:
            logits = logits + extra_bias.unsqueeze(0)
        logits = logits.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.einsum("hnm,mhd->nhd", attn, h)                       # (N, H, d)
        return out.reshape(n, self.out_dim)


class TCNBlock(nn.Module):
    """Dilated causal 1-D temporal conv, parallel alternative to the sequential BiGRU
    branch (efficiency motivation, see SOURCE_WEAKNESS_ANALYSIS.md)."""

    def __init__(self, in_ch=1, hidden=16, out_dim=16):
        super().__init__()
        self.c1 = nn.Conv1d(in_ch, hidden, kernel_size=3, padding=2, dilation=2)
        self.c2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, dilation=1)
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, hist):  # hist: (N, T)
        x = hist.unsqueeze(1)              # (N, 1, T)
        x = F.relu(self.c1(x))[:, :, : hist.shape[1]]
        x = F.relu(self.c2(x))
        pooled = x.mean(-1)                # (N, hidden)
        return self.proj(pooled)
