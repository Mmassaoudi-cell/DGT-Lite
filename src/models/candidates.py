"""
The four hybrid candidate models designed to fix the weaknesses diagnosed in
SOURCE_WEAKNESS_ANALYSIS.md (imitation label-ceiling, decoupled temporal/spatial
processing, absence of a deadline-aware training signal, and the sequential-BiGRU +
dense-GCN inference cost). See MODEL_CANDIDATES.md for the full rationale for each.

  C1 DA-STGAT-RL   : joint spatio-temporal graph-attention encoder, deadline-urgency
                     attention bias, trained end-to-end with actor-critic RL directly
                     against the deadline/fairness/energy reward (removes the
                     imitation label ceiling).
  C2 DGT-Sched     : parallel TCN temporal encoder + sparse graph attention over the
                     true VM topology + a deadline-aware supervised loss (CE + a
                     differentiable slack-violation penalty), replacing the BiGRU
                     (sequential) and the plain MSE-on-softmax loss.
  C3 MoE-GAT-AC    : spatial-only graph attention with a 2-expert (CPU-bound /
                     IO-bound) mixture-of-experts head, trained with actor-critic RL;
                     tests whether temporal modeling is even necessary once the
                     objective and task-type specialization are fixed.
  C4 Distilled-DGT : compact single-GRUCell + linear student distilled from C2,
                     for the efficiency/Pareto analysis.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import AttentionPool, GATLayer, TCNBlock, normalize_adj
from .flexible import FEAT_DIM


# ---------------------------------------------------------------------------
# C1: DA-STGAT-RL
# ---------------------------------------------------------------------------
class DA_STGAT_RL(nn.Module):
    name = "DA-STGAT-RL"

    def __init__(self, feat_dim=FEAT_DIM, hidden=32):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True, bidirectional=False)
        self.attn_pool = AttentionPool(hidden)
        self.in_proj = nn.Linear(feat_dim + hidden, hidden)
        self.gat1 = GATLayer(hidden, hidden, n_heads=4)
        self.gat2 = GATLayer(hidden, hidden, n_heads=4)
        self.actor = nn.Linear(hidden, 1)
        self.critic = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, feat, hist, adj):
        h_seq, _ = self.gru(hist.unsqueeze(-1))
        ctx, _ = self.attn_pool(h_seq)
        x = torch.relu(self.in_proj(torch.cat([feat, ctx], -1)))
        # deadline-urgency attention bias: slack column (index 2 of feat) -> higher
        # urgency (lower slack) increases attention logits directed toward that node.
        slack = feat[:, 2]
        urgency = torch.sigmoid(-slack)  # (N,)
        bias = urgency.unsqueeze(0).expand(feat.shape[0], -1)  # broadcast as dst bias (N,N)
        x = torch.relu(self.gat1(x, adj, extra_bias=bias))
        x = torch.relu(self.gat2(x, adj, extra_bias=bias))
        logits = self.actor(x).squeeze(-1)
        value = self.critic(x.mean(0, keepdim=True)).squeeze()
        return logits, value


# ---------------------------------------------------------------------------
# C2: DGT-Sched
# ---------------------------------------------------------------------------
class DGTSched(nn.Module):
    name = "DGT-Sched"

    def __init__(self, feat_dim=FEAT_DIM, hidden=32):
        super().__init__()
        self.tcn = TCNBlock(in_ch=1, hidden=16, out_dim=hidden)
        self.in_proj = nn.Linear(feat_dim + hidden, hidden)
        self.gat1 = GATLayer(hidden, hidden, n_heads=4)
        self.gat2 = GATLayer(hidden, hidden, n_heads=4)
        self.out = nn.Linear(hidden, 1)

    def forward(self, feat, hist, adj):
        t = self.tcn(hist)
        x = torch.relu(self.in_proj(torch.cat([feat, t], -1)))
        x = torch.relu(self.gat1(x, adj))
        x = torch.relu(self.gat2(x, adj))
        return self.out(x).squeeze(-1)


def dgt_loss(logits, label, feat, lam=0.5):
    ce = F.cross_entropy(logits.unsqueeze(0), torch.tensor([label]))
    probs = torch.softmax(logits, dim=-1)
    slack = feat[:, 2]  # projected slack per VM if assigned there
    expected_violation = torch.sum(probs * torch.relu(-slack))
    return ce + lam * expected_violation


# ---------------------------------------------------------------------------
# C3: MoE-GAT-AC
# ---------------------------------------------------------------------------
class MoE_GAT_AC(nn.Module):
    name = "MoE-GAT-AC"

    def __init__(self, feat_dim=FEAT_DIM, hidden=32):
        super().__init__()
        self.in_proj = nn.Linear(feat_dim, hidden)
        self.gat1 = GATLayer(hidden, hidden, n_heads=4)
        self.gat2 = GATLayer(hidden, hidden, n_heads=4)
        self.expert_cpu = nn.Linear(hidden, 1)
        self.expert_io = nn.Linear(hidden, 1)
        self.critic = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, feat, hist, adj):
        x = torch.relu(self.in_proj(feat))
        x = torch.relu(self.gat1(x, adj))
        x = torch.relu(self.gat2(x, adj))
        task_type = feat[0, 8]  # broadcast task-type feature (identical across VM rows)
        gate = torch.sigmoid((task_type - 0.5) * 20.0)  # ~hard gate: 0 -> cpu expert, 1 -> io
        logits = (1 - gate) * self.expert_cpu(x).squeeze(-1) + gate * self.expert_io(x).squeeze(-1)
        value = self.critic(x.mean(0, keepdim=True)).squeeze()
        return logits, value


# ---------------------------------------------------------------------------
# C4: Distilled-DGT
# ---------------------------------------------------------------------------
class DistilledDGT(nn.Module):
    name = "Distilled-DGT"

    def __init__(self, feat_dim=FEAT_DIM, hidden=16):
        super().__init__()
        self.cell = nn.GRUCell(1, hidden)
        self.lin1 = nn.Linear(feat_dim + hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, feat, hist, adj=None):
        n, T = hist.shape
        h = torch.zeros(n, self.cell.hidden_size)
        for t in range(T):
            h = self.cell(hist[:, t : t + 1], h)
        x = torch.relu(self.lin1(torch.cat([feat, h], -1)))
        return self.out(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training routines
# ---------------------------------------------------------------------------
def train_actor_critic(model, scale, seeds, epochs=1, gamma=0.97, lr=1e-3,
                        entropy_coef=0.01, seed=0, log_every=None):
    from src.simulator.core import make_scenario

    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist_log = []
    for ep in range(epochs):
        for s in seeds:
            sim = make_scenario(scale, s)
            log_probs, values, rewards, entropies = [], [], [], []
            while not sim.done():
                task = sim.current_task()
                feat = torch.tensor(sim.all_vm_features(task), dtype=torch.float32)
                hist = torch.tensor(sim.history_tensor(), dtype=torch.float32)
                adj = torch.tensor(sim.adjacency, dtype=torch.float32)
                logits, value = model(feat, hist, adj)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                action = dist.sample()
                loads_before = np.array([vm.n_assigned for vm in sim.vms], dtype=np.float32)
                out = sim.step(int(action.item()))
                loads_after = np.array([vm.n_assigned for vm in sim.vms], dtype=np.float32)
                imbalance = float(np.std(loads_after) / (np.mean(loads_after) + 1e-6))
                r = (1.0 if out["met"] else -1.5) - 0.3 * imbalance
                log_probs.append(dist.log_prob(action))
                values.append(value)
                rewards.append(r)
                entropies.append(dist.entropy())
            returns = []
            G = 0.0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32)
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)
            values_t = torch.stack(values)
            log_probs_t = torch.stack(log_probs)
            entropies_t = torch.stack(entropies)
            advantage = returns - values_t.detach()
            policy_loss = -(log_probs_t * advantage).mean() - entropy_coef * entropies_t.mean()
            value_loss = F.mse_loss(values_t, returns)
            loss = policy_loss + 0.5 * value_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(f"    {model.name} epoch {ep}: mean_reward={np.mean(rewards):.3f}")
        hist_log.append(float(np.mean(rewards)))
    return hist_log


def train_dgt_supervised(model, samples, adj_by_ep, epochs=15, lr=1e-3, batch_size=32,
                          lam=0.5, weight_decay=1e-5, seed=0, log_every=None):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(samples))
    for ep in range(epochs):
        rng.shuffle(idx_all)
        opt.zero_grad()
        running, correct = 0.0, 0
        for step, i in enumerate(idx_all):
            s = samples[i]
            adj = adj_by_ep[s.ep_id]
            logits = model(s.feat, s.hist, adj)
            loss = dgt_loss(logits, s.label, s.feat, lam=lam)
            (loss / batch_size).backward()
            running += loss.item()
            correct += int(torch.argmax(logits).item() == s.label)
            if (step + 1) % batch_size == 0:
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(f"    DGT-Sched epoch {ep}: loss={running/len(samples):.4f} "
                  f"top1={correct/len(samples):.3f}")


def distill_train(student, teacher, samples, adj_by_ep, epochs=15, lr=1e-3, batch_size=32,
                   alpha=0.5, T=2.0, seed=0, log_every=None):
    teacher.eval()
    torch.manual_seed(seed)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(samples))
    for ep in range(epochs):
        rng.shuffle(idx_all)
        opt.zero_grad()
        running = 0.0
        for step, i in enumerate(idx_all):
            s = samples[i]
            adj = adj_by_ep[s.ep_id]
            with torch.no_grad():
                t_logits = teacher(s.feat, s.hist, adj)
            s_logits = student(s.feat, s.hist, adj)
            ce = F.cross_entropy(s_logits.unsqueeze(0), torch.tensor([s.label]))
            kd = F.kl_div(F.log_softmax(s_logits / T, dim=-1),
                          F.softmax(t_logits / T, dim=-1), reduction="batchmean") * (T * T)
            loss = alpha * ce + (1 - alpha) * kd
            (loss / batch_size).backward()
            running += loss.item()
            if (step + 1) % batch_size == 0:
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(f"    Distilled-DGT epoch {ep}: loss={running/len(samples):.4f}")


class ACScheduler:
    def __init__(self, model, name):
        self.model = model
        self.name = name
        self.model.eval()

    def select(self, sim, task):
        feat = torch.tensor(sim.all_vm_features(task), dtype=torch.float32)
        hist = torch.tensor(sim.history_tensor(), dtype=torch.float32)
        adj = torch.tensor(sim.adjacency, dtype=torch.float32)
        with torch.no_grad():
            logits, _ = self.model(feat, hist, adj)
        return int(torch.argmax(logits).item())


class LogitScheduler:
    """For plain-logit models (DGT-Sched, Distilled-DGT) that don't return a value head."""
    def __init__(self, model, name):
        self.model = model
        self.name = name
        self.model.eval()

    def select(self, sim, task):
        feat = torch.tensor(sim.all_vm_features(task), dtype=torch.float32)
        hist = torch.tensor(sim.history_tensor(), dtype=torch.float32)
        adj = torch.tensor(sim.adjacency, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(feat, hist, adj)
        return int(torch.argmax(logits).item())
