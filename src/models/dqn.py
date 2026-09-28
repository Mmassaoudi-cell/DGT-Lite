"""DQN baseline: a per-VM Q-value head (shared MLP over instantaneous features, same
variable-action-space trick used by every other scheduler here) trained online via
1-step Q-learning directly against the simulator (no oracle labels -- a genuine RL
baseline, unlike the imitation-learned MLP/LSTM/Transformer/GCN-only/DCLD-net models)."""
from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .flexible import FEAT_DIM


class QNet(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat):  # (N, F) -> (N,)
        return self.net(feat).squeeze(-1)


def reward_fn(step_out: dict, task, imbalance_after: float) -> float:
    r = 1.0 if step_out["met"] else -1.5
    r -= 0.3 * imbalance_after
    return r


def train_dqn(scale, seeds, epochs=1, gamma=0.95, lr=1e-3, eps_start=0.4, eps_end=0.05,
              buffer_size=4000, batch_size=32, target_sync=200, seed=0, log_every=None):
    from src.simulator.core import make_scenario

    torch.manual_seed(seed)
    qnet = QNet()
    target = QNet()
    target.load_state_dict(qnet.state_dict())
    opt = torch.optim.Adam(qnet.parameters(), lr=lr)
    buf = deque(maxlen=buffer_size)
    rng = np.random.default_rng(seed)
    step_count = 0
    history = []
    for ep in range(epochs):
        eps = eps_start + (eps_end - eps_start) * (ep / max(1, epochs - 1))
        for s in seeds:
            sim = make_scenario(scale, s)
            prev_feat = None
            while not sim.done():
                task = sim.current_task()
                feat = torch.tensor(sim.all_vm_features(task), dtype=torch.float32)
                with torch.no_grad():
                    q = qnet(feat)
                if rng.random() < eps:
                    a = int(rng.integers(0, sim.n_vms))
                else:
                    a = int(torch.argmax(q).item())
                loads_before = np.array([vm.n_assigned for vm in sim.vms], dtype=np.float32)
                out = sim.step(a)
                loads_after = np.array([vm.n_assigned for vm in sim.vms], dtype=np.float32)
                imbalance = float(np.std(loads_after) / (np.mean(loads_after) + 1e-6))
                r = reward_fn(out, task, imbalance)
                next_feat = (torch.tensor(sim.all_vm_features(sim.current_task()), dtype=torch.float32)
                             if not sim.done() else None)
                buf.append((feat, a, r, next_feat))
                step_count += 1

                if len(buf) >= batch_size:
                    batch = random.sample(buf, batch_size)
                    loss = 0.0
                    opt.zero_grad()
                    for bf, ba, br, bnf in batch:
                        q_sa = qnet(bf)[ba]
                        if bnf is not None:
                            with torch.no_grad():
                                q_next = target(bnf).max()
                            tgt = br + gamma * q_next
                        else:
                            tgt = torch.tensor(br)
                        loss = loss + (q_sa - tgt) ** 2
                    loss = loss / batch_size
                    loss.backward()
                    opt.step()
                if step_count % target_sync == 0:
                    target.load_state_dict(qnet.state_dict())
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(f"    DQN epoch {ep}: buffer={len(buf)} eps={eps:.2f}")
        history.append(ep)
    return qnet


class DQNScheduler:
    name = "DQN"

    def __init__(self, qnet):
        self.qnet = qnet
        self.qnet.eval()

    def select(self, sim, task):
        feat = torch.tensor(sim.all_vm_features(task), dtype=torch.float32)
        with torch.no_grad():
            q = self.qnet(feat)
        return int(torch.argmax(q).item())
