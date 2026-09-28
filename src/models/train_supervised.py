from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def train_model(model, samples, adj_by_ep, epochs=15, lr=1e-3, batch_size=32,
                 loss_type="ce", weight_decay=1e-5, seed=0, log_every=None):
    """Generic imitation-learning trainer. Each forward pass is already batched over the
    n_vms nodes of one decision instance (variable graph size across episodes prevents
    naive multi-sample batching), so we honor the paper's batch_size=32 (Table II) via
    gradient accumulation over 32 samples before each optimizer step."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(samples))
    history = []
    for ep in range(epochs):
        rng.shuffle(idx_all)
        opt.zero_grad()
        running_loss, correct, n_seen = 0.0, 0, 0
        for step, i in enumerate(idx_all):
            s = samples[i]
            adj = adj_by_ep[s.ep_id]
            logits = model(s.feat, s.hist, adj)
            if loss_type == "mse_softmax":
                probs = torch.softmax(logits, dim=-1)
                target = F.one_hot(torch.tensor(s.label), num_classes=probs.shape[0]).float()
                loss = torch.mean((probs - target) ** 2)
            else:
                loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([s.label]))
            (loss / batch_size).backward()
            running_loss += loss.item()
            correct += int(torch.argmax(logits).item() == s.label)
            n_seen += 1
            if (step + 1) % batch_size == 0:
                opt.step()
                opt.zero_grad()
        opt.step()
        opt.zero_grad()
        avg_loss = running_loss / max(1, n_seen)
        acc = correct / max(1, n_seen)
        history.append((ep, avg_loss, acc))
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(f"    epoch {ep:3d}  loss={avg_loss:.4f}  train_top1={acc:.3f}")
    return history


class NeuralScheduler:
    """Wraps a trained FlexibleScheduler/candidate module for closed-loop evaluation."""
    def __init__(self, model, name):
        self.model = model
        self.name = name
        self.model.eval()
        self._adj_cache = None
        self._adj_id = None

    def select(self, sim, task):
        feat = torch.tensor(sim.all_vm_features(task), dtype=torch.float32)
        hist = torch.tensor(sim.history_tensor(), dtype=torch.float32)
        if self._adj_id is not id(sim):
            self._adj_cache = torch.tensor(sim.adjacency, dtype=torch.float32)
            self._adj_id = id(sim)
        with torch.no_grad():
            logits = self.model(feat, hist, self._adj_cache)
        return int(torch.argmax(logits).item())
