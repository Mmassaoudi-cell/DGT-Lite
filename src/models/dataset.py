"""Imitation-learning dataset collection (documented assumption #3, SOURCE_PAPER_AUDIT.md):
the source paper never defines ground-truth schedule labels, so we generate them with a
strong deadline-aware lookahead oracle (LookaheadOracle) and collect (state, label) pairs
under a lightweight DAgger-style mixed behavior policy (oracle w.p. 1-eps, random w.p. eps)
to reduce train/deploy distribution mismatch for the closed-loop-evaluated policies."""
from __future__ import annotations

import numpy as np
import torch

from src.simulator.core import make_scenario
from src.simulator.schedulers import LookaheadOracle


class Sample:
    __slots__ = ("feat", "hist", "label", "scale", "ep_id")

    def __init__(self, feat, hist, label, scale, ep_id):
        self.feat = feat
        self.hist = hist
        self.label = label
        self.scale = scale
        self.ep_id = ep_id


def collect_dataset(scale: str, seeds, eps: float = 0.3, power_mode="dvfs", seed_offset=0):
    samples = []
    adj_by_ep = {}
    oracle = LookaheadOracle()
    rng = np.random.default_rng(1234 + seed_offset)
    for seed in seeds:
        sim = make_scenario(scale, seed, power_mode)
        adj_by_ep[seed] = torch.tensor(sim.adjacency, dtype=torch.float32)
        while not sim.done():
            task = sim.current_task()
            feat = sim.all_vm_features(task)
            hist = sim.history_tensor()
            oracle_vm = oracle.select(sim, task)
            samples.append(Sample(
                feat=torch.tensor(feat, dtype=torch.float32),
                hist=torch.tensor(hist, dtype=torch.float32),
                label=oracle_vm, scale=scale, ep_id=seed,
            ))
            if rng.random() < eps:
                act = int(rng.integers(0, sim.n_vms))
            else:
                act = oracle_vm
            sim.step(act)
    return samples, adj_by_ep
