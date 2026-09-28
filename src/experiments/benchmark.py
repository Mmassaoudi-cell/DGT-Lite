"""
Unified benchmark runner. For each (method, scale, seed_index i):
  - learned methods are retrained with torch seed = i (dataset reused across i to save
    compute; RL methods reuse the same training-scenario seeds across i as well, only
    network init/sampling stochasticity varies, which is standard multi-seed DL practice)
  - the resulting policy (or the non-learned heuristic) is evaluated closed-loop on
    TEST_SEEDS[i] (or VAL_SEEDS[i] during screening)
  - one row is appended per (method, scale, i) to the output CSV

This pairing (train_seed_i <-> eval_seed_i) gives n paired samples per method per scale,
enabling the paired significance tests in stats.py.
"""
from __future__ import annotations

import argparse
import time
import json
import numpy as np
import pandas as pd
import torch

from src.simulator.core import make_scenario, SPLIT_SEEDS
from src.simulator.schedulers import HEURISTIC_REGISTRY, RoundRobin
from src.models.dataset import collect_dataset
from src.models import flexible
from src.models.train_supervised import train_model, NeuralScheduler
from src.models.dqn import train_dqn, DQNScheduler
from src.models import candidates as C

TRAIN_SEEDS = {"small": list(range(0, 40)), "medium": list(range(0, 10)),
               "large": list(range(0, 5)), "table2": list(range(0, 10))}
SUPERVISED_EPOCHS = {"small": 14, "medium": 6, "large": 3, "table2": 8}
# RL training budget equalized against the supervised methods' scenario x epoch coverage
# (revision: previously 3/2 epochs over only 10 scenarios, an unequal-budget confound
# identified in review; now scaled to comparable total episode-updates).
AC_EPOCHS = {"small": 6, "medium": 4, "large": 1, "table2": 2}
AC_TRAIN_SCENARIOS = {"small": 15, "medium": 10, "large": 5, "table2": 10}


def run_episode(scale, seed, sched, power_mode="dvfs"):
    sim = make_scenario(scale, seed, power_mode)
    t0 = time.time()
    n_decisions = 0
    while not sim.done():
        task = sim.current_task()
        v = sched.select(sim, task)
        sim.step(v)
        n_decisions += 1
    dt = time.time() - t0
    summ = sim.summary()
    summ["latency_ms_per_decision"] = 1000.0 * dt / max(1, n_decisions)
    return summ


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def build_heuristic(name, sim_n_vms, seed):
    if name == "RR":
        return RoundRobin(sim_n_vms)
    cls = HEURISTIC_REGISTRY[name]
    if name in ("HIWIGOA-LB-approx", "GA", "PSO"):
        return cls(seed=seed)
    return cls()


FLEX_NAMES = ["MLP", "LSTM", "Transformer", "GCN-only", "DCLD-net-repro", "DCLD-net-CE",
              "DCLD-net-noattn", "DCLD-net-nogcn", "DCLD-net-notemporal"]
CE_LOSS_OVERRIDE = {"DCLD-net-CE"}  # same architecture as DCLD-net-repro, CE loss instead of MSE
HEUR_NAMES = ["RR", "Greedy-LL", "Min-Min", "Max-Min", "MrLBA-approx",
              "HIWIGOA-LB-approx", "GA", "PSO"]


def run_all(scale, methods, split="test", n_seeds=8, power_mode="dvfs",
            cache=None, verbose=True):
    rows = []
    eval_seeds = SPLIT_SEEDS[split][:n_seeds]
    cache = cache if cache is not None else {}

    need_dataset = any(m in FLEX_NAMES or m in ("DGT-Sched", "Distilled-DGT")
                        for m in methods)
    if need_dataset and "dataset" not in cache:
        if verbose:
            print(f"  [dataset] collecting supervised dataset for scale={scale} ...")
        cache["dataset"] = collect_dataset(scale, TRAIN_SEEDS[scale], eps=0.3)
    samples, adj_by_ep = cache.get("dataset", (None, None))

    for method in methods:
        for i in range(n_seeds):
            eval_seed = eval_seeds[i]
            t_train0 = time.time()
            latency_probe = None
            n_par = None
            if method in HEUR_NAMES:
                sim0 = make_scenario(scale, eval_seed, power_mode)
                sched = build_heuristic(method, sim0.n_vms, seed=i)
            elif method in FLEX_NAMES:
                model = flexible.build(method)
                if method in CE_LOSS_OVERRIDE:
                    loss_type = "ce"
                else:
                    loss_type = "mse_softmax" if method.startswith("DCLD-net") else "ce"
                train_model(model, samples, adj_by_ep, epochs=SUPERVISED_EPOCHS[scale],
                            loss_type=loss_type, seed=i)
                n_par = n_params(model)
                sched = NeuralScheduler(model, method)
            elif method == "DQN":
                qnet = train_dqn(scale, TRAIN_SEEDS[scale][: min(15, len(TRAIN_SEEDS[scale]))],
                                  epochs=1, seed=i)
                n_par = n_params(qnet)
                sched = DQNScheduler(qnet)
            elif method == "DA-STGAT-RL":
                model = C.DA_STGAT_RL()
                C.train_actor_critic(model, scale, TRAIN_SEEDS[scale][:AC_TRAIN_SCENARIOS[scale]],
                                      epochs=AC_EPOCHS[scale], seed=i)
                n_par = n_params(model)
                sched = C.ACScheduler(model, method)
            elif method == "MoE-GAT-AC":
                model = C.MoE_GAT_AC()
                C.train_actor_critic(model, scale, TRAIN_SEEDS[scale][:AC_TRAIN_SCENARIOS[scale]],
                                      epochs=AC_EPOCHS[scale], seed=i)
                n_par = n_params(model)
                sched = C.ACScheduler(model, method)
            elif method == "DGT-Sched":
                model = C.DGTSched()
                C.train_dgt_supervised(model, samples, adj_by_ep,
                                        epochs=SUPERVISED_EPOCHS[scale], seed=i)
                n_par = n_params(model)
                sched = C.LogitScheduler(model, method)
                cache[("teacher", scale, i)] = model
            elif method == "Distilled-DGT":
                teacher = cache.get(("teacher", scale, i))
                if teacher is None:
                    teacher = C.DGTSched()
                    C.train_dgt_supervised(teacher, samples, adj_by_ep,
                                            epochs=SUPERVISED_EPOCHS[scale], seed=i)
                    cache[("teacher", scale, i)] = teacher
                model = C.DistilledDGT()
                C.distill_train(model, teacher, samples, adj_by_ep,
                                 epochs=SUPERVISED_EPOCHS[scale], seed=i)
                n_par = n_params(model)
                sched = C.LogitScheduler(model, method)
            else:
                raise ValueError(method)
            train_time = time.time() - t_train0

            summ = run_episode(scale, eval_seed, sched, power_mode)
            row = dict(method=method, scale=scale, seed_idx=i, train_seed=i,
                       eval_seed=eval_seed, train_time_s=train_time, n_params=n_par, **summ)
            rows.append(row)
            if verbose:
                print(f"  {scale:7s} {method:20s} seed{i}: "
                      f"compliance={summ['deadline_compliance']:.3f} "
                      f"fairness={summ['load_balance_fairness']:.3f} "
                      f"energy={summ['energy']:.1f} lat={summ['latency_ms_per_decision']:.3f}ms "
                      f"(train {train_time:.1f}s)")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="small")
    ap.add_argument("--methods", default="RR,Min-Min,DCLD-net-repro")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    methods = args.methods.split(",")
    df = run_all(args.scale, methods, split=args.split, n_seeds=args.n_seeds)
    if args.out:
        df.to_csv(args.out, index=False)
    print(df.groupby("method")[["deadline_compliance", "load_balance_fairness", "energy"]].mean())
