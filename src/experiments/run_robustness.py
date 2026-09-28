"""Robustness / distribution-shift evaluation (master protocol section 20). Policies are
trained ONCE on the nominal training distribution, then evaluated zero-shot (no retraining)
under four perturbed test conditions applied at the SIMULATOR/scenario level so every method
(heuristic or learned) is probed identically and fairly:
  - nominal            : standard test distribution (sanity reference row)
  - bursty_arrivals     : arrival_scale 0.55 -> 0.30 (much higher contention)
  - tight_deadlines     : slack_range (1.0,3.2) -> (0.6,1.8) (harsher deadlines)
  - noisy_hosts         : host_capacity_noise_std 0 -> 0.30 (unreliable/heterogeneous hardware)
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import pandas as pd

from src.simulator.core import SPLIT_SEEDS
from src.experiments.benchmark import run_episode, build_heuristic, n_params, HEUR_NAMES
from src.models.dataset import collect_dataset
from src.models import flexible
from src.models.train_supervised import train_model, NeuralScheduler
from src.models import candidates as C

PERTURBATIONS = {
    "nominal": dict(),
    "bursty_arrivals": dict(arrival_scale=0.30),
    "tight_deadlines": dict(slack_range=(0.6, 1.8)),
    "noisy_hosts": dict(host_capacity_noise_std=0.30),
}

# Methods probed: representative heuristic, the reproduced source method, and the frozen
# final model (trained once per seed on the NOMINAL distribution only).
METHODS = ["RR", "MrLBA-approx", "DCLD-net-repro", "FINAL_MODEL"]


def build_and_train(method, scale, seed, samples, adj_by_ep):
    if method in HEUR_NAMES or method == "RR":
        return None  # heuristics are stateless; built fresh per episode
    if method in flexible.__dict__.get("_", {}):
        pass
    if method.startswith("DCLD-net") or method in ("MLP", "LSTM", "Transformer", "GCN-only"):
        model = flexible.build(method)
        loss_type = "mse_softmax" if method.startswith("DCLD-net") else "ce"
        train_model(model, samples, adj_by_ep, epochs=12, loss_type=loss_type, seed=seed)
        return NeuralScheduler(model, method)
    if method == "FINAL_MODEL":
        raise RuntimeError("Set FINAL_MODEL_BUILDER before running (see bottom of file).")
    raise ValueError(method)


def run(scale, methods, final_builder, n_seeds=5):
    rows = []
    from src.experiments.benchmark import TRAIN_SEEDS
    samples, adj_by_ep = collect_dataset(scale, TRAIN_SEEDS[scale], eps=0.3)
    test_seeds = SPLIT_SEEDS["test"][:n_seeds]
    for method in methods:
        for i in range(n_seeds):
            eval_seed = test_seeds[i]
            if method == "RR" or method in HEUR_NAMES:
                sched = build_heuristic(method, None, seed=i)
            elif method == "FINAL_MODEL":
                sched = final_builder(scale, i, samples, adj_by_ep)
            else:
                sched = build_and_train(method, scale, i, samples, adj_by_ep)
            for pert_name, pert_kwargs in PERTURBATIONS.items():
                summ = run_episode_perturbed(scale, eval_seed, sched, **pert_kwargs)
                rows.append(dict(method=method, scale=scale, seed_idx=i,
                                  perturbation=pert_name, **summ))
                print(f"  {scale} {method:16s} seed{i} {pert_name:16s} "
                      f"compliance={summ['deadline_compliance']:.3f} "
                      f"fairness={summ['load_balance_fairness']:.3f}")
    return pd.DataFrame(rows)


def run_episode_perturbed(scale, seed, sched, **pert_kwargs):
    from src.simulator.core import make_scenario
    sim = make_scenario(scale, seed, **pert_kwargs)
    while not sim.done():
        task = sim.current_task()
        v = sched.select(sim, task)
        sim.step(v)
    return sim.summary()


if __name__ == "__main__":
    os.makedirs("results/raw", exist_ok=True)
    # Placeholder final-model builder wired up after model selection (see
    # run_robustness_final.py, generated once FINAL_MODEL_CONFIG.yaml is frozen).
    print("This module provides `run()`; invoke via run_robustness_final.py once the "
          "final model is selected and frozen.")
