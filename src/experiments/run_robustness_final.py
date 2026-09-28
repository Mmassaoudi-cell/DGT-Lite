"""Robustness / distribution-shift evaluation of the FROZEN final model (Distilled-DGT /
"DGT-Lite") against RR, MrLBA-approx, and the DCLD-net reproduction. Policies are trained once
per seed on the nominal training distribution and evaluated zero-shot under 3 perturbed test
conditions (see PERTURBATIONS below) applied at the simulator level so every method is probed
identically."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import pandas as pd

from src.simulator.core import SPLIT_SEEDS, make_scenario
from src.experiments.benchmark import build_heuristic, n_params, TRAIN_SEEDS, HEUR_NAMES
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
METHODS = ["RR", "MrLBA-approx", "DCLD-net-repro", "DGT-Sched", "Distilled-DGT"]


def run_episode_perturbed(scale, seed, sched, **pert_kwargs):
    sim = make_scenario(scale, seed, **pert_kwargs)
    while not sim.done():
        task = sim.current_task()
        v = sched.select(sim, task)
        sim.step(v)
    return sim.summary()


def build_sched(method, scale, seed, samples, adj_by_ep, teacher_cache):
    if method in HEUR_NAMES or method == "RR":
        from src.simulator.core import SCALE_CONFIGS
        return build_heuristic(method, SCALE_CONFIGS[scale]["n_vms"], seed=seed)
    if method == "DCLD-net-repro":
        model = flexible.build(method)
        train_model(model, samples, adj_by_ep, epochs=12, loss_type="mse_softmax", seed=seed)
        return NeuralScheduler(model, method)
    if method == "DGT-Sched":
        model = C.DGTSched()
        C.train_dgt_supervised(model, samples, adj_by_ep, epochs=14, seed=seed)
        teacher_cache[seed] = model
        return C.LogitScheduler(model, method)
    if method == "Distilled-DGT":
        teacher = teacher_cache.get(seed)
        if teacher is None:
            teacher = C.DGTSched()
            C.train_dgt_supervised(teacher, samples, adj_by_ep, epochs=14, seed=seed)
            teacher_cache[seed] = teacher
        model = C.DistilledDGT()
        C.distill_train(model, teacher, samples, adj_by_ep, epochs=12, seed=seed)
        return C.LogitScheduler(model, method)
    raise ValueError(method)


def run(scale="small", n_seeds=5):
    samples, adj_by_ep = collect_dataset(scale, TRAIN_SEEDS[scale], eps=0.3)
    test_seeds = SPLIT_SEEDS["test"][:n_seeds]
    rows = []
    teacher_cache = {}
    for method in METHODS:
        for i in range(n_seeds):
            sched = build_sched(method, scale, i, samples, adj_by_ep, teacher_cache)
            for pert_name, pert_kwargs in PERTURBATIONS.items():
                summ = run_episode_perturbed(scale, test_seeds[i], sched, **pert_kwargs)
                rows.append(dict(method=method, scale=scale, seed_idx=i,
                                  perturbation=pert_name, **summ))
                print(f"  {scale} {method:16s} seed{i} {pert_name:16s} "
                      f"compliance={summ['deadline_compliance']:.3f} "
                      f"fairness={summ['load_balance_fairness']:.3f}", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs("results/raw", exist_ok=True)
    dfs = []
    for scale in ["small", "medium"]:
        print(f"=== ROBUSTNESS: scale={scale} ===", flush=True)
        dfs.append(run(scale, n_seeds=10))
    full = pd.concat(dfs, ignore_index=True)
    full.to_csv("results/raw/robustness.csv", index=False)
    print("ROBUSTNESS DONE")
