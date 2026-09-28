"""
Direct diagnostic for the GCN over-smoothing mechanism proposed in the ablation discussion
(reviewer-requested: the ablation shows WHERE the failure is localized -- the GCN branch --
but not WHY; this script measures node-embedding similarity after each GCN layer to test the
over-smoothing hypothesis directly, rather than only inferring it from the ablation).

Metric: mean pairwise cosine similarity between per-VM node embeddings at the output of each
GCN layer, averaged over decision steps of several validation episodes. A value near 1.0 means
essentially all VMs get near-identical representations (over-smoothed, no information left to
discriminate which VM to pick); a value near 0 means embeddings remain well separated.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import torch

from src.simulator.core import make_scenario, SPLIT_SEEDS
from src.models.dataset import collect_dataset
from src.models import flexible
from src.models.train_supervised import train_model
from src.experiments.benchmark import TRAIN_SEEDS, SUPERVISED_EPOCHS


def mean_pairwise_cosine(x: torch.Tensor) -> float:
    xn = torch.nn.functional.normalize(x, dim=-1)
    sim = xn @ xn.T
    n = sim.shape[0]
    off_diag = (sim.sum() - torch.diagonal(sim).sum()) / (n * (n - 1))
    return float(off_diag.item())


def layer_similarities(model, feat, hist, adj):
    with torch.no_grad():
        h_seq, _ = model.temporal(hist.unsqueeze(-1))
        ctx, _ = model.attn_pool(h_seq)
        x0 = torch.relu(model.input_proj(torch.cat([feat, ctx], dim=-1)))
        from src.models.blocks import normalize_adj
        a_norm = normalize_adj(adj)
        x1 = torch.relu(model.g1(x0, a_norm))
        x2 = torch.relu(model.g2(x1, a_norm))
    return dict(input_layer=mean_pairwise_cosine(x0),
                gcn_layer1=mean_pairwise_cosine(x1),
                gcn_layer2=mean_pairwise_cosine(x2))


def measure(scale, n_eval_episodes=5, seed=0):
    samples, adj_by_ep = collect_dataset(scale, TRAIN_SEEDS[scale], eps=0.3)
    model = flexible.build("DCLD-net-repro")
    train_model(model, samples, adj_by_ep, epochs=SUPERVISED_EPOCHS[scale],
                loss_type="mse_softmax", seed=seed)
    model.eval()

    all_sims = {"input_layer": [], "gcn_layer1": [], "gcn_layer2": []}
    for ep_seed in SPLIT_SEEDS["val"][:n_eval_episodes]:
        sim_env = make_scenario(scale, ep_seed)
        adj = torch.tensor(sim_env.adjacency, dtype=torch.float32)
        step = 0
        while not sim_env.done() and step < 20:  # sample early decisions of each episode
            task = sim_env.current_task()
            feat = torch.tensor(sim_env.all_vm_features(task), dtype=torch.float32)
            hist = torch.tensor(sim_env.history_tensor(), dtype=torch.float32)
            sims = layer_similarities(model, feat, hist, adj)
            for k, v in sims.items():
                all_sims[k].append(v)
            # advance with a fixed, arbitrary action just to move state forward for sampling
            sim_env.step(int(np.argmax(feat[:, 2].numpy())))  # pick max-slack VM, cheap probe
            step += 1
    return {k: float(np.mean(v)) for k, v in all_sims.items()}


if __name__ == "__main__":
    results = {}
    for scale in ["small", "medium"]:
        print(f"=== over-smoothing diagnostic: scale={scale} ===", flush=True)
        results[scale] = measure(scale)
        print(scale, results[scale], flush=True)
    import json, os
    os.makedirs("results/aggregate", exist_ok=True)
    with open("results/aggregate/oversmoothing_diagnostic.json", "w") as f:
        json.dump(results, f, indent=2)
    print("DONE", results)
