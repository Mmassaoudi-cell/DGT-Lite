import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import pandas as pd
from src.experiments.benchmark import run_all, HEUR_NAMES

MAIN_METHODS = HEUR_NAMES + [
    "MLP", "LSTM", "Transformer", "GCN-only", "DCLD-net-repro", "DCLD-net-CE",
    "DQN", "DA-STGAT-RL", "MoE-GAT-AC", "DGT-Sched", "Distilled-DGT",
]

if __name__ == "__main__":
    os.makedirs("results/raw", exist_ok=True)
    all_dfs = []
    for scale in ["small", "medium"]:
        print(f"=== MAIN GRID: scale={scale} ===", flush=True)
        t0 = time.time()
        cache = {}
        df = run_all(scale, MAIN_METHODS, split="test", n_seeds=10, cache=cache, verbose=True)
        df.to_csv(f"results/raw/main_{scale}.csv", index=False)
        print(f"=== scale={scale} done in {time.time()-t0:.0f}s ===", flush=True)
        all_dfs.append(df)
    full = pd.concat(all_dfs, ignore_index=True)
    full.to_csv("results/raw/main_benchmark.csv", index=False)
    print("ALL DONE", flush=True)
