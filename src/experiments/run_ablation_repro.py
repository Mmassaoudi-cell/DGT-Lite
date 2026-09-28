"""Architectural ablation of the DCLD-net REPRODUCTION (the ablation the source paper itself
never ran -- see SOURCE_PAPER_AUDIT.md section 11). Full model vs. each component removed,
using the identical FlexibleScheduler harness so the only thing that changes is the ablated
component."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import pandas as pd
from src.experiments.benchmark import run_all

METHODS = ["DCLD-net-repro", "DCLD-net-CE", "DCLD-net-noattn", "DCLD-net-nogcn",
           "DCLD-net-notemporal"]

if __name__ == "__main__":
    os.makedirs("results/raw", exist_ok=True)
    dfs = []
    for scale in ["small", "medium"]:
        print(f"=== ABLATION (repro): scale={scale} ===", flush=True)
        df = run_all(scale, METHODS, split="test", n_seeds=10, cache={}, verbose=True)
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    full.to_csv("results/raw/ablation_repro.csv", index=False)
    print("ABLATION DONE", flush=True)
