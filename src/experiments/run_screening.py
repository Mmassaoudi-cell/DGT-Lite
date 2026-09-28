"""Stage 2 validation-only candidate screening (master protocol section 9). Runs the four
hybrid candidates plus the DCLD-net-repro reference on VALIDATION seeds only (never test),
3 seeds, at small+medium scale. Output feeds MODEL_SELECTION_REPORT.md and the final-model
selection decision -- this run, not the exploratory test-split numbers collected elsewhere,
is the basis for which candidate is selected."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import pandas as pd
from src.experiments.benchmark import run_all

CANDIDATES = ["DCLD-net-repro", "DA-STGAT-RL", "MoE-GAT-AC", "DGT-Sched", "Distilled-DGT"]

if __name__ == "__main__":
    os.makedirs("results/raw", exist_ok=True)
    dfs = []
    for scale in ["small", "medium"]:
        print(f"=== SCREENING (val): scale={scale} ===", flush=True)
        df = run_all(scale, CANDIDATES, split="val", n_seeds=3, cache={}, verbose=True)
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    full.to_csv("results/raw/screening_val.csv", index=False)
    print("SCREENING DONE", flush=True)
