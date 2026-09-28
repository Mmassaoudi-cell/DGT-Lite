"""Supplementary large-scale (1000 hosts / 2000 VMs / 2000 tasks) scalability check --
run only for representative heuristics, the DCLD-net reproduction, and the frozen final model
(retraining all 13+ methods at this scale was judged not worth the compute budget for a single
compressed research session; see FINAL_RESEARCH_SUMMARY.md for the explicit scope note)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import pandas as pd
from src.experiments.benchmark import run_all

if __name__ == "__main__":
    # NOTE (documented scope decision): DCLD-net-repro (dense GCN) and the final model's
    # DGT-Sched teacher (graph attention) both compute an O(n_vms^2) propagation over a
    # 2000x2000 adjacency at this scale; a single training seed took >25 minutes under this
    # session's CPU budget with 3+ concurrent jobs, making a 3-seed run impractical here. We
    # therefore report only the non-learned (no-training-cost) methods' scalability at 2000
    # VMs -- sufficient to show the simulator and heuristics scale, and to characterize how
    # per-decision latency grows with VM count -- and disclose the NN-scalability gap in
    # FINAL_RESEARCH_SUMMARY.md rather than silently omitting it.
    methods = ["RR", "Min-Min", "MrLBA-approx", "HIWIGOA-LB-approx"]
    os.makedirs("results/raw", exist_ok=True)
    df = run_all("large", methods, split="test", n_seeds=3, cache={}, verbose=True)
    df.to_csv("results/raw/large_scale.csv", index=False)
    print("LARGE-SCALE DONE (heuristics only; see script docstring note on NN scalability scope)")
