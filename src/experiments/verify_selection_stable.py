"""Post-hoc robustness check (NOT a re-selection): confirms the validation-based final-model
choice (Distilled-DGT) is not an artifact of the previously under-trained RL candidates, using
the revised, budget-equalized TEST-set results. This does not change FINAL_MODEL_CONFIG.yaml;
it only verifies the selection was not sensitive to the training-budget asymmetry raised in
review."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import pandas as pd
from src.utils.metrics import sqs

CANDIDATES = ["DA-STGAT-RL", "MoE-GAT-AC", "DGT-Sched", "Distilled-DGT"]

if __name__ == "__main__":
    df = pd.read_csv("results/raw/main_benchmark.csv")
    df = df[df.method.isin(CANDIDATES)]
    ref_energy = df.groupby("scale")["energy"].transform("mean")
    df = df.copy()
    df["sqs"] = [sqs(r.deadline_compliance, r.load_balance_fairness, r.energy, e)
                 for r, e in zip(df.itertuples(), ref_energy)]
    agg = df.groupby("method")["sqs"].mean().sort_values(ascending=False)
    print("SQS ranking on equalized-budget TEST data (post-hoc stability check):")
    print(agg)
    winner = agg.index[0]
    print(f"\nWinner under equalized RL budget: {winner}")
    print(f"Matches frozen FINAL_MODEL_CONFIG.yaml selection (Distilled-DGT)? "
          f"{'YES' if winner == 'Distilled-DGT' else 'NO -- flag for discussion'}")
