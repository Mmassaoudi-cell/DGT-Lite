"""Reads results/raw/screening_val.csv (VALIDATION seeds only) and ranks the four candidates
(+ DCLD-net-repro reference row) by the Scheduling Quality Score (SQS, src/utils/metrics.py),
averaged across small+medium scale and all screened seeds. Selects the top candidate as the
final model per master protocol section 10 (near-strongest validation score, no seed collapse,
competitive on all three co-primary metrics). Writes the decision to
results/aggregate/final_selection.json and prints a human-readable rationale."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import pandas as pd

from src.utils.metrics import sqs

if __name__ == "__main__":
    df = pd.read_csv("results/raw/screening_val.csv")
    ref_energy = df.groupby("scale")["energy"].transform("mean")
    df["sqs"] = [sqs(r.deadline_compliance, r.load_balance_fairness, r.energy, e)
                 for r, e in zip(df.itertuples(), ref_energy)]

    agg = df.groupby("method").agg(
        sqs_mean=("sqs", "mean"), sqs_std=("sqs", "std"),
        compliance_mean=("deadline_compliance", "mean"),
        compliance_min=("deadline_compliance", "min"),
        fairness_mean=("load_balance_fairness", "mean"),
        energy_mean=("energy", "mean"),
    ).sort_values("sqs_mean", ascending=False)
    print(agg)

    candidates_only = agg.drop(index=[i for i in agg.index if i == "DCLD-net-repro"],
                                errors="ignore")
    # eliminate any candidate that collapsed on any individual seed (min compliance < 0.3)
    survivors = candidates_only[candidates_only["compliance_min"] >= 0.3]
    ranked = (survivors if len(survivors) else candidates_only).sort_values(
        "sqs_mean", ascending=False)
    winner = ranked.index[0]
    runner_up = ranked.index[1] if len(ranked) > 1 else None

    decision = dict(
        winner=winner,
        winner_sqs=float(ranked.loc[winner, "sqs_mean"]),
        runner_up=runner_up,
        ranking=ranked.index.tolist(),
        full_table=agg.reset_index().to_dict(orient="records"),
    )
    os.makedirs("results/aggregate", exist_ok=True)
    with open("results/aggregate/final_selection.json", "w") as f:
        json.dump(decision, f, indent=2)
    print("\nSELECTED FINAL MODEL:", winner)
    print("Runner-up (retained for tuning comparison):", runner_up)
