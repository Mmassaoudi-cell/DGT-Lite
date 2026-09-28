"""Generates the manuscript's figures directly from results CSVs (master protocol section 31:
figures must be generated from real result files, never hand-drawn)."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})


def fig_main_benchmark(main_csv, champion, out="fig3_main_benchmark.png"):
    df = pd.read_csv(main_csv)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    metrics = [("deadline_compliance", "Deadline Compliance"),
               ("load_balance_fairness", "Load-Balance Fairness (Jain)"),
               ("energy", "Energy (normalized units)")]
    order = df.groupby("method")["deadline_compliance"].mean().sort_values().index.tolist()
    for ax, (m, title) in zip(axes, metrics):
        piv = df.groupby(["method", "scale"])[m].mean().unstack()
        piv = piv.reindex(order)
        piv.plot(kind="barh", ax=ax, legend=(m == metrics[0][0]))
        for lbl in ax.get_yticklabels():
            if lbl.get_text() == champion:
                lbl.set_fontweight("bold")
        ax.set_title(title)
        ax.set_xlabel(m)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, out))
    plt.close()


def fig_ablation(ablation_csv, out="fig5_ablation.png"):
    df = pd.read_csv(ablation_csv)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, m in zip(axes, ["deadline_compliance", "load_balance_fairness"]):
        piv = df.groupby(["method", "scale"])[m].mean().unstack()
        piv.plot(kind="bar", ax=ax)
        ax.set_title(m)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, out))
    plt.close()


def fig_pareto(main_csv, out="fig6_pareto.png"):
    df = pd.read_csv(main_csv)
    agg = df.groupby("method").agg(
        compliance=("deadline_compliance", "mean"),
        latency=("latency_ms_per_decision", "mean"),
        params=("n_params", "mean"),
    ).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].scatter(agg["latency"], agg["compliance"])
    for _, r in agg.iterrows():
        axes[0].annotate(r["method"], (r["latency"], r["compliance"]), fontsize=6)
    axes[0].set_xlabel("Latency per decision (ms)")
    axes[0].set_ylabel("Deadline compliance")
    axes[0].set_title("Performance vs. Latency")

    agg2 = agg.dropna(subset=["params"])
    axes[1].scatter(agg2["params"], agg2["compliance"])
    for _, r in agg2.iterrows():
        axes[1].annotate(r["method"], (r["params"], r["compliance"]), fontsize=6)
    axes[1].set_xlabel("Parameters")
    axes[1].set_ylabel("Deadline compliance")
    axes[1].set_xscale("log")
    axes[1].set_title("Performance vs. Model Size")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, out))
    plt.close()


def fig_robustness(robustness_csv, out="fig4_robustness.png"):
    df = pd.read_csv(robustness_csv)
    fig, ax = plt.subplots(figsize=(7, 4))
    piv = df.groupby(["method", "perturbation"])["deadline_compliance"].mean().unstack()
    piv.plot(kind="bar", ax=ax)
    ax.set_ylabel("Deadline compliance")
    ax.set_title("Robustness under distribution shift")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, out))
    plt.close()


if __name__ == "__main__":
    import sys
    champion = sys.argv[1] if len(sys.argv) > 1 else "DGT-Sched"
    if os.path.exists("results/raw/main_benchmark.csv"):
        fig_main_benchmark("results/raw/main_benchmark.csv", champion)
        fig_pareto("results/raw/main_benchmark.csv")
    if os.path.exists("results/raw/ablation_repro.csv"):
        fig_ablation("results/raw/ablation_repro.csv")
    if os.path.exists("results/raw/robustness.csv"):
        fig_robustness("results/raw/robustness.csv")
    print("figures written to", OUT)
