"""Generates IEEE-ready LaTeX tables directly from result CSVs (master protocol section 34:
never hand-type numbers that can be generated automatically)."""
from __future__ import annotations
import sys, os
import pandas as pd

OUT = "manuscript/tables"
os.makedirs(OUT, exist_ok=True)


def main_benchmark_table(main_csv, champion, out="tab_main_benchmark.tex"):
    df = pd.read_csv(main_csv)
    n_seeds = df.groupby(["method", "scale"]).size().max()
    agg = df.groupby(["method", "scale"]).agg(
        compliance=("deadline_compliance", "mean"),
        fairness=("load_balance_fairness", "mean"),
        energy=("energy", "mean"),
        latency=("latency_ms_per_decision", "mean"),
        params=("n_params", "mean"),
    ).reset_index()
    scales = sorted(agg["scale"].unique())
    methods = agg["method"].unique().tolist()
    methods = sorted(methods, key=lambda m: (m != champion, m))

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Main benchmark comparison (mean over %d paired seeds). "
                  r"Bold = proposed final model.}" % n_seeds)
    lines.append(r"\label{tab:main}")
    col_spec = "l" + "ccc" * len(scales)
    lines.append(r"\begin{tabular}{%s}" % col_spec)
    lines.append(r"\toprule")
    header = ["Method"] + sum([[r"\multicolumn{3}{c}{%s}" % s.capitalize()] for s in scales], [])
    lines.append(" & ".join(["Method"] + [r"\multicolumn{3}{c}{%s}" % s.capitalize()
                                           for s in scales]) + r" \\")
    lines.append(" & ".join([""] + ["Compl.", "Fair.", "Energy"] * len(scales)) + r" \\")
    lines.append(r"\midrule")
    for m in methods:
        row = [m if m != champion else r"\textbf{%s}" % m]
        for s in scales:
            sub = agg[(agg.method == m) & (agg.scale == s)]
            if len(sub):
                r_ = sub.iloc[0]
                row += [f"{r_.compliance:.3f}", f"{r_.fairness:.3f}", f"{r_.energy:.1f}"]
            else:
                row += ["--", "--", "--"]
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    with open(os.path.join(OUT, out), "w") as f:
        f.write("\n".join(lines))
    print("wrote", out)


def _bucket(outcome: str) -> str:
    o = outcome.lower()
    if o.startswith("win"):
        return "win"
    if o.startswith("loss"):
        return "loss"
    return "tie"


def wtl_summary_table(wtl_csv, out="tab_wtl.tex"):
    df = pd.read_csv(wtl_csv)
    n_seeds = int(df["n"].max()) if "n" in df.columns else "N"
    df["bucket"] = df["outcome"].map(_bucket)
    sig = df["outcome"].str.contains("significant\\)", case=False, regex=True) & ~df[
        "outcome"].str.contains("not significant", case=False)
    df["sig_win"] = sig & (df["bucket"] == "win")
    counts = df.groupby(["scale", "bucket"]).size().unstack(fill_value=0)
    sig_counts = df.groupby("scale")["sig_win"].sum()
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Win/tie/loss vs. all baselines on SQS (%s paired seeds). "
             r"``Sig.'' = also significant after Holm correction (see Sec.~VII-C).}" % n_seeds,
             r"\label{tab:wtl}", r"\begin{tabular}{lcccc}", r"\toprule",
             r"Scale & Wins & Ties & Losses & Sig. wins \\", r"\midrule"]
    for scale, row in counts.iterrows():
        lines.append(f"{scale} & {row.get('win', 0)} & {row.get('tie', 0)} & "
                      f"{row.get('loss', 0)} & {int(sig_counts.get(scale, 0))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(OUT, out), "w") as f:
        f.write("\n".join(lines))
    print("wrote", out)


def ablation_table(ablation_csv, out="tab_ablation.tex"):
    df = pd.read_csv(ablation_csv)
    agg = df.groupby(["method", "scale"]).agg(
        compliance=("deadline_compliance", "mean"), fairness=("load_balance_fairness", "mean"),
    ).reset_index()
    scales = sorted(agg["scale"].unique())
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Architectural ablation of the DCLD-net reproduction.}",
             r"\label{tab:ablation}",
             r"\begin{tabular}{l%s}" % ("cc" * len(scales)),
             r"\toprule"]
    lines.append(" & ".join(["Variant"] + sum([[f"Compl.({s})", f"Fair.({s})"]
                                                for s in scales], [])) + r" \\")
    lines.append(r"\midrule")
    for m in agg["method"].unique():
        row = [m]
        for s in scales:
            sub = agg[(agg.method == m) & (agg.scale == s)]
            if len(sub):
                r_ = sub.iloc[0]
                row += [f"{r_.compliance:.3f}", f"{r_.fairness:.3f}"]
            else:
                row += ["--", "--"]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(OUT, out), "w") as f:
        f.write("\n".join(lines))
    print("wrote", out)


def robustness_table(robustness_csv, champion, out="tab_robustness.tex"):
    df = pd.read_csv(robustness_csv)
    agg = df.groupby(["method", "scale", "perturbation"]).agg(
        compliance=("deadline_compliance", "mean"), fairness=("load_balance_fairness", "mean"),
    ).reset_index()
    perts = ["nominal", "bursty_arrivals", "tight_deadlines", "noisy_hosts"]
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Robustness under distribution shift (deadline compliance / fairness), "
             r"medium scale, zero-shot (no retraining).}",
             r"\label{tab:robustness}",
             r"\begin{tabular}{l%s}" % ("cc" * len(perts)), r"\toprule"]
    lines.append(" & ".join(["Method"] + [p.replace("_", " ") for p in perts]) + r" \\")
    lines.append(r"\midrule")
    for m in df.method.unique():
        row = [m if m != champion else r"\textbf{%s}" % m]
        for p in perts:
            sub = agg[(agg.method == m) & (agg.scale == "medium") & (agg.perturbation == p)]
            if len(sub):
                r_ = sub.iloc[0]
                row.append(f"{r_.compliance:.2f}/{r_.fairness:.2f}")
            else:
                row.append("--")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    with open(os.path.join(OUT, out), "w") as f:
        f.write("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    champion = sys.argv[1] if len(sys.argv) > 1 else "DGT-Sched"
    if os.path.exists("results/raw/main_benchmark.csv"):
        main_benchmark_table("results/raw/main_benchmark.csv", champion)
    if os.path.exists("BENCHMARK_WTL.csv"):
        wtl_summary_table("BENCHMARK_WTL.csv")
    if os.path.exists("results/raw/ablation_repro.csv"):
        ablation_table("results/raw/ablation_repro.csv")
    if os.path.exists("results/raw/robustness.csv"):
        robustness_table("results/raw/robustness.csv", champion)
