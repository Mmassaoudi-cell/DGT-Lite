"""Assembles REPRODUCTION_REPORT.md from real CSV results (never hand-typed numbers)."""
import os
import pandas as pd

TEMPLATE_HEAD = """# Reproduction Report

Per `SOURCE_PAPER_AUDIT.md`, the source paper's absolute power-consumption units are
unrecoverable (constants `C`, `V_max`, `f_max` in Eqs. 19-21 are never given), and its headline
abstract claims (27% deadline compliance, 29.3% load-balancing efficiency) have no supporting
numeric table anywhere in the paper body. Consequently this report compares **qualitative
trends and rankings** (which the paper *does* support, via Figs. 4-7 and Tables III-IV) against
our own reproduction, rather than claiming to recover the paper's absolute figures.

## 1. Table II configuration (50 VMs / 1000 tasks) -- direct fidelity check

The paper's own numeric tables (III: Power-Aware configuration; IV: DVFS, large-size
datacenters) rank methods as **DCLD-net < MrLBA < HIWIGOA-LB < RR** in power consumption (lower
is better; DCLD-net lowest). Our reproduction below uses the identical VM/task counts (Table II)
under the corresponding power-management regime.

"""


def fmt_table(df, metric_cols, label):
    lines = [f"### {label}\n", "| Method | " + " | ".join(metric_cols) + " |",
             "|---|" + "---|" * len(metric_cols)]
    for _, r in df.iterrows():
        vals = " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c])
                           for c in metric_cols)
        lines.append(f"| {r['method']} | {vals} |")
    return "\n".join(lines) + "\n"


def main():
    out = [TEMPLATE_HEAD]
    for mode in ["pa", "dvfs"]:
        path = f"results/raw/reproduction_table2_{mode}.csv"
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        agg = df.groupby("method").agg(
            energy=("energy", "mean"), energy_std=("energy", "std"),
            compliance=("deadline_compliance", "mean"), fairness=("load_balance_fairness", "mean"),
        ).reset_index()
        agg = agg.sort_values("energy")
        out.append(fmt_table(agg, ["energy", "energy_std", "compliance", "fairness"],
                              f"Power mode = {mode.upper()} (our simulator's normalized energy units)"))
        ranking = agg.sort_values("energy")["method"].tolist()
        out.append(f"\n**Reproduced energy ranking ({mode.upper()}, lowest first):** "
                    f"{' < '.join(ranking)}\n")
        matches = ranking[0] == "DCLD-net-repro"
        out.append(f"\n**Matches paper's own ranking (DCLD-net lowest)?** "
                    f"{'YES' if matches else 'NO -- see discussion below'}\n\n")

    out.append("\n## 2. Section VI three-scale configuration (small/medium/large)\n\n")
    out.append("Deadline compliance, load-balance fairness, and energy for RR, MrLBA-approx, "
                "HIWIGOA-LB-approx and DCLD-net-repro at the small and medium scales (see "
                "`results/raw/main_small.csv`, `results/raw/main_medium.csv`):\n\n")
    for scale in ["small", "medium"]:
        path = f"results/raw/main_{scale}.csv"
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df = df[df.method.isin(["RR", "MrLBA-approx", "HIWIGOA-LB-approx", "DCLD-net-repro"])]
        agg = df.groupby("method").agg(
            deadline_compliance=("deadline_compliance", "mean"),
            load_balance_fairness=("load_balance_fairness", "mean"),
            energy=("energy", "mean"),
        ).reset_index()
        out.append(fmt_table(agg, ["deadline_compliance", "load_balance_fairness", "energy"],
                              f"Scale = {scale}"))

    out.append("\n## 2b. Loss-only minimal fix (DCLD-net architecture, cross-entropy loss)\n\n")
    out.append("Same BiGRU+attention+GCN architecture as the reproduction, with only the loss "
                "changed from Eq. 7's MSE-on-softmax to standard cross-entropy against the same "
                "oracle labels -- no other change:\n\n")
    for scale in ["small", "medium"]:
        path = f"results/raw/main_{scale}.csv"
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df = df[df.method.isin(["DCLD-net-repro", "DCLD-net-CE"])]
        if len(df) == 0:
            continue
        agg = df.groupby("method").agg(
            deadline_compliance=("deadline_compliance", "mean"),
            load_balance_fairness=("load_balance_fairness", "mean"),
            energy=("energy", "mean"),
        ).reset_index()
        out.append(fmt_table(agg, ["deadline_compliance", "load_balance_fairness", "energy"],
                              f"Scale = {scale}"))

    if os.path.exists("results/aggregate/oversmoothing_diagnostic.json"):
        import json
        with open("results/aggregate/oversmoothing_diagnostic.json") as f:
            osm = json.load(f)
        out.append("\n## 2c. Direct over-smoothing measurement\n\n")
        out.append("Mean pairwise cosine similarity between per-VM node embeddings, by GCN "
                    "layer (5 held-out validation episodes, DCLD-net-repro):\n\n")
        out.append("| Scale | Input proj. | After GCN layer 1 | After GCN layer 2 |\n"
                    "|---|---|---|---|\n")
        for scale, vals in osm.items():
            out.append(f"| {scale} | {vals['input_layer']:.3f} | {vals['gcn_layer1']:.3f} | "
                        f"{vals['gcn_layer2']:.3f} |\n")
        out.append("\nAt 200 VMs, similarity saturates to numerically 1.0 after both GCN layers, "
                    "confirming complete representational collapse; at 20 VMs it stays well "
                    "below 1.0 and decreases across layers.\n")

    # Build the energy-rank-vs-quality table dynamically from the actual current CSVs
    # (never hand-typed -- this table previously went stale after a re-run).
    rank_rows = []
    four = ["RR", "MrLBA-approx", "HIWIGOA-LB-approx", "DCLD-net-repro"]
    for label, path, mode_filter in [
        ("Table II (50 VM, PA)", "results/raw/reproduction_table2_pa.csv", None),
        ("Table II (50 VM, DVFS)", "results/raw/reproduction_table2_dvfs.csv", None),
        ("Medium (200 VM)", "results/raw/main_medium.csv", None),
        ("Small (20 VM)", "results/raw/main_small.csv", None),
    ]:
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        d = d[d.method.isin(four)]
        a = d.groupby("method").agg(energy=("energy", "mean"),
                                     compliance=("deadline_compliance", "mean"),
                                     fairness=("load_balance_fairness", "mean")).reset_index()
        a = a.sort_values("energy").reset_index(drop=True)
        rank = int(a[a.method == "DCLD-net-repro"].index[0]) + 1
        row = a[a.method == "DCLD-net-repro"].iloc[0]
        rank_word = "Lowest" if rank == 1 else ("Highest" if rank == len(a) else f"{rank} of {len(a)}")
        rank_rows.append(f"| {label} | {rank_word} ({'wins' if rank == 1 else 'loses'}) | "
                          f"{row.compliance:.3f} | {row.fairness:.3f} |")
    rank_table = "\n".join(["| Scale | DCLD-net-repro energy rank | Compliance | Fairness |",
                             "|---|---|---:|---:|"] + rank_rows)

    out.append(f"""
## 3. Discussion

The paper's claim that DCLD-net achieves the lowest power consumption among {{RR, MrLBA,
HIWIGOA-LB, DCLD-net}} **is reproducible at the paper's own Table-II scale (50 VMs, both PA and
DVFS) and at our "medium" scale (200 VMs)** -- but not at "small" scale (20 VMs).

**This apparent win is hollow, and the pattern of exactly when it appears is the key finding of
this reproduction study.** Cross-referencing energy rank against deadline compliance and fairness
at each operating point:

{rank_table}

**DCLD-net-repro "wins" on energy only in the configurations where it has collapsed to a
degenerate policy** (compliance and fairness both near zero) -- a degenerate policy that ignores
deadlines and concentrates tasks on a few VMs trivially lowers measured power under a PA/DVFS
model, since idle VMs/hosts draw less power. At the one scale where it trains into a genuinely
functional policy (small, 20 VMs), it no longer has the lowest energy. **Our reproduction of the
source paper's own headline energy comparison is real, but it is reproducing an artifact of
policy collapse, not a genuine efficiency advantage** -- something the source paper could not
have detected because it never reports deadline compliance or fairness alongside its energy
figures (Table III/IV report only power consumption and one unlabeled second column; see
`SOURCE_PAPER_AUDIT.md` section 9). This is itself a reproducible, quantitative finding, not
merely an architectural critique (see `SOURCE_WEAKNESS_ANALYSIS.md` weakness #10bis).

Separately, our faithful reproduction of the paper's literal MSE-on-softmax loss (Eq. 7) fails to
train a usable policy specifically once the GCN branch is active at >=50 VMs; component-wise
ablation (`results/raw/ablation_repro.csv`) localizes this to the GCN branch specifically, not
attention or the temporal branch (see `SOURCE_WEAKNESS_ANALYSIS.md` weakness #1). The paper never
tests at a scale where this failure mode's effect on compliance would be visible in its own
reported metrics, because it never reports compliance in its numeric tables at all.

## 4. Summary judgment

| Aspect | Verdict |
|---|---|
| Architecture (BiGRU+attention+GCN) reproducible | Yes, exactly as described |
| Table II-scale (50 VM) power *ranking* reproducible | Yes (DCLD-net lowest, both PA and DVFS) |
| ...but that ranking reflects genuine scheduling quality | **No** -- it coincides exactly with policy collapse (Section 3) |
| Abstract's 27%/29.3% claims reproducible | Not assessable -- no supporting table exists in the source paper |
| Reproduction stable across scales | **No** -- degrades sharply from 20 to >=50 VMs (Section 3; ablation isolates cause to the GCN branch) |
""")

    with open("REPRODUCTION_REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("wrote REPRODUCTION_REPORT.md")


if __name__ == "__main__":
    main()
