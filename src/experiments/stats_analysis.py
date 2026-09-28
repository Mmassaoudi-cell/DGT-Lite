"""Statistical validation (master protocol section 22): paired Wilcoxon signed-rank test
(paired by seed_idx across methods, since every method is evaluated on the same 5 (train_seed,
eval_seed) pairs -- see benchmark.py docstring), Holm correction across baselines, effect size
(matched-pairs rank-biserial correlation), plus mean/std/median/95% CI per method/metric/scale.
Produces BENCHMARK_WTL.csv (win/tie/loss classification) and results/aggregate/stats_tests.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sstats

PRIMARY_METRIC = "sqs"
SECONDARY_METRICS = ["deadline_compliance", "load_balance_fairness", "energy",
                      "mean_response_time"]
ALPHA = 0.05


def add_sqs(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the Scheduling Quality Score (src/utils/metrics.py) as column 'sqs', using the
    per-scale mean energy across all methods/seeds as the normalization reference -- fixed
    before any comparison is drawn, not tuned per-method."""
    from src.utils.metrics import sqs
    ref = df.groupby("scale")["energy"].transform("mean")
    df = df.copy()
    df["sqs"] = [sqs(r.deadline_compliance, r.load_balance_fairness, r.energy, e)
                 for r, e in zip(df.itertuples(), ref)]
    return df


def rank_biserial(diffs: np.ndarray) -> float:
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return 0.0
    n_pos = np.sum(diffs > 0)
    n_neg = np.sum(diffs < 0)
    return (n_pos - n_neg) / len(diffs)


def summarize(df: pd.DataFrame, group_cols=("method", "scale")) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(list(group_cols)):
        row = dict(zip(group_cols, keys))
        for m in [PRIMARY_METRIC] + SECONDARY_METRICS:
            vals = g[m].values.astype(float)
            n = len(vals)
            mean, std, med = vals.mean(), vals.std(ddof=1) if n > 1 else 0.0, np.median(vals)
            se = std / np.sqrt(n) if n > 1 else 0.0
            ci = 1.96 * se
            row.update({f"{m}_mean": mean, f"{m}_std": std, f"{m}_median": med,
                        f"{m}_ci95": ci, f"{m}_n": n})
        rows.append(row)
    return pd.DataFrame(rows)


def paired_tests(df: pd.DataFrame, champion: str, scale: str, metric: str = PRIMARY_METRIC,
                  higher_is_better: bool = True) -> pd.DataFrame:
    champ = df[(df.method == champion) & (df.scale == scale)].set_index("seed_idx")[metric]
    baselines = [m for m in df.method.unique() if m != champion]
    rows = []
    pvals = []
    for b in baselines:
        base = df[(df.method == b) & (df.scale == scale)].set_index("seed_idx")[metric]
        common = champ.index.intersection(base.index)
        if len(common) < 3:
            continue
        c_vals = champ.loc[common].values
        b_vals = base.loc[common].values
        diffs = c_vals - b_vals if higher_is_better else b_vals - c_vals
        if np.allclose(diffs, 0):
            p = 1.0
        else:
            try:
                _, p = sstats.wilcoxon(c_vals, b_vals)
            except ValueError:
                p = 1.0
        effect = rank_biserial(diffs)
        mean_diff = diffs.mean()
        rows.append(dict(baseline=b, scale=scale, metric=metric, n=len(common),
                          champion_mean=c_vals.mean(), baseline_mean=b_vals.mean(),
                          mean_diff=mean_diff, p_raw=p, effect_size=effect))
        pvals.append(p)
    r = pd.DataFrame(rows)
    if len(r):
        order = np.argsort(r["p_raw"].values)
        m = len(r)
        adj = np.empty(m)
        running_max = 0.0
        for rank, idx in enumerate(order):
            corrected = (m - rank) * r["p_raw"].values[idx]
            running_max = max(running_max, corrected)
            adj[idx] = min(1.0, running_max)
        r["p_holm"] = adj

        def classify(row):
            """Master protocol section 15 classification: statistically superior /
            practically superior / statistically indistinguishable ("tie") / inferior."""
            sig = row["p_holm"] < ALPHA
            n_ = int(row["n"])
            if sig and row["mean_diff"] > 0.01:
                return "win (statistically significant)"
            if sig and row["mean_diff"] < -0.01:
                return "loss (statistically significant)"
            if abs(row["mean_diff"]) < 0.01:
                return "tie"
            return (f"win (practical, not significant at n={n_})" if row["mean_diff"] > 0
                    else f"loss (practical, not significant at n={n_})")
        r["outcome"] = r.apply(classify, axis=1)
    return r


def wtl_table(df: pd.DataFrame, champion: str, scales, metric=PRIMARY_METRIC) -> pd.DataFrame:
    parts = [paired_tests(df, champion, s, metric) for s in scales]
    all_r = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return all_r


if __name__ == "__main__":
    import sys
    df = pd.read_csv(sys.argv[1])
    df = add_sqs(df)
    champion = sys.argv[2]
    scales = df.scale.unique().tolist()
    summ = summarize(df)
    summ.to_csv("results/aggregate/summary_stats.csv", index=False)
    wtl = wtl_table(df, champion, scales)
    wtl.to_csv("BENCHMARK_WTL.csv", index=False)
    print(wtl[["baseline", "scale", "mean_diff", "p_holm", "outcome"]])
    print("\nWTL counts:", wtl["outcome"].value_counts().to_dict())
