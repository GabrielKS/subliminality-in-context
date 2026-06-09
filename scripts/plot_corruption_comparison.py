#!/usr/bin/env python
"""Compare a base QA run against corruption runs (random + entangled).

Reads one **base** run (`run_qa_inference.py` output, e.g.
`data/qa_inference_small.parquet`), one **random** corruption run, and any number
of **entangled** corruption runs (`run_cot_corruption.py` outputs, e.g.
`data/corrupted/corrupted_0.5_{random_numerical,correct_top10}.parquet`), and
writes a PNG with three horizontal bar panels:

1. **fraction correct** — mean of `--correct-col` (default `model_correct`) per
   run, with Wilson 95% CIs (one bar per file: base, random, each entangled).
2. **correct_diff** — each *non-base* run's fraction correct minus the base's,
   paired by `qid` (mean ± 1.96·SEM of the per-question difference).
3. **logit difference** — mean `logprob_diff` per run (= logit(correct) −
   logit(incorrect); the log-softmax normalizer cancels, so logprob- and
   logit-differences are identical), mean ± 1.96·SEM.

Paired t-tests (`scipy.stats.ttest_rel`, paired on `qid`) compare the **random**
run against **each entangled** run; entangled bars with p < 0.05 get a `*`. The
correctness test (used for panels 1 & 2) is on the per-question `--correct-col`;
the logit test (panel 3) is on per-question `logprob_diff`.

Runs are aligned on the intersection of `qid`s; duplicate `qid`s within a run are
dropped (keep-first, logged). Example:

    uv run scripts/plot_corruption_comparison.py \\
        --base data/qa_inference_small.parquet \\
        --random data/corrupted/corrupted_0.5_random_numerical.parquet \\
        --entangled data/corrupted/corrupted_0.5_correct_top10.parquet \\
                    data/corrupted/corrupted_0.5_incorrect_top10.parquet
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: write a PNG, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

Z = 1.96  # 95%
BASE_COLOR, RANDOM_COLOR, ENT_COLOR = "#8C8C8C", "#DD8452", "#4C72B0"


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float, float]:
    "Wilson score interval for a proportion -> (p_hat, lo, hi)."
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, center - half, center + half


def mean_ci(x, z: float = Z) -> tuple[float, float, float]:
    "Mean +/- z*SEM for a numeric array -> (mean, lo, hi)."
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    m = float(x.mean()) if len(x) else 0.0
    sem = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
    return m, m - z * sem, m + z * sem


def paired_p(a, b) -> float:
    "Two-sided paired t-test p-value; 1.0 if undefined (e.g. zero-variance diff)."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = stats.ttest_rel(np.asarray(a, float), np.asarray(b, float)).pvalue
    return float(p) if np.isfinite(p) else 1.0


def load_run(path: Path, role: str) -> tuple[str, pd.DataFrame]:
    "Load a run, dedup qids (keep-first), index by qid; return (label, df)."
    df = pd.read_parquet(path)
    n0 = len(df)
    df = df.drop_duplicates("qid", keep="first")
    if n0 != len(df):
        print(f"  note: {path.name} had {n0 - len(df)} duplicate qid row(s) — dropped (kept first)")
    label = "base" if role == "base" else re.sub(r"^corrupted_[0-9.]+_", "", path.stem)
    return label, df.set_index("qid")


def bar_panel(ax, labels, vals, err_lo, err_hi, colors, sig, title, ylabel, *,
              ref=None, ylim=None, pct=False):
    "One bar panel with asymmetric error bars, a reference line, and `*` on sig bars."
    x = np.arange(len(labels))
    ax.bar(x, vals, yerr=[err_lo, err_hi], capsize=5, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ref is not None:
        ax.axhline(ref, ls=":", c="grey", lw=1)
    if ylim is not None:
        ax.set_ylim(*ylim)
    # value labels + significance asterisks
    tops = [v + eh for v, eh in zip(vals, err_hi)]
    bots = [v - el for v, el in zip(vals, err_lo)]
    span = (max(tops + [0]) - min(bots + [0])) or 1.0
    for i, (v, eh, el, s) in enumerate(zip(vals, err_hi, err_lo, sig)):
        txt = f"{v:.0%}" if pct else f"{v:+.2f}"
        if v >= 0:
            ax.text(i, v + eh + 0.02 * span, txt, ha="center", va="bottom", fontsize=8)
            if s:
                ax.text(i, v + eh + 0.07 * span, "*", ha="center", va="bottom", fontsize=18)
        else:
            ax.text(i, v - el - 0.02 * span, txt, ha="center", va="top", fontsize=8)
            if s:
                ax.text(i, v - el - 0.07 * span, "*", ha="center", va="top", fontsize=18)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True, help="base QA-inference run.")
    ap.add_argument("--random", type=Path, required=True, help="random-token corruption run.")
    ap.add_argument("--entangled", type=Path, nargs="+", required=True,
                    help="one or more entangled corruption runs.")
    ap.add_argument("--correct-col", default="model_correct",
                    help="boolean column used for 'fraction correct' (default: model_correct).")
    ap.add_argument("--out", type=Path, default=repo_root / "plots/corruption_comparison.png")
    args = ap.parse_args()

    cc = args.correct_col
    base_label, base = load_run(args.base, "base")
    rnd_label, rnd = load_run(args.random, "random")
    ent = [load_run(p, "entangled") for p in args.entangled]

    # Cutoff fraction for the title: all entangled runs must share one value.
    ent_cutoffs = {}
    for lab, df in ent:
        if "cutoff_frac" not in df.columns:
            raise ValueError(f"entangled run {lab!r} has no 'cutoff_frac' column")
        ent_cutoffs[lab] = sorted(df["cutoff_frac"].dropna().unique().tolist())
    distinct = {c for cs in ent_cutoffs.values() for c in cs}
    if len(distinct) != 1:
        raise ValueError(f"entangled runs disagree on cutoff_frac: {ent_cutoffs}")
    cutoff_fraction = distinct.pop()

    # Align everything on the common qid set (paired by question).
    common = base.index
    for _, df in [(rnd_label, rnd), *ent]:
        common = common.intersection(df.index)
    common = common.sort_values()
    if len(common) == 0:
        raise SystemExit("no shared qids across the runs")
    base, rnd = base.loc[common], rnd.loc[common]
    ent = [(lab, df.loc[common]) for lab, df in ent]
    print(f"aligned on {len(common)} shared qids "
          f"(base {len(base.index.unique())}, random {len(rnd)}, {len(ent)} entangled run(s))")

    base_correct = base[cc].astype(float).to_numpy()

    # Per-run aggregates + the per-run vs-random paired tests.
    def run_stats(df):
        correct = df[cc].astype(float).to_numpy()
        fc = wilson(int(correct.sum()), len(correct))            # (p, lo, hi) Wilson
        diff = correct - base_correct                            # paired vs base
        cd = mean_ci(diff)                                       # (mean, lo, hi)
        ld = mean_ci(df["logprob_diff"].to_numpy())              # (mean, lo, hi)
        return dict(correct=correct, fc=fc, cd=cd, ld=ld,
                    logit=df["logprob_diff"].to_numpy())

    rows = [("base", base, BASE_COLOR), (rnd_label, rnd, RANDOM_COLOR),
            *[(lab, df, ENT_COLOR) for lab, df in ent]]
    stat = {lab: run_stats(df) for lab, df, _ in rows}

    # Significance vs random (entangled only).
    p_correct, p_logit = {}, {}
    for lab, df in ent:
        p_correct[lab] = paired_p(df[cc].astype(float), rnd[cc].astype(float))
        p_logit[lab] = paired_p(df["logprob_diff"], rnd["logprob_diff"])

    # ---- stdout summary ----
    print(f"\nfraction correct ({cc}):")
    for lab, _, _ in rows:
        print(f"  {lab:<22} {stat[lab]['fc'][0]:.1%}")
    print("\nvs-random paired t-tests (entangled):")
    for lab, _ in ent:
        print(f"  {lab:<22} correct p={p_correct[lab]:.4f}{' *' if p_correct[lab] < 0.05 else ''}"
              f"   logit p={p_logit[lab]:.4f}{' *' if p_logit[lab] < 0.05 else ''}")

    # ---- figure ----
    fig, (ax_fc, ax_cd, ax_ld) = plt.subplots(1, 3, figsize=(15, 4.8))

    # Panel 1: fraction correct (all runs), Wilson CI.
    labels = [lab for lab, _, _ in rows]
    colors = [c for _, _, c in rows]
    vals = [stat[l]["fc"][0] for l in labels]
    elo = [stat[l]["fc"][0] - stat[l]["fc"][1] for l in labels]
    ehi = [stat[l]["fc"][2] - stat[l]["fc"][0] for l in labels]
    sig = [l in p_correct and p_correct[l] < 0.05 for l in labels]
    bar_panel(ax_fc, labels, vals, elo, ehi, colors, sig,
              "Fraction correct", "rate", ref=0.5, ylim=(0, 1.12), pct=True)

    # Panel 2: correct_diff (non-base runs), paired mean ± SEM.
    nb = [(lab, c) for (lab, _, c) in rows if lab != "base"]
    labels2 = [lab for lab, _ in nb]
    colors2 = [c for _, c in nb]
    vals2 = [stat[l]["cd"][0] for l in labels2]
    elo2 = [stat[l]["cd"][0] - stat[l]["cd"][1] for l in labels2]
    ehi2 = [stat[l]["cd"][2] - stat[l]["cd"][0] for l in labels2]
    sig2 = [l in p_correct and p_correct[l] < 0.05 for l in labels2]
    bar_panel(ax_cd, labels2, vals2, elo2, ehi2, colors2, sig2,
              "Change in fraction correct", "Δ fraction correct", ref=0.0)

    # Panel 3: logit difference (all runs), mean ± SEM.
    vals3 = [stat[l]["ld"][0] for l in labels]
    elo3 = [stat[l]["ld"][0] - stat[l]["ld"][1] for l in labels]
    ehi3 = [stat[l]["ld"][2] - stat[l]["ld"][0] for l in labels]
    sig3 = [l in p_logit and p_logit[l] < 0.05 for l in labels]
    bar_panel(ax_ld, labels, vals3, elo3, ehi3, colors, sig3,
              "Correct vs. incorrect logit difference", "logit(correct) − logit(incorrect)", ref=0.0)

    fig.suptitle(f"Subliminal chain of thought, cutoff fraction = {cutoff_fraction:.2f}  ·  "
                 f"n={len(common)}  ·  * = p<0.05 vs. random (paired t-test)", y=1.02)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
