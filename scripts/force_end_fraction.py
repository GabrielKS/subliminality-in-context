#!/usr/bin/env python
"""Print the fraction of transcripts force-ended for one (cutoff, condition).

A trace is *force-ended* (`forced_close`) when it hits the token budget without
emitting `</think>` on its own and the closing tag is appended. For condition `base`
the value is read from the uncorrupted run (`--base`) and is the same for every
cutoff; for any other condition it is read from
`{--corrupted-dir}/corrupted_{cutoff}_{condition}.parquet`.

    uv run scripts/force_end_fraction.py --cutoff 0.5 --condition base
    uv run scripts/force_end_fraction.py --cutoff 0.5 --condition correct_top10
"""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", type=float, required=True, help="cutoff fraction, e.g. 0.25 / 0.5 / 0.75.")
    ap.add_argument("--condition", required=True,
                    help="'base' or a token source (e.g. random_numerical, correct_top10).")
    ap.add_argument("--base", type=Path, default=repo / "data/qa_inference_small.parquet",
                    help="uncorrupted run, used when --condition base (default: %(default)s).")
    ap.add_argument("--corrupted-dir", type=Path, default=repo / "data/corrupted",
                    help="directory of corrupted runs (default: %(default)s).")
    args = ap.parse_args()

    if args.condition == "base":
        path = args.base  # uncorrupted run: cutoff is for labelling only
    else:
        path = args.corrupted_dir / f"corrupted_{args.cutoff:g}_{args.condition}.parquet"
    df = pd.read_parquet(path)
    n = len(df)
    k = int(df["forced_close"].sum())
    frac = k / n if n else float("nan")
    print(f"cutoff={args.cutoff:g} condition={args.condition}: forced_close = {frac:.3f} ({frac:.1%}, {k}/{n})")


if __name__ == "__main__":
    main()
