#!/usr/bin/env python
"""Build the SCoT question table from the Arcuschin/chainscope dataset.

We consume the ``non-ambiguous-hard-2`` subset of the World-Model comparison
questions (Arcuschin et al., "Chain-of-Thought Reasoning in the Wild Is Not
Always Faithful" — the main paper dataset: 29 prop_ids, 4834 question pairs =
9668 questions, 16 models). The aggregated pickle
(``df-wm-non-ambiguous-hard-2.pkl.gz``) holds per-question×model CoT eval stats
but is missing the field the SCoT sprint needs — ``q_str_open_ended``, the
entity-answer phrasing ("Which work has more pages: X or Y?"). That field lives
in the per-dataset YAML question files, so those YAMLs are the source of truth
here; the pickle is used only to cross-validate that we recovered exactly the
right qid set.

Output: one Parquet row per question (deduplicated to the unique ``qid``; the
pickle's per-model rollout stats are about *other* models and irrelevant to a
sprint where we run DeepSeek-R1-Distill-Llama-8B ourselves). Each row carries
the open-ended question plus the entity names/values and the derived
correct/incorrect answer entity. We deliberately include more columns than
strictly necessary.

Run from the repo root (so the default relative paths resolve):

    uv run scripts/build_scot_dataset.py

See ``SCOTSPRINT.md`` for how this table feeds the experiment.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

# The four question variants are one folder each, named
# ``{comparison}_{expected_answer}_{num_comparisons}``.
VARIANT_RE = re.compile(r"^(?P<comparison>gt|lt)_(?P<answer>YES|NO)_(?P<ncmp>\d+)$")
# Optional "about <topic>:" preamble that prefixes the wm questions.
PREAMBLE_RE = re.compile(r"^\s*(about [^:\n]+):\s*\n", re.IGNORECASE)


def _topic(q_str: str) -> str | None:
    "The 'about <topic>:' preamble of a question, if present."
    m = PREAMBLE_RE.match(q_str or "")
    return m.group(1).strip() if m else None


def _rows_from_yaml(path: Path) -> list[dict]:
    """Parse one chainscope question YAML into per-question row dicts."""
    doc = yaml.safe_load(path.read_text())
    params = doc["params"]
    comparison = params["comparison"]        # gt | lt
    expected = params["answer"]              # YES | NO  (answer to the Yes/No q_str)
    variant = path.parent.name               # e.g. gt_YES_1
    dataset_id = path.stem                   # filename without .yaml

    rows = []
    for qid, q in doc["question_by_qid"].items():
        x_name, y_name = q["x_name"], q["y_name"]
        x_value, y_value = q.get("x_value"), q.get("y_value")

        # The open-ended answer is the entity that wins the comparison.
        # Answer-derived: YES => the first-named entity (x), NO => the second (y).
        # This holds regardless of gt/lt because q_str_open_ended is phrased to
        # match the comparison direction. We cross-check it against the values.
        correct_side = "x" if expected == "YES" else "y"
        # Value-derived (independent check): gt => larger value wins, lt => smaller.
        value_side: str | None = None
        if x_value is not None and y_value is not None and x_value != y_value:
            x_wins = (x_value > y_value) if comparison == "gt" else (x_value < y_value)
            value_side = "x" if x_wins else "y"

        def pick(side: str, xv, yv):
            return xv if side == "x" else yv

        rows.append({
            "qid": qid,
            "dataset_id": dataset_id,
            "variant": variant,
            "prop_id": params["prop_id"],
            "comparison": comparison,
            "expected_answer": expected,
            "suffix": params.get("suffix"),
            "uuid": params.get("uuid"),
            "topic": _topic(q.get("q_str", "")),
            "q_str": q.get("q_str"),
            "q_str_open_ended": q.get("q_str_open_ended"),
            "x_name": x_name,
            "y_name": y_name,
            "x_value": x_value,
            "y_value": y_value,
            "correct_side": correct_side,
            "correct_name": pick(correct_side, x_name, y_name),
            "correct_value": pick(correct_side, x_value, y_value),
            "incorrect_side": "y" if correct_side == "x" else "x",
            "incorrect_name": pick("y" if correct_side == "x" else "x", x_name, y_name),
            "incorrect_value": pick("y" if correct_side == "x" else "x", x_value, y_value),
            "value_side": value_side,
            "polarity_ok": (value_side is None) or (value_side == correct_side),
            "value_diff": (abs(x_value - y_value)
                           if x_value is not None and y_value is not None else None),
            "source_path": str(path),
        })
    return rows


def build(questions_dir: Path, suffix: str) -> pd.DataFrame:
    yamls = sorted(questions_dir.glob(f"*/*_{suffix}.yaml"))
    if not yamls:
        raise FileNotFoundError(
            f"No '*_{suffix}.yaml' files under {questions_dir}. "
            "Is the chainscope repo cloned? (see README)")

    bad_variants = [p for p in yamls if not VARIANT_RE.match(p.parent.name)]
    if bad_variants:
        raise ValueError(f"Unexpected variant folder(s): {bad_variants[:3]}")

    rows = [r for p in yamls for r in _rows_from_yaml(p)]
    df = pd.DataFrame(rows).sort_values(["dataset_id", "qid"]).reset_index(drop=True)

    if df["qid"].duplicated().any():
        dups = int(df["qid"].duplicated().sum())
        raise ValueError(f"{dups} duplicate qid(s) across YAML files — unexpected.")
    return df


def cross_check_pickle(df: pd.DataFrame, pickle_path: Path, suffix: str) -> None:
    """Best-effort: confirm our qid set matches the aggregated pickle's subset."""
    if not pickle_path.exists():
        print(f"[cross-check] pickle not found at {pickle_path}; skipping.")
        return
    agg = pd.read_pickle(pickle_path)
    agg = agg[agg["dataset_suffix"] == suffix]
    pkl_qids, our_qids = set(agg["qid"]), set(df["qid"])
    print(f"[cross-check] pickle unique qids: {len(pkl_qids)}; ours: {len(our_qids)}")
    if pkl_qids == our_qids:
        print("[cross-check] OK — qid sets match exactly.")
    else:
        print(f"[cross-check] MISMATCH — only in pickle: {len(pkl_qids - our_qids)}, "
              f"only in ours: {len(our_qids - pkl_qids)}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions-dir", type=Path,
                    default=repo_root / "chainscope/chainscope/data/questions",
                    help="chainscope questions dir (contains gt_YES_1/, ...).")
    ap.add_argument("--suffix", default="non-ambiguous-hard-2",
                    help="dataset suffix to select (default: the main paper subset).")
    ap.add_argument("--pickle", type=Path,
                    default=repo_root / "chainscope/chainscope/data/df-wm-non-ambiguous-hard-2.pkl.gz",
                    help="aggregated pickle, used only to cross-validate the qid set.")
    ap.add_argument("--out", type=Path, default=repo_root / "data/wm-non-ambiguous-hard-2.parquet",
                    help="output Parquet path.")
    args = ap.parse_args()

    df = build(args.questions_dir, args.suffix)
    print(f"Parsed {len(df)} questions "
          f"({df['qid'].nunique()} unique qids, "
          f"{df['dataset_id'].nunique()} datasets, "
          f"{df['prop_id'].nunique()} prop_ids).")

    n_bad = int((~df["polarity_ok"]).sum())
    if n_bad:
        print(f"WARNING: {n_bad} questions where answer-derived correct entity "
              "disagrees with value-derived. Inspect 'polarity_ok'==False rows.")
    else:
        print("Polarity check OK — answer- and value-derived correct entities agree "
              "for every question with comparable values.")

    cross_check_pickle(df, args.pickle, args.suffix)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Wrote {args.out} ({len(df)} rows, {len(df.columns)} columns).")


if __name__ == "__main__":
    main()
