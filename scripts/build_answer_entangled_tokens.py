#!/usr/bin/env python
"""Precompute number tokens entangled with each comparison answer entity.

For every question in ``data/wm-non-ambiguous-hard-2.parquet`` we resolve each
entity's *answer-candidate token* — the exact token the reasoning model emits for
that entity at the ``\\boxed{`` read point
(:func:`subliminality.answer_candidate_token`) — and find the top-10 / bottom-10
**number** tokens entangled with it by **unembedding cosine similarity**
(:func:`subliminality.compute_entanglements` with ``method="unembedding"``,
masked to number tokens). These are the candidate steering tokens the SCoT sprint
splices into the chain of thought (see ``SCOTSPRINT.md``); precomputing them once
lets the experiment read them off a table instead of recomputing per run.

Mirrors the entangled-token prior work in
``notebooks/subliminal_prompting_demo.ipynb`` (` owl` → top/bottom digit tokens).
Loads ``deepseek-ai/DeepSeek-R1-Distill-Llama-8B`` for its unembedding matrix
only — **no** generation/forward passes — so it is fast and GPU-light.

Run from the repo root:

    uv run scripts/build_answer_entangled_tokens.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from subliminality import (
    answer_candidate_token,
    answer_tokens_collide,
    compute_entanglements,
    get_device,
    is_number,
    token_mask,
)

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"


def load_model(model_id: str, device):
    """Load the reasoning model + tokenizer (cf. notebooks/demo_basic_qa.ipynb)."""
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
    # #45488: AutoTokenizer would silently drop spaces; PreTrainedTokenizerFast
    # respects tokenizer.json's ByteLevel pre-tokenizer. Verify it round-trips.
    assert tokenizer("a b").input_ids != tokenizer("ab").input_ids, \
        "tokenizer dropped the space (see transformers #45488)"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map=device, dtype=torch.bfloat16).eval()
    return model, tokenizer


def entangled_numbers(model, tokenizer, candidate_ids: list[int], digits: torch.Tensor,
                      *, topk: int, bottomk: int, chunk_size: int) -> dict[int, dict]:
    """Map each candidate token id -> {"top": [...], "bottom": [...]} number tokens.

    Each list entry is ``{"token_id", "token", "score"}`` (cosine similarity),
    ranked over **number** tokens only (``digits`` mask). Computed in chunks to
    bound the transient ``[chunk, vocab]`` cosine tensor; ``top_bottom``
    (tokens.py) only returns strings, so we take the masked top-k directly here to
    keep the token *ids* needed for exact downstream injection.
    """
    keep = digits.to(model.get_output_embeddings().weight.device)
    result: dict[int, dict] = {}

    def entries(vals: torch.Tensor, idx: torch.Tensor) -> list[dict]:
        ids = idx.tolist()
        toks = tokenizer.convert_ids_to_tokens(ids)
        return [{"token_id": int(i), "token": t, "score": float(v)}
                for i, t, v in zip(ids, toks, vals.tolist())]

    for start in range(0, len(candidate_ids), chunk_size):
        chunk = candidate_ids[start:start + chunk_size]
        sim = compute_entanglements(
            model, torch.tensor(chunk), method="unembedding")  # [chunk, vocab], on device
        # Push non-number tokens out of contention (mirrors tokens.top_bottom).
        top_src = sim.masked_fill(~keep, float("-inf"))
        bot_src = sim.masked_fill(~keep, float("inf"))
        top_vals, top_idx = top_src.topk(topk, dim=-1, largest=True)
        bot_vals, bot_idx = bot_src.topk(bottomk, dim=-1, largest=False)
        for row, tok_id in enumerate(chunk):
            result[tok_id] = {
                "top": entries(top_vals[row], top_idx[row]),
                "bottom": entries(bot_vals[row], bot_idx[row]),
            }
    return result


def build(df: pd.DataFrame, model, tokenizer, *, topk: int, bottomk: int,
          chunk_size: int) -> pd.DataFrame:
    # 1. Resolve per-entity candidate tokens + the pair-collision flag.
    x_cand = [answer_candidate_token(tokenizer, n) for n in df["x_name"]]
    y_cand = [answer_candidate_token(tokenizer, n) for n in df["y_name"]]
    collide = [answer_tokens_collide(tokenizer, x, y)
               for x, y in zip(df["x_name"], df["y_name"])]

    # 2. Entangled numbers, computed once per unique non-None candidate id.
    unique_ids = sorted({c for c in (*x_cand, *y_cand) if c is not None})
    n_none = sum(c is None for c in (*x_cand, *y_cand))
    print(f"{len(df)} questions; {len(unique_ids)} unique candidate tokens; "
          f"{n_none} unresolved (None) candidates; {sum(collide)} colliding pairs.")
    digits = token_mask(tokenizer, is_number)
    print(f"number-token vocab size (after NUMBER_BLOCKLIST): {int(digits.sum())}")
    ent = entangled_numbers(model, tokenizer, unique_ids, digits,
                            topk=topk, bottomk=bottomk, chunk_size=chunk_size)

    def top(c, key):  # [] for unresolved candidates
        return ent[c][key] if c is not None else []

    return pd.DataFrame({
        "qid": df["qid"].to_numpy(),
        "x_name": df["x_name"].to_numpy(),
        "y_name": df["y_name"].to_numpy(),
        "x_answer_candidate_token": pd.array(x_cand, dtype="Int64"),
        "x_top10": [top(c, "top") for c in x_cand],
        "x_bottom10": [top(c, "bottom") for c in x_cand],
        "y_answer_candidate_token": pd.array(y_cand, dtype="Int64"),
        "y_top10": [top(c, "top") for c in y_cand],
        "y_bottom10": [top(c, "bottom") for c in y_cand],
        "answer_tokens_collide": collide,
    })


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path,
                    default=repo_root / "data/wm-non-ambiguous-hard-2.parquet")
    ap.add_argument("--out", type=Path,
                    default=repo_root / "data/answer-entangled-tokens.parquet")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--bottomk", type=int, default=10)
    ap.add_argument("--chunk-size", type=int, default=256,
                    help="candidate tokens per unembedding matmul (bounds GPU memory).")
    ap.add_argument("--device", default=None, help="override get_device() (cuda/mps/cpu).")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    device = args.device or get_device()
    print(f"device: {device} | model: {args.model}")
    model, tokenizer = load_model(args.model, device)

    out = build(df, model, tokenizer,
                topk=args.topk, bottomk=args.bottomk, chunk_size=args.chunk_size)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"Wrote {args.out} ({len(out)} rows, {len(out.columns)} columns).")


if __name__ == "__main__":
    main()
