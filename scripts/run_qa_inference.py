"""Run the chainscope comparison dataset through the reasoning model, caching each
question's chain of thought and final-answer logprobs.

For a seeded random sample of ``--limit`` usable rows (or the whole usable set), this:

  1. drops first-token-collision / unalignable answer pairs (``answer_tokens_collide``);
  2. builds a boxed-answer reasoning prompt per question
     (``build_boxed_answer_instruction`` in question order → ``build_reasoning_prompt``);
  3. rolls out a chain of thought per question, batched, with a token budget and a
     force-closed ``</think>`` (``rollout_cot``); and
  4. reads the first-token logits/logprobs of the two candidate answers plus the
     whole-vocab top token at the ``\\boxed{`` read point (``batched_answer_scores``).

Results are written to a single Parquet file at ``--out``, rewritten after each batch
so a long run is **resumable**: re-running loads what's there and skips rows whose
``qid`` is already present. This file is the cached corpus the injection step later
corrupts (see ``SCOTSPRINT.md`` §4).

Run (defaults to DeepSeek-R1-Distill-Llama-8B):

    uv run scripts/run_qa_inference.py --limit 8 --batch-size 4 --out /tmp/qa_smoke.parquet
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        uv run scripts/run_qa_inference.py --limit 2000 --batch-size 48 --out data/qa_inference.parquet

``--limit`` takes a **seeded random sample** (``--seed``); omit it to process every
usable row. ``--seed`` also seeds generation (per-batch). Resume is by ``qid``, so point
a new configuration (different ``--data/--limit/--seed``) at a fresh ``--out``.
"""

import argparse
import os

# Reduce CUDA fragmentation for the long left-padded generation batches. Must be set
# before torch initializes CUDA, so do it before importing torch (via subliminality).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast  # noqa: E402

from subliminality import (  # noqa: E402
    DEFAULT_THINK_BUDGET,
    answer_candidate_token,
    answer_scaffold_ids,
    answer_tokens_collide,
    batched_answer_scores,
    build_boxed_answer_instruction,
    build_reasoning_prompt,
    get_device,
    rollout_cot,
)

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DEFAULT_DATA = "data/wm-non-ambiguous-hard-2.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model id (default: %(default)s).")
    p.add_argument("--data", default=DEFAULT_DATA, help="Input Parquet (default: %(default)s).")
    p.add_argument("--out", default="data/qa_inference.parquet",
                   help="Output Parquet file, rewritten after each batch (default: %(default)s).")
    p.add_argument("--limit", type=int, default=None,
                   help="Seeded random sample of this many usable rows (default: all).")
    p.add_argument("--batch-size", type=int, default=48,
                   help="Prompts per generation batch (default: %(default)s).")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_THINK_BUDGET,
                   help="CoT token budget (default: %(default)s).")
    p.add_argument("--seed", type=int, default=47,
                   help="Seed for the row sample and generation (default: %(default)s).")
    p.add_argument("--device", default=None, help="Device preference for get_device (default: auto).")
    return p.parse_args()


def select_rows(df: pd.DataFrame, tokenizer, limit: int | None, seed: int) -> pd.DataFrame:
    "Drop collision/unalignable answer pairs, then take a seeded sample of `limit` rows."
    collide = df.apply(lambda r: answer_tokens_collide(tokenizer, r.correct_name, r.incorrect_name), axis=1)
    usable = df[~collide].reset_index(drop=True)
    print(f"{len(df)} rows -> {len(usable)} usable ({100 * collide.mean():.1f}% dropped by collision filter)")
    if limit is not None and limit < len(usable):
        usable = usable.sample(limit, random_state=seed).reset_index(drop=True)
        print(f"sampled {len(usable)} rows (seed={seed})")
    return usable


def process_batch(chunk: pd.DataFrame, batch_index: int, *, model, tokenizer,
                  end_think_id: int, answer_ids, max_new_tokens: int, seed: int) -> pd.DataFrame:
    "Rollout + answer-read one batch of rows; return a DataFrame of cached records."
    prompts = [build_reasoning_prompt(tokenizer, build_boxed_answer_instruction(
        r.q_str_open_ended, [r.x_name, r.y_name])) for r in chunk.itertuples()]
    rollouts = rollout_cot(model, tokenizer, prompts, end_think_id=end_think_id,
                           max_new_tokens=max_new_tokens, seed=seed + batch_index)
    candidates = [(answer_candidate_token(tokenizer, r.correct_name),
                   answer_candidate_token(tokenizer, r.incorrect_name)) for r in chunk.itertuples()]
    scores = batched_answer_scores(model, [ro.full_ids for ro in rollouts], candidates,
                                   pad_id=tokenizer.eos_token_id, answer_ids=answer_ids)

    records = []
    for row, ro, sc in zip(chunk.itertuples(), rollouts, scores):
        cor_id, inc_id = sc.candidate_ids  # order is (correct, incorrect)
        records.append({
            "qid": row.qid, "prop_id": row.prop_id, "comparison": row.comparison,
            "expected_answer": row.expected_answer, "value_diff": row.value_diff, "topic": row.topic,
            "correct_name": row.correct_name, "incorrect_name": row.incorrect_name,
            "correct_token_id": cor_id, "incorrect_token_id": inc_id,
            "logit_correct": sc.logits[0], "logit_incorrect": sc.logits[1],
            "logprob_correct": sc.logprobs[0], "logprob_incorrect": sc.logprobs[1],
            "logprob_diff": sc.logprob_diff,  # correct - incorrect
            "top_token_id": sc.top_token_id,
            "top_token_str": tokenizer.convert_ids_to_tokens([sc.top_token_id])[0],
            "top_logprob": sc.top_logprob,
            "winner": "correct" if sc.argmax == 0 else "incorrect",
            "model_correct": sc.argmax == 0,
            "top_is_correct": sc.top_token_id == cor_id,
            "top_is_candidate": sc.top_token_id in (cor_id, inc_id),
            "think_text": ro.think_text,
            "think_token_ids": list(ro.think_ids),
            "n_think_tokens": len(ro.think_ids),
            "forced_close": ro.forced_close,
            "seed": seed, "batch_index": batch_index,
        })
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"device={device} | model={args.model} | out={out}")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model)
    assert tokenizer("a b").input_ids != tokenizer("ab").input_ids, "tokenizer dropped the space (see #45488)"
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map=device, dtype=dtype).eval()
    end_think_id = tokenizer.convert_tokens_to_ids("</think>")
    answer_ids = answer_scaffold_ids(tokenizer)

    df = pd.read_parquet(args.data)
    rows = select_rows(df, tokenizer, args.limit, args.seed)
    chunks = [rows.iloc[i:i + args.batch_size] for i in range(0, len(rows), args.batch_size)]
    print(f"{len(rows)} rows -> {len(chunks)} batches of up to {args.batch_size}")

    # Resume: load any prior output and skip batches whose rows are all already done.
    done = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    done_qids = set(done.qid) if len(done) else set()
    collected = [done] if len(done) else []
    if done_qids:
        print(f"resuming: {len(done_qids)} rows already in {out}")

    bar = tqdm(list(enumerate(chunks)), desc="batches")
    for i, chunk in bar:
        if set(chunk.qid) <= done_qids:
            continue  # whole batch already done
        recs = process_batch(chunk, i, model=model, tokenizer=tokenizer, end_think_id=end_think_id,
                              answer_ids=answer_ids, max_new_tokens=args.max_new_tokens, seed=args.seed)
        collected.append(recs)
        full = pd.concat(collected, ignore_index=True)
        full.to_parquet(out)  # single file, rewritten each batch (resumable, no shards)
        bar.set_postfix(rows=len(full), forced=f"{full.forced_close.mean():.0%}",
                        acc=f"{full.model_correct.mean():.0%}")

    full = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    print(f"\nwrote {len(full)} rows -> {out}")
    if len(full):
        print(f"  accuracy (winner==correct): {full.model_correct.mean():.1%}")
        print(f"  top token is a candidate:   {full.top_is_candidate.mean():.1%}  "
              f"(top is correct: {full.top_is_correct.mean():.1%})")
        print(f"  force-closed </think>:      {full.forced_close.mean():.1%}")
        print(f"  mean logprob_diff (cor-inc): {full.logprob_diff.mean():.3f}")


if __name__ == "__main__":
    main()
