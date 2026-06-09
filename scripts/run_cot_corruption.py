"""Corrupt cached chains of thought by injecting number tokens, then regenerate.

Takes the cached uncorrupted CoTs (``--cots``, an output of ``run_qa_inference.py``),
joins back to the source questions (``--source``) and the per-question entangled-number
table (``--entangled``, from ``build_answer_entangled_tokens.py``), and for each row:

  1. picks a pool of number tokens per ``--tokens-source`` (entangled top/bottom-10 for
     the correct/incorrect entity — mapped to the x/y side by ground truth — or random
     ``is_number`` tokens for a control);
  2. truncates the CoT to ``--cutoff-frac`` and splices one sampled pool token in at each
     sentence boundary (``build_injection_prefill``, exact-id splice);
  3. regenerates the rest (``rollout_cot``) and re-reads the two-answer logprobs at the
     ``\\boxed{`` scaffold (``batched_answer_scores``), reusing the baseline's candidate
     tokens so the corrupted ``logprob_diff`` is directly comparable.

Output is a single (unsharded) Parquet shaped like the ``--cots`` table — recomputed for
the regenerated trace — plus corruption columns (``cutoff_frac``, ``tokens_source``,
``n_injected``, ``pool_token_ids``, …) and the ``baseline_*`` values for easy deltas. It
is rewritten after each batch and resumes by ``qid``.

Run (entangled-toward-correct vs. a random control, at 50% cutoff):

    uv run scripts/run_cot_corruption.py --cots data/qa_inference_small.parquet \\
        --cutoff-frac 0.5 --tokens-source incorrect_top10
    uv run scripts/run_cot_corruption.py --cots data/qa_inference_small.parquet \\
        --cutoff-frac 0.5 --tokens-source random_numerical --limit 100

``--cutoff-frac`` and ``--tokens-source`` are required; the default ``--out`` is
``data/corrupted/corrupted_{cutoff_frac}_{tokens_source}.parquet``.
"""

import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import random  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast  # noqa: E402

from subliminality import (  # noqa: E402
    DEFAULT_THINK_BUDGET,
    answer_scaffold_ids,
    batched_answer_scores,
    build_boxed_answer_instruction,
    build_injection_prefill,
    build_reasoning_prompt,
    get_device,
    is_number,
    rollout_cot,
    token_mask,
)

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
TOKENS_SOURCES = ["correct_top10", "correct_bottom10", "incorrect_top10",
                  "incorrect_bottom10", "random_numerical"]
POOL_SIZE = 10  # matches the entangled top/bottom-10 pools


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cots", default="data/qa_inference.parquet",
                   help="Uncorrupted CoTs to corrupt (run_qa_inference output; default: %(default)s).")
    p.add_argument("--source", default="data/wm-non-ambiguous-hard-2.parquet",
                   help="Source question table (default: %(default)s).")
    p.add_argument("--entangled", default="data/answer-entangled-tokens.parquet",
                   help="Per-question entangled-number table (default: %(default)s).")
    p.add_argument("--cutoff-frac", type=float, required=True,
                   help="Trace fraction to keep before the (injected) cutoff, 0..1 (required).")
    p.add_argument("--tokens-source", required=True, choices=TOKENS_SOURCES,
                   help="Which number tokens to inject (required).")
    p.add_argument("--out", default=None,
                   help="Output Parquet (default: data/corrupted/corrupted_{cutoff}_{source}.parquet).")
    p.add_argument("--limit", type=int, default=None,
                   help="Seeded random sample of this many usable rows (default: all).")
    p.add_argument("--batch-size", type=int, default=48, help="Prompts per batch (default: %(default)s).")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_THINK_BUDGET,
                   help="CoT regeneration budget (default: %(default)s).")
    p.add_argument("--seed", type=int, default=0, help="Seed for sampling/injection/generation (default: %(default)s).")
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model id (default: %(default)s).")
    p.add_argument("--device", default=None, help="Device preference for get_device (default: auto).")
    return p.parse_args()


def row_seed(base: int, qid: str) -> int:
    "A deterministic per-row seed from the run seed and the (hex) qid."
    return (base * 1_000_003 + int(qid[:8], 16)) % (2**31 - 1)


def pool_for_row(row, tokens_source: str, number_ids, seed: int) -> list[int]:
    "The number-token pool to inject for one row, per --tokens-source."
    if tokens_source == "random_numerical":
        return random.Random(seed).sample(number_ids, POOL_SIZE)
    which, rank = tokens_source.split("_")               # ("correct"|"incorrect", "top10"|"bottom10")
    side = row.correct_side if which == "correct" else row.incorrect_side  # "x" or "y"
    return [int(e["token_id"]) for e in getattr(row, f"{side}_{rank}")]


def select_rows(args, tokenizer) -> pd.DataFrame:
    "Join cots ⋈ source ⋈ entangled on qid, drop collided pairs, optionally subsample."
    cots = pd.read_parquet(args.cots)
    src = pd.read_parquet(args.source)[["qid", "q_str_open_ended", "x_name", "y_name",
                                        "correct_side", "incorrect_side"]]
    ent = pd.read_parquet(args.entangled)[["qid", "x_top10", "x_bottom10", "y_top10",
                                           "y_bottom10", "answer_tokens_collide"]]
    df = cots.merge(src, on="qid").merge(ent, on="qid")
    df = df[~df.answer_tokens_collide].reset_index(drop=True)
    print(f"{len(cots)} cached CoTs -> {len(df)} usable (collision/None pairs dropped)")
    if args.limit is not None and args.limit < len(df):
        df = df.sample(args.limit, random_state=args.seed).reset_index(drop=True)
        print(f"sampled {len(df)} rows (seed={args.seed})")
    return df


def process_batch(chunk: pd.DataFrame, batch_index: int, *, model, tokenizer, end_think_id,
                  answer_ids, args, number_ids) -> pd.DataFrame:
    "Inject + regenerate + re-read one batch; return a DataFrame of corrupted records."
    prefills, pools = [], []
    prompts = []
    for row in chunk.itertuples():
        rs = row_seed(args.seed, row.qid)
        pool = pool_for_row(row, args.tokens_source, number_ids, rs)
        pf = build_injection_prefill(list(row.think_token_ids), insert_ids=pool,
                                     cutoff_frac=args.cutoff_frac, seed=rs,
                                     tokenizer=tokenizer, end_think_id=end_think_id)
        prefills.append(pf)
        pools.append(pool)
        instr = build_boxed_answer_instruction(row.q_str_open_ended, [row.x_name, row.y_name])
        prompts.append(build_reasoning_prompt(tokenizer, instr, prefill=pf.prefill_ids))

    rollouts = rollout_cot(model, tokenizer, prompts, end_think_id=end_think_id,
                           max_new_tokens=args.max_new_tokens, seed=args.seed + batch_index)
    candidates = [(int(row.correct_token_id), int(row.incorrect_token_id)) for row in chunk.itertuples()]
    scores = batched_answer_scores(model, [ro.full_ids for ro in rollouts], candidates,
                                   pad_id=tokenizer.eos_token_id, answer_ids=answer_ids)

    records = []
    for row, ro, sc, pf, pool in zip(chunk.itertuples(), rollouts, scores, prefills, pools):
        cor_id, inc_id = sc.candidate_ids
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
            "think_text": ro.think_text, "think_token_ids": list(ro.think_ids),
            "n_think_tokens": len(ro.think_ids), "forced_close": ro.forced_close,
            # --- corruption metadata ---
            "cutoff_frac": args.cutoff_frac, "tokens_source": args.tokens_source,
            "correct_side": row.correct_side, "n_injected": pf.n_injected,
            "cutoff_token": pf.cutoff_token, "n_sentences": pf.n_sentences,
            "pool_token_ids": [int(t) for t in pool],
            # --- baseline (uncorrupted) for easy deltas ---
            "baseline_logprob_diff": row.logprob_diff, "baseline_model_correct": row.model_correct,
            "seed": args.seed, "batch_index": batch_index,
        })
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    if args.out is None:
        args.out = f"data/corrupted/corrupted_{args.cutoff_frac:g}_{args.tokens_source}.parquet"
    device = get_device(args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"device={device} | model={args.model} | tokens={args.tokens_source} | cutoff={args.cutoff_frac} | out={out}")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model)
    assert tokenizer("a b").input_ids != tokenizer("ab").input_ids, "tokenizer dropped the space (see #45488)"
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map=device, dtype=dtype).eval()
    end_think_id = tokenizer.convert_tokens_to_ids("</think>")
    answer_ids = answer_scaffold_ids(tokenizer)
    number_ids = (torch.nonzero(token_mask(tokenizer, is_number), as_tuple=False).squeeze(-1).tolist()
                  if args.tokens_source == "random_numerical" else None)

    df = select_rows(args, tokenizer)

    # Resume: load prior output and drop already-done rows *before* batching, so a
    # partially-overlapping batch can't re-process (and duplicate) a done qid.
    done = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    done_qids = set(done.qid) if len(done) else set()
    if done_qids:
        df = df[~df.qid.isin(done_qids)].reset_index(drop=True)
        print(f"resuming: {len(done_qids)} rows already in {out}; {len(df)} left to do")
    chunks = [df.iloc[i:i + args.batch_size] for i in range(0, len(df), args.batch_size)]
    print(f"{len(df)} rows -> {len(chunks)} batches of up to {args.batch_size}")

    collected = [done] if len(done) else []
    bar = tqdm(list(enumerate(chunks)), desc="batches")
    for i, chunk in bar:
        recs = process_batch(chunk, i, model=model, tokenizer=tokenizer, end_think_id=end_think_id,
                             answer_ids=answer_ids, args=args, number_ids=number_ids)
        collected.append(recs)
        full = pd.concat(collected, ignore_index=True)
        full.to_parquet(out)  # single file, rewritten each batch (resumable, no shards)
        bar.set_postfix(rows=len(full), forced=f"{full.forced_close.mean():.0%}",
                        acc=f"{full.model_correct.mean():.0%}", inj=f"{full.n_injected.mean():.1f}")

    full = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    print(f"\nwrote {len(full)} rows -> {out}")
    if len(full):
        d = full.logprob_diff - full.baseline_logprob_diff
        print(f"  accuracy: {full.model_correct.mean():.1%} (baseline {full.baseline_model_correct.mean():.1%})")
        print(f"  mean logprob_diff: {full.logprob_diff.mean():+.3f} (baseline {full.baseline_logprob_diff.mean():+.3f}; "
              f"mean Δ = {d.mean():+.3f})")
        print(f"  mean injected/row: {full.n_injected.mean():.1f} | force-closed: {full.forced_close.mean():.1%}")


if __name__ == "__main__":
    main()
