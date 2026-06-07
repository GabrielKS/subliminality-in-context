# Handoff — DeepSeek run COMPLETE (working snapshot)

Transient working-state doc. Durable guidance lives in `CLAUDE.md`. The DeepSeek gen-think run is
now **done**, so the original purpose of this file is fulfilled — it can be deleted, or its findings
folded into a proper writeup. Kept here as the record of what the run showed.

## Status: done
`notebooks/subliminal_prompting_demo.ipynb` is **fully executed end-to-end** (committed in
"Update notebook with new results!"). All four DeepSeek tables — ratio/unembedding discovery ×
empty-think/generated-think measurement — have outputs. `uv run pytest` → 26 passing.

## The batching paid off (the point of this work)
Per-animal batching (1 `generate` + 1 read per animal, batch = 21 instructions × 3 samples = 63
sequences) replaced the old 210 serial generations per table:

| gen-think table | wall-clock | per animal |
|---|---|---|
| ratio discovery | **4:00** | 24 s |
| unembedding discovery | **3:31** | 21 s |

vs. the previous estimate of ~30–40 min per table — roughly a **10× speedup**, realized. (Empty-think
tables are ~free: 2.5 s and <1 s.) GPU during the run: ~97% SM, power-capped at 310 W — i.e. near
the hardware envelope; further gains would come from more power budget, not code.

## Findings
Headline metric: **geomean uplift (top-k)** = outlier-robust "typical" uplift of the target animal's
probability when the model is told to love its top-k entangled numbers, normalized by the
unconditioned base prob. `top-k vs bottom-k` tells whether *entanglement ranking* matters (vs.
loving any number at all).

| discovery × condition | geomean uplift top-k | bottom-k |
|---|---|---|
| ratio × empty-think | 0.29 | 0.28 |
| ratio × **gen-think** | 0.58 | 1.03 |
| unembedding × empty-think | 0.30 | 0.28 |
| unembedding × **gen-think** | **1.37** | 0.84 |

1. **A real (generated) think block matters.** Across both discovery methods, gen-think uplifts are
   ~2–4× the empty-think ones. Empty-think shows uplift well **below 1** (loving the numbers slightly
   *suppresses* the animal) with no top/bottom discrimination; gen-think pushes the typical uplift up
   to ~0.6–1.4. So *whether the reasoning model actually thinks* changes the picture — which was the
   question this run existed to answer.
2. **But the effect is weak and inconsistent.** Even in gen-think, the top-k vs bottom-k ordering
   **flips** between discovery methods (ratio: top < bottom; unembedding: top > bottom), and only
   unembedding×gen-think clears geomean uplift > 1 (1.37, median 1.20). No robust "the *entangled*
   numbers specifically raise the target" signal. Consistent with the overall project conclusion that
   naive entanglement → subliminal prompting is weak once you divide out the base prob.
3. **`panda` dominates DeepSeek** (base 0.199 empty / 0.002 gen — the model's strong default animal),
   inflating several arithmetic-mean rows; the geomean/median rows are the ones to trust.

Reproducibility: gen-think tables use `n_samples=3`, `max_new_tokens=512`, `seed=0` (seeded once per
animal-batch — see the note in `measure_ds_gen`).

## Code state (durable bits are in CLAUDE.md)
- Library: `batched_answer_probs`, `batched_generate_truncated` in
  `src/subliminality/generation.py`, exported via `__all__`; covered by `tests/test_generation.py`.
- Notebook: batched `measure` contract — `measure(animal, instructions, *, model, tokenizer) ->
  list[float]`, instructions = `[None] + top-k + bottom-k`. The user added a top-of-notebook cell
  setting CPU thread caps for the cgroup-throttled pod (see the CPU-quota section in CLAUDE.md).

## Optional next steps (none required)
- **Cross-animal batching** would widen the batch beyond 63, but the GPU was already ~97% SM and
  power-capped during the run, so the wall-clock gain would be marginal. Not worth it unless a single
  table feels slow.
- Fold the findings above into a proper writeup and delete this file.
