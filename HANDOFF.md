# Handoff — current state (working snapshot)

Transient working-state doc. Durable guidance lives in `CLAUDE.md`; this is the "where we are /
what's next" snapshot and can be deleted once the DeepSeek gen-think run is done.

## TL;DR — the one thing left to do
**Run the DeepSeek section of `notebooks/subliminal_prompting_demo.ipynb` end-to-end.** The
cross-number batching it needs is now **implemented and verified** (see below), so the two
generated-think tables should finish in ~minutes, not ~30–40 min. The section's outputs are
currently **cleared** (the run was deferred because the H100 was occupied by other GPU sessions
holding ~70 GiB — DeepSeek wouldn't fit alongside the 1B+8B models). When the GPU is free:

1. `uv run pytest` — should be **26 passing** (env sanity).
2. Open the notebook and run it top to bottom (or just re-run from the Reasoning models section;
   1B/8B outputs above are already fresh from the batched code). DeepSeek is gated-free; the Llamas
   need HF auth (`HF_TOKEN` / `huggingface-cli login`).
3. The DeepSeek load needs ~16 GiB + a batch-63 KV cache (~7 GiB); make sure ~24 GiB is free.

## What changed this session (batching — DONE)
The slow path was `measure_ds_gen` running **one `model.generate` per (animal, number)** = 210
serial generations per gen-think table. Now batched per-animal: **1 generate + 1 read per animal**.

- **New library module `src/subliminality/generation.py`** (exported via `__init__.__all__`):
  - `batched_answer_probs(model, prefixes, target, *, pad_id, answer_ids=())` — left-pads ragged
    prefixes, appends a shared answer, one forward, reads `P(target)` at `-1` (float32 softmax).
    Derives `position_ids` from the attention mask, so it's correct for both RoPE (Llama/DeepSeek)
    and learned-positional (GPT-2) models.
  - `batched_generate_truncated(model, prompts, *, stop_id, pad_id, gen_kwargs, n_samples, seed)` —
    one left-padded batched `generate`, each output truncated at the first `stop_id` (`</think>`).
- **`tests/test_generation.py`** (distilgpt2, CPU): `batched_answer_probs` is `allclose` to the
  unpadded single-sequence loop (the deterministic guard on the left-padding/masking/position_ids);
  plus shape/truncation checks for `batched_generate_truncated`.
- **Notebook**: the `measure` contract is now **batched** — `measure(animal, instructions, *, model,
  tokenizer) -> list[float]` (instructions = `[None] + top-k + bottom-k`; `None` = base).
  `animal_entanglement_table`, `query_animal_preference`, `measure_ds_empty`, `measure_ds_gen`
  rewritten on the primitives; the two gen-think cells are **uncommented**.
- **Seeding note (behavioral):** `measure_ds_gen` now seeds once per animal-batch (was per
  (animal, number)). Still reproducible; the sampled draws — hence the gen-think probabilities —
  differ from a hypothetical per-call run. Neither is "more correct."

## Verification done
- `uv run pytest` → **26 passed** (21 prior + 5 new).
- Real-DeepSeek smoke test (batched gen path, tiny `max_new_tokens`) — passed.
- The **1B and 8B** sections re-ran under the batched contract and reproduce the prior committed
  numbers within bf16 batched-vs-single noise (e.g. 1B-ratio quokka uplift 30.75 → 30.86). Those
  fresh outputs are committed; the DeepSeek section is cleared, awaiting the run above.

## Where we are (library)
- Public API (`__init__.__all__`): `get_device`, `seed_everything`, `compute_entanglements`,
  **`batched_answer_probs`**, **`batched_generate_truncated`**, `SENTINEL`, `build_input_ids`,
  `first_token`, `is_number`, `token_mask`, `top_bottom`.
- `scripts/perf_entanglement.py` now imports `seed_everything` from the library (its private copy
  was removed) — the duplicate-seed cleanup is done.

## DeepSeek specifics (also in CLAUDE.md)
- **Tokenizer via `PreTrainedTokenizerFast.from_pretrained(...)`** — `AutoTokenizer` drops spaces
  under transformers v5 (bug [#45488](https://github.com/huggingface/transformers/issues/45488)).
  Check: `tok("a b").input_ids != tok("ab").input_ids`.
- Instructions in the **user** turn (no system prompt); `add_generation_prompt=True` opens
  `<think>\n`, then raw text is appended; `</think>`=128014.

## Open items / decisions pending
- **Cross-animal batching (further, optional).** We batch per-animal (63 seqs/call). Batching all
  10 animals into one ~630-seq generate is possible but was deliberately not done — OOM risk on a
  shared GPU, padding waste from length variance, and it needs per-sequence targets. Only worth it
  if the per-animal run is still too slow, which it shouldn't be.
- **Performance knobs** for the gen-think run: `max_new_tokens=512` (caps think length — biggest
  lever) and `n_samples=3` (already batched, ~free).
