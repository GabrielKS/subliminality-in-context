# Handoff — current state (working snapshot)

Transient working-state doc for picking up after moving to an **H100**. Durable guidance lives in
`CLAUDE.md`; this file is the "where we are / what's next" snapshot and can be deleted once the
DeepSeek run is done.

## TL;DR — what to do next on the H100
1. Set up the env (`uv sync`); authenticate to Hugging Face for the **gated Llama** models
   (`huggingface-cli login` or `HF_TOKEN`). DeepSeek and distilgpt2 are ungated.
2. `uv run pytest` — should be 21 passing (sanity that the env is good).
3. Open `notebooks/subliminal_prompting_demo.ipynb` and run **the Reasoning models section**
   (DeepSeek). The four DeepSeek table cells have **never been executed** — that's the main goal.
   The two **generated-think** sweeps are the slow part (est. ~30–40 min total on an H100 as
   currently written; see "Performance" below).
4. Decide whether to implement **cross-number batching** first (big H100 speedup) — see "Open".

## Where we are
- **Library `subliminality` is built and tested (21 passing).** Public API (`__init__.__all__`):
  `get_device`, `seed_everything`, `compute_entanglements`, `SENTINEL`, `build_input_ids`,
  `first_token`, `is_number`, `token_mask`, `top_bottom`.
  - `entanglement.py`: `compute_entanglements(model, tokens, *, method, tokenizer=None, prompt=…,
    return_components=False)` — `"unembedding"` (cosine, no forward) and `"output_distribution"`
    (returns `guided/base` ratio by default; `(guided, base)` with `return_components=True`).
    Prompt assumptions are module constants, overridable via `prompt=`.
  - `tokens.py`: splice/first-token/filter/select primitives + `NUMBER_BLOCKLIST` (default-excluded
    "loaded" numbers, via `is_number(s, blocklist=…)`).
  - `device.py`: `get_device`, `seed_everything`.
- **Notebook `notebooks/subliminal_prompting_demo.ipynb`** runs end-to-end through the 8B section
  (outputs present, reproduce prior numbers). Sections: Basic reproduction (`087`→`owl`) →
  Computing entangled tokens (both methods, owl) → animal sweep tables (1B) → A Larger Model (8B)
  → **Reasoning models (DeepSeek)** ← new, not yet executed.
- **DeepSeek section cells** (in order): model load (uses `PreTrainedTokenizerFast` — see below) →
  helpers (`deepseek_entangle_prompt`, `ds_guided_scores`, `ds_ratio_scores`, `_ds_thinking_prefix`,
  `measure_ds_empty`, `measure_ds_gen`) → 4 table cells:
  `ratio×empty-think`, `ratio×gen-think`, `unembedding×empty-think`, `unembedding×gen-think`.
- `animal_entanglement_table(score_fn, …, measure=…)` was generalized to take a `measure` callable
  so any model/prompting style plugs in. DeepSeek uses `measure_ds_empty` / `measure_ds_gen`.

## DeepSeek specifics (also in CLAUDE.md)
- **Tokenizer must load via `PreTrainedTokenizerFast.from_pretrained(...)`** — `AutoTokenizer`
  silently drops spaces under transformers v5 (bug
  [#45488](https://github.com/huggingface/transformers/issues/45488); Metaspace overrides ByteLevel).
  Already done in the load cell. Quick check: `tok("a b").input_ids != tok("ab").input_ids`.
- Prompting: instructions in the **user** turn (no system prompt); `add_generation_prompt=True`
  opens `<think>\n`, then we append raw text; `</think>`=128014.
- `measure_ds_gen` currently batches the `n_samples=3` traces into one `generate` call
  (`num_return_sequences`); reads after `</think>` are looped.

## Performance / the slow run
The `ratio×gen-think` table = **210 `generate` calls** (10 animals × (1 base + 10 top + 10 bottom)),
each a batched-3 generation up to `max_new_tokens=512`. The `unembedding×gen-think` table is another
210. Empty-think tables are cheap (one forward per measurement). Rough estimates (HF `transformers`
`generate`, 8B bf16):
- **M5 Pro (MPS):** ~2–6 h per gen-think table.
- **H100:** ~30–40 min per gen-think table as written.
- Knobs: `max_new_tokens` (caps think length — biggest lever), `n_samples` (already batched, ~free).

## Open items / decisions pending
- **Cross-number batching (the big H100 win, NOT implemented).** Currently `measure` is
  per-instruction, so generations run one (animal, number) at a time. Batching all ~20 numbers
  (× n_samples) per animal with **left-padded** batched generation + per-sequence `</think>`
  truncation would cut the H100 gen-think run to ~a few minutes (decode is bandwidth-bound; many
  sequences decode in ~the same wall-clock as one). Needs: a batch-of-instructions `measure`
  interface (touches the table + the Instruct `query_animal_preference` path) and careful
  left-padding/attention-mask/chunking. We deliberately did *not* ship this untested into a long
  run; suggested approach is to write it then smoke-test with tiny `max_new_tokens` + 2 animals
  before the full sweep.
- **Duplicate `seed_everything`.** `scripts/perf_entanglement.py` has its own `seed_everything`
  that now duplicates `subliminality.seed_everything`. Per the "library is single source of truth"
  convention the script should import it from the library (signature differs slightly: the script's
  takes `(seed, device)`). Minor cleanup, deprioritized earlier.

## Key decisions/findings so far (so we don't relitigate)
- Entanglement: **first-token handle** for multi-token words (`" " + word`); acknowledged leaky
  (`sea turtle`/`sea otter` collide on ` sea`).
- Output-distribution: the **guided/base ratio** beats guided alone (notebook conclusion), so the
  DeepSeek discovery uses `ds_ratio_scores`.
- Number filtering excludes a `NUMBER_BLOCKLIST` of culturally-loaded numbers by default.
- Tables report base prob + uplift, with **mean / geomean / median** rows; geomean is the
  outlier-robust "typical uplift" (arithmetic mean is dominated by low-base animals).
- Probes read raw logits in **float32** (bf16 underflows tails); temperature only matters for the
  gen-think generation, not the probes.
- Across Llama 1B/8B the entanglement→animal effect is weak/baseline-dominated once you divide by
  the unconditioned base prob; the DeepSeek run is to see whether a reasoning model differs and
  whether a real think block changes the picture vs an empty one.
