# SCoT Sprint — Subliminal Chain of Thought

**Status:** active sprint plan. Read this first if you're picking up SCoT work.
**Audience:** future agents (and humans) joining the sprint mid-flight.

**Progress so far:**
- ✅ **Question table** — `data/wm-non-ambiguous-hard-2.parquet` (9,668 questions)
  via `scripts/build_scot_dataset.py` (§5).
- ✅ **Per-answer entangled numbers** — `data/answer-entangled-tokens.parquet`
  (keyed by `qid`; top/bottom-10 cosine-entangled number tokens for each entity's
  answer-candidate token) via `scripts/build_answer_entangled_tokens.py` (§5.1).
- ✅ **Reasoning primitives** — `src/subliminality/reasoning_generation.py` +
  baseline `notebooks/demo_basic_qa.ipynb` (roll out a CoT, read the two-answer
  logits at the `\boxed{` scaffold).
- ⏳ **Next** — truncate a baseline trace → splice entangled numbers in as a
  `prefill` → regenerate → compare `logprob_diff` vs. a random-number control (§7).

---

## 1. One-paragraph summary

We're testing whether **numerical tokens injected into a reasoning model's
chain of thought can subliminally steer its final answer** toward the number's
*entangled* concept — here, one of two entities in a comparison question. This
is **Experiment 1** of `subliminality-resources/GabrielKS Technical Writing
Sample -- Subliminal Chain of Thought.md`, adapted to this repo's tooling and
tightened in five ways (Section 3). If injecting numbers entangled with the
*wrong* answer flips the model's answer (or shifts the correct-vs-incorrect
logit gap) significantly more than injecting random numbers, we've **elicited
subliminal chain of thought**.

---

## 2. Where this sits in the literature

Full summaries of all source papers are in
[`subliminality-resources/resources-summary.md`](subliminality-resources/resources-summary.md).
The short version of the lineage this sprint extends:

- **Subliminal learning** (Cloud et al. 2025): fine-tuning a student on a
  teacher's *numbers* transmits the teacher's traits (owl-love, misalignment).
  Signal is model-specific, survives filtering.
- **Token entanglement + subliminal prompting** (Zur et al. 2025, the repo's
  direct parent): the trait↔number link is **token entanglement** (one token's
  representation raises another's probability, partly a softmax-bottleneck
  effect). Putting *one* entangled number in the **system prompt** —
  *subliminal prompting* — reproduces the effect with no fine-tuning. They give
  three detection methods; we use **unembedding cosine** (and maybe **output
  distribution**).
- **CoT (un)faithfulness** (Arcuschin et al. 2025 "in the wild"; Lanham et al.
  taxonomy via the proposal sample): natural-language CoT is not always a
  faithful account of the answer-producing process. *Encoded reasoning* = the
  CoT is a real computation step whose significance to the model ≠ its plain
  meaning. SCoT would be a concrete, **mechanism-known instance** of encoded
  reasoning: meaning smuggled through entangled tokens.
- **Thought anchors** (Bogdan et al. 2025): sentence-level resampling /
  attention tools for *where* in a trace influence lives — relevant if we later
  want to localize or explain the effect (e.g. the distance-decay extension).

SCoT moves the entangled token one more step toward inference time:
**system prompt (Zur) → chain of thought (us).**

This sprint is **Phase 1** of the broader agenda in `GabrielKS Research
Proposal Sample -- Chain of Thought Perturbation and Interpretability.md`.

---

## 3. How this sprint differs from GabrielKS Experiment 1

| # | Writing-sample Exp 1 | This sprint |
|---|---|---|
| 1 | LLM-reword Arcuschin Y/N questions so the answer is an entity | **No rewriting** — chainscope already ships an entity-answer field (`q_str_open_ended`); see §5 |
| 2 | Entangled tokens of any kind | **Numeric tokens only** — the number vocabulary this repo already works with (`is_number` + `NUMBER_BLOCKLIST`) |
| 3 | Zur et al.'s first *two* methods (cosine + logit) | **Unembedding cosine** as the primary metric; **output correlations a possible expansion** |
| 4 | Start with Qwen QwQ-32B | **`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`** (baseline pipeline in `notebooks/demo_basic_qa.ipynb`; see §6) |
| 5 | Track answer-flip rate | Track answer-flip **and** the **logit difference between the first tokens of the correct vs. incorrect answer entities** (a continuous signal, not just a top-1 flip) |

Everything else (inject into the first X% of the trace, random-token control,
distance-decay extension) follows the writing sample.

---

## 4. Experimental design

### 4.1 Core question
Does inserting numbers entangled with the **non-chosen** entity into the early
part of the CoT bias the final answer toward that entity, **more than** random
numbers do?

### 4.2 Procedure (per question)
1. **Baseline rollout.** Prompt DeepSeek-R1-Distill-Llama-8B on the open-ended
   comparison question, generate a real `<think>…</think>` trace, force-close
   if it hits the budget, read the answer (§4.4). This gives (a) the model's
   *naturally chosen* entity and (b) a trace to corrupt.
2. **Pick the target.** The **non-chosen** entity is the steering target (we try
   to flip the model *toward* it). Its top/bottom entangled numbers are
   **precomputed** in `data/answer-entangled-tokens.parquet` (§5.1) — look up the
   non-chosen side's (`x`/`y`) `*_top10` by `qid`; no need to recompute.
3. **Inject.** Truncate the trace to its first **X%** (start **X=50**; the point
   is to intervene *before* the model commits). Insert the top entangled numbers
   into that prefix — start with random-position injection; optionally also an
   LLM-smoothed rewrite variant.
4. **Force-close + read.** `rollout_cot` force-closes `</think>`; then
   `batched_answer_scores` reads the two entities' first-token logits/logprobs at
   the `\boxed{` scaffold (§4.4). Record `logit_diff`/`logprob_diff` and the winner.
5. **Control.** Repeat step 3–4 injecting **random numbers** (same count, same
   positions) instead of entangled ones.
6. **Verdict.** Entangled injection shifting the answer toward the non-chosen
   entity significantly more than the random control ⇒ subliminal CoT elicited.

### 4.3 Metrics
- **Primary:** the signed first-token gap at the `\boxed{` read point —
  `logit(correct) − logit(incorrect)` (or the log-prob version), a continuous
  measure of the model's confidence in the right entity. `batched_answer_scores`
  returns it as `AnswerScores.logit_diff` / `.logprob_diff` when candidates are
  ordered **(correct, incorrect)** — **positive ⇒ favors the correct entity**, so a
  subliminal push toward the wrong entity *lowers* it. Read in float32 (the primitive
  handles it). Compare entangled vs. random control, paired per question.
- **Secondary:** top-1 answer-flip rate (`AnswerScores.argmax`), and **free-choice
  accuracy** from `AnswerScores.top_token_id` — the whole-vocab argmax at the read
  point (regardless of the two candidates): whether the model's actual top token is
  the correct answer / either candidate.
- Aggregate across questions; report effect sizes with CIs, entangled vs. control.

### 4.4 Reading the answer (via `subliminality.reasoning_generation`)
Use the prototyped reasoning primitives — baseline pipeline in
`notebooks/demo_basic_qa.ipynb`:
1. `build_boxed_answer_instruction(question, [x_name, y_name])` constrains the
   answer to the two named entities (Arcuschin-style "YES/NO" analogue, options in
   question order); `build_reasoning_prompt(tok, …)` wraps it into an opened-think
   prompt (no system prompt). For the injection step pass `prefill=<truncated +
   spliced ids>`.
2. `rollout_cot(model, tok, prompts, end_think_id=...)` → generates each CoT
   under the 1024-token budget and **force-closes** `</think>` (returns
   `CoTRollout`s; log the `forced_close` rate).
3. `batched_answer_scores(model, [r.full_ids …], candidates, answer_ids=…)`
   teacher-forces the `\boxed{` scaffold (`DEFAULT_ANSWER_SCAFFOLD` /
   `answer_scaffold_ids`) and reads each candidate's logits + logprobs at the
   answer position. Candidates come from `answer_candidate_token` (the exact
   in-context token — no leading-space/casing guesswork).
- **Cheap variant:** the closed-empty-think block (`<think>\n\n</think>`) reads
  the answer with no generation — sanity baselines only (no trace to corrupt).

### 4.5 Extension — distance decay
Inject into a *small* early window (e.g. X=10%) at varying distances from the
answer and check whether the effect shrinks with distance (the writing sample's
hypothesis). Thought-anchors-style sentence attribution could deepen this later.

---

## 5. Dataset — chainscope (Arcuschin et al.)

Source cloned alongside the repo (see README): `git clone
git@github.com:jettjaniak/chainscope.git` → `./chainscope/` (**not** a Python
dependency we import; we read its data files directly).

**Use the prebuilt table — don't re-parse the source.** We use the
**`non-ambiguous-hard-2`** subset (the main paper dataset: 29 prop_ids, **9668
questions = 4834 pairs**, balanced YES/NO and gt/lt). It has been built into a
flat Parquet table:

- **Artifact:** `data/wm-non-ambiguous-hard-2.parquet` (9668 rows, 25 columns).
- **Builder:** `uv run scripts/build_scot_dataset.py` (re-run to regenerate;
  `--suffix`/`--out` to target another subset).
- **Why a custom build:** the aggregated pickle
  `chainscope/.../df-wm-non-ambiguous-hard-2.pkl.gz` is per-question×model CoT
  *eval stats* (183k rows, other models) and **lacks `q_str_open_ended`**. That
  entity-answer field lives only in the per-dataset YAMLs
  (`data/questions/{gt,lt}_{YES,NO}_1/*_non-ambiguous-hard-2.yaml`), so the
  builder parses those and uses the pickle only to cross-validate the qid set
  (confirmed: exact match, 9668).

**Key columns** (see the builder docstring for the full list):
- `q_str_open_ended` — *the entity-answer question we prompt with* ("Which work
  has more pages: X or Y?"). `q_str` is the original Yes/No phrasing.
- `x_name`/`y_name`/`x_value`/`y_value` — the two entities and ground-truth values.
- `correct_name`/`incorrect_name` (+ `_side`, `_value`) — **`incorrect_name` is
  the steering target** (the entity we try to flip the model toward).
- `prop_id`, `comparison`, `expected_answer`, `variant`, `qid`, `dataset_id`,
  `value_diff` (difficulty proxy), `topic`.
- **Correct-entity rule** = `x_name` iff `expected_answer == "YES"` else
  `y_name`. This was **verified** against the value-based rule (gt→larger value,
  lt→smaller value) for all 9668 questions — **0 mismatches** (`polarity_ok`
  column, all True). The earlier "verify per template" caveat is resolved.

**First-token collisions — handled.** The answer "handle" is the exact token each
entity emits right after the `\boxed{` scaffold (`answer_candidate_token`, derived
in context — *not* `first_token(" "+name)`). When the two entities don't map to
distinct, cleanly-alignable tokens, `answer_tokens_collide` flags the pair so it
can be dropped. Both are **precomputed** per question in
`answer-entangled-tokens.parquet` (§5.1), so downstream just filters on
`answer_tokens_collide`.

### 5.1 Precomputed entangled numbers — `data/answer-entangled-tokens.parquet`
For each `qid`, the top-10 / bottom-10 **number** tokens entangled (unembedding
cosine) with each entity's `\boxed{` answer-candidate token — the candidate
steering numbers to splice into the CoT.

- **Builder:** `uv run scripts/build_answer_entangled_tokens.py` (loads
  DeepSeek-R1-Distill-Llama-8B for its unembedding matrix only — no generation).
- **Columns** (keyed by `qid`): `x_name`/`y_name`,
  `x_answer_candidate_token`/`y_answer_candidate_token` (nullable `Int64`),
  `x_top10`/`x_bottom10`/`y_top10`/`y_bottom10` (each a list of
  `{token_id, token, score}`, ranked over number tokens only — same
  `is_number` / `NUMBER_BLOCKLIST` filter as the demo notebook), and
  `answer_tokens_collide`.
- **Coverage (R1-Distill-Llama-8B, `\boxed{` scaffold):** 260 of 9,668 entities
  don't pin to a clean token (`None` candidate → empty lists) and 424 pairs
  collide — drop both downstream. The candidate is the entity's *first* token at
  the read point, often a short/generic prefix for author-prefixed names (e.g.
  `S`, `Amy`), so per-entity number sets can be coarse — keep in mind when
  interpreting results.

---

## 6. Model — DeepSeek-R1-Distill-Llama-8B (gotchas)

`notebooks/demo_basic_qa.ipynb` already loads and prompts this model; copy its
setup. Key points (full detail in `CLAUDE.md` → "Reasoning models"):

- **No system prompt.** Put the question in the **user** turn.
- **Think tags are single tokens:** `<think>` = **128013**, `</think>` = **128014**.
- **Prompt build:** `add_generation_prompt=True` (template auto-appends
  `<｜Assistant｜><think>\n`), then append raw text. Do **not** pass a
  `<think>…</think>` block as a `continue_final_message` assistant turn (the
  template raises).
- **Budget + force-close:** R1-Distill overthinks; a 1024-token budget is the
  affordable floor and many traces won't close on their own. **Log the unclosed
  / force-close rate** and report results as "think for up to N tokens."
- **Temperature** (0.6 / top_p 0.95) only affects *generation*; the logit probes
  read the raw distribution and are temperature-independent. Seed via
  `seed_everything`.
- **Tokenizer:** load via `PreTrainedTokenizerFast` (the #45488 ByteLevel/Llama
  gotcha — see `CLAUDE.md`). Sanity-check `tok("a b") != tok("ab")`.

---

## 7. Library primitives (the building blocks)

Import everything from the installed `subliminality` package — never reach into
submodules or hack `sys.path`. Keep heavy logic in the library; keep notebook
cells thin (per `CLAUDE.md`).

Already available (`src/subliminality/`):
- `compute_entanglements(model, tokens, *, method, tokenizer=None, prompt=…,
  return_components=False)` — `method="unembedding"` (cheap, one matmul) is our
  primary; `method="output_distribution"` is the possible expansion. Returns a
  trailing `[…, vocab]` entanglement dim.
- `tokens`: `build_input_ids` (exact-token splice via `SENTINEL`), `first_token`
  (the entity/answer handle), `is_number` + `NUMBER_BLOCKLIST`, `token_mask`,
  `top_bottom` (rank entangled numbers; mask to numbers).
- `generation`: `batched_generate_truncated` (batched trace generation,
  truncate at `</think>`) and `batched_answer_probs` (left-padded `P(target)`
  read after a teacher-forced answer prefix, float32 softmax). Both
  device-agnostic.
- `reasoning_generation` (the SCoT answer-reading layer): `build_reasoning_prompt`
  (opened-think prompt + optional CoT `prefill`), `build_boxed_answer_instruction`
  (constrain the answer to the two named entities), `rollout_cot` (budgeted,
  force-closed CoT → `CoTRollout`), `batched_answer_scores` (two-candidate
  logits/logprobs **+ whole-vocab top token** at a scaffold → `AnswerScores`),
  `answer_candidate_token` / `answer_tokens_collide`, `answer_scaffold_ids`,
  `build_injection_prefill` (truncate a CoT to a sentence boundary + splice exact
  tokens at each boundary → `InjectionPrefill`, `default_split_sentences`),
  `DEFAULT_ANSWER_SCAFFOLD` / `DEFAULT_ANSWER_FORMAT`, `DEFAULT_THINK_BUDGET`.
  `rollout_cot` and `batched_answer_scores` take `batch_size` / `progress` to chunk
  arbitrarily many prompts (bounded VRAM, per-chunk seeding).
- `get_device`, `seed_everything` (device-aware: CUDA/MPS/CPU).

Remaining code this sprint needs (keep reusable logic in the library, test under
`tests/`; dataset/experiment scaffolding stays in scripts/notebooks):
- ~~Chainscope loader~~ **done** — `scripts/build_scot_dataset.py` →
  `data/wm-non-ambiguous-hard-2.parquet` (§5).
- ~~Entangled-number precompute~~ **done** —
  `scripts/build_answer_entangled_tokens.py` →
  `data/answer-entangled-tokens.parquet` (§5.1).
- ~~Answer scaffold + two-candidate logit/logprob read~~ **done** —
  `batched_answer_scores` / `answer_candidate_token` in `reasoning_generation`
  (baseline in `notebooks/demo_basic_qa.ipynb`).
- ~~Batched inference run + cache (CoT + answer logprobs)~~ **done** —
  `scripts/run_qa_inference.py` rolls out a seeded `--limit` sample (or all usable
  rows) and writes one resumable Parquet of CoT (text + exact ids) and answer
  logits/logprobs/diff/winner/top-token — the corpus the injection step consumes.
- ~~Trace truncation to first X% + number injection~~ **done (primitive)** —
  `build_injection_prefill` splits the CoT into sentences, cuts at `cutoff_frac`
  (snapped to a boundary), and splices the (entangled or random-control) tokens in
  at each boundary as exact ids → `prefill` for `build_reasoning_prompt`, then
  `rollout_cot` regenerates. (LLM-smoothed-rewrite variant still optional/TODO.)
- **TODO — per-question driver** running entangled vs. random-number control over
  the cached rollouts (truncate → inject → regenerate → re-read) and aggregating the
  metrics (§4.3), batched at `run_qa_inference` scale.

Reference existing usage: `notebooks/demo_basic_qa.ipynb` (the SCoT baseline
pipeline), `notebooks/subliminal_prompting_demo.ipynb` (entanglement methods), and
`scripts/perf_entanglement.py`.

---

## 8. Open decisions (resolve as we go)

- **k** (how many top entangled numbers per entity) and **how many to inject**.
- **Injection style:** raw random-position splice vs. LLM-smoothed rewrite
  (writing sample offers both). Start raw.
- **X** schedule (default 50% for elicitation; sweep for the decay extension).
- ~~Answer scaffold / first-token vs. span~~ **decided** — `\boxed{` scaffold,
  single in-context first token (`answer_candidate_token`); collided/unresolved
  pairs dropped.
- ~~Constrain the output format~~ **decided** — append a boxed-answer instruction
  (`build_boxed_answer_instruction`, two entities in question order) so the model
  boxes a name verbatim and the first-token read stays faithful for long names.
- ~~Metric sign~~ **decided** — `logprob_diff` = correct − incorrect (candidates
  ordered `(correct, incorrect)`); positive ⇒ favors correct.
- ~~Entangle against which token(s) of a multi-token name~~ **decided (v1)** — the
  single `\boxed{` answer-candidate token (often a short prefix; revisit if too
  coarse — see §5.1).
- **Question subset / size** for a first cheap run vs. the full sweep.
- Whether to add the **output-distribution** entanglement method or stay
  unembedding-only for v1.

---

## 9. Conventions reminder (from CLAUDE.md)

- Use **`uv`** for everything (`uv run …`, `uv sync`, `uv add`); never bare
  `python`/`pip`/`pytest`.
- Library is the **single source of truth**; expose new functionality via
  `src/subliminality/__init__.py` `__all__`; keep public functions typed.
- **Device-agnostic** code (CUDA/MPS/CPU) via `get_device`; no hardcoded `cuda`.
- **Git read-only by default** — don't commit/stage/push unless asked this turn.
- On the H100 pod, the thread-cap `.env` matters (cgroup ~20 vCPU); notebooks
  load it in their first cell.
