# SCoT Sprint — Subliminal Chain of Thought

**Status:** active sprint plan. Read this first if you're picking up SCoT work.
**Audience:** future agents (and humans) joining the sprint mid-flight.

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
| 4 | Start with Qwen QwQ-32B | **`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`** (already supported in the demo notebook; see §6) |
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
   to flip the model *toward* it). Compute the numeric tokens most entangled
   with that entity's token(s) via **unembedding cosine similarity**
   (`compute_entanglements(..., method="unembedding")`), restricted to number
   tokens (`token_mask(tok, is_number)`), take the top-k.
3. **Inject.** Truncate the trace to its first **X%** (start **X=50**; the point
   is to intervene *before* the model commits). Insert the top entangled numbers
   into that prefix — start with random-position injection; optionally also an
   LLM-smoothed rewrite variant.
4. **Force-close + read.** Append `</think>`, teacher-force a fixed answer-prefix
   scaffold, and read `P(first token of each entity)` (§4.4). Record the
   logit/prob gap and which entity wins.
5. **Control.** Repeat step 3–4 injecting **random numbers** (same count, same
   positions) instead of entangled ones.
6. **Verdict.** Entangled injection shifting the answer toward the non-chosen
   entity significantly more than the random control ⇒ subliminal CoT elicited.

### 4.3 Metrics
- **Primary (new this sprint):** `logit(first_token(incorrect)) −
  logit(first_token(correct))` at the answer read point — a signed, continuous
  measure of how far the injection pushed the model toward the wrong entity.
  Compare entangled vs. random control (paired per question). Use logits (or
  log-probs); read the distribution in float32 (bf16 underflows tails — the
  library already `.float()`s before softmax).
- **Secondary:** top-1 answer-flip rate (did the argmax entity change toward the
  injected target?).
- Aggregate across questions; report effect sizes with CIs, entangled vs.
  control.

### 4.4 Reading the answer (two options)
- **Generated-think (primary):** generate a real trace so there's something to
  inject into, force-close `</think>`, then teacher-force the answer prefix and
  read `P(target)`. Use `batched_generate_truncated` (stop/truncate at
  `</think>` = **128014**) then `batched_answer_probs`.
- **Closed-empty-think (cheap variant):** skip generation; read the answer right
  after `<think>\n\n</think>`. Cheaper but no trace to corrupt, so only useful
  for sanity baselines, not the injection itself.

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

**Caveat — first-token collisions.** Our entity "handle" is `first_token(" " +
name)`. If the two entities share a first token (e.g. both "Boeing …"), the
logit-difference metric is degenerate — **filter those pairs out**, or
disambiguate with a multi-token read.

---

## 6. Model — DeepSeek-R1-Distill-Llama-8B (gotchas)

The demo notebook already loads and prompts this model; copy its setup. Key
points (full detail in `CLAUDE.md` → "Reasoning models"):

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
- `get_device`, `seed_everything` (device-aware: CUDA/MPS/CPU).

Likely **new** code this sprint needs (keep it in the library, test under
`tests/`):
- ~~A chainscope loader~~ **done** — `scripts/build_scot_dataset.py` →
  `data/wm-non-ambiguous-hard-2.parquet` (§5). Just `pd.read_parquet` it; apply
  first-token-collision filtering per-model downstream (tokenizer-dependent, so
  intentionally kept out of the model-agnostic table).
- Trace truncation to first X% + number injection (random-position; optional
  LLM-smoothed rewrite) + force-close.
- An answer-prefix scaffold + a read that returns the correct/incorrect
  first-token logits and their difference.
- A per-question driver that runs entangled vs. random-control and aggregates.

Reference existing usage: `notebooks/subliminal_prompting_demo.ipynb` (the best
example of the library in use) and `scripts/perf_entanglement.py`.

---

## 8. Open decisions (resolve as we go)

- **k** (how many top entangled numbers per entity) and **how many to inject**.
- **Injection style:** raw random-position splice vs. LLM-smoothed rewrite
  (writing sample offers both). Start raw.
- **X** schedule (default 50% for elicitation; sweep for the decay extension).
- **Answer-prefix scaffold** wording for the open-ended question, and whether to
  read the bare first token or a short disambiguating span.
- **Entangle against which tokens** of a multi-token entity name — average over
  all its tokens (Zur et al. average over concept tokens) vs. first token only.
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
