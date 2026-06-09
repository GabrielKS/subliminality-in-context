# CLAUDE.md

Guidance for working in this repository.

## Always use `uv`

This project is managed entirely with `uv`. **Never invoke bare `python`,
`pip`, or `pytest`.** Always go through `uv`:

- `uv run <cmd>` — run anything inside the project env (e.g. `uv run pytest`,
  `uv run scripts/run_example.py`, `uv run python -c "..."`).
- `uv sync` — create/update the environment from `pyproject.toml` + `uv.lock`.
- `uv add <pkg>` — add a runtime dependency (edits `pyproject.toml` + lockfile).
- `uv add --dev <pkg>` — add a dev/tooling dependency to the `dev` group.

Do not hand-edit dependency lists in `pyproject.toml`; use `uv add` /
`uv remove` so the lockfile stays consistent.

## Git: read-only by default

**Default to read-only Git operations.** Inspecting history is always fine
(`git status`, `git diff`, `git log`, `git show`, `git blame`). Do **not** run
state-changing Git commands — `git commit`, `git add`/stage, `git push`, `git
reset`, `git checkout`/`switch`, `git rebase`, `git merge`, branch creation — unless
the user explicitly asks in the current turn (e.g. "commit this", "push it"). A plan
or task option that merely mentions committing is **not** standing authorization;
confirm first. When work is done, leave it staged-or-not as the user prefers and let
them commit.

## Repository layout

- `src/subliminality/` — the library (import name `subliminality`; distribution
  name `subliminality-in-context`). Uses a `src/` layout.
- `scripts/` — runnable scripts; invoke via `uv run scripts/<name>.py`.
- `notebooks/` — Jupyter notebooks.
- `tests/` — pytest tests, discovered under `tests/`.
- `data/` — generated data artifacts (Parquet); **gitignored**, regenerate via the
  `scripts/build_*` scripts. Source data is the `chainscope` repo, cloned alongside
  the repo (see README and `SCOTSPRINT.md` §5).

## Conventions

- **The library is the single source of truth.** Scripts and notebooks import
  from `subliminality`; never duplicate logic into a script or notebook cell.
  Keep heavy logic in the library and keep notebook cells thin.
- **Maintain the public API** in `src/subliminality/__init__.py` via `__all__`.
  Expose new functionality there rather than having callers reach into
  submodules.
- **Imports always go through the installed `subliminality` package.** The
  library is installed (editable) into the venv, so `import subliminality` /
  `from subliminality.x import y` resolves everywhere — scripts, notebooks, and
  tests alike. From outside the library, **never** manipulate `sys.path`, add
  `src/` to the path by hand, or use relative imports to reach into the package.
  If an import fails, the fix is `uv sync` (or `uv run`), not a path hack.
- **Tests:** run with `uv run pytest`; add new tests under `tests/` named
  `test_*.py`.
- **Notebooks** run against the project venv (the `dev` group ships
  `ipykernel`). Launch via `uv run jupyter lab` so the kernel matches.
- The package ships type information (`py.typed`); keep public functions typed.

## What this project studies

We probe "naïve entanglement" between vocabulary tokens in instruct LLMs: which
tokens does *encouraging* one token drag along? The motivating example (from the
"owls" subliminal-learning work) is that the number `087` is entangled with
` owl` in Llama-3.2-1B — a system prompt professing love for `087` raises the
probability the model names the owl as its favorite bird. We measure entanglement
two ways and then test whether the entangled numbers actually move a target
token's probability (the "subliminal prompting" effect).

## Core abstraction — `subliminality.compute_entanglements`

`compute_entanglements(model, tokens, *, method, tokenizer=None, prompt=..., return_components=False)`
takes token ids of any shape `[...]` and returns entanglement against the whole
vocabulary (a trailing `[..., vocab]` dim). `method` is required:

- `method="unembedding"` — cosine similarity of unembedding-matrix rows. No
  forward pass (one matmul); cheap at any scale.
- `method="output_distribution"` — shift in the next-token distribution when the
  token is injected as the model's "favorite". One forward pass *per query token*
  (+ one shared "base" pass), so keep query sets small. Returns the `guided/base`
  ratio by default; `return_components=True` returns the `(guided, base)` pieces.

Supporting primitives in `subliminality.tokens`: `build_input_ids` (exact-token
splice), `first_token`, `token_mask` + `is_number`, `top_bottom`.

Generation/reading primitives: `subliminality.generation`
(`batched_answer_probs`, `batched_generate_truncated`) and — for reasoning models
— `subliminality.reasoning_generation` (`build_reasoning_prompt`,
`build_boxed_answer_instruction`, `rollout_cot`, `batched_answer_scores`,
`answer_candidate_token`, `answer_tokens_collide`, `DEFAULT_ANSWER_SCAFFOLD`; see
`SCOTSPRINT.md`). `rollout_cot` / `batched_answer_scores` take `batch_size` /
`progress` to chunk arbitrarily many prompts (bounded VRAM, per-chunk seeding);
`AnswerScores` carries both candidates' logits/logprobs **and** the whole-vocab top
token (`top_token_id`/`top_logprob`, for free-choice accuracy).

The library is **domain-agnostic** — no animals/birds. Experiment scaffolding
(animal/bird preference sweeps, pandas tables) lives in the notebook, built on
these primitives. New experiments should follow suit and stay thin.

## Modeling conventions & gotchas (learned the hard way)

- **Softmax in float32.** Models load in **bfloat16**; bf16 softmax underflows
  tiny tail probabilities to 0 and quantizes ratios into integers. `.float()` the
  logits before softmax (the library does this internally).
- **Exact-token injection.** Decoding a token to a string and re-encoding does
  NOT reliably preserve it (~18% of the vocab breaks; e.g. ` 087` splits). Inject
  the exact id via the `SENTINEL` splice in `build_input_ids`, re-encoding parts
  with `add_special_tokens=False` (the chat template already adds BOS — else you
  double it).
- **First-token handle.** Multi-token words are represented by the first token of
  `" " + word`. Cheap, but a leaky proxy: `sea turtle`/`sea otter` share first
  token ` sea`, and prefixes like ` oct` are weak.
- **Prompt assumptions are overridable.** The "favorite token" prompt lives in
  module constants in `entanglement.py`, overridable per call via the `prompt=`
  builder — the seam for non-chat / reasoning models.
- **Findings are model-specific.** Re-derive entangled tokens per model.
  Tokenizers differ too.
- **DeepSeek (and other ByteLevel-but-declared-Llama) tokenizers drop spaces under
  transformers v5** ([#45488](https://github.com/huggingface/transformers/issues/45488)):
  `LlamaTokenizerFast.__init__` forces a Metaspace pre-tokenizer over the `tokenizer.json`
  ByteLevel one, so `AutoTokenizer` silently encodes `"a b c"` identically to `"abc"`.
  Genuine Llama tokenizers are unaffected. Workaround (used in the notebook's DeepSeek
  load): load via `PreTrainedTokenizerFast.from_pretrained(...)`, which respects
  `tokenizer.json`. Sanity-check any new tokenizer with
  `tok("a b").input_ids != tok("ab").input_ids`.

## Models & tests

- Experiments use `meta-llama/Llama-3.2-1B-Instruct` and
  `meta-llama/Llama-3.1-8B-Instruct` (gated; bf16; ~2.5GB / ~16GB; ~24GB peak with
  both resident — fine on a 48GB machine).
- Tests use **`distilgpt2`** (ungated, CPU, seconds) so `uv run pytest` stays
  cheap. distilgpt2 has **no chat template**, so output-distribution tests pass a
  plain-text `prompt` builder.

## The demo notebook

`notebooks/subliminal_prompting_demo.ipynb` is the working experiment scaffolding
and the best example of the library in use — start here for any entanglement /
subliminal-prompting task. Its arc:

1. **Basic reproduction** — `087` → ` owl`: a system prompt loving `087` raises the
   owl probability in "My favorite bird is the ___".
2. **Computing entangled tokens** — both methods (`output_distribution`,
   `unembedding`), filtered to number tokens, on the ` owl` target.
3. **Subliminal-prompting test** — does loving an entangled *number* raise a target
   animal? A sweep over a list of animals producing base-prob + uplift tables
   (mean / geomean / median), one per method/metric.
4. **A larger model** — the same sweep repeated on Llama-3.1-8B for comparison.
5. **Reasoning models** — `DeepSeek-R1-Distill-Llama-8B`, with different prompting
   (see below) and a generated-think measurement condition.

Keep cells thin: heavy logic belongs in `subliminality`, the notebook just composes
primitives and presents results.

For the **SCoT (subliminal chain-of-thought) sprint**, see `SCOTSPRINT.md` (plan +
current status) and `notebooks/demo_basic_qa.ipynb` — the baseline reasoning-QA
pipeline (`build_reasoning_prompt` → `rollout_cot` → `batched_answer_scores` at a
`\boxed{` scaffold) on the chainscope comparison questions.

## Reasoning models (DeepSeek R1-Distill)

`deepseek-ai/DeepSeek-R1-Distill-Llama-8B` (Llama-3.1-8B distilled on R1 traces) needs
different handling from the Instruct models:

- **No system prompt.** Put instructions in the *user* turn (the notebook injects the
  favorite-token / love-number text there, not in a system message).
- **Thinking block.** The model emits `<think>…</think>` then the answer. Its chat template
  auto-appends `<｜Assistant｜><think>\n` when `add_generation_prompt=True`, and **raises** if a
  `continue_final_message` assistant turn contains `<think>…</think>` (the template strips think
  content from messages). So build prompts with `add_generation_prompt=True` and then *append
  raw text*. `<think>`=128013, `</think>`=128014 are single tokens.
- **Two ways to read the answer.** Discovery and the cheap measurement condition use a *closed
  empty* think block (`<think>\n\n</think>`) so the answer position is read directly, no
  generation. The other condition *generates* a real think block (sampled), stops at `</think>`,
  then teacher-forces the answer prefix and reads `P(target)`.
- **Budget the think block, then force-close (generated-think).** R1-Distill *overthinks* even
  trivial prompts ("what's your favorite animal?"), so most traces blow past a short
  `max_new_tokens` budget without ever emitting `</think>` — and reading the answer after an
  **open** think block is malformed. Two practices, learned the hard way:
  1. **Don't trust a round-number budget.** `512` is far too short (the majority of traces never
     close); `1024` is the affordable floor we use, but the budget is an *empirical axis*, not a
     "sufficient" constant (R1's own design ceiling is 32k). Always **log the unclosed rate** and
     pick the budget by a stated tolerance — and report results as "think for up to N tokens",
     since at a tight budget you are measuring *truncated* reasoning, not free generation.
  2. **Force-close.** Append `</think>` to any trace that hit the budget,
     so every measured trace is read in a closed block — naturally or artificially (a natural close
     ends on exactly that token, so the forced one mirrors it). Keep the force-close **rate**
     visible (real-time + an end-of-run %). This both fixes the malformed read and, empirically,
     **strengthened** the measured subliminal effect vs. the old open-block reads.
  3. **Never shorten via prompt text** ("be brief", "answer in one sentence"): it's unreliable
     *and* it contaminates the very prompt you're probing. Budget/force-close are decode-time only.
  Note the generation prompt carries **no target token** and is seeded per batch, so the think
  blocks are identical across targets within a run (only the final `P(target)` read differs) —
  hence the force-close rate is the same for every target.
- **Temperature is irrelevant to the logit probes** (they read the raw distribution). DeepSeek's
  temp 0.6 / top_p 0.95 recommendation is for *generation* — used only in the gen-think
  condition, seeded via `seed_everything`.
- **Tokenizer:** load via `PreTrainedTokenizerFast` (see the #45488 gotcha above).

## Reading a constrained answer (the `\boxed{` convention)

For comparison QA (the SCoT sprint) the answer is read at a teacher-forced
`\boxed{` scaffold after the closed `</think>`. Learned the hard way:

- **Scaffold = `DEFAULT_ANSWER_SCAFFOLD` (`"\n\n\\boxed{"`).** The trailing `{`
  tokenizes as its own token → a hard *no-space* boundary, so the next token is the
  entity's first token with no leading space and natural casing. Empirically
  R1-Distill puts ~0.999 of its mass on that single token there (the competing entity
  ~8 logits below — the signal we read).
- **Derive the candidate token *in context*** with `answer_candidate_token` (tokenize
  `scaffold + name`, take the first token past the scaffold) — never
  `first_token(" " + name)`, which guesses the space/casing wrong. It returns `None`
  on a cross-boundary BPE merge (e.g. `{i` for "iPhone"); `answer_tokens_collide`
  treats `None`-or-equal pairs as degenerate and drops them. `DEFAULT_ANSWER_FORMAT`
  (`"\\boxed{{{}}}"`) stays coupled to the scaffold.
- **Constrain the output** with `build_boxed_answer_instruction(question, [x, y])`
  (Arcuschin et al.'s "give a YES / NO answer" analogue, options in **question
  order**). It makes the model box one of the two names *verbatim*, so the first-token
  read is faithful even for long author-prefixed names (which the model would
  otherwise box as a short surface form, e.g. "Charles Band's Hideous!" → `Hideous!`).
  Only the final token is constrained; the chain of thought stays free.
- **Diff sign:** order candidates **(correct, incorrect)** so `AnswerScores.logit_diff`
  / `.logprob_diff` = correct − incorrect — **positive ⇒ favors the correct entity**.
- **Free-choice accuracy:** `AnswerScores.top_token_id` is the whole-vocab argmax at
  the read point (regardless of the two candidates) — use it to ask whether the
  model's actual pick is the correct answer / either candidate.

## Scripts

`scripts/perf_entanglement.py` is a performance test for the two entanglement
methods. For `--n` random tokens (default 1000) it computes entanglements by both
methods, then for the top-10 unembedding neighbours of each token reads the top-20
downstream logits after `"<token>. I'd like to say:"` (result shape `(n, 10, 20)`).
It shows tqdm bars and prints per-step timings. `output_distribution` is one
forward pass per token, so it's chunked (`--od-batch`); bump `--od-batch`/`--b-batch`
on large-VRAM GPUs. Run via `uv run scripts/perf_entanglement.py` (defaults to
Llama-3.1-8B-Instruct; `--model`/`--device` override).

`scripts/build_scot_dataset.py` builds the SCoT question table
`data/wm-non-ambiguous-hard-2.parquet` from the chainscope question YAMLs (the
`non-ambiguous-hard-2` subset: 9,668 questions carrying the `q_str_open_ended`
entity-answer field plus a derived `correct_name`/`incorrect_name`, with polarity
cross-checked against the aggregated pickle).

`scripts/build_answer_entangled_tokens.py` builds
`data/answer-entangled-tokens.parquet` (keyed by `qid`): per question, the top-10
/ bottom-10 number tokens entangled (unembedding cosine) with each entity's
`\boxed{` answer-candidate token, plus the `answer_tokens_collide` flag. Loads
DeepSeek-R1-Distill-Llama-8B for its unembedding matrix only (no generation).

`scripts/run_qa_inference.py` runs the reasoning-QA pipeline over the dataset:
collision-filter → seeded `--limit` sample (or all usable rows) → batched
`rollout_cot` → `batched_answer_scores` at the `\boxed{` scaffold. Writes a **single
resumable Parquet** (`--out`, rewritten each batch; resume skips rows already present
by `qid` — not sharded), caching per question the CoT (`think_text` + exact
`think_token_ids`), both candidates' logits/logprobs, `logprob_diff` (correct −
incorrect), the free-choice top token, and accuracy/force-close flags. ~0.7 rows/s at
`--batch-size 48` (KV-cache-bound; B≈48 peaks ~37 GB, B=96 OOMs);
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` recommended. This is the baseline
corpus the injection step later corrupts.

All three write under `data/` (gitignored) and run from the repo root; see
`SCOTSPRINT.md` for how the tables feed the experiment.

## Device handling (CUDA / MPS / CPU)

This project must run on both **NVIDIA CUDA** machines and **Apple Silicon
(MPS)**. Write device-agnostic code:

- **Never hardcode a device.** Don't call `.cuda()`, hardcode `"cuda"`, or call
  `torch.cuda.*` unconditionally. Resolve the device once and pass it around
  explicitly; move tensors/modules with `.to(device)`.
- **Single source of truth.** Device selection lives in `get_device()`
  (`src/subliminality/device.py`, exported as `subliminality.get_device`). It
  prefers `cuda` → `mps` → `cpu` and accepts an explicit override (the `prefer`
  arg or the `SUBLIMINALITY_DEVICE` env var). Scripts and notebooks call it —
  they don't re-detect the device themselves.
- **Detection differs per backend:** `torch.cuda.is_available()` vs
  `torch.backends.mps.is_available()`. Gate any CUDA-only path on the resolved
  device, not on an assumption.
- **MPS gotchas to code around:**
  - No `float64` on MPS — use `float32` (double silently breaks). Don't rely on
    a global default dtype that assumes CUDA.
  - Not every op is implemented on MPS. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` to
    fall back to CPU for missing ops, and prefer documented-supported ops.
  - Mixed precision: `torch.cuda.amp` is CUDA-only. Use
    `torch.autocast(device_type=...)`, and note AMP support on MPS is limited.
  - `pin_memory=True` / `non_blocking=True` and `torch.cuda.synchronize()` are
    CUDA concepts — guard them on the device type.
  - Seeding differs per backend; seed in a device-aware way for reproducibility.
    Use `subliminality.seed_everything(seed)` (seeds Python/NumPy/torch + the active
    CUDA/MPS backend).

## CPU quota & thread caps (cgroup-limited hosts like the H100 pod)

The H100 pod is a Docker container with a **cgroup CPU quota of ~20 vCPU**
(`cpu.cfs_quota_us/cpu.cfs_period_us = 2040000/100000`), even though the host
exposes **192 cores** — `nproc`, `lscpu`, and `os.cpu_count()` all report 192,
and nothing clamps to the quota. So native thread pools (OpenMP, MKL, OpenBLAS,
NumExpr, HF `tokenizers`/Rayon, `torch.get_num_threads()` — all default to 192)
oversubscribe the quota, and CFS **throttles the whole cgroup** in ~80 ms bursts.
That is why cold, CPU-bound, latency-sensitive work (Python imports, `pytest`
collection, Claude/Node startup) feels slow on the pod while GPU ("hot")
computation — which doesn't touch the CPU quota — is fast. It is **not** a disk or
memory bottleneck: verified IO pressure-stall ~0, no iowait, no swap, ~1 TB warm
page cache. Diagnose with `cat /sys/fs/cgroup/cpu/cpu.stat` (cgroup v1 here) and
watch `nr_throttled`/`throttled_time` climb under load.

**Fix — cap the thread pools below the quota.** The repo `.env` sets
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=16` and
`TOKENIZERS_PARALLELISM=false` (drop lower, e.g. 4, for pure-GPU runs). **`uv`
does not auto-load `.env`** — there is no `[tool.uv]`/`uv.toml` config key for it;
the only native triggers are `uv run --env-file .env …` or exporting
`UV_ENV_FILE=.env` (e.g. in the shell rc or the pod's container env). For a
committed, no-flags-each-time setup, use direnv (`.envrc`) or a wrapper target.

**Notebooks (VSCode) load it themselves.** Because the thread-count vars are read
at *import* time, the demo notebook's **first cell** loads the `.env` via
`python-dotenv` (`load_dotenv(find_dotenv(usecwd=True))`, `override=False` so a real
env wins) *before* any `import torch`/`import subliminality`, with a
`torch.set_num_threads(...)` backstop for re-runs. Keep that cell first. This is why
VSCode kernels don't need the `uv run --env-file` launch dance.

