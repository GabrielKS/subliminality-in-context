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

## Repository layout

- `src/subliminality/` — the library (import name `subliminality`; distribution
  name `subliminality-in-context`). Uses a `src/` layout.
- `scripts/` — runnable scripts; invoke via `uv run scripts/<name>.py`.
- `notebooks/` — Jupyter notebooks.
- `tests/` — pytest tests, discovered under `tests/`.

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

Keep cells thin: heavy logic belongs in `subliminality`, the notebook just composes
primitives and presents results.

## Scripts

`scripts/perf_entanglement.py` is a performance test for the two entanglement
methods. For `--n` random tokens (default 1000) it computes entanglements by both
methods, then for the top-10 unembedding neighbours of each token reads the top-20
downstream logits after `"<token>. I'd like to say:"` (result shape `(n, 10, 20)`).
It shows tqdm bars and prints per-step timings. `output_distribution` is one
forward pass per token, so it's chunked (`--od-batch`); bump `--od-batch`/`--b-batch`
on large-VRAM GPUs. Run via `uv run scripts/perf_entanglement.py` (defaults to
Llama-3.1-8B-Instruct; `--model`/`--device` override).

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

