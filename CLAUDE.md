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

