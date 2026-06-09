# subliminality-in-context

Research code for studying subliminal learning-like effects in context.

The repo is organized as:

- `src/subliminality/` — the pip-installable library and its public API.
- `scripts/` — runnable scripts that depend on the library.
- `notebooks/` — Jupyter notebooks that depend on the library.
- `tests/` — pytest test suite.

## Setup

This project is managed with [uv](https://docs.astral.sh/uv/). Create/sync the
environment (installs the library in editable mode plus dev tooling):

```bash
uv sync
```

Also clone another project from which we reuse data:

```bash
git clone git@github.com:jettjaniak/chainscope.git
```

## Common tasks

```bash
uv run pytest                      # run the test suite
uv run scripts/run_example.py      # run a script
uv add <package>                   # add a runtime dependency
uv add --dev <package>             # add a dev/tooling dependency
```

To work in notebooks, you can launch Jupyter through uv so it uses the project venv:

```bash
uv run jupyter lab
```

## SCoT Sprint
To reproduce:
```bash
uv run scripts/build_scot_dataset.py
uv run scripts/build_answer_entangled_tokens.py
uv run scripts/run_qa_inference.py --limit 96 --batch-size 48 --out data/qa_inference_small.parquet
uv run scripts/run_qa_inference.py --limit 1000 --batch-size 48 --out data/qa_inference_medium.parquet
```
