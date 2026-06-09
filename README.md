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
# Build input datasets:
uv run scripts/build_scot_dataset.py
uv run scripts/build_answer_entangled_tokens.py

# Run uncorrupted inference:
uv run scripts/run_qa_inference.py --limit 96 --batch-size 48 --out data/qa_inference_small.parquet
uv run scripts/run_qa_inference.py --limit 1000 --batch-size 48 --out data/qa_inference_medium.parquet

# Run corrupted inference:
# (each run_cot_corruption is roughly 2 minutes on an H100)
uv run scripts/run_cot_corruption.py --cots data/qa_inference_small.parquet --cutoff-frac 0.5 --tokens-source random_numerical
uv run scripts/run_cot_corruption.py --cots data/qa_inference_small.parquet --cutoff-frac 0.5 --tokens-source correct_top10
uv run scripts/run_cot_corruption.py --cots data/qa_inference_small.parquet --cutoff-frac 0.5 --tokens-source incorrect_top10
uv run scripts/run_cot_corruption.py --cots data/qa_inference_small.parquet --cutoff-frac 0.5 --tokens-source correct_bottom10
uv run scripts/run_cot_corruption.py --cots data/qa_inference_small.parquet --cutoff-frac 0.5 --tokens-source incorrect_bottom10

uv run scripts/run_cot_corruption.py --cots data/qa_inference_small.parquet --cutoff-frac 0.25 --tokens-source random_numerical
uv run scripts/run_cot_corruption.py --cots data/qa_inference_small.parquet --cutoff-frac 0.25 --tokens-source correct_top10
uv run scripts/run_cot_corruption.py --cots data/qa_inference_small.parquet --cutoff-frac 0.25 --tokens-source incorrect_top10
uv run scripts/run_cot_corruption.py --cots data/qa_inference_small.parquet --cutoff-frac 0.25 --tokens-source correct_bottom10
uv run scripts/run_cot_corruption.py --cots data/qa_inference_small.parquet --cutoff-frac 0.25 --tokens-source incorrect_bottom10

uv run scripts/run_cot_corruption.py --cots data/qa_inference_small.parquet --cutoff-frac 0.75 --tokens-source random_numerical
uv run scripts/run_cot_corruption.py --cots data/qa_inference_small.parquet --cutoff-frac 0.75 --tokens-source correct_top10
uv run scripts/run_cot_corruption.py --cots data/qa_inference_small.parquet --cutoff-frac 0.75 --tokens-source incorrect_top10
uv run scripts/run_cot_corruption.py --cots data/qa_inference_small.parquet --cutoff-frac 0.75 --tokens-source correct_bottom10
uv run scripts/run_cot_corruption.py --cots data/qa_inference_small.parquet --cutoff-frac 0.75 --tokens-source incorrect_bottom10

# Plot results:
uv run scripts/plot_corruption_comparison.py --base data/qa_inference_small.parquet --random data/corrupted/corrupted_0.5_random_numerical.parquet --out plots/small_0.5.png\
  --entangled data/corrupted/corrupted_0.5_correct_top10.parquet data/corrupted/corrupted_0.5_incorrect_top10.parquet data/corrupted/corrupted_0.5_correct_bottom10.parquet data/corrupted/corrupted_0.5_incorrect_bottom10.parquet

uv run scripts/plot_corruption_comparison.py --base data/qa_inference_small.parquet --random data/corrupted/corrupted_0.25_random_numerical.parquet --out plots/small_0.25.png\
  --entangled data/corrupted/corrupted_0.25_correct_top10.parquet data/corrupted/corrupted_0.25_incorrect_top10.parquet data/corrupted/corrupted_0.25_correct_bottom10.parquet data/corrupted/corrupted_0.25_incorrect_bottom10.parquet

uv run scripts/plot_corruption_comparison.py --base data/qa_inference_small.parquet --random data/corrupted/corrupted_0.75_random_numerical.parquet --out plots/small_0.75.png\
  --entangled data/corrupted/corrupted_0.75_correct_top10.parquet data/corrupted/corrupted_0.75_incorrect_top10.parquet data/corrupted/corrupted_0.75_correct_bottom10.parquet data/corrupted/corrupted_0.75_incorrect_bottom10.parquet
```
