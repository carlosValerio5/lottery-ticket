# Constraint-Driven Training for Modular Edge Models

Research codebase for training convolutional networks from scratch with sparsity-oriented losses (see `tasks.md` for the full roadmap). **Phase 1** provides a CIFAR-style ResNet-18 builder, explicit Kaiming initialization, and named submodules for later activation penalties. **Phase 2** adds `SparseLoss` (CE + weight L1 + activation L1), CIFAR-10 loaders, a training/eval loop, optional λ/γ grid search, and **observability** (console, TensorBoard, CSV/JSONL artifacts, and Python logging).

### Running the CLI

The package lives under `src/pia`. Until you install the project in editable mode, set `PYTHONPATH` so `python -m pia.cli.train_cifar` resolves imports:

```bash
cd /path/to/pia
PYTHONPATH=src pipenv run python -m pia.cli.train_cifar --epochs 1 --data-root ./data --logdir ./runs --run-name smoke
```

### Training commands

```bash
PYTHONPATH=src pipenv run python -m pia.cli.train_cifar --data-root ./data --logdir ./runs --run-name exp1 \
  --epochs 10 --lambda-weight 1e-5 --gamma-activation 1e-5

pipenv run tensorboard --logdir ./runs/exp1/tb
```

Grid search over `--lambdas` and `--gammas` (one subdirectory per combination under `--logdir`, plus `grid_index.json`):

```bash
PYTHONPATH=src pipenv run python -m pia.cli.train_cifar --grid --logdir ./runs/grid --epochs 3 \
  --lambdas 1e-6 1e-5 --gammas 1e-6 1e-5
```

Optional structured log file for the `pia` logger (see below):

```bash
PYTHONPATH=src pipenv run python -m pia.cli.train_cifar --json-log ./runs/exp1/pia_structured.logl ...
```

### Logs and observability

Training uses a **composite observer** (`build_default_observer` in `src/pia/training/grid_search.py`): each epoch writes the same scalar metrics to several sinks so you can watch progress live, audit a run later, or ship logs to another system.

#### Console (live progress)

- **`tqdm`** bars for the training and validation passes each epoch (`train`, `val`). The training bar’s postfix shows **total loss**, **cross-entropy (task) term**, and **batch accuracy** so you can spot instability without opening files.
- The CLI also logs start messages on the **`pia.cli`** logger (INFO).

#### Run directory layout (`--logdir` / per-run folder)

For a single run (`--logdir ./runs --run-name exp1`), artifacts land under **`./runs/exp1/`**:

| Artifact | Role |
|----------|------|
| `metrics.csv` | One **CSV row per epoch**. The header is fixed on the **first** epoch from all keys in that row: `epoch`, run config fields (`run_id`, `lambda_weight`, `gamma_activation`, `epochs`, `batch_size`, `lr`, `data_root`, `device`, optional `git_sha`), and aggregated metrics (see below). |
| `events.jsonl` | **Append-only JSON lines**: each line is one epoch, merging the same run **config** snapshot with `epoch` and all **float metrics** (good for `jq`, log aggregators, or diffing across retries). |
| `live_batches.jsonl` | **Per-batch JSON lines** while the **current** epoch runs: `phase` (`train` / `val`), `epoch`, `batch`, `loss`, `loss_task`, `acc`. Truncated at the start of each training phase; the Streamlit dashboard reads it for intra-epoch charts. |
| `tb/` | **TensorBoard** event files. Each metric name is a scalar; **`global_step` is the epoch index** (1-based in the loop). Point TensorBoard at this directory (as in the command above). |
| `summary.json` | Written **once at the end** of the run: `{"config": {...}, "metrics": {...}}` with the **last epoch’s** merged train+val scalars. |

**Scalar names** logged each epoch (from `fit` / `train_one_epoch` / `evaluate`):

- Training: `train/loss`, `train/loss_task`, `train/loss_weight_l1`, `train/loss_activation_l1`, `train/acc`, `train/weight_sparsity_ratio`
- Validation: `val/loss`, `val/loss_task`, `val/loss_weight_l1`, `val/loss_activation_l1`, `val/acc`, `val/weight_sparsity_ratio`

`weight_sparsity_ratio` is the fraction of parameters with \(|w| < 10^{-3}\) (cheap sparsity proxy for the accuracy–sparsity tradeoff in `tasks.md`).

#### Python logging (`pia` tree)

`setup_pia_logging` (called from the CLI) attaches:

1. **stderr** — human-readable lines: `timestamp`, `LEVEL`, `logger`, `message`.
2. **Optional file** (`--json-log`) — one JSON object per log record via `JsonFormatter`: fields include `level`, `logger`, `message`, and `exc_info` if present.

The **`LoggingMetricsObserver`** writes **INFO** lines on the **`pia`** logger:

- At run start: `train_begin` plus a JSON blob of the **run config** (hyperparameters, paths, device string, optional **`git_sha`** if the repo is a git checkout).
- After each epoch: `epoch_end` with `epoch=` and a JSON blob of **all epoch metrics** (same keys as TensorBoard/CSV).
- At run end: `train_end` with a small JSON **summary** (currently includes `last_metrics`).

Separate loggers used elsewhere:

- **`pia.training`** — **WARNING** if validation accuracy drops sharply versus the previous epoch, or if **`val/acc`** falls below a floor (default 0.1); **ERROR** if a non-finite loss is detected during train or validation.
- **`pia.inference`** — used by `run_inference_batch` (see below).

#### Grid search

Under `--logdir ./runs/grid`, each `(λ, γ)` combination gets its own subdirectory (name derived from the hyperparameters). After all runs, **`grid_index.json`** lists each run’s directory, config, and final metrics for quick comparison.

#### Inference / “production-style” logging

`pia.inference.run_inference_batch` runs the model in **eval** mode (restores train mode afterward if it was training), times one forward with **`time.perf_counter`**, and logs a single **INFO** line on **`pia.inference`** whose message embeds JSON with:

- `event`: `"inference_batch"`
- `batch_size`, `latency_ms`, `device`
- `mean_entropy_nats` — mean softmax entropy across the batch (higher often means less confident predictions; useful as a coarse health signal in deployment).

Non-finite logits trigger **ERROR** and raise `RuntimeError`. Configure logging (level, file handlers) the same way as for training if you want JSON lines from inference shipped alongside training logs.

### Live training dashboard (Streamlit)

The dashboard polls **`live_batches.jsonl`** (per-batch metrics **during** the current train/val pass) and **`events.jsonl`** (one line **after** each full epoch). Curves update on a short disk poll interval without coupling training to the UI. It is **opt-in** via environment variable so the app is not exposed by mistake.

**Requirements:** dev install includes **Streamlit ≥ 1.33** (for `@st.fragment(run_every=...)`) and **pandas** (see [`Pipfile`](Pipfile) `[dev-packages]`).

**Feature flag:** set `PIA_STREAMLIT_DASHBOARD` to `1`, `true`, `yes`, or `on`. The Streamlit page also checks this and stops with an error if it is missing (e.g. if someone opens the app URL without the flag).

**Optional default run directory:** `PIA_RUN_DIR` points the sidebar default to your run folder (e.g. `./runs/exp1`).

**Terminal A — training:**

```bash
cd /path/to/pia
PYTHONPATH=src pipenv run python -m pia.cli.train_cifar --logdir ./runs --run-name exp1 --epochs 20
```

**Terminal B — dashboard** (after the first epoch has appended to `events.jsonl`, or leave it open and it will show “waiting” until data exists):

```bash
cd /path/to/pia
export PIA_STREAMLIT_DASHBOARD=1
PYTHONPATH=src pipenv run python -m pia.cli.dashboard --run-dir ./runs/exp1
```

Or pass port: `--port 8502`.

**Spawn from training (single run only):** Streamlit starts in a **detached** subprocess (new session) so **training logs stay on your terminal**; Streamlit’s stdout/stderr go to a file (default `<run_dir>/dashboard_streamlit.log`, override with `--dashboard-log`). Tail that file if you need the server’s own messages. Ignored with `--grid`.

```bash
PYTHONPATH=src pipenv run python -m pia.cli.train_cifar --logdir ./runs --run-name exp1 \
  --spawn-dashboard --dashboard-port 8501
# optional custom log path:
#   ... --dashboard-log /tmp/streamlit_pia.log
```

Then open `http://localhost:8501` (or your chosen port). The child receives `PIA_STREAMLIT_DASHBOARD=1` and `PIA_RUN_DIR` pointing at the run directory.

The dashboard shows **live line charts** (refreshed from disk each poll interval): one block for **train/acc vs val/acc**, one for **train/loss vs val/loss** (total sparse loss), and an expander for **CE-only** curves (`train/loss_task`, `val/loss_task`) when present.

## Prerequisites

- **Python 3.13** (matches `Pipfile` `[requires]`)
- [pipenv](https://pipenv.pypa.io/) for dependency management
- Optional: **CUDA**-capable GPU for later training phases; Phase 1 tests run on CPU.

## Setup

From the repository root:

```bash
pipenv install --dev
```

That creates a virtualenv and installs **torch**, **torchvision**, **numpy**, **tensorboard**, **tqdm**, **scikit-learn**, **matplotlib**, plus dev tools **pytest**, **black**, **ruff**, **pandas**, and **streamlit** (dashboard).

If virtualenv creation fails because the default location under your home directory is not writable, keep the environment inside the project:

```bash
export PIPENV_VENV_IN_PROJECT=1
pipenv install --dev
```

Activate the shell when you want the venv on your `PATH`:

```bash
pipenv shell
```

Or run one-off commands with:

```bash
pipenv run pytest
```

## Project layout

| Path | Purpose |
|------|---------|
| `src/pia/` | Installable package (`pia`): models and training utilities. |
| `src/pia/models/resnet_cifar.py` | ResNet-18 for small images, `apply_he_init`, monitored layers API. |
| `src/pia/losses/sparse_loss.py` | `SparseLoss` and per-term breakdown. |
| `src/pia/data/cifar10.py` | CIFAR-10 train/val/test `DataLoader` builders. |
| `src/pia/training/` | `fit`, observers, optional `grid_search` in `grid_search.py`. |
| `src/pia/observability/logging_config.py` | `setup_pia_logging` (stderr + optional JSON file). |
| `src/pia/inference/predict.py` | `run_inference_batch` for production-style latency logs. |
| `src/pia/cli/train_cifar.py` | CLI entry (`python -m pia.cli.train_cifar`). |
| `src/pia/cli/dashboard.py` | CLI entry for Streamlit (`python -m pia.cli.dashboard`). |
| `src/pia/dashboard/` | `io.py` (JSONL → DataFrame), `app.py` (Streamlit UI). |
| `context/` | Design notes (e.g. CIFAR stem and which layers are monitored). |
| `tests/` | Pytest suite; `pyproject.toml` sets `pythonpath = ["src"]` so `import pia` works. |

## Common commands

```bash
pipenv run pytest          # run tests
pipenv run ruff check src tests
pipenv run black --check src tests
pipenv run black src tests # apply formatting
```

## Conventions

See `AGENTS.md` for collaboration principles, tooling (black, ruff, pytest), and expectations around docstrings and structure.
