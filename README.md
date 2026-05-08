# Constraint-Driven Training for Modular Edge Models

Research codebase for training convolutional networks from scratch with sparsity-oriented losses (see `tasks.md` for the full roadmap). **The primary workflow is iterative magnitude pruning (IMP)**—the *lottery ticket* experiment: train a CIFAR ResNet-18 with `SparseLoss`, prune a fraction of surviving **Conv/Linear weights** by global magnitude (BatchNorm affine params and biases stay dense), optionally **rewind** to `θ₀`, to a **late-epoch checkpoint `θ_k`**, or **not at all**, then repeat. Each phase logs the same observability stack as single-run training (CSV/JSONL, TensorBoard, live batch traces).

Baseline metrics and pass/fail heuristics for comparing runs (e.g. vs `large-epoch`): [`context/imp_baseline_large_epoch.md`](context/imp_baseline_large_epoch.md). A compact hyperparameter sweep template: [`scripts/imp_compact_sweep.sh`](scripts/imp_compact_sweep.sh).

## Lottery ticket (IMP): quick start

From the repo root, with dependencies installed (`pipenv install --dev`), run:

```bash
cd /path/to/pia
PYTHONPATH=src pipenv run python -m pia.cli.lottery_ticket \
  --output-dir ./runs/lt --run-name my_imp \
  --num-rounds 5 --prune-per-round 0.2 --epochs-per-round 10 \
  --data-root ./data
```

**What this does:** there are `num_rounds + 1` training phases—round `0` is dense, then each subsequent round prunes `prune_per_round` of the *remaining* masked-in weights (globally across selected tensors), optionally reloads weights per `--rewind-mode`, reapplies the growing mask, and trains for `epochs_per_round` again.

**Recommended knobs when validation collapses after the first prune:** try `--rewind-mode late_k --rewind-epoch-k 3` or `4`, lower `--prune-per-round` (e.g. `0.1`), `--lr-scheduler cosine`, smaller `--lambda-weight` / `--gamma-activation`, `--weight-l1-aggregation mean` or `mean_per_param`, and optionally `--exclude-conv1-from-pruning` / `--exclude-fc-from-pruning`.

| Flag | Purpose |
|------|---------|
| `--rewind-mode theta0` | After each prune, reset to initial `θ₀` (default). |
| `--rewind-mode late_k` | Reset to `θ_k` saved at end of epoch `k` in round 0 (`theta_late_k.pt`). |
| `--rewind-mode none` | No weight reset; train from post-prune weights with mask applied. |
| `--rewind-epoch-k` | Epoch index (1-based) for `late_k`; must be ≤ `epochs-per-round`. |
| `--exclude-conv1-from-pruning` | Keep stem `conv1.weight` dense. |
| `--exclude-fc-from-pruning` | Keep classifier `fc.weight` dense. |
| `--lr-scheduler none\|cosine\|step` | Per-round LR schedule after each epoch. |
| `--weight-l1-aggregation sum\|mean\|mean_per_param` | Scale of weight L1 in `SparseLoss`. |
| `--abort-on-val-acc-drop` | If ≥ 0, stop the current phase when `val/acc` drops more than this between consecutive epochs. |

**Sparsity metrics (logged each epoch):** `pruning/mask_sparsity` and `sparsity/mask_zero_fraction` are the **IMP mask** zero fraction on pruned parameter tensors. `train/weight_sparsity_ratio` / `val/weight_sparsity_ratio` count **all** model parameters with \|w\| \< 1e-3 (different notion; useful but not the mask).

**Run directory** (example: `./runs/lt/my_imp/`):

| Path | Role |
|------|------|
| `theta_0.pt` | Initial `state_dict` (CPU); used when `--rewind-mode theta0`. |
| `theta_late_k.pt` | Checkpoint at epoch `k` of round 0 when using `late_k`. |
| `imp_index.json` | Versioned index: `schema_version`, `run_status`, `meta`, and `rounds` (list of per-round entries with `status`). Legacy list-only files are still supported by the dashboard reader. |
| `round_XX/` | Per-round artifacts (same layout as a normal training run): `metrics.csv`, `events.jsonl`, `live_batches.jsonl`, `tb/`, `summary.json`, `round_summary.json`. |
| `masks_round_XX.pt` | Saved masks after a prune (not written after the final round). |

Optional structured logging for the `pia` logger:

```bash
PYTHONPATH=src pipenv run python -m pia.cli.lottery_ticket \
  --output-dir ./runs/lt --run-name my_imp --json-log ./runs/lt/my_imp/pia_structured.logl \
  ...
```

### Lottery ticket dashboard (Streamlit)

The IMP dashboard is a **separate** Streamlit app from the single-run trainer (`pia.cli.dashboard` runs `dashboard/app.py`). For lottery ticket, Streamlit loads `src/pia/dashboard/lottery_ticket_app.py`.

**Requirements:** dev install includes **Streamlit ≥ 1.33**, **pandas**, and **Altair** (see [`Pipfile`](Pipfile) `[dev-packages]`).

**Feature flag:** set `PIA_STREAMLIT_DASHBOARD` to `1`, `true`, `yes`, or `on`. The app refuses to run without it.

**Default run directory:** set `PIA_RUN_DIR` to the IMP run folder (e.g. `./runs/lt/my_imp`) so the sidebar opens on the right experiment.

**Terminal A — IMP training:**

```bash
cd /path/to/pia
PYTHONPATH=src pipenv run python -m pia.cli.lottery_ticket \
  --output-dir ./runs/lt --run-name my_imp --epochs-per-round 20
```

**Terminal B — dashboard** (after `round_00` exists or once training has started):

```bash
cd /path/to/pia
export PIA_STREAMLIT_DASHBOARD=1
export PIA_RUN_DIR="$(pwd)/runs/lt/my_imp"
PYTHONPATH=src pipenv run python -m streamlit run src/pia/dashboard/lottery_ticket_app.py --server.port 8501
```

Open `http://localhost:8501` (or change `--server.port`).

**Spawn from the lottery CLI** (detached process; Streamlit logs default to `<run_dir>/dashboard_streamlit.log`):

```bash
PYTHONPATH=src pipenv run python -m pia.cli.lottery_ticket \
  --output-dir ./runs/lt --run-name my_imp \
  --spawn-dashboard --dashboard-port 8501
# Optional: --dashboard-log /tmp/streamlit_lt.log
```

**Using the UI:**

- **Sidebar — “Directorio del run IMP”:** path to the run root (must contain `imp_index.json` and/or `round_XX` folders). Pre-filled from `PIA_RUN_DIR`.
- **Sidebar — poll interval:** how often `imp_index.json` is re-read for the global summary (live batch charts refresh about every second regardless).
- **“Resumen por ronda”:** metrics and Altair charts for target vs achieved sparsity and `val/acc` across IMP rounds, plus a table of all rounds.
- **“Ronda a inspeccionar”:** pick a round; the main panel shows **live** train/val curves from that round’s `live_batches.jsonl` (current epoch) and **closed-epoch** curves from `events.jsonl` (accuracy, loss, foldable CE-only loss).

Caption on the page summarizes the three files: `live_batches.jsonl` (intra-epoch), `events.jsonl` (per epoch), `imp_index.json` (cross-round summary).

---

## Single-run CIFAR training (`train_cifar`)

For one training job without IMP, use:

```bash
cd /path/to/pia
PYTHONPATH=src pipenv run python -m pia.cli.train_cifar --epochs 1 --data-root ./data --logdir ./runs --run-name smoke
```

Grid search, TensorBoard, JSON logging, and the **single-run** dashboard (`python -m pia.cli.dashboard --run-dir ./runs/exp1`) behave as before; see the sections below for artifact layout and observability details that also apply under each `round_XX/` directory during IMP.

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

Optional structured log file for the `pia` logger:

```bash
PYTHONPATH=src pipenv run python -m pia.cli.train_cifar --json-log ./runs/exp1/pia_structured.logl ...
```

### Live training dashboard (single run)

The composite observer writes **`live_batches.jsonl`** and **`events.jsonl`** under the run directory. The **single-run** app is opt-in via `PIA_STREAMLIT_DASHBOARD` and launched with:

```bash
export PIA_STREAMLIT_DASHBOARD=1
PYTHONPATH=src pipenv run python -m pia.cli.dashboard --run-dir ./runs/exp1
```

**Spawn from training (single run only; ignored with `--grid`):**

```bash
PYTHONPATH=src pipenv run python -m pia.cli.train_cifar --logdir ./runs --run-name exp1 \
  --spawn-dashboard --dashboard-port 8501
```

### Logs and observability

Training uses a **composite observer** (`build_default_observer` in `src/pia/training/grid_search.py`): each epoch writes the same scalar metrics to several sinks so you can watch progress live, audit a run later, or ship logs to another system.

#### Console (live progress)

- **`tqdm`** bars for the training and validation passes each epoch (`train`, `val`). The training bar’s postfix shows **total loss**, **cross-entropy (task) term**, and **batch accuracy** so you can spot instability without opening files.
- The CLI also logs start messages on the **`pia.cli`** logger (INFO).

#### Run directory layout (`--logdir` / per-run folder)

For a single run (`--logdir ./runs --run-name exp1`), artifacts land under **`./runs/exp1/`** (and the same filenames appear under each IMP `round_XX/`):

| Artifact | Role |
|----------|------|
| `metrics.csv` | One **CSV row per epoch**. The header is fixed on the **first** epoch from all keys in that row: `epoch`, run config fields (`run_id`, `lambda_weight`, `gamma_activation`, `epochs`, `batch_size`, `lr`, `data_root`, `device`, optional `git_sha`), and aggregated metrics (see below). |
| `events.jsonl` | **Append-only JSON lines**: each line is one epoch, merging the same run **config** snapshot with `epoch` and all **float metrics** (good for `jq`, log aggregators, or diffing across retries). |
| `live_batches.jsonl` | **Per-batch JSON lines** while the **current** epoch runs: `phase` (`train` / `val`), `epoch`, `batch`, `loss`, `loss_task`, `acc`. Truncated at the start of each training phase; the Streamlit dashboards read it for intra-epoch charts. |
| `tb/` | **TensorBoard** event files. Each metric name is a scalar; **`global_step` is the epoch index** (1-based in the loop). Point TensorBoard at this directory (as in the command above). |
| `summary.json` | Written **once at the end** of the run: `{"config": {...}, "metrics": {...}}` with the **last epoch’s** merged train+val scalars. |

During IMP, epoch metrics also include `pruning/imp_round`, `pruning/mask_sparsity`, and `pruning/prune_fraction_per_step`.

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

## Prerequisites

- **Python 3.13** (matches `Pipfile` `[requires]`)
- [pipenv](https://pipenv.pypa.io/) for dependency management
- Optional: **CUDA**-capable GPU for later training phases; Phase 1 tests run on CPU.

## Setup

From the repository root:

```bash
pipenv install --dev
```

That creates a virtualenv and installs **torch**, **torchvision**, **numpy**, **tensorboard**, **tqdm**, **scikit-learn**, **matplotlib**, plus dev tools **pytest**, **black**, **ruff**, **pandas**, **streamlit**, and **altair** (dashboards).

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
| `src/pia/pruning/lottery_ticket.py` | IMP orchestration (`iterative_magnitude_pruning`). |
| `src/pia/pruning/` | Masks and magnitude pruning helpers. |
| `src/pia/training/` | `fit`, observers, optional `grid_search` in `grid_search.py`. |
| `src/pia/observability/logging_config.py` | `setup_pia_logging` (stderr + optional JSON file). |
| `src/pia/inference/predict.py` | `run_inference_batch` for production-style latency logs. |
| `src/pia/cli/train_cifar.py` | CLI entry for single-run training (`python -m pia.cli.train_cifar`). |
| `src/pia/cli/lottery_ticket.py` | CLI entry for IMP (`python -m pia.cli.lottery_ticket`). |
| `src/pia/cli/dashboard.py` | CLI entry for the **single-run** Streamlit app (`python -m pia.cli.dashboard`). |
| `src/pia/dashboard/app.py` | Streamlit UI for one training run. |
| `src/pia/dashboard/lottery_ticket_app.py` | Streamlit UI for IMP / lottery ticket runs. |
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
