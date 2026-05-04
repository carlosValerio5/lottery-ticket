# Constraint-Driven Training for Modular Edge Models

Research codebase for training convolutional networks from scratch with sparsity-oriented losses (see `tasks.md` for the full roadmap). **Phase 1** provides a CIFAR-style ResNet-18 builder, explicit Kaiming initialization, and named submodules for later activation penalties.

## Prerequisites

- **Python 3.13** (matches `Pipfile` `[requires]`)
- [pipenv](https://pipenv.pypa.io/) for dependency management
- Optional: **CUDA**-capable GPU for later training phases; Phase 1 tests run on CPU.

## Setup

From the repository root:

```bash
pipenv install --dev
```

That creates a virtualenv and installs **torch**, **torchvision**, **numpy**, plus dev tools **pytest**, **black**, and **ruff**.

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
