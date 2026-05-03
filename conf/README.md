# Per-model training configs

This directory holds per-architecture training settings — one TOML file per model registered in `models.MODELS`. `train.py` reads `conf/<MODEL_NAME>.toml` at startup via the standard-library `tomllib`. Editing a hyperparameter does not require touching Python.

## Convention

- One file per model, named `<model>.toml` where `<model>` matches a key in `models.MODELS` (e.g. `unet.toml`, `vit.toml`).
- Required fields per file: `learning_rate` (float), `num_epochs` (int).
- Adding a new model to the registry requires creating a matching file here. A missing or malformed config file makes `train.py` fail at startup, before any data generation or GPU allocation.

## What does not belong here

- **Dataset / training-loop constants** (`BATCH_SIZE`, `SEED`, `NUM_SAMPLES`, `IMAGE_SIZE`, `NUM_LABELED`, `MAX_LABELS`) — these describe the experiment, not the architecture, and stay as module-level constants in `train.py`.
- **Run metadata** (timestamps, git SHAs, output paths) — those are produced by the run, not configured.
- **Layered overrides, environment-variable injection, CLI flags** — out of scope; this is a single-file, single-source-of-truth setup.
