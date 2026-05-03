# baseline-experiment Specification

## Purpose
TBD - created by archiving change fix-and-finish-baseline. Update Purpose after archive.
## Requirements
### Requirement: README accurately describes the runnable repository

The `README.md` SHALL only reference files, scripts, and commands that exist in the repository at the same revision. Every install command, project-structure entry, and quick-start example MUST resolve against the actual filesystem.

#### Scenario: Quick-start references resolve

- **WHEN** a contributor follows the README's "Quick start" section verbatim from a fresh clone
- **THEN** every file path or script invocation referenced in that section MUST exist in the repository or be created by a step earlier in the same section

#### Scenario: Project structure listing matches reality

- **WHEN** a contributor reads the README's "Project structure" block
- **THEN** every file or directory listed MUST exist at the repository root or in the path shown

### Requirement: `evaluate.py` runs end-to-end against a freshly trained model with no extra flags

After running the training entrypoint to completion, the evaluation entrypoint with default arguments (no `--checkpoint` override) SHALL successfully load the most-recently-saved final checkpoint for the given model.

#### Scenario: Default evaluation invocation succeeds

- **WHEN** a contributor runs `python train.py` to completion with the default `MODEL_NAME`, then runs `python evaluate.py --model <same-name>`
- **THEN** `evaluate.py` MUST successfully load the model checkpoint and produce evaluation metrics, without the contributor needing to pass `--checkpoint`

#### Scenario: Explicit checkpoint path still honored

- **WHEN** a contributor runs `python evaluate.py --model <name> --checkpoint <some/other/path>`
- **THEN** `evaluate.py` MUST attempt to load from `<some/other/path>` and not the default

### Requirement: Training uses a batch size where BatchNorm statistics are valid

Any model in the registry that uses `BatchNorm` SHALL be trained with a batch size strictly greater than 1. The training script's default `BATCH_SIZE` constant MUST satisfy this.

#### Scenario: Default BATCH_SIZE supports BatchNorm

- **WHEN** the training script is loaded with its default `BATCH_SIZE` constant
- **THEN** `BATCH_SIZE` MUST be `>= 8`, ensuring per-batch mean/variance estimates have at least 8 samples per dimension

### Requirement: `dataset.render_sample` paints labels at the correct pixels

The `render_sample` function in `dataset.py` SHALL produce, for each input row, an image whose foreground pixels and a mask whose label pixels exactly match the disk's geometry: pixel `(r, c)` is foreground when its squared distance from the disk center is strictly less than the disk radius squared, and carries the disk's label only if the disk has a non-zero label.

#### Scenario: A single labeled disk paints its label only inside its radius

- **WHEN** `render_sample` is called with a DataFrame containing one disk at `(cx, cy)` with radius `r > 0` and label `L > 0`, on a canvas of size `(W, H)`
- **THEN** the returned mask MUST satisfy: `mask[y, x] == L` for every `(x, y)` with `(x - cx)^2 + (y - cy)^2 < r^2`, and `mask[y, x] == 0` everywhere else

#### Scenario: Unlabeled disks contribute to the image but not to the mask

- **WHEN** `render_sample` is called with a DataFrame whose row has `label == 0`
- **THEN** the returned image MUST mark that disk's interior as foreground (`1.0`), AND the returned mask MUST remain `0` at those same pixels

#### Scenario: Multiple disks render without label collision

- **WHEN** `render_sample` is called with two disjoint disks `A` (label 1) and `B` (label 2)
- **THEN** the returned mask MUST equal `1` inside `A`, `2` inside `B`, and `0` everywhere else

### Requirement: Training hyperparameters that genuinely differ per architecture are configured per model

The training script SHALL source per-architecture hyperparameters from per-model TOML files located under a top-level `conf/` directory, with one file per registered model named `conf/<model>.toml`. Each file MUST contain at least `learning_rate` and `num_epochs`. The active model's file SHALL be the source of truth at training time; the in-file `MODEL_CONFIGS` mapping previously used MUST NOT exist alongside the per-model files. A missing or unreadable `conf/<MODEL_NAME>.toml`, or one missing a required key, MUST cause `train.py` to fail at startup before any data generation or model construction.

#### Scenario: Each registered model has its own learning rate and epoch count in `conf/`

- **WHEN** the training script is loaded with `MODEL_NAME` set to any model name registered in `models.MODELS`
- **THEN** the script MUST read `learning_rate` and `num_epochs` by parsing `conf/<MODEL_NAME>.toml` via the standard-library `tomllib`, AND MUST NOT fall back to module-level `LEARNING_RATE` / `NUM_EPOCHS` constants or to an in-file `MODEL_CONFIGS` mapping

#### Scenario: All registry models have a config file

- **WHEN** any model is added to `models.MODELS`
- **THEN** a file `conf/<model>.toml` MUST exist at the repository root with both `learning_rate` and `num_epochs` populated, so that switching `MODEL_NAME` to that model does not raise `FileNotFoundError` or `KeyError`

#### Scenario: A missing config file fails fast

- **WHEN** the training script is started with a `MODEL_NAME` for which no matching `conf/<model>.toml` exists
- **THEN** the script MUST raise `FileNotFoundError` (or equivalent) before constructing the model, generating data, or allocating GPU memory

#### Scenario: A malformed config file fails fast

- **WHEN** the training script reads a `conf/<model>.toml` that is missing `learning_rate` or `num_epochs`
- **THEN** the script MUST raise `KeyError` (or equivalent) before training begins

#### Scenario: Config path is anchored to the script, not the working directory

- **WHEN** `train.py` is invoked from any working directory (e.g. `python /abs/path/to/train.py`)
- **THEN** the loader MUST resolve `conf/<MODEL_NAME>.toml` relative to the location of `train.py` itself, so behavior is independent of the caller's CWD

### Requirement: Per-run output for each model is isolated under `logs/<model>/`

The training script SHALL write all per-run output files (Python log, metrics CSV, training-curve plot) for a given run into a `logs/<model>/` subdirectory keyed by the active `MODEL_NAME`. Output for one model MUST NOT share a filesystem path with output for another model.

#### Scenario: UNet and ViT runs write to disjoint folders

- **WHEN** a contributor runs `python train.py` once with `MODEL_NAME = "unet"` and once with `MODEL_NAME = "vit"`
- **THEN** the unet run's metrics CSV, training plot, and per-run log file MUST be located under `logs/unet/`, AND the vit run's same artifacts MUST be located under `logs/vit/`, AND no file written by the unet run MUST share a path with any file written by the vit run

#### Scenario: Per-run log file preserves a timestamp

- **WHEN** the training script starts and obtains its logger
- **THEN** the resulting per-run log file MUST be located at `logs/<model>/<name>_<timestamp>.log`, where `<name>` is the logger name (`"train"`) and `<timestamp>` matches the existing `YYYYMMDD_HHMMSS` format

