## MODIFIED Requirements

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
