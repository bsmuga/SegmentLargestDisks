## ADDED Requirements

### Requirement: Training hyperparameters that genuinely differ per architecture are configured per model

The training script SHALL maintain a single, in-file mapping that supplies architecture-specific hyperparameters for each model in the registry. The mapping MUST cover at least the learning rate and the number of epochs. The active model's entry SHALL be the source of truth at training time; module-level constants for these per-model values MUST NOT exist alongside the mapping.

#### Scenario: Each registered model has its own learning rate and epoch count

- **WHEN** the training script is loaded with `MODEL_NAME` set to any model name registered in `models.MODELS`
- **THEN** the script MUST resolve `learning_rate` and `num_epochs` from a `MODEL_CONFIGS` mapping keyed by that model name, and MUST NOT fall back to a globally shared `LEARNING_RATE` or `NUM_EPOCHS` constant

#### Scenario: All registry models have a config entry

- **WHEN** any model is added to `models.MODELS`
- **THEN** `MODEL_CONFIGS` MUST contain an entry under the same key with both `learning_rate` and `num_epochs` populated, so that switching `MODEL_NAME` to that model does not raise `KeyError`

### Requirement: Per-run output for each model is isolated under `logs/<model>/`

The training script SHALL write all per-run output files (Python log, metrics CSV, training-curve plot) for a given run into a `logs/<model>/` subdirectory keyed by the active `MODEL_NAME`. Output for one model MUST NOT share a filesystem path with output for another model.

#### Scenario: UNet and ViT runs write to disjoint folders

- **WHEN** a contributor runs `python train.py` once with `MODEL_NAME = "unet"` and once with `MODEL_NAME = "vit"`
- **THEN** the unet run's metrics CSV, training plot, and per-run log file MUST be located under `logs/unet/`, AND the vit run's same artifacts MUST be located under `logs/vit/`, AND no file written by the unet run MUST share a path with any file written by the vit run

#### Scenario: Per-run log file preserves a timestamp

- **WHEN** the training script starts and obtains its logger
- **THEN** the resulting per-run log file MUST be located at `logs/<model>/<name>_<timestamp>.log`, where `<name>` is the logger name (`"train"`) and `<timestamp>` matches the existing `YYYYMMDD_HHMMSS` format
