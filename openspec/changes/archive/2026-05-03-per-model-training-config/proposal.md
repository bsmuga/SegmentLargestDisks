## Why

`train.py` currently has one set of hyperparameters (`LEARNING_RATE = 1e-3`, `NUM_EPOCHS = 100`) and one set of output filenames (`logs/{MODEL_NAME}_metrics.csv`, `logs/{MODEL_NAME}_training.png`, plus a flat `logs/train_<timestamp>.log`). UNet and ViT have meaningfully different training dynamics — ViT typically wants a smaller learning rate — so a single shared LR is wrong for at least one of them. Output files use the model name as a *prefix*, so successive runs of the same model overwrite each other and runs of different models share the same `logs/` namespace, which makes comparisons noisy.

Both problems have the same fix: introduce a tiny per-model lookup of the settings that genuinely differ, and route per-run output into `logs/<model>/`. Keep the rest shared.

## What Changes

- Add a `MODEL_CONFIGS` mapping in `train.py` keyed by model name with two fields per entry: `learning_rate`, `num_epochs`. Replace the module-level `LEARNING_RATE` and `NUM_EPOCHS` constants with lookups against the active `MODEL_NAME`.
- Route per-run output for the active model into a `logs/<model>/` subdirectory:
  - `logs/<model>/metrics.csv` (was `logs/{model}_metrics.csv`)
  - `logs/<model>/training.png` (was `logs/{model}_training.png`)
  - `logs/<model>/<name>_<timestamp>.log` (was `logs/<name>_<timestamp>.log`)
- Update `logger.get_logger` to accept an optional subdirectory so the log file lands in the per-model folder.

Non-goals (explicitly out of scope to keep this minimal):

- No per-model `BATCH_SIZE` (stays at `8` for both — BatchNorm-safety constraint already established).
- No per-model `SEED`, `NUM_SAMPLES`, `IMAGE_SIZE`, `NUM_LABELED`, `MAX_LABELS` (these describe the data, not the model).
- No CLI flags, YAML/TOML configs, or `argparse` for `train.py`.
- No experiment-tracking integration (mlflow/wandb).
- No restructuring of `checkpoints/` (already per-model under `checkpoints/<model>/`).
- No `runs/<model>/` unified folder — top-level layout stays `checkpoints/<model>/` + `logs/<model>/`.

## Capabilities

### New Capabilities

<!-- None — no new capability folders. -->

### Modified Capabilities

- `baseline-experiment`: Add two new requirements covering per-model training configuration and per-model output isolation. The existing four requirements remain unchanged.

## Impact

- **Code**: `train.py` (introduce `MODEL_CONFIGS`, swap two constants for lookups, route output through `logs/<model>/`). `logger.py` (add an optional subdirectory argument; default behavior unchanged for unrelated callers).
- **On-disk**: New per-run output lands under `logs/<model>/`. Pre-existing flat `logs/<model>_metrics.csv` etc. are not migrated — they were per-run anyway and are gitignored.
- **No breaking changes** to `evaluate.py`, the model registry, the dataset, or test suite.
- **CI**: No changes; existing tests remain valid.
