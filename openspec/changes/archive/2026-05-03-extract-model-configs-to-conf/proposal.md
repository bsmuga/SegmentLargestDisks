## Why

Once `per-model-training-config` lands, per-architecture hyperparameters (learning rate, number of epochs) live as an in-file `MODEL_CONFIGS` dict inside `train.py`. That works, but it conflates *training-loop code* with *experiment knobs*: editing a hyperparameter requires touching a Python module, every config-only diff shows up as a code change, and there is no separation between "the recipe" and "the engine that runs it." Externalizing per-model configs into a dedicated `conf/` subfolder keeps `train.py` focused on the training loop and gives configs a stable, easy-to-find home as more architectures (or experiment variants) are added later.

This change builds **on top of** `per-model-training-config`. It replaces the inline `MODEL_CONFIGS` dict introduced there with per-model TOML files. The behavior visible to a contributor running `python train.py` is unchanged; only the source of truth for hyperparameters moves.

## What Changes

- Add a top-level `conf/` directory that holds one TOML file per registered model, named `conf/<model>.toml` (e.g. `conf/unet.toml`, `conf/vit.toml`). Each file carries the same fields as the inline `MODEL_CONFIGS` entry it replaces (`learning_rate`, `num_epochs`).
- Remove the `MODEL_CONFIGS` dict from `train.py`. Replace it with a small loader that reads `conf/<MODEL_NAME>.toml` at startup using the standard-library `tomllib` (Python ≥ 3.11; the project already requires ≥ 3.14).
- Add a one-line note to `README.md` "Project structure" pointing to `conf/`.

Non-goals (explicitly out of scope to keep this minimal):

- No new file format support beyond TOML (no YAML, JSON, or Python config modules).
- No environment-variable overrides, CLI flags, or layered config (base + override).
- No nested layout like `conf/training/<model>.toml` or `conf/<model>/training.toml` — flat is enough at this scale.
- No fields beyond what `MODEL_CONFIGS` already covers; this change is a pure source-of-truth move, not a settings expansion.
- No restructuring of `checkpoints/` or `logs/` (still per-model under their own roots).
- No conf for non-model parameters (`BATCH_SIZE`, `SEED`, `NUM_SAMPLES`, `IMAGE_SIZE`, etc.) — those are dataset/training-loop constants, not per-architecture knobs.
- No new dependency on PyYAML, Hydra, OmegaConf, pydantic, or any other config library.

## Capabilities

### New Capabilities

<!-- None — this change does not introduce a new capability folder. -->

### Modified Capabilities

- `baseline-experiment`: Modifies the existing requirement "Training hyperparameters that genuinely differ per architecture are configured per model" so that the source of truth is `conf/<model>.toml`, not an in-file `MODEL_CONFIGS` mapping. The other requirements (output isolation, README accuracy, etc.) remain unchanged.

## Impact

- **Code**: `train.py` (drop the `MODEL_CONFIGS` dict, add a small `tomllib`-based loader). No other source files change.
- **Files added**: `conf/unet.toml`, `conf/vit.toml`. Optionally `conf/README.md` describing the convention (one paragraph).
- **Files modified**: `README.md` — one new line under "Project structure".
- **Dependencies**: None added. `tomllib` is in the standard library on the project's required Python version.
- **Ordering**: This change assumes `per-model-training-config` has been applied and archived first. Without that, there is no `MODEL_CONFIGS` dict in `train.py` to extract; the loader's target schema (`learning_rate`, `num_epochs`) comes from that earlier change.
- **No breaking changes** to `evaluate.py`, the model registry, the dataset, or the test suite.
- **CI**: No changes; existing tests remain valid. A small new test may verify the loader picks up the right values for each registered model.
