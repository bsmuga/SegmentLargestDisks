## 1. Pre-flight

- [x] 1.1 Confirm `per-model-training-config` is applied (the inline `MODEL_CONFIGS` dict exists in `train.py`) and archived under `openspec/changes/archive/`. If it is not, stop — this change depends on it.
- [x] 1.2 Note the current values of `MODEL_CONFIGS["unet"]` and `MODEL_CONFIGS["vit"]` from `train.py`; these become the seed contents of the new TOML files

## 2. Create the `conf/` catalog

- [x] 2.1 Create the `conf/` directory at the repository root
- [x] 2.2 Create `conf/README.md`: one paragraph stating that `conf/` holds per-model training configs, the file-naming convention (`<model>.toml` matching keys in `models.MODELS`), the required schema (`learning_rate`, `num_epochs`), and an out-of-scope note (no dataset or training-loop constants live here)

## 3. Author per-model TOML files

- [x] 3.1 Create `conf/unet.toml` with `learning_rate` and `num_epochs` values copied verbatim from `MODEL_CONFIGS["unet"]`
- [x] 3.2 Create `conf/vit.toml` with `learning_rate` and `num_epochs` values copied verbatim from `MODEL_CONFIGS["vit"]`

## 4. Update `train.py` to load from `conf/`

- [x] 4.1 In `train.py`, add `import tomllib` (alphabetical placement among standard-library imports)
- [x] 4.2 Replace the `MODEL_CONFIGS` dict with a small loader that opens `os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf", f"{MODEL_NAME}.toml")` in binary mode and calls `tomllib.load`
- [x] 4.3 Bind `LEARNING_RATE` and `NUM_EPOCHS` (or whatever local names the training loop currently uses post `per-model-training-config`) from the loaded config dict, preserving the rest of `main()` unchanged
- [x] 4.4 Verify there are no remaining references to `MODEL_CONFIGS` in `train.py` after the edit

## 5. Wire `conf/` into the README

- [x] 5.1 In `README.md`, add a single line under "Project structure" pointing to `conf/` with a one-phrase description (e.g., "per-model training configs (TOML)")

## 6. Verification

- [x] 6.1 Run `python -c "import train"` (or the equivalent module-load smoke check) with `MODEL_NAME = "unet"` and confirm no exception
- [x] 6.2 Repeat with `MODEL_NAME = "vit"` and confirm no exception
- [x] 6.3 Temporarily rename `conf/vit.toml` aside, run `train.py` with `MODEL_NAME = "vit"`, confirm a `FileNotFoundError` is raised before model construction; restore the file
- [x] 6.4 Add or modify a key in `conf/unet.toml` to drop `learning_rate`, run with `MODEL_NAME = "unet"`, confirm `KeyError`; restore the file
- [x] 6.5 Run `python train.py` with default `MODEL_NAME` for a single epoch (e.g., temporarily lower `NUM_EPOCHS` in the conf file) and confirm training proceeds end-to-end with no shape or import errors
- [x] 6.6 Run `pytest tests/` and confirm all existing tests still pass
- [x] 6.7 Run `flake8 .` and confirm no new violations are introduced by the loader edit in `train.py`
