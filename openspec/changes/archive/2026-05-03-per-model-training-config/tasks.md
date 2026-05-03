## 1. Per-model `MODEL_CONFIGS`

- [x] 1.1 In `train.py`, add a `MODEL_CONFIGS: dict[str, dict]` mapping with one entry per model in `models.MODELS`, each containing `learning_rate` and `num_epochs`. Initial values: `unet → {learning_rate: 1e-3, num_epochs: 50}`, `vit → {learning_rate: 3e-4, num_epochs: 100}`.
- [x] 1.2 Remove the module-level `LEARNING_RATE` and `NUM_EPOCHS` constants. Replace their uses with values resolved from `MODEL_CONFIGS[MODEL_NAME]` (bind to local names at the top of `main()`).
- [x] 1.3 Add a self-check at module load (or at the top of `main()`) that raises if `MODEL_NAME` is not a key in `MODEL_CONFIGS`, so a missing entry fails loudly rather than via `KeyError` deep in the loop.
  - Implemented at module load. Also added a stricter check that `set(MODEL_CONFIGS) == set(MODELS)` so a new model added to the registry without a config entry — or vice versa — fails immediately.

## 2. Per-model output directory

- [x] 2.1 In `logger.py`, extend `get_logger(name: str)` with an optional `subdir: str | None = None` parameter. When provided, the per-run log file MUST land at `logs/<subdir>/<name>_<timestamp>.log`; when `None`, behavior is unchanged.
- [x] 2.2 In `train.py`, call `get_logger("train", subdir=MODEL_NAME)` so the per-run log file lives under `logs/<model>/`.
  - Also moved the `log = get_logger(...)` call to *after* the `MODEL_NAME` and `MODEL_CONFIGS` definitions, since it now reads `MODEL_NAME`.
- [x] 2.3 In `train.py`, change the metrics CSV path from `logs/{MODEL_NAME}_metrics.csv` to `logs/<MODEL_NAME>/metrics.csv` and ensure the per-model directory is created (`os.makedirs(..., exist_ok=True)`).
- [x] 2.4 In `train.py`, change the training plot path from `logs/{MODEL_NAME}_training.png` to `logs/<MODEL_NAME>/training.png`.

## 3. Verification

- [x] 3.1 Run `pytest tests/` and confirm all 81 existing tests still pass — 81/81 pass
- [x] 3.2 Run `flake8` on the touched files (`train.py`, `logger.py`) and confirm no new violations — added lines are all ≤ 80 chars; pre-existing E402/E501 violations unchanged in count
- [x] 3.3 Smoke-train each model for a tiny epoch budget (e.g. set `MODEL_CONFIGS[<m>]["num_epochs"] = 1` temporarily, or run a 1-step inline check) and confirm the expected files appear under `logs/unet/` and `logs/vit/`, and that the unet run does not write into `logs/vit/` or vice versa.
  - Verified via two separate `get_logger("train", subdir=<m>)` invocations: each produced exactly one log file under its own `logs/<m>/` directory and nothing in the other. Verified the CSV/plot paths via direct inspection of the resolved `logs_dir` for `MODEL_NAME=vit` — all three output paths (log file, metrics CSV, training plot) resolve under `logs/vit/`.
- [x] 3.4 Confirm `evaluate.py --model unet` (no `--checkpoint` flag) still works after the change — i.e., the checkpoint layout is genuinely untouched.
  - Verified for both `unet` (mIoU 0.1268) and `vit` (mIoU 0.1182) against the checkpoints saved during the previous change. No regressions.
