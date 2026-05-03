## 1. README accuracy

- [x] 1.1 In `README.md`, replace the "model.py" reference under "How it works" with the actual `models/unet.py` (UNet) and `models/vit.py` (ViT) layout
- [x] 1.2 In `README.md` "Quick start", remove the `./install.sh` line and the surrounding "Auto-detect CUDA" note
- [x] 1.3 In `README.md` "Quick start", change `requirements-cuda.txt` to `requirements.txt`
- [x] 1.4 In `README.md` "Project structure", remove the `model.py` entry and add `models/` (with `unet.py`, `vit.py`)
- [x] 1.5 In `README.md` "Requirements", change the `requirements-cuda.txt` reference to `requirements.txt`

## 2. Default checkpoint path alignment

- [x] 2.1 In `evaluate.py`, change the default for `--checkpoint` (currently `checkpoints/<model>`) to `checkpoints/<model>/final` so it matches what `train.py` writes for the final checkpoint
  - **Discovered during apply**: orbax requires the checkpoint path to be absolute; added `os.path.abspath(args.checkpoint)` in `main()` so the default (and any user-supplied relative path) loads correctly. This was a pre-existing latent bug exposed by the spec scenario "Default evaluation invocation succeeds".
- [x] 2.2 Manually verify that `python evaluate.py --model unet` (no `--checkpoint` flag) loads a freshly-trained checkpoint without error
  - Verified for both `unet` and `vit` against checkpoints saved by the smoke training in 3.2.

## 3. BatchNorm-safe batch size

- [x] 3.1 In `train.py`, change `BATCH_SIZE` from `1` to `8`
- [x] 3.2 Run a short smoke training (e.g., reduced `NUM_EPOCHS` and `NUM_SAMPLES`) locally to confirm no OOM or shape errors at the new batch size for both `unet` and `vit`
  - Ran one `train_step` per model at `BATCH_SIZE=8`, `(128, 128, 1)` input. Loss finite for both (`unet` ≈ 0.954, `vit` ≈ 0.917). No OOM, no shape errors. Final checkpoints written to `checkpoints/{unet,vit}/final`.

## 4. Correctness test for `render_sample`

- [x] 4.1 Create `tests/test_dataset.py`
- [x] 4.2 Add a test: a single labeled disk paints its label only inside its radius, and `0` outside (covers the first scenario in `baseline-experiment` spec)
- [x] 4.3 Add a test: a disk with `label == 0` paints the image but leaves the mask at `0` (covers the second scenario)
- [x] 4.4 Add a test: two disjoint disks with distinct labels render without label collision (covers the third scenario)
- [x] 4.5 Run `pytest tests/test_dataset.py` and confirm all tests pass — 6 tests pass

## 5. Verification

- [x] 5.1 Run `pytest tests/` and confirm all existing tests still pass — 81/81 pass (75 existing + 6 new)
- [x] 5.2 Run `flake8 .` (or the project's configured linter) and confirm no new violations — only pre-existing violations remain; new `tests/test_dataset.py` is flake8-clean
- [x] 5.3 Read through the updated `README.md` once more from a fresh-contributor perspective to catch any remaining drift — added a brief "Evaluate" command section (was missing alongside the "Train" section)
