## Why

The baseline disk-segmentation experiment has drifted: the README references files and scripts that no longer exist, `evaluate.py`'s default checkpoint path does not match where `train.py` writes checkpoints, and training uses `BATCH_SIZE=1` with `BatchNorm` — a combination that makes BatchNorm statistics meaningless and silently degrades results. The repo's stated identity is "a minimal experiment"; right now a fresh contributor cannot run it end-to-end without trip-hazards. This change closes the gap without adding new features.

## What Changes

- Update `README.md` to match the actual code:
  - Replace the `model.py` reference with the `models/unet.py` + `models/vit.py` layout
  - Remove or correct the `./install.sh` reference (file does not exist)
  - Correct `requirements-cuda.txt` to `requirements.txt` (or rename the file — see design)
  - Remove `model.py` from the "Project structure" listing
- Align `evaluate.py`'s default checkpoint path with where `train.py` actually writes the final checkpoint, so `python evaluate.py --model unet` works without manual `--checkpoint` flag.
- Raise `train.py`'s `BATCH_SIZE` from `1` to a value where `BatchNorm` is statistically valid (target: `8`).
- Add one correctness test for `dataset.render_sample` — that labels land at the right pixels for the right disks. This is the only piece of load-bearing logic currently untested.

Non-goals (explicitly out of scope to keep this minimal):

- No `argparse` for `train.py` hyperparameters
- No best-checkpoint selection / early stopping
- No new loss functions or class-imbalance handling
- No vectorization of `render_sample` or `validate_disjoint`
- No new tooling (mypy, pre-commit, lockfiles)

## Capabilities

### New Capabilities

- `baseline-experiment`: The end-to-end "generate data → train → evaluate" workflow that is the project's stated purpose. Captures the runnability and correctness invariants that must hold for the experiment to be reproducible by a fresh contributor.

### Modified Capabilities

<!-- None. No existing specs in openspec/specs/. -->

## Impact

- **Code**: `README.md`, `train.py` (one constant), `evaluate.py` (one default), `tests/` (one new test file or one added test in `test_dataset.py`).
- **Behavior**: Training runs with a meaningful BatchNorm; default `evaluate.py` invocation succeeds against a freshly trained model; documented commands match reality.
- **No breaking changes** to the public interface — checkpoints already written under the old path remain loadable via `--checkpoint`.
- **CI**: No changes; the existing `python-tests.yml` continues to run all tests including the new one.
