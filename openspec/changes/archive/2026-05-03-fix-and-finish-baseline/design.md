## Context

The repo is a small experimental ML project (UNet/ViT segmentation of the N largest disks). It is small enough that the friction of any drift between docs, training, and evaluation has outsized impact on a fresh run. Today three independent issues conspire to break the "clone → run" path:

1. The README mentions files that do not exist (`model.py`, `install.sh`, `requirements-cuda.txt`).
2. `evaluate.py`'s default `--checkpoint` is `checkpoints/<model>`, but `train.py` writes the final checkpoint to `checkpoints/<model>/final` — so the default invocation fails.
3. `train.py` sets `BATCH_SIZE = 1` while both models use `BatchNorm`. Per-batch normalization with a single sample produces unstable activations, noisy running statistics, and a meaningful train/eval gap. This is silent: training "works" but converges worse than it should.

There are no existing specs in `openspec/specs/`, so this is the first capability to be defined for the project.

## Goals / Non-Goals

**Goals:**

- A fresh contributor can run `python train.py` then `python evaluate.py --model <name>` end-to-end without reading source.
- Training BatchNorm receives a statistically valid batch.
- Mask rendering — the only untested load-bearing piece of logic — gains a correctness test.
- Stay minimal: change as little code as possible.

**Non-Goals:**

- Refactoring `train.py` to take CLI arguments.
- Adding best-checkpoint selection, schedulers, early stopping.
- Changing the loss function or addressing class imbalance.
- Performance work on `render_sample` or `validate_disjoint`.
- New tooling (mypy, pre-commit, lockfiles).
- Replacing BatchNorm with LayerNorm/GroupNorm (a valid alternative — see Decisions).

## Decisions

### Decision 1: Fix checkpoint mismatch in `evaluate.py`, not `train.py`

`train.py` writes both periodic (`checkpoints/<model>/epoch_N`) and final (`checkpoints/<model>/final`) checkpoints. Changing `evaluate.py`'s default to `checkpoints/<model>/final` is a one-line edit and preserves the existing on-disk layout (the `epoch_N` snapshots remain useful). Changing `train.py` would either lose the periodic-snapshot layout or require a more invasive layout shuffle.

**Alternative considered**: have `train.py` save the final checkpoint as `checkpoints/<model>/` directly. Rejected — collides with the periodic-snapshot subdirectories under the same parent.

### Decision 2: Set `BATCH_SIZE = 8`

`8` is the smallest power of two where BatchNorm running statistics become statistically reasonable for our input shape (`128×128×1`). It fits comfortably in 4 GB GPU memory for both UNet (~7 M params) and ViT (~2.3 M params). Larger batches (16, 32) would also work; `8` is chosen as the most conservative value that still fixes the bug.

**Alternative considered**: replace `BatchNorm` with `LayerNorm` or `GroupNorm` in both models. This decouples training stability from batch size and is the architecturally cleaner fix. Rejected for this change — it touches four BatchNorm sites in `unet.py` and four in `vit.py`, changes parameter shapes (so existing checkpoints become unloadable), and is a research-flavored decision better made on its own. Re-open if a future change wants single-sample training.

### Decision 3: Update the README rather than renaming `requirements.txt`

The README references `requirements-cuda.txt`; the file is named `requirements.txt` and contains `jax[cuda12]`. Two options:

| Option | Diff size | Side effects |
|--------|-----------|-------------|
| Rename `requirements.txt` → `requirements-cuda.txt` | 1 file rename + README edit | Any external doc/script referencing `requirements.txt` breaks |
| Update README to reference `requirements.txt` | README edit only | None |

Choosing the README edit. It is more minimal and has zero blast radius beyond docs. The `install.sh` reference is removed entirely (the script does not exist).

### Decision 4: Place the `render_sample` test in a new `tests/test_dataset.py`

`tests/` follows a `test_<module>.py` convention (`test_generate_data.py`, `test_model.py`, `test_train.py`, `test_vit.py`). Adding `tests/test_dataset.py` keeps the convention. The test covers correctness only (label values land at the correct pixels for the correct disks); no batched-iterator or property-based tests in scope.

### Decision 5: Not delete the existing periodic checkpoint snapshots

The user's `checkpoints/` directory is gitignored and may contain prior runs. This change touches only code; it does not move or remove on-disk data.

## Risks / Trade-offs

- **[Risk]** Bumping `BATCH_SIZE` from 1 to 8 may change training memory peak by ~8×. → Mitigation: 128×128×1 inputs are tiny; 8× peak is still well under 4 GB on either model. If a user is constrained, they can lower it manually — the constant remains a single-line edit.
- **[Risk]** A user with previously trained checkpoints under `checkpoints/<model>/` (no `/final`) will see the new default fail. → Mitigation: this layout is what `train.py` produces *today*; documenting the change in the README's evaluation section, and `--checkpoint` flag still works for any explicit path.
- **[Trade-off]** We leave the BatchNorm-with-tiny-batch architectural smell in place. If anyone wants `BATCH_SIZE=1` again (e.g., for debugging), it will silently regress. → Mitigation: out of scope; if it becomes a real need, propose a separate change to swap norms.
- **[Trade-off]** README is written by hand and can drift again. → Mitigation: the new `render_sample` test plus existing tests catch most code drift; README drift is intrinsic to docs and requires reviewer discipline, not tooling.

## Migration Plan

No migration. Code changes are local; no schema, no API, no on-disk format changes. Users who want to evaluate a checkpoint at a non-default path use the existing `--checkpoint` flag.
