## ADDED Requirements

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
