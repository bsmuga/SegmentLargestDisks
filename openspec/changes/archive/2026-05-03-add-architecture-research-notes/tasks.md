## 1. Create the `research/` catalog

- [x] 1.1 Create the `research/` directory at the repository root
- [x] 1.2 Create `research/README.md` with: a one-paragraph statement of purpose (a catalog for analytical notes about the project), the convention (kebab-case `*.md` filenames, free-form structure, optional last-reviewed date), an explicit out-of-scope list (status updates, roadmaps, OpenSpec proposals, transient TODOs), and a one-line index entry for `architecture-notes.md`

## 2. Author the architecture analysis note

- [x] 2.1 Create `research/architecture-notes.md` with a top-of-file metadata block containing the title and a `Last reviewed: YYYY-MM-DD` line
- [x] 2.2 Add a "Problem framing" section that explicitly identifies per-pixel labels as depending on the *rank* of the containing disk among all disks, and states that rank is a global (not local) property
- [x] 2.3 Add a subsection on the asymmetric label structure produced by `generate_data.py` (labels 1–4 unique vs. label 5 covering ranks 5–25 vs. label 0 mixing background and the smallest disks) and what that asymmetry implies for difficulty
- [x] 2.4 Add a "UNet through this lens" section: receptive-field reasoning, which sub-problems convolutions solve naturally, and where ranking pressure breaks the inductive bias (especially the rank-4 vs rank-5 boundary)
- [x] 2.5 Add a "ViT through this lens" section: why self-attention naturally fits cross-disk comparison, and why the configuration in `models/vit.py` (192-dim, 4 layers, patch 16 on 128×128, conv-transpose decoder) imposes capacity and pixel-precision limits
- [x] 2.6 Add a side-by-side comparison table covering: foreground/background detection, single-disk size measurement, cross-disk ranking, pixel precision, sample efficiency, training stability
- [x] 2.7 Add an "Alternatives" section mentioning at least classical connected-components + sort, and a hybrid UNet-encoder + transformer-bottleneck + UNet-decoder
- [x] 2.8 Close with a short "Open questions / what would change this analysis" list (e.g., capacity bumps to ViT, replacing BatchNorm with LayerNorm, switching to instance segmentation framing)

## 3. Wire the catalog into the README

- [x] 3.1 In `README.md`, add a single line under "Project structure" pointing to `research/` with a one-phrase description (e.g., "research notes / analysis")
- [x] 3.2 Verify the README change does not conflict with the in-flight `fix-and-finish-baseline` README edits (those rewrite `model.py` references and the install commands; the new line should be additive)

## 4. Verification

- [x] 4.1 Run `ls research/` and confirm `README.md` and `architecture-notes.md` are both present
- [x] 4.2 Read `research/README.md` cold and confirm it explains what the directory is for, what belongs in it, and what does not
- [x] 4.3 Read `research/architecture-notes.md` cold and confirm it covers all sections required by the spec (problem framing, both architectures, comparison table, alternatives)
- [x] 4.4 From `README.md`, confirm there is a discoverable pointer to `research/`
- [x] 4.5 Run `pytest tests/` and `flake8 .` to confirm no regressions (expected: nothing changes — these are doc-only edits)
