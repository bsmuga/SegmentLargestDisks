# research-notes Specification

## Purpose
TBD - created by archiving change add-architecture-research-notes. Update Purpose after archive.
## Requirements
### Requirement: Repository hosts a `research/` catalog at the root

The repository SHALL contain a top-level directory named `research/` that holds analytical notes about the project (problem framing, architecture comparisons, open questions). The directory MUST be discoverable from a single `ls` at the repository root.

#### Scenario: `research/` exists at the repo root

- **WHEN** a contributor lists files at the repository root
- **THEN** a directory named `research/` MUST be present

#### Scenario: `research/` is referenced from the README

- **WHEN** a contributor reads `README.md` from top to bottom
- **THEN** they MUST encounter at least one reference to the `research/` directory that allows them to navigate there

### Requirement: `research/` includes a self-describing index

The `research/` directory SHALL contain a `README.md` that states the directory's purpose, lists the current notes with one-line descriptions, and documents what kind of content belongs in `research/` versus elsewhere.

#### Scenario: Index explains the directory's purpose

- **WHEN** a contributor opens `research/README.md`
- **THEN** the file MUST describe the directory as a catalog for analytical notes (not specs, not status, not transient TODOs)

#### Scenario: Index enumerates the current notes

- **WHEN** a contributor opens `research/README.md`
- **THEN** every note file in `research/` (other than `README.md` itself) MUST be listed with a brief description

### Requirement: First note documents the UNet vs. ViT architecture analysis

The `research/` directory SHALL contain a note (named `architecture-notes.md`) that captures the analysis of whether UNet and ViT can solve the largest-disk segmentation task. The note MUST cover: the problem's local-vs-global framing, why disk ranking is a global property, per-architecture fit and failure modes for both UNet and ViT, the asymmetric label structure (labels 1–4 unique, label 5 as a fuzzy band, label 0 mixing background and small disks), and at least one alternative approach (e.g., classical connected-components or a hybrid architecture).

#### Scenario: Note covers the global-ranking framing

- **WHEN** a contributor reads `research/architecture-notes.md`
- **THEN** the note MUST explicitly identify that per-pixel labels depend on the *rank* of the containing disk among all disks, and that this is a global rather than local property

#### Scenario: Note compares both architectures with rationale

- **WHEN** a contributor reads `research/architecture-notes.md`
- **THEN** the note MUST contain analysis of UNet (its convolutional inductive bias and where it struggles with global ranking) AND analysis of ViT (self-attention's natural fit for cross-instance comparison, and capacity/decoder limits at the configuration in `models/vit.py`)

#### Scenario: Note acknowledges alternatives

- **WHEN** a contributor reads `research/architecture-notes.md`
- **THEN** the note MUST mention at least one non-deep-learning baseline (e.g., connected components + sort by area) AND at least one hybrid architecture (e.g., UNet encoder + transformer bottleneck + UNet decoder), so a reader sees what the learned approach is being compared against

### Requirement: Research notes follow a lightweight, free-form convention

Files in `research/` other than `README.md` SHALL be markdown files whose names are kebab-case topic slugs. Notes MUST capture analysis, framing, or comparison; they MUST NOT be used as status updates, roadmaps, OpenSpec proposals, or transient TODO lists. Notes are living documents — they may be edited or deleted as the analysis evolves; no historical preservation is required.

#### Scenario: Note filenames are kebab-case slugs

- **WHEN** a contributor lists `*.md` files in `research/`
- **THEN** every filename other than `README.md` MUST be lowercase, kebab-case (words separated by `-`), and end in `.md`

#### Scenario: Status content does not belong in `research/`

- **WHEN** a contributor proposes adding a file containing build status, sprint plans, or task tracking to `research/`
- **THEN** the convention documented in `research/README.md` MUST direct that content elsewhere (commit messages, OpenSpec changes, or issue tracker)
