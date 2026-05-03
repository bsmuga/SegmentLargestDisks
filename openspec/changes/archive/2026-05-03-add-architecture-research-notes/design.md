## Context

The repository is a small experimental ML project comparing UNet and ViT on a segmentation task whose surface form (pixel classification) hides a relational core: per-pixel labels depend on each disk's *rank* among all disks in the image. That framing — and the consequences for which architecture is naturally suited — exists today only as conversational analysis. The repo has no place to put this kind of thinking; `README.md` is for runnability, OpenSpec proposals are for upcoming changes, and code comments are deliberately kept minimal per project convention. As a result, the analysis is at risk of being lost or repeated.

This change introduces a thin new convention: a `research/` directory at the repo root, holding markdown notes that capture *analysis and framing* (the "why behind decisions"), distinct from `README.md` (runnability) and `openspec/` (proposed changes and specs).

## Goals / Non-Goals

**Goals:**

- A new contributor or returning collaborator can find the architecture analysis in under 30 seconds from the README.
- The first note (`architecture-notes.md`) captures the substance of the explore-mode discussion: problem framing, per-architecture fit, the asymmetric label structure, the comparison table, alternative approaches.
- The convention for what belongs in `research/` is documented in one place (`research/README.md`) so future notes follow a consistent style.
- Stay minimal: documentation only, no code or test changes, no automation.

**Non-Goals:**

- Migrating prose from elsewhere in the repo (CLAUDE-style notes, comments, etc.) into `research/`.
- Adding doc tooling (link-checkers, lint, MkDocs/Sphinx, generated TOCs).
- Establishing a formal RFC/ADR process — `research/` is for analytical notes, not decision records that gate implementation.
- Coupling notes to OpenSpec changes. A research note may inform a future proposal, but it is not itself a proposal and does not require artifacts.
- Versioning or freezing notes. They are living documents; if the analysis becomes wrong, edit it.

## Decisions

### Decision 1: Top-level `research/` rather than `docs/research/` or `openspec/research/`

The repo is flat: code, tests, generated samples, and `README.md` all live at the root, with `openspec/` as the only nested subsystem. Putting notes at the top level (`research/`) keeps them discoverable from a single `ls` of the repo root, signals that they are first-class artifacts, and avoids implying a parent `docs/` tree that does not exist.

**Alternative considered**: `docs/research/`. Rejected — there is no `docs/`, and creating one for a single note implies a documentation hierarchy that is not warranted by the project's size.

**Alternative considered**: `openspec/research/`. Rejected — OpenSpec is for *proposed and archived changes*; research notes are not changes and have no schema. Conflating them would muddy what `openspec/` means.

### Decision 2: Free-form markdown, not a templated note schema

OpenSpec already supplies a heavyweight artifact system (proposal/design/specs/tasks). Imposing a similar template on research notes would discourage writing them. A short note convention (kebab-case filename by topic, one-line title, free-form body) is the lowest friction format that still produces grep-able, navigable files.

**Alternative considered**: ADR (Architecture Decision Record) format with Status/Context/Decision/Consequences. Rejected — ADRs are for *decisions that gate code*; the architecture analysis is a comparison, not a decision. If a future note actually records a decision, it can use ADR structure inside the same `research/` folder without requiring it for everything else.

### Decision 3: Single index file (`research/README.md`) describing the convention

A one-paragraph index serves two purposes: it explains to a new reader what this folder is for, and it lists the current notes. It is updated by hand when a note is added or removed. With the project's small scale this is cheaper than any generated index.

### Decision 4: README pointer is one line under "Project structure"

The existing `README.md` has a "Project structure" code block listing top-level files. Adding `research/` there is the minimal change that makes the catalog discoverable without restructuring the README.

**Alternative considered**: a new "Notes" or "Background" section in the README. Rejected — adds README surface area for a project whose README is already in the process of being trimmed (`fix-and-finish-baseline`).

### Decision 5: First note is `architecture-notes.md`, not split per architecture

The UNet and ViT analyses are most useful *side by side* — the comparison table and the rank-asymmetry framing apply to both. Splitting into `unet-notes.md` and `vit-notes.md` would force readers to context-switch and duplicate the shared framing. One note keeps the comparative shape intact.

### Decision 6: Notes capture thinking, not status

A research note explains *how the problem was framed and which approaches looked promising*. It is not a status update ("we tried X, it got 0.83 mIoU") and not a roadmap ("we plan to ship Y next quarter"). Status belongs in commit messages and OpenSpec changes; roadmaps belong in proposals. Keeping `research/` scoped to analysis prevents it from drifting into a stale dumping ground.

## Risks / Trade-offs

- **[Risk]** Notes drift out of date as the code evolves (e.g., a note saying "ViT has 4 layers" becomes wrong after a refactor). → Mitigation: every note carries a "last-reviewed" date in its header; readers treat that as a freshness signal. Drift is intrinsic to docs; the alternative (no notes) is worse.
- **[Risk]** `research/` becomes a junk drawer for any markdown someone does not want to delete. → Mitigation: `research/README.md` documents what belongs there (analysis, framing, comparisons) and what does not (status, TODOs, decision-with-consequences records, transient logs). Reviewers enforce this when notes are added.
- **[Trade-off]** No tooling means broken links and stale facts will not be flagged automatically. → Mitigation: out of scope for this change; revisit only if the catalog grows past ~5 notes.
- **[Trade-off]** Free-form structure makes notes inconsistent in shape. → Mitigation: accepted. Consistency in *prose* is less valuable than the lower friction to write at all.

## Migration Plan

No migration. This change adds files; it does not move or modify existing code or docs other than a one-line addition to `README.md`. There is no data, no schema, no API, and no on-disk format to preserve.
