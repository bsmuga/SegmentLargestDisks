# Research notes

This directory is the catalog for analytical notes about the project — problem framing, architecture comparisons, and open questions. Notes capture the *why* behind decisions and the shape of the problem, written so a contributor opening the repo cold can pick up the reasoning without having to reconstruct it from chat logs or commits.

## Conventions

- Files are markdown.
- Filenames are kebab-case topic slugs (e.g., `architecture-notes.md`, `loss-function-tradeoffs.md`).
- Each note begins with a short metadata block (title + a `Last reviewed: YYYY-MM-DD` line). Treat that date as a freshness signal: if the code or task has moved on, the note may be stale.
- Structure inside a note is free-form — write what serves the analysis. No mandatory sections.
- Notes are living documents. Edit or delete them as understanding evolves; there is no requirement to preserve historical versions (git history covers that).

## What belongs here

- Analyses of how a problem is framed and why a given approach is or is not a good fit.
- Side-by-side comparisons of architectures, algorithms, or design alternatives.
- Open questions and the conditions under which the current analysis would change.

## What does not belong here

- **Specs and proposed changes** — those go in `openspec/changes/` and `openspec/specs/`.
- **Status updates, sprint plans, roadmaps** — those belong in commit messages, OpenSpec proposals, or the issue tracker.
- **Transient TODOs** — track those in your editor, an issue, or a task list, not in the repo.
- **Decision records that gate code** (ADRs) — these notes are comparative thinking, not blocking decisions. If a future note actually records a binding decision, write it as an ADR inside the same folder; the format does not need to be uniform across notes.

## Index

- [`architecture-notes.md`](architecture-notes.md) — UNet vs. ViT for the largest-disk segmentation task: framing the per-pixel-rank problem as a global comparison, where each architecture fits or struggles, and what alternatives bound the deep-learning approach.
