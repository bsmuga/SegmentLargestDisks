## Why

The repo trains both UNet and ViT on a task that *looks* like pixel classification but actually has a global, relational core (per-pixel labels depend on a disk's size-rank among all disks in the image). That framing — and why each architecture is well-matched or mismatched to it — currently lives only in chat history. Without writing it down, future contributors (and future-me) repeat the same exploratory thinking from scratch every time the question "should we try X instead?" comes up. A short, in-repo research note keeps the analysis discoverable next to the code it is about.

## What Changes

- Add a top-level `research/` directory as the catalog for in-repo research and analysis notes.
- Add `research/architecture-notes.md` containing the UNet vs. ViT analysis: problem framing (local vs. global signal), per-architecture fit and failure modes, the asymmetric label structure (labels 1–4 unique vs. label 5 as a fuzzy band), comparison table, and pointers to alternative approaches (classical connected-components, hybrid UNet-with-Transformer-bottleneck) that bound what a learned solution is worth doing.
- Add a one-line pointer in `README.md` linking to `research/` so contributors can find the catalog without spelunking.
- Establish a lightweight convention (documented in the spec): research notes are markdown, named by topic in kebab-case, and capture *thinking* (analyses, comparisons, open questions) — not specs, not implementation plans, not transient TODOs.

Non-goals:

- No code changes (no new model, no training tweaks, no test changes).
- No automated link-checking or doc tooling.
- No migration of existing prose elsewhere in the repo into `research/`.

## Capabilities

### New Capabilities

- `research-notes`: An in-repo catalog (`research/`) of analytical notes that capture *why* decisions were made and how the problem is framed, written for a future contributor opening the repo cold. Establishes where these notes live, what belongs in them (analysis, comparisons, framing) versus what does not (specs, task lists, ephemeral status), and the naming convention.

### Modified Capabilities

<!-- None. The existing baseline-experiment capability is unaffected; this change adds documentation only. -->

## Impact

- **Code**: None.
- **Files added**: `research/architecture-notes.md`, `research/README.md` (one-paragraph index of what's in the directory and the conventions).
- **Files modified**: `README.md` — one new line under "Project structure" pointing to `research/`.
- **Behavior**: No runtime behavior changes. Tests, training, and evaluation are untouched.
- **Onboarding**: A fresh contributor reading the README has a discoverable path to the architecture rationale without asking anyone.
