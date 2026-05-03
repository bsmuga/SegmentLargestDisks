## Context

After `fix-and-finish-baseline`, `train.py` reads hyperparameters from module-level constants:

```python
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
MODEL_NAME: Literal["unet", "vit"] = "vit"
```

Switching `MODEL_NAME` is one edit, but `LEARNING_RATE` and `NUM_EPOCHS` are tuned for one architecture and unconsciously inherited by the other. Output paths are `logs/{MODEL_NAME}_metrics.csv` etc., which use the model name as a filename prefix — same-name runs overwrite, different-model runs scatter into the same flat folder.

The repo's identity is "minimal experiment." The fix should do the smallest sensible thing: a tiny lookup, per-model output folders, no new infrastructure.

## Goals / Non-Goals

**Goals:**

- UNet and ViT each train with hyperparameters appropriate to their architecture.
- The two models' run outputs do not share filenames or step on each other.
- The configuration surface for a contributor remains a single Python file (no new YAML/TOML/CLI).
- The change is small enough that a reader can understand it from the diff alone.

**Non-Goals:**

- A general experiment-config system (parameter sweeps, YAML, mlflow).
- Per-model `BATCH_SIZE` — already pinned at 8 for BatchNorm safety in both models, and there is no current memory pressure.
- Per-model dataset settings (`NUM_SAMPLES`, `IMAGE_SIZE`, etc.) — those describe the *task*, not the model.
- Migrating any existing `logs/*.csv` / `logs/*.png` files; they're gitignored.

## Decisions

### Decision 1: Per-model config lives in a `MODEL_CONFIGS` dict in `train.py`

```python
MODEL_CONFIGS: dict[str, dict] = {
    "unet": {"learning_rate": 1e-3, "num_epochs": 50},
    "vit":  {"learning_rate": 3e-4, "num_epochs": 100},
}
```

Why a dict, not a class or `dataclass`: a dict is the smallest representation that solves the problem. Two keys per entry, two entries. A `dataclass` adds an import and ceremony with no behavioral upside at this size.

Why these two keys only:

- `learning_rate`: Genuinely different per architecture. ViT typically benefits from a smaller LR; UNet tolerates a larger one. This is the single most impactful per-model knob.
- `num_epochs`: ViT typically needs more epochs to converge on small synthetic datasets. UNet usually saturates earlier.
- `BATCH_SIZE` deliberately left shared at module scope — it is governed by BatchNorm safety (existing spec requirement) and current memory budget, neither of which differ per model here.

**Alternative considered**: a `dataclass` per model (`UnetConfig`, `ViTConfig`) with all training knobs. Rejected — fixing two values in two dict entries does not earn a class hierarchy.

**Alternative considered**: a YAML/TOML config file. Rejected — adds a parser dependency and a second source of truth for the smallest possible payload (4 numbers).

### Decision 2: Output goes to `logs/<model>/`, not `runs/<model>/`

The user-visible output split is:

```
checkpoints/<model>/{epoch_N, final}/   # already exists, untouched
logs/<model>/<name>_<timestamp>.log     # per-run python log
logs/<model>/metrics.csv                # per-run CSV
logs/<model>/training.png               # per-run plot
```

Moving everything under a single `runs/<model>/` would be cleaner conceptually but renames `checkpoints/<model>/` → `runs/<model>/checkpoints/` and forces another change in `evaluate.py`'s default path. We just stabilized that default in the previous change. Keeping the top-level layout (`checkpoints/`, `logs/`) and partitioning each by model is the smaller diff with the same isolation benefit.

### Decision 3: Filenames inside `logs/<model>/` drop the redundant model prefix

Old: `logs/{model}_metrics.csv` → New: `logs/<model>/metrics.csv`. The model is already in the directory name; repeating it in the filename is noise. Same for `training.png` and the per-run logger filename (which keeps the `<name>_<timestamp>` part for log-stream identification, but lives under the per-model folder).

### Decision 4: Extend `logger.get_logger` minimally with an optional `subdir` parameter

Signature change:

```python
def get_logger(name: str, subdir: str | None = None) -> logging.Logger:
```

When `subdir` is provided, the log file is written to `logs/<subdir>/<name>_<timestamp>.log`. When `None`, behavior is unchanged (log file at `logs/<name>_<timestamp>.log`). This preserves the contract for the only other caller, `evaluate.py:get_logger("evaluate")`.

**Alternative considered**: Have `train.py` reach into the `logs/` directory directly and bypass the logger's filename logic. Rejected — it would duplicate path-building and split log-routing across two files.

### Decision 5: Same-model re-runs still overwrite `metrics.csv` and `training.png`

A new run of `train.py --model unet` overwrites `logs/unet/metrics.csv` and `logs/unet/training.png`. We do not append a timestamp to these. Reason: this matches the existing pre-change behavior (which also overwrote), and the per-run timestamped log file already preserves the full history of any specific run. Adding timestamps to all artifacts would creep toward an experiment tracker, which is a non-goal.

## Risks / Trade-offs

- **[Risk]** Initial per-model defaults (UNet `lr=1e-3, epochs=50`; ViT `lr=3e-4, epochs=100`) are educated guesses, not measured optima. → Mitigation: they are local edits in one dict; tuning is left to follow-up runs and is not gated by this change.
- **[Risk]** `evaluate.py` reads from `logs/...` for nothing today, but a future change might. → Mitigation: none needed now; this change touches only `train.py` and `logger.py`.
- **[Trade-off]** A reader skimming `train.py` now sees `MODEL_CONFIGS[MODEL_NAME]["learning_rate"]` instead of `LEARNING_RATE`. Slightly less ergonomic in three places. → Mitigation: bind to local names (`learning_rate = MODEL_CONFIGS[MODEL_NAME]["learning_rate"]`) once at the top of `main()`.
- **[Trade-off]** Same-model re-run overwrites `metrics.csv` / `training.png`. Anyone wanting to keep them must rename manually. → Acceptable given the experiment is "minimal" and the per-run timestamped log file is preserved.

## Migration Plan

No on-disk migration required. Pre-existing flat `logs/*_metrics.csv` / `logs/*_training.png` are gitignored and per-run; they can stay in place or be deleted by the user. New runs write to the new location.
