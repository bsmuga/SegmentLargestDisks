## Context

The change `per-model-training-config` introduces an in-file `MODEL_CONFIGS` dict in `train.py`:

```python
MODEL_CONFIGS: dict[str, dict] = {
    "unet": {"learning_rate": 1e-3, "num_epochs": 50},
    "vit":  {"learning_rate": 3e-4, "num_epochs": 100},
}
```

That solves "use the right LR per architecture," but it leaves training-loop code and tunable knobs entangled. As more architectures or variants get added, every config tweak shows up as a Python diff and `train.py` keeps absorbing data that is not behavior. This change extracts those values into per-model TOML files under a new `conf/` directory and makes `train.py` read from them at startup.

This is a refactor of the source of truth, not a behavior change. After both `per-model-training-config` and this change are applied, training behavior is identical to running `per-model-training-config` alone — the difference is that hyperparameters live in `conf/<model>.toml` instead of inline.

Ordering matters: this change presupposes that `per-model-training-config` has been applied and archived first.

## Goals / Non-Goals

**Goals:**

- Per-model training settings live in `conf/`, one file per model, with the same fields the inline `MODEL_CONFIGS` carries (`learning_rate`, `num_epochs`).
- Adding or tuning a model's hyperparameters does not require editing `train.py`.
- No new runtime dependency. The TOML loader is the standard library.
- The diff is small and reviewable as a pure refactor.

**Non-Goals:**

- Layered configs (base + override), CLI overrides, environment-variable injection.
- Schema validation libraries (pydantic, attrs). Loose typing is fine for two known keys.
- Generalized config support beyond per-model training settings.
- Dataset/training-loop constants (`BATCH_SIZE`, `SEED`, `NUM_SAMPLES`, `IMAGE_SIZE`, `NUM_LABELED`, `MAX_LABELS`) — they describe the experiment, not the architecture, and stay as module-level constants in `train.py`.

## Decisions

### Decision 1: TOML files, one per model, flat at `conf/<model>.toml`

```
conf/
  unet.toml
  vit.toml
```

Each file:

```toml
# conf/unet.toml
learning_rate = 1e-3
num_epochs = 50
```

Why TOML:

- Standard-library `tomllib` is available on Python ≥ 3.11; project requires ≥ 3.14 (per `README.md`). No new dependency.
- The project already uses TOML for `pyproject.toml`; readers are familiar with the syntax.
- Numeric/string scalars are well-typed in TOML — no need for type coercion in the loader.

**Alternative considered**: YAML. Rejected — adds PyYAML (or pyyaml-tng) as a dependency for a payload of two scalars per file, and YAML parser quirks (booleans, version pinning) outweigh the syntactic prettiness here.

**Alternative considered**: JSON. Rejected — comments are not allowed and trailing-comma errors are common; this is a config humans edit, not a wire format.

**Alternative considered**: Python module per model (`conf/unet.py` exporting variables). Rejected — re-introduces "configs are code" that this change is trying to undo, and makes accidental imports/side-effects possible.

### Decision 2: Flat layout (`conf/<model>.toml`), not nested (`conf/training/<model>.toml`)

The only thing in `conf/` for now is per-model training settings. Pre-emptively nesting under `conf/training/` would imply other categories (`conf/data/`, `conf/eval/`) that do not exist and may never. Flat is cheaper to navigate; it can be reorganized later if a real second category appears.

**Alternative considered**: `conf/<model>/training.toml`. Rejected — same anti-speculation reasoning, plus it doubles the file-tree depth without benefit.

### Decision 3: Inline loader in `train.py`, not a new `config.py` module

The loader is small enough that exporting it as a module is overkill:

```python
import tomllib

with open(f"conf/{MODEL_NAME}.toml", "rb") as f:
    cfg = tomllib.load(f)
LEARNING_RATE = cfg["learning_rate"]
NUM_EPOCHS = cfg["num_epochs"]
```

Roughly four lines, no new module. If a second consumer of these configs ever appears (e.g., `evaluate.py` wanting to know the LR a checkpoint was trained at), promote to a module then.

**Alternative considered**: A `config.py` at the repo root with a `load_model_config(name) -> dict` helper. Rejected for now — premature abstraction for one caller. Worth revisiting once a second caller exists.

### Decision 4: Keep field names matching the in-file dict keys (`learning_rate`, `num_epochs`)

Same names as the inline `MODEL_CONFIGS` so the diff is a pure file-location move. This also keeps the spec requirement ("MUST contain `learning_rate` and `num_epochs`") syntactically stable.

### Decision 5: Path is computed relative to the script's own location, not CWD

`train.py` already uses `os.path.dirname(os.path.abspath(__file__))` to anchor the `checkpoints/` and `logs/` paths. The conf-file path uses the same anchor: `os.path.join(base_dir, "conf", f"{MODEL_NAME}.toml")`. This means `python train.py` works the same regardless of the working directory the user invokes it from.

### Decision 6: A missing or malformed conf file is a hard failure at import time

If `conf/<MODEL_NAME>.toml` does not exist or is missing required keys, `train.py` raises immediately — before the dataset is generated, before any GPU memory is allocated. No silent fallbacks, no defaults baked into the loader. Reason: a missing config is always a contributor mistake; surfacing it as a clean `FileNotFoundError` or `KeyError` at startup is the cheapest debugging path.

### Decision 7: One small `conf/README.md` describing the convention

Mirrors the pattern set by `research/README.md`: a one-paragraph index that says what `conf/` is for, the file naming convention (one file per model in the registry, named `<model>.toml`), and the required schema. Kept tiny — it is documentation, not a generated index.

## Risks / Trade-offs

- **[Risk]** A user adding a new model to `models.MODELS` forgets to create the matching `conf/<name>.toml`. → Mitigation: `train.py` raises `FileNotFoundError` immediately on `MODEL_NAME` switch; the spec also requires this constraint, so any test that exercises model selection against the registry would catch the gap. Optionally, a one-line check at startup could enumerate `MODELS` and warn for missing files (out of scope here).
- **[Risk]** TOML's float parsing differs subtly from Python's `float(...)` in edge cases (e.g., `1e-3` is a valid TOML float). → Mitigation: stick to plain decimal floats and integers; the two keys we need are well-supported.
- **[Trade-off]** Splitting `train.py` and `conf/` means a contributor reading `train.py` cold has to open a second file to see the LR. → Acceptable: the locator (`f"conf/{MODEL_NAME}.toml"`) is a single line; the configs are explicitly *meant* to be edited without touching code, which is the value being bought.
- **[Trade-off]** No schema validation means a typo (`learning_Rate = 1e-3`) raises `KeyError` rather than a friendly message. → Acceptable for now; a tiny `assert {"learning_rate", "num_epochs"} <= cfg.keys()` is cheap to add later if needed.

## Migration Plan

1. Apply and archive `per-model-training-config` first. After that change, `train.py` contains the inline `MODEL_CONFIGS` dict.
2. This change adds `conf/unet.toml` and `conf/vit.toml` populated with the same numbers `MODEL_CONFIGS` carries, then removes the dict and replaces lookups with the TOML loader.
3. No on-disk migration of any artifacts (`logs/`, `checkpoints/`) — those layouts are unchanged.
4. Rollback: revert this change. The previous in-file `MODEL_CONFIGS` is restored from git history; the `conf/` directory can be deleted.
