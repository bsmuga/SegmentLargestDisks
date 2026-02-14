# Segment Largest Disks

Can a UNet learn to pick out the **N largest disks** from a cluttered image and
label them by size order?

This project is a minimal experiment that answers that question. We
procedurally generate images with many non-overlapping disks, train a standard
UNet to produce a segmentation mask where only the N largest disks are labeled
(1 = largest, 2 = second-largest, ..., rest = background), and measure how well
the network recovers that ordering.

![Generated samples](samples.png)

## How it works

1. **Data generation** (`generate_data.py`) &mdash; random non-overlapping disks
   are placed on a canvas using a greedy algorithm with a full pairwise distance
   matrix. The N largest disks receive distinct labels sorted by radius; the
   rest are background.
2. **Model** (`model.py`) &mdash; a standard UNet (32&rarr;64&rarr;128&rarr;256&rarr;512
   bottleneck) implemented in Flax NNX. Input: single-channel binary image.
   Output: per-pixel class logits.
3. **Training** (`train.py`) &mdash; soft dice loss, Adam optimizer, trained
   with JAX on CPU or GPU.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate

# Auto-detect CUDA and install the right JAX variant:
./install.sh

# Or install manually:
pip install -r requirements-cuda.txt   # GPU (CUDA 12)
pip install -r requirements-cpu.txt    # CPU only
```

Generate sample data and preview:

```bash
python generate_data.py
python plot_samples.py
```

Train the model:

```bash
python train.py
```

Run tests:

```bash
pip install -r requirements-dev.txt
pytest tests/
```

## Project structure

```
generate_data.py   # circle generation + DataFrame export + batch parallel
dataset.py         # dataset helpers for training (images & masks)
model.py           # UNet (Flax NNX)
train.py           # training loop with dice loss + mIoU logging
plot_samples.py    # visualise generated samples
tests/             # unit tests for generate_data.py
```

## Requirements

- Python >= 3.14
- JAX, Flax, Optax
- **GPU**: `requirements-cuda.txt` installs `jax[cuda12]`
- **CPU**: `requirements-cpu.txt` installs `jax[cpu]`
