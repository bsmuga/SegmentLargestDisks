# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepCircleCounter is a PyTorch-based computer vision project that trains neural networks to segment and identify the largest disks/circles in synthetically generated images, recognizing them in order from largest to smallest.

## Key Commands

### Development Environment
```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Format code
black .

# Lint code
flake8

# Run tests
python -m pytest tests/
# or
python -m unittest
```

### Training
```bash
# Train model with configuration
python -m src.main --config configs/unet.yml
```

### Visualization
```bash
# Plot generated circles
python scripts/plot_circles.py
```

## Architecture Overview

The codebase follows a modular structure using PyTorch Lightning:

1. **Data Pipeline** (`src/data/`):
   - `dataset.py`: Generates synthetic disk images on-the-fly with non-overlapping circles
   - `dm.py`: Lightning DataModule managing train/val/test splits

2. **Model Training** (`src/`):
   - `main.py`: Entry point that loads config and initiates training
   - `segmentation_module.py`: Lightning module wrapping segmentation models from the `segmentation_models_pytorch` library
   - `confusion_matrix.py`: Custom IoU-based confusion matrix metric for evaluating object detection

3. **Configuration** (`configs/`):
   - YAML files define model architecture, data parameters, and training settings
   - Default uses UNet with ResNet18 encoder

4. **Experiment Tracking**:
   - MLflow logs experiments to `/tmp/logs` by default
   - Tracks metrics including Dice coefficient and custom confusion matrix

## Code Style

- Black formatter with line length 88
- Flake8 linter with max line length 88
- Type hints are used throughout the codebase
- Tests use unittest framework in `tests/` directory

## Current Development Focus

The project is actively developing metrics and loss functions. Current branch `add-metric` implements confusion matrix logging. Next priorities include replacing MSE loss with Dice + Cross Entropy loss and experimenting with different architectures.