"""Evaluate a trained UNet checkpoint: compute metrics and visualize predictions."""

import argparse

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from tqdm import tqdm

from dataset import make_dataset
from logger import get_logger
from models import create_model
from train import compute_iou, dice_loss

log = get_logger("evaluate")

NUM_CLASSES = 6
IMAGE_SIZE = (128, 128)

LABEL_CMAP = mcolors.ListedColormap(
    ["black", "red", "orange", "green", "purple", "cyan"]
)
LABEL_NORM = mcolors.BoundaryNorm(range(NUM_CLASSES + 1), NUM_CLASSES)


def load_model(checkpoint_path: str, model_name: str = "unet") -> nnx.Module:
    """Load a model from an orbax checkpoint."""
    model = create_model(model_name, num_classes=NUM_CLASSES, rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    state = checkpointer.restore(checkpoint_path, target=state)
    model = nnx.merge(graphdef, state)
    model.eval()
    return model


def evaluate(model: nnx.Module, images: np.ndarray, masks: np.ndarray) -> dict:
    """Run evaluation over an entire dataset.

    Returns
    -------
    dict with keys:
        dice_loss : float
        miou : float
        per_class_iou : np.ndarray of shape (NUM_CLASSES,)
        pixel_accuracy : float
        per_class_accuracy : np.ndarray of shape (NUM_CLASSES,)
    """
    all_ious = []
    total_loss = 0.0
    total_correct = 0
    total_pixels = 0
    class_correct = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_total = np.zeros(NUM_CLASSES, dtype=np.int64)

    # Process in batches of 8 to avoid OOM
    batch_size = 4
    n_batches = 0
    for start in tqdm(range(0, len(images), batch_size), desc="Evaluating"):
        end = min(start + batch_size, len(images))
        batch_imgs = jnp.array(images[start:end])
        batch_masks = jnp.array(masks[start:end])

        logits = model(batch_imgs)
        preds = jnp.argmax(logits, axis=-1)

        total_loss += float(dice_loss(logits, batch_masks, NUM_CLASSES))
        all_ious.append(compute_iou(logits, batch_masks, NUM_CLASSES))

        total_correct += int(jnp.sum(preds == batch_masks))
        total_pixels += int(batch_masks.size)

        for c in range(NUM_CLASSES):
            gt_c = batch_masks == c
            class_total[c] += int(jnp.sum(gt_c))
            class_correct[c] += int(jnp.sum((preds == c) & gt_c))

        n_batches += 1

    mean_ious = jnp.nanmean(jnp.stack(all_ious), axis=0)
    per_class_acc = np.where(
        class_total > 0, class_correct / class_total, np.nan
    )

    return {
        "dice_loss": total_loss / max(n_batches, 1),
        "miou": float(jnp.nanmean(mean_ious)),
        "per_class_iou": np.array(mean_ious),
        "pixel_accuracy": total_correct / max(total_pixels, 1),
        "per_class_accuracy": per_class_acc,
    }


def plot_predictions(
    model: nnx.Module,
    images: np.ndarray,
    masks: np.ndarray,
    num_samples: int = 4,
    save_path: str = "evaluation.png",
):
    """Plot a grid of input | ground truth | prediction for visual inspection."""
    indices = np.linspace(0, len(images) - 1, num_samples, dtype=int)
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 3 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        img = images[idx]
        gt = masks[idx]
        logits = model(jnp.array(img[np.newaxis]))
        pred = np.array(jnp.argmax(logits, axis=-1)[0])

        axes[row, 0].imshow(img[:, :, 0], cmap="gray")
        axes[row, 0].set_title(f"Input #{idx}", fontsize=9)

        axes[row, 1].imshow(gt, cmap=LABEL_CMAP, norm=LABEL_NORM, interpolation="nearest")
        axes[row, 1].set_title("Ground truth", fontsize=9)

        axes[row, 2].imshow(pred, cmap=LABEL_CMAP, norm=LABEL_NORM, interpolation="nearest")
        axes[row, 2].set_title("Prediction", fontsize=9)

        for ax in axes[row]:
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    log.info(f"Saved {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation model")
    parser.add_argument("--model", default="unet", choices=["unet", "vit"], help="Model architecture")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to orbax checkpoint directory (default: checkpoints/<model>)",
    )
    parser.add_argument("--num-samples", type=int, default=25, help="Number of eval samples to generate")
    parser.add_argument("--seed", type=int, default=999, help="Seed for eval data (different from training)")
    parser.add_argument("--plot", type=int, default=4, help="Number of samples to visualize (0 to skip)")
    parser.add_argument("--save-path", default=None, help="Path for visualization output (default: evaluation_<model>.png)")
    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = f"checkpoints/{args.model}"
    if args.save_path is None:
        args.save_path = f"evaluation_{args.model}.png"

    log.info("Loading model...")
    model = load_model(args.checkpoint, model_name=args.model)

    log.info("Generating evaluation dataset...")
    images, masks = make_dataset(
        num_samples=args.num_samples,
        image_size=IMAGE_SIZE,
        seed=args.seed,
    )
    log.info(f"Eval set: {images.shape[0]} samples")

    log.info("Running evaluation...")
    metrics = evaluate(model, images, masks)

    log.info(f"{'─' * 50}")
    log.info(f"  Dice loss:       {metrics['dice_loss']:.4f}")
    log.info(f"  Pixel accuracy:  {metrics['pixel_accuracy']:.4f}")
    log.info(f"  mIoU:            {metrics['miou']:.4f}")
    log.info(f"{'─' * 50}")
    log.info("  Per-class breakdown:")
    class_names = ["background"] + [f"label {i}" for i in range(1, NUM_CLASSES)]
    for i, name in enumerate(class_names):
        iou = metrics["per_class_iou"][i]
        acc = metrics["per_class_accuracy"][i]
        iou_str = f"{iou:.4f}" if not np.isnan(iou) else "  N/A "
        acc_str = f"{acc:.4f}" if not np.isnan(acc) else "  N/A "
        log.info(f"    {name:>12s}  IoU={iou_str}  Acc={acc_str}")
    log.info(f"{'─' * 50}")

    if args.plot > 0:
        log.info(f"Plotting {args.plot} samples...")
        plot_predictions(model, images, masks, num_samples=args.plot, save_path=args.save_path)


if __name__ == "__main__":
    main()
