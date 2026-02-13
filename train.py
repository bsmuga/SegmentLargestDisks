"""Training script for UNet segmentation of largest disks."""

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from dataset import make_dataset, batched_iterator
from model import UNet

# ── Hyperparameters ──────────────────────────────────────────────────────────
NUM_SAMPLES = 200
NUM_CIRCLES = 30
IMAGE_SIZE = (256, 256)
NUM_LABELED = 25
MAX_LABELS = 5
NUM_CLASSES = MAX_LABELS + 1  # 0=background + 1..5 labels

BATCH_SIZE = 8
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
SEED = 42


def dice_loss(logits: jnp.ndarray, labels: jnp.ndarray, num_classes: int, smooth: float = 1.0) -> jnp.ndarray:
    """Soft dice loss averaged over classes.

    Parameters
    ----------
    logits : (B, H, W, C) float
    labels : (B, H, W) int
    num_classes : int
    smooth : float
        Smoothing term to avoid division by zero.
    """
    probs = jax.nn.softmax(logits, axis=-1)  # (B, H, W, C)
    one_hot = jax.nn.one_hot(labels, num_classes)  # (B, H, W, C)

    # Flatten spatial dims: (B, H*W, C)
    probs = probs.reshape(probs.shape[0], -1, num_classes)
    one_hot = one_hot.reshape(one_hot.shape[0], -1, num_classes)

    intersection = jnp.sum(probs * one_hot, axis=1)  # (B, C)
    cardinality = jnp.sum(probs + one_hot, axis=1)  # (B, C)

    dice_per_class = (2.0 * intersection + smooth) / (cardinality + smooth)  # (B, C)
    return 1.0 - dice_per_class.mean()


def compute_iou(logits: jnp.ndarray, labels: jnp.ndarray, num_classes: int) -> jnp.ndarray:
    """Per-class IoU. Returns array of shape (num_classes,)."""
    preds = jnp.argmax(logits, axis=-1)  # (B, H, W)
    ious = []
    for c in range(num_classes):
        pred_c = preds == c
        label_c = labels == c
        intersection = jnp.sum(pred_c & label_c)
        union = jnp.sum(pred_c | label_c)
        iou = jnp.where(union > 0, intersection / union, jnp.nan)
        ious.append(iou)
    return jnp.stack(ious)


@nnx.jit
def train_step(model, optimizer, images, labels):
    def loss_fn(model):
        logits = model(images)
        return dice_loss(logits, labels, NUM_CLASSES), logits

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model)
    optimizer.update(model, grads)
    return loss, logits


@nnx.jit
def eval_step(model, images):
    return model(images)


def main():
    print("Generating dataset...")
    images, masks = make_dataset(
        num_samples=NUM_SAMPLES,
        num_circles=NUM_CIRCLES,
        image_size=IMAGE_SIZE,
        num_labeled=NUM_LABELED,
        max_labels=MAX_LABELS,
        seed=0,
    )
    print(f"Dataset: images {images.shape}, masks {masks.shape}")

    # Split into train/val (80/20)
    n_train = int(0.8 * NUM_SAMPLES)
    train_images, val_images = images[:n_train], images[n_train:]
    train_masks, val_masks = masks[:n_train], masks[n_train:]
    print(f"Train: {len(train_images)}, Val: {len(val_images)}")

    # Initialize model and optimizer
    rngs = nnx.Rngs(SEED)
    model = UNet(num_classes=NUM_CLASSES, rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt=nnx.Param)

    # Verify shapes with a dummy forward pass
    dummy = jnp.zeros((1, IMAGE_SIZE[1], IMAGE_SIZE[0], 1))
    out = model(dummy)
    print(f"Model output shape: {out.shape} (expected (1, {IMAGE_SIZE[1]}, {IMAGE_SIZE[0]}, {NUM_CLASSES}))")

    rng = jax.random.key(SEED)

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── Training ────────────────────────────────────────────────────
        model.train()
        rng, epoch_rng = jax.random.split(rng)
        epoch_loss = 0.0
        n_batches = 0

        for batch_imgs, batch_masks in batched_iterator(
            train_images, train_masks, BATCH_SIZE, epoch_rng
        ):
            loss, _ = train_step(model, optimizer, batch_imgs, batch_masks)
            epoch_loss += float(loss)
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # ── Validation ──────────────────────────────────────────────────
        model.eval()
        all_ious = []
        val_loss = 0.0
        n_val = 0

        for batch_imgs, batch_masks in batched_iterator(
            val_images, val_masks, BATCH_SIZE, epoch_rng
        ):
            logits = eval_step(model, batch_imgs)
            val_loss += float(dice_loss(logits, batch_masks, NUM_CLASSES))
            ious = compute_iou(logits, batch_masks, NUM_CLASSES)
            all_ious.append(ious)
            n_val += 1

        avg_val_loss = val_loss / max(n_val, 1)
        mean_ious = jnp.nanmean(jnp.stack(all_ious), axis=0)
        miou = float(jnp.nanmean(mean_ious))

        per_class = ", ".join(f"c{i}={float(mean_ious[i]):.3f}" for i in range(NUM_CLASSES))
        print(
            f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={avg_val_loss:.4f} | "
            f"mIoU={miou:.4f} [{per_class}]"
        )

    # ── Save checkpoint ─────────────────────────────────────────────────
    import orbax.checkpoint as ocp

    checkpointer = ocp.StandardCheckpointer()
    _, state = nnx.split(model)
    ckpt_path = "./checkpoints/unet"
    checkpointer.save(ckpt_path, state)
    print(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
