"""Dataset utilities: render circles to images/masks and provide batched iterators."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from generate_data import generate_batch


def render_sample(df: pd.DataFrame, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Render a single sample's DataFrame rows into an image and segmentation mask.

    Parameters
    ----------
    df : pd.DataFrame
        Rows for one sample, must contain columns: x, y, r, label.
    image_size : tuple[int, int]
        (width, height) of the canvas.

    Returns
    -------
    image : np.ndarray, float32, shape (H, W, 1)
        Binary circle image (1.0 inside any circle, 0.0 outside).
    mask : np.ndarray, int32, shape (H, W)
        Segmentation labels (0 = background/unlabeled circles).
    """
    w, h = image_size
    image = np.zeros((h, w), dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.int32)

    yy, xx = np.mgrid[:h, :w]

    for _, row in df.iterrows():
        cx, cy, r, label = int(row["x"]), int(row["y"]), int(row["r"]), int(row["label"])
        inside = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        image[inside] = 1.0
        if label > 0:
            mask[inside] = label

    return image[:, :, np.newaxis], mask


def make_dataset(
    num_samples: int,
    num_circles: int = 30,
    image_size: tuple[int, int] = (256, 256),
    num_labeled: int = 25,
    max_labels: int = 5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a full dataset of rendered images and masks.

    Returns
    -------
    images : np.ndarray, float32, shape (N, H, W, 1)
    masks : np.ndarray, int32, shape (N, H, W)
    """
    batch_df = generate_batch(
        num_samples=num_samples,
        num_circles=num_circles,
        image_size=image_size,
        num_labeled=num_labeled,
        max_labels=max_labels,
        seed=seed,
    )

    images_list = []
    masks_list = []
    for sample_id in range(num_samples):
        sample_df = batch_df[batch_df["sample_id"] == sample_id]
        img, msk = render_sample(sample_df, image_size)
        images_list.append(img)
        masks_list.append(msk)

    return np.stack(images_list), np.stack(masks_list)


def batched_iterator(
    images: np.ndarray,
    masks: np.ndarray,
    batch_size: int,
    rng: jax.Array,
):
    """Yield shuffled batches as JAX arrays.

    Parameters
    ----------
    images : np.ndarray, shape (N, H, W, 1)
    masks : np.ndarray, shape (N, H, W)
    batch_size : int
    rng : jax.Array
        JAX PRNG key for shuffling.

    Yields
    ------
    (jnp.ndarray, jnp.ndarray)
        Batches of (images, masks).
    """
    n = len(images)
    perm = jax.random.permutation(rng, n)
    images = images[perm]
    masks = masks[perm]

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield jnp.array(images[start:end]), jnp.array(masks[start:end])
