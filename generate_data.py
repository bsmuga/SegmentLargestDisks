import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd


def _generate_disks(
    image_size: tuple[int, int],
    num_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate non-overlapping disks using a greedy algorithm.

    1. Sample random candidate centers.
    2. Compute the *full* pairwise distance matrix.
    3. For each center, pick a random radius that fits inside the image
       and does not overlap any previously placed disk.

    Returns an (M, 3) array of [x, y, r] for the M accepted disks.
    """
    centers = np.column_stack(
        [
            rng.integers(0, image_size[0], num_points),
            rng.integers(0, image_size[1], num_points),
        ]
    )
    centers = np.unique(centers, axis=0)

    # Full pairwise distance matrix (NaN on diagonal).
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    distances = np.sqrt((diff**2).sum(axis=2))
    np.fill_diagonal(distances, np.nan)

    disks: list[list[int]] = []
    for i, center in enumerate(centers):
        r_max = np.nanmin(
            [
                *distances[i, :],
                center[0],
                center[1],
                image_size[0] - center[0],
                image_size[1] - center[1],
            ]
        )
        if r_max > 1:
            r = int(rng.integers(1, int(np.floor(r_max)) + 1))
            disks.append([int(center[0]), int(center[1]), r])
            distances[:, i] -= r
            distances[i, :] -= r

    if not disks:
        return np.empty((0, 3), dtype=int)
    return np.asarray(disks, dtype=int)


NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


def _uid_from_center(x: int, y: int) -> str:
    """Deterministic UUID derived from circle center coordinates."""
    return str(uuid.uuid5(NAMESPACE, f"{x},{y}"))


def generate_circles_dataframe(
    num_circles: int,
    image_size: tuple[int, int],
    num_labeled: int,
    max_labels: int,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate non-overlapping circles and return as a DataFrame.

    Parameters
    ----------
    num_circles : int
        Maximum number of candidate center points (actual circles may be fewer).
    image_size : tuple[int, int]
        Size of the canvas (width, height).
    num_labeled : int
        Number of largest circles to assign distinct labels.
    max_labels : int
        Maximum label value. Labels are clamped to [1, max_labels].
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: uid, x, y, r, label, image_w, image_h,
        num_circles, num_labeled.
    """
    rng = np.random.default_rng(seed)
    arr = _generate_disks(image_size, num_circles, rng)

    columns = [
        "uid", "x", "y", "r", "label",
        "image_w", "image_h", "num_circles", "num_labeled",
    ]
    if len(arr) == 0:
        return pd.DataFrame(columns=columns)

    # Sort by radius descending for labelling.
    order = np.argsort(-arr[:, 2])
    arr = arr[order]

    labels = np.zeros(len(arr), dtype=int)
    for i in range(min(num_labeled, len(arr))):
        labels[i] = min(i + 1, max_labels)

    records = []
    for i, (x, y, r) in enumerate(arr):
        records.append({
            "uid": _uid_from_center(x, y),
            "x": x,
            "y": y,
            "r": r,
            "label": int(labels[i]),
            "image_w": image_size[0],
            "image_h": image_size[1],
            "num_circles": num_circles,
            "num_labeled": num_labeled,
        })

    return pd.DataFrame(records)


def generate_batch(
    num_samples: int,
    num_circles: int,
    image_size: tuple[int, int],
    num_labeled: int,
    max_labels: int,
    seed: int | None = None,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Generate multiple samples in parallel using ThreadPoolExecutor.

    Parameters
    ----------
    num_samples : int
        Number of independent circle sets to generate.
    num_circles : int
        Maximum candidate centers per sample.
    image_size : tuple[int, int]
        Canvas size (width, height).
    num_labeled : int
        Number of largest circles to label per sample.
    max_labels : int
        Maximum label value.
    seed : int or None
        Base seed. Each sample uses seed + i for reproducibility.
    max_workers : int or None
        Thread pool size. None lets the executor pick a default.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame for all samples with an extra ``sample_id`` column.
    """
    def _worker(i: int) -> pd.DataFrame:
        sample_seed = seed + i if seed is not None else None
        df = generate_circles_dataframe(
            num_circles=num_circles,
            image_size=image_size,
            num_labeled=num_labeled,
            max_labels=max_labels,
            seed=sample_seed,
        )
        df["sample_id"] = i
        return df

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        frames = list(executor.map(_worker, range(num_samples)))

    return pd.concat(frames, ignore_index=True)


def validate_disjoint(df: pd.DataFrame) -> tuple[bool, list[tuple[int, int]]]:
    """Check whether all circles in the DataFrame are pairwise disjoint.

    Two circles overlap when the Euclidean distance between their centers
    is strictly less than the sum of their radii.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns x, y, r.

    Returns
    -------
    tuple[bool, list[tuple[int, int]]]
        (True, []) if all circles are disjoint.
        (False, overlapping_pairs) otherwise, where each pair is a tuple
        of row indices.
    """
    xs = df["x"].to_numpy(dtype=float)
    ys = df["y"].to_numpy(dtype=float)
    rs = df["r"].to_numpy(dtype=float)
    n = len(df)

    overlaps: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
            if dist < rs[i] + rs[j]:
                overlaps.append((i, j))

    return (len(overlaps) == 0, overlaps)


if __name__ == "__main__":
    IMAGE_SIZE = (256, 256)
    NUM_CIRCLES = 30
    NUM_LABELED = 25
    MAX_LABELS = 5
    NUM_TRIALS = 500

    # --- parallel batch generation ---
    batch = generate_batch(
        num_samples=NUM_TRIALS,
        num_circles=NUM_CIRCLES,
        image_size=IMAGE_SIZE,
        num_labeled=NUM_LABELED,
        max_labels=MAX_LABELS,
        seed=0,
    )
    print(f"Generated {len(batch)} circles across {batch['sample_id'].nunique()} samples")
    print(batch.head(10))
    print()

    # --- validate disjointness per sample ---
    failures = 0
    for sample_id, group in batch.groupby("sample_id"):
        ok, overlapping = validate_disjoint(group.reset_index(drop=True))
        if not ok:
            failures += 1

    print(f"Result: {failures}/{NUM_TRIALS} trials had overlapping circles")
    if failures == 0:
        print("Algorithm always produced disjoint circles in this test.")
    else:
        print("Algorithm does NOT always produce disjoint circles.")
