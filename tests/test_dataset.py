"""Correctness tests for dataset.render_sample.

These verify that labels and foreground pixels land exactly where the disk
geometry says they should — the load-bearing invariant the rest of training
depends on.
"""

import numpy as np
import pandas as pd

from dataset import render_sample


IMAGE_SIZE = (32, 32)


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal DataFrame with the columns render_sample needs."""
    return pd.DataFrame(rows, columns=["x", "y", "r", "label"])


def _expected_disk_mask(
    cx: int, cy: int, r: int, image_size: tuple[int, int]
) -> np.ndarray:
    """Boolean mask: True where (x - cx)^2 + (y - cy)^2 < r^2."""
    w, h = image_size
    yy, xx = np.mgrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2


class TestRenderSampleLabeledDisk:
    def test_label_lands_only_inside_radius(self):
        cx, cy, r, label = 16, 16, 5, 3
        df = _make_df([{"x": cx, "y": cy, "r": r, "label": label}])

        image, mask = render_sample(df, IMAGE_SIZE)

        inside = _expected_disk_mask(cx, cy, r, IMAGE_SIZE)
        assert np.all(mask[inside] == label)
        assert np.all(mask[~inside] == 0)

    def test_image_foreground_matches_mask_extent(self):
        cx, cy, r = 10, 12, 4
        df = _make_df([{"x": cx, "y": cy, "r": r, "label": 1}])

        image, _ = render_sample(df, IMAGE_SIZE)

        inside = _expected_disk_mask(cx, cy, r, IMAGE_SIZE)
        assert np.all(image[inside, 0] == 1.0)
        assert np.all(image[~inside, 0] == 0.0)

    def test_output_shapes_and_dtypes(self):
        df = _make_df([{"x": 8, "y": 8, "r": 3, "label": 2}])

        image, mask = render_sample(df, IMAGE_SIZE)

        w, h = IMAGE_SIZE
        assert image.shape == (h, w, 1)
        assert mask.shape == (h, w)
        assert image.dtype == np.float32
        assert mask.dtype == np.int32


class TestRenderSampleUnlabeledDisk:
    def test_zero_label_paints_image_but_not_mask(self):
        cx, cy, r = 16, 16, 6
        df = _make_df([{"x": cx, "y": cy, "r": r, "label": 0}])

        image, mask = render_sample(df, IMAGE_SIZE)

        inside = _expected_disk_mask(cx, cy, r, IMAGE_SIZE)
        assert np.all(image[inside, 0] == 1.0)
        assert np.all(mask == 0)


class TestRenderSampleMultipleDisks:
    def test_two_disjoint_labels_do_not_collide(self):
        a = {"x": 8, "y": 8, "r": 3, "label": 1}
        b = {"x": 24, "y": 24, "r": 3, "label": 2}
        df = _make_df([a, b])

        _, mask = render_sample(df, IMAGE_SIZE)

        inside_a = _expected_disk_mask(a["x"], a["y"], a["r"], IMAGE_SIZE)
        inside_b = _expected_disk_mask(b["x"], b["y"], b["r"], IMAGE_SIZE)

        assert np.all(mask[inside_a] == 1)
        assert np.all(mask[inside_b] == 2)
        assert np.all(mask[~(inside_a | inside_b)] == 0)

    def test_unlabeled_disk_does_not_overwrite_labeled_neighbour(self):
        """Unlabeled disks should not zero out labels written for labeled disks,
        regardless of iteration order in the DataFrame."""
        labeled = {"x": 8, "y": 8, "r": 3, "label": 1}
        unlabeled = {"x": 24, "y": 24, "r": 3, "label": 0}
        df = _make_df([labeled, unlabeled])

        _, mask = render_sample(df, IMAGE_SIZE)

        inside_labeled = _expected_disk_mask(
            labeled["x"], labeled["y"], labeled["r"], IMAGE_SIZE
        )
        assert np.all(mask[inside_labeled] == 1)
