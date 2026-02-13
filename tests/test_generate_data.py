import uuid
import unittest

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generate_data import (
    _generate_disks,
    _uid_from_center,
    generate_batch,
    generate_circles_dataframe,
    validate_disjoint,
)

EXPECTED_COLUMNS = [
    "uid", "x", "y", "r", "label",
    "image_w", "image_h", "num_circles", "num_labeled",
]


# ── _generate_disks ─────────────────────────────────────────────────────

class TestGenerateDisks(unittest.TestCase):
    def test_output_shape(self):
        arr = _generate_disks((200, 200), 20, np.random.default_rng(0))
        self.assertEqual(arr.ndim, 2)
        self.assertEqual(arr.shape[1], 3)

    def test_radii_positive(self):
        arr = _generate_disks((200, 200), 30, np.random.default_rng(1))
        self.assertTrue((arr[:, 2] >= 1).all())

    def test_disks_inside_image(self):
        size = (150, 200)
        arr = _generate_disks(size, 25, np.random.default_rng(2))
        for x, y, r in arr:
            self.assertGreaterEqual(x - r, 0)
            self.assertGreaterEqual(y - r, 0)
            self.assertLessEqual(x + r, size[0])
            self.assertLessEqual(y + r, size[1])

    def test_disjoint_100_seeds(self):
        for seed in range(100):
            arr = _generate_disks((256, 256), 30, np.random.default_rng(seed))
            n = len(arr)
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.sqrt(
                        (arr[i, 0] - arr[j, 0]) ** 2
                        + (arr[i, 1] - arr[j, 1]) ** 2
                    )
                    self.assertGreaterEqual(
                        dist, arr[i, 2] + arr[j, 2],
                        f"seed={seed}: disks {i},{j} overlap",
                    )

    def test_reproducibility(self):
        a = _generate_disks((200, 200), 20, np.random.default_rng(7))
        b = _generate_disks((200, 200), 20, np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)

    def test_empty_when_no_candidates(self):
        arr = _generate_disks((200, 200), 0, np.random.default_rng(5))
        self.assertEqual(arr.shape, (0, 3))


# ── _uid_from_center ────────────────────────────────────────────────────

class TestUidFromCenter(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(_uid_from_center(10, 20), _uid_from_center(10, 20))

    def test_different_for_different_coords(self):
        self.assertNotEqual(_uid_from_center(10, 20), _uid_from_center(20, 10))

    def test_valid_uuid(self):
        uid = _uid_from_center(1, 2)
        uuid.UUID(uid)


# ── generate_circles_dataframe ──────────────────────────────────────────

class TestGenerateCirclesDataframe(unittest.TestCase):
    def test_columns(self):
        df = generate_circles_dataframe(20, (200, 200), 3, 5, seed=0)
        self.assertEqual(list(df.columns), EXPECTED_COLUMNS)

    def test_metadata_values(self):
        df = generate_circles_dataframe(25, (300, 200), 4, 5, seed=0)
        self.assertTrue((df["image_w"] == 300).all())
        self.assertTrue((df["image_h"] == 200).all())
        self.assertTrue((df["num_circles"] == 25).all())
        self.assertTrue((df["num_labeled"] == 4).all())

    def test_labels_assignment(self):
        df = generate_circles_dataframe(20, (256, 256), 3, 5, seed=0)
        labeled = df[df["label"] > 0]
        self.assertLessEqual(len(labeled), 3)
        self.assertTrue(labeled["label"].isin([1, 2, 3]).all())
        self.assertTrue((df[df["label"] == 0]["label"] == 0).all())

    def test_sorted_by_radius_descending(self):
        df = generate_circles_dataframe(20, (256, 256), 3, 5, seed=0)
        if len(df) > 1:
            radii = df["r"].tolist()
            self.assertEqual(radii, sorted(radii, reverse=True))

    def test_max_labels_clamped(self):
        df = generate_circles_dataframe(30, (256, 256), 10, 3, seed=0)
        self.assertLessEqual(df["label"].max(), 3)

    def test_empty_result(self):
        df = generate_circles_dataframe(0, (200, 200), 3, 5, seed=0)
        self.assertEqual(len(df), 0)
        self.assertEqual(list(df.columns), EXPECTED_COLUMNS)

    def test_uids_unique(self):
        df = generate_circles_dataframe(30, (256, 256), 3, 5, seed=0)
        self.assertEqual(df["uid"].nunique(), len(df))

    def test_reproducibility(self):
        a = generate_circles_dataframe(20, (200, 200), 3, 5, seed=42)
        b = generate_circles_dataframe(20, (200, 200), 3, 5, seed=42)
        pd.testing.assert_frame_equal(a, b)


# ── validate_disjoint ──────────────────────────────────────────────────

class TestValidateDisjoint(unittest.TestCase):
    def test_disjoint(self):
        df = pd.DataFrame({"x": [0, 100], "y": [0, 0], "r": [10, 10]})
        ok, overlaps = validate_disjoint(df)
        self.assertTrue(ok)
        self.assertEqual(overlaps, [])

    def test_overlapping(self):
        df = pd.DataFrame({"x": [0, 10], "y": [0, 0], "r": [10, 10]})
        ok, overlaps = validate_disjoint(df)
        self.assertFalse(ok)
        self.assertEqual(overlaps, [(0, 1)])

    def test_tangent_is_disjoint(self):
        df = pd.DataFrame({"x": [0, 20], "y": [0, 0], "r": [10, 10]})
        ok, _ = validate_disjoint(df)
        self.assertTrue(ok)

    def test_single_circle(self):
        ok, overlaps = validate_disjoint(
            pd.DataFrame({"x": [50], "y": [50], "r": [10]})
        )
        self.assertTrue(ok)
        self.assertEqual(overlaps, [])

    def test_empty(self):
        ok, overlaps = validate_disjoint(
            pd.DataFrame({"x": [], "y": [], "r": []})
        )
        self.assertTrue(ok)
        self.assertEqual(overlaps, [])


# ── generate_batch ──────────────────────────────────────────────────────

class TestGenerateBatch(unittest.TestCase):
    def test_sample_id_present(self):
        batch = generate_batch(5, 10, (100, 100), 2, 3, seed=0)
        self.assertIn("sample_id", batch.columns)
        self.assertEqual(batch["sample_id"].nunique(), 5)

    def test_reproducibility(self):
        a = generate_batch(5, 10, (100, 100), 2, 3, seed=0)
        b = generate_batch(5, 10, (100, 100), 2, 3, seed=0)
        pd.testing.assert_frame_equal(a, b)

    def test_each_sample_disjoint(self):
        batch = generate_batch(50, 20, (200, 200), 3, 5, seed=0)
        for _, group in batch.groupby("sample_id"):
            ok, _ = validate_disjoint(group.reset_index(drop=True))
            self.assertTrue(ok)

    def test_max_workers(self):
        batch = generate_batch(4, 10, (100, 100), 2, 3, seed=0, max_workers=2)
        self.assertEqual(batch["sample_id"].nunique(), 4)


if __name__ == "__main__":
    unittest.main()
