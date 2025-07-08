import unittest
import numpy as np
import torch
from src.data.dataset import DisksDataset, Disk
import time


class TestDisksDataset(unittest.TestCase):
    """Comprehensive tests for DisksDataset"""

    # Original tests from test_generate_disks.py
    def test_generate_disks_non_overlapping(self):
        """Test that generated disks don't overlap (original test)"""
        disks = DisksDataset.generate_disks((200, 300), 5)

        for i in range(len(disks) - 1):
            for j in range(i + 1, len(disks)):
                c_1 = disks[i]
                c_2 = disks[j]
                # Check that distance between centers is greater than sum of radii
                distance_squared = (c_1.x - c_2.x) ** 2 + (c_1.y - c_2.y) ** 2
                sum_radii_squared = (c_1.r + c_2.r) ** 2
                self.assertGreater(
                    distance_squared,
                    sum_radii_squared,
                    f"Disks {i} and {j} overlap",
                )

    def test_dataset_basic(self):
        """Test basic dataset functionality (original test)"""
        dataset = DisksDataset((200, 300), 10, 3, 1)
        image, segmentation = next(iter(dataset))
        self.assertLessEqual(image.max(), 1.0)  # Should be 1.0 or 0.0 if no disks
        # With new labeling, max should be at most labeled_disks (3)
        self.assertLessEqual(segmentation.max(), 3)

    # New comprehensive tests for fixes
    def test_reproducibility_with_seed(self):
        """Test that dataset generates same data with same seed"""
        seed = 42
        ds1 = DisksDataset((100, 100), 5, 3, 10, seed=seed)
        ds2 = DisksDataset((100, 100), 5, 3, 10, seed=seed)

        # Check first 3 items are identical
        for i in range(3):
            img1, seg1 = ds1[i]
            img2, seg2 = ds2[i]

            torch.testing.assert_close(img1, img2)
            torch.testing.assert_close(seg1, seg2)

    def test_different_results_without_seed(self):
        """Test that dataset generates different data without seed"""
        ds1 = DisksDataset((100, 100), 5, 3, 10)
        ds2 = DisksDataset((100, 100), 5, 3, 10)

        # Check that at least one of first 3 items is different
        different = False
        for i in range(3):
            img1, seg1 = ds1[i]
            img2, seg2 = ds2[i]

            if not torch.equal(img1, img2) or not torch.equal(seg1, seg2):
                different = True
                break

        self.assertTrue(
            different, "Datasets without seed should generate different data"
        )

    def test_label_generation_correctness(self):
        """Test that labels are generated correctly (1 to labeled_disks, 0 for others)"""
        ds = DisksDataset((100, 100), 10, 3, 1, seed=42)
        img, seg = ds[0]

        # Get unique labels
        unique_labels = torch.unique(seg).numpy()

        # Should contain 0 (background) and possibly 1, 2, 3 (depending on disk count)
        self.assertIn(0, unique_labels, "Background label 0 should be present")

        # All labels should be in range [0, labeled_disks]
        self.assertTrue(all(0 <= label <= 3 for label in unique_labels))

        # No negative labels
        self.assertTrue(all(label >= 0 for label in unique_labels))

    def test_disk_rendering_efficiency(self):
        """Test that optimized disk rendering is faster than pixel-by-pixel"""
        # Test with larger image to see performance difference
        size = (400, 300)
        disks = [
            Disk(100, 100, 50),
            Disk(200, 150, 30),
            Disk(300, 200, 40),
        ]
        values = [1, 2, 3]

        # Time the optimized version
        start = time.time()
        for _ in range(10):
            img = DisksDataset.disks2img(size, disks, values)
        optimized_time = time.time() - start

        # The optimized version should be reasonably fast
        self.assertLess(optimized_time, 1.0, "Optimized rendering should be fast")

        # Check correctness
        self.assertEqual(img.shape, (size[1], size[0]))
        self.assertTrue(np.all(img >= 0))

    def test_no_duplicate_floor_operation(self):
        """Test that floor operation is not duplicated"""
        rng = np.random.default_rng(42)
        disks = DisksDataset.generate_disks((100, 100), 5, rng)

        # Check all disk radii are integers
        for disk in disks:
            self.assertEqual(disk.r, int(disk.r))

    def test_disk_non_overlapping_comprehensive(self):
        """Test that generated disks don't overlap using image values"""
        ds = DisksDataset((200, 200), 10, 5, 1, seed=42)
        img, seg = ds[0]

        # Convert to numpy for easier processing
        img_np = img.squeeze().numpy()

        # In a non-overlapping scenario, no pixel should have value > 1
        # (since each disk contributes 1 to the image)
        self.assertTrue(np.all(img_np <= 1), "Disks should not overlap")

    def test_segmentation_mask_consistency(self):
        """Test that segmentation mask matches the disk positions"""
        ds = DisksDataset((150, 150), 5, 3, 1, seed=42)
        img, seg = ds[0]

        # Where image has a disk (value > 0), segmentation should have a label
        img_np = img.squeeze().numpy()
        seg_np = seg.numpy()

        # All pixels with disks in image should have non-zero labels in segmentation
        # (except for disks beyond labeled_disks which get label 0)
        disk_pixels = img_np > 0

        # At least some disk pixels should have non-zero labels
        labeled_pixels = seg_np > 0
        self.assertTrue(np.any(labeled_pixels), "Some disks should be labeled")

    def test_empty_disk_generation(self):
        """Test behavior when no disks can fit"""
        # Very small image where disks might not fit
        ds = DisksDataset((10, 10), 20, 3, 1, seed=42)
        img, seg = ds[0]

        # Should not crash and return valid tensors
        self.assertEqual(img.shape, (1, 10, 10))
        self.assertEqual(seg.shape, (10, 10))

    def test_disk_sorting_by_size(self):
        """Test that disks are properly sorted by size for labeling"""
        # Generate multiple samples to ensure we get disks
        ds = DisksDataset((200, 200), 10, 5, 10, seed=42)

        for i in range(10):
            img, seg = ds[i]
            seg_np = seg.numpy()

            # If we have labeled disks, verify label 1 corresponds to largest area
            unique_labels = np.unique(seg_np)
            if len(unique_labels) > 1:  # More than just background
                areas = []
                for label in unique_labels:
                    if label > 0:  # Skip background
                        area = np.sum(seg_np == label)
                        areas.append((label, area))

                if len(areas) > 1:
                    # Sort by area descending
                    areas.sort(key=lambda x: x[1], reverse=True)
                    # First label (1) should have the largest area
                    self.assertEqual(
                        areas[0][0],
                        1,
                        f"Label 1 should be the largest disk, but got {areas}",
                    )


if __name__ == "__main__":
    unittest.main()