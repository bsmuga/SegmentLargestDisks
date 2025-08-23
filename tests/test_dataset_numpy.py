import unittest
import numpy as np
from src.dataset import DisksDataset, Disk


class TestDisksDatasetNumpy(unittest.TestCase):
    """Comprehensive tests for DisksDataset using numpy"""

    def test_generate_disks_non_overlapping(self):
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
        dataset = DisksDataset((200, 300), 10, 3, 1)
        image, segmentation = dataset[0]
        
        # Check shapes and types
        self.assertEqual(image.shape, (1, 300, 200))  # (C, H, W)
        self.assertEqual(segmentation.shape, (300, 200))  # (H, W)
        self.assertEqual(image.dtype, np.float32)
        self.assertEqual(segmentation.dtype, np.int64)
        
        self.assertLessEqual(image.max(), 1.0)
        self.assertLessEqual(segmentation.max(), 3)

    def test_reproducibility_with_seed(self):
        seed = 42
        ds1 = DisksDataset((100, 100), 5, 3, 10, seed=seed)
        ds2 = DisksDataset((100, 100), 5, 3, 10, seed=seed)

        # Check first 3 items are identical
        for i in range(3):
            img1, seg1 = ds1[i]
            img2, seg2 = ds2[i]

            np.testing.assert_array_equal(img1, img2)
            np.testing.assert_array_equal(seg1, seg2)

    def test_different_results_without_seed(self):
        ds1 = DisksDataset((100, 100), 5, 3, 10)
        ds2 = DisksDataset((100, 100), 5, 3, 10)

        # Check that at least one of first 3 items is different
        different = False
        for i in range(3):
            img1, seg1 = ds1[i]
            img2, seg2 = ds2[i]

            if not np.array_equal(img1, img2) or not np.array_equal(seg1, seg2):
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
        unique_labels = np.unique(seg)

        # Should contain 0 (background) and possibly 1, 2, 3 (depending on disk count)
        self.assertIn(0, unique_labels, "Background label 0 should be present")

        # All labels should be in range [0, labeled_disks]
        self.assertTrue(all(0 <= label <= 3 for label in unique_labels))

        # No negative labels
        self.assertTrue(all(label >= 0 for label in unique_labels))

    def test_disk_rendering_efficiency(self):
        # Test with larger image to see performance difference
        size = (400, 300)
        disks = [
            Disk(100, 100, 50),
            Disk(200, 150, 30),
            Disk(300, 200, 40),
        ]
        values = [1, 2, 3]

        # Test the optimized version
        import time
        start = time.time()
        for _ in range(10):
            img = DisksDataset.disks2img(size, disks, values)
        elapsed = time.time() - start

        # The optimized version should be reasonably fast
        self.assertLess(elapsed, 1.0, "Optimized rendering should be fast")

        # Check correctness
        self.assertEqual(img.shape, (size[1], size[0]))
        self.assertTrue(np.all(img >= 0))

    def test_disk_sorting_by_size(self):
        """Test that disks are properly sorted by size for labeling"""
        ds = DisksDataset((200, 200), 10, 5, 10, seed=42)

        for i in range(10):
            img, seg = ds[i]

            # If we have labeled disks, verify label 1 corresponds to largest area
            unique_labels = np.unique(seg)
            if len(unique_labels) > 1:  # More than just background
                areas = []
                for label in unique_labels:
                    if label > 0:  # Skip background
                        area = np.sum(seg == label)
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

    def test_edge_cases(self):
        """Test various edge cases"""
        # Very small image
        ds_small = DisksDataset((10, 10), 20, 3, 1, seed=42)
        img, seg = ds_small[0]
        self.assertEqual(img.shape, (1, 10, 10))
        self.assertEqual(seg.shape, (10, 10))

        # Zero disks
        ds_zero = DisksDataset((100, 100), 0, 0, 1, seed=42)
        img, seg = ds_zero[0]
        self.assertEqual(np.sum(img), 0)
        self.assertEqual(np.sum(seg), 0)

        # More labels than possible disks
        ds_more_labels = DisksDataset((50, 50), 2, 10, 1, seed=42)
        img, seg = ds_more_labels[0]
        max_label = np.max(seg)
        self.assertLessEqual(max_label, 10)

    def test_disk_dataclass(self):
        """Test Disk dataclass"""
        disk = Disk(50, 60, 20)
        self.assertEqual(disk.x, 50)
        self.assertEqual(disk.y, 60)
        self.assertEqual(disk.r, 20)

    def test_dataset_length(self):
        """Test dataset length"""
        items = 25
        ds = DisksDataset((100, 100), 5, 3, items, seed=42)
        self.assertEqual(len(ds), items)

        # Should be able to access all items
        for i in range(items):
            img, seg = ds[i]
            self.assertIsInstance(img, np.ndarray)
            self.assertIsInstance(seg, np.ndarray)


if __name__ == "__main__":
    unittest.main()