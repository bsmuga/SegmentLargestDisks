import unittest
import torch
from src.data.dm import DiskDataModule


class TestDiskDataModule(unittest.TestCase):
    """Tests for DiskDataModule"""

    def test_datamodule_initialization(self):
        """Test that DataModule initializes correctly"""
        dm = DiskDataModule(
            image_size=(100, 100),
            disk_max_num=5,
            labeled_disks=3,
            train_items=10,
            valid_items=5,
            test_items=3,
            batch_size=2,
            num_workers=1,
        )
        
        self.assertEqual(dm.image_size, (100, 100))
        self.assertEqual(dm.disk_max_num, 5)
        self.assertEqual(dm.labeled_disks, 3)
        self.assertEqual(dm.train_items, 10)
        self.assertEqual(dm.valid_items, 5)
        self.assertEqual(dm.test_items, 3)
        self.assertEqual(dm.batch_size, 2)
        self.assertEqual(dm.num_workers, 1)
        self.assertIsNone(dm.seed)

    def test_datamodule_with_seed(self):
        """Test that DataModule works with seed parameter"""
        seed = 42
        dm = DiskDataModule(
            image_size=(100, 100),
            disk_max_num=5,
            labeled_disks=3,
            train_items=10,
            valid_items=5,
            test_items=3,
            batch_size=2,
            num_workers=1,
            seed=seed,
        )
        
        self.assertEqual(dm.seed, seed)
        
        # Setup datasets
        dm.setup("fit")
        
        # Check that datasets were created
        self.assertIsNotNone(dm.ds_train)
        self.assertIsNotNone(dm.ds_valid)
        self.assertIsNotNone(dm.ds_test)
        
        # Check that datasets have correct seeds
        self.assertEqual(dm.ds_train.seed, seed)
        self.assertEqual(dm.ds_valid.seed, seed + 1000)
        self.assertEqual(dm.ds_test.seed, seed + 2000)

    def test_datamodule_reproducibility(self):
        """Test that DataModule with same seed produces same data"""
        seed = 42
        
        # Create two identical DataModules
        dm1 = DiskDataModule(
            image_size=(50, 50),
            disk_max_num=3,
            labeled_disks=2,
            train_items=5,
            valid_items=3,
            test_items=2,
            batch_size=1,
            num_workers=1,
            seed=seed,
        )
        
        dm2 = DiskDataModule(
            image_size=(50, 50),
            disk_max_num=3,
            labeled_disks=2,
            train_items=5,
            valid_items=3,
            test_items=2,
            batch_size=1,
            num_workers=1,
            seed=seed,
        )
        
        # Setup both
        dm1.setup("fit")
        dm2.setup("fit")
        
        # Check that first training samples are identical
        img1, seg1 = dm1.ds_train[0]
        img2, seg2 = dm2.ds_train[0]
        
        torch.testing.assert_close(img1, img2)
        torch.testing.assert_close(seg1, seg2)

    def test_dataloader_creation(self):
        """Test that DataLoaders are created correctly"""
        dm = DiskDataModule(
            image_size=(50, 50),
            disk_max_num=3,
            labeled_disks=2,
            train_items=4,
            valid_items=2,
            test_items=2,
            batch_size=2,
            num_workers=1,
        )
        
        dm.setup("fit")
        
        # Test DataLoader creation
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        test_loader = dm.test_dataloader()
        
        # Check that loaders are not None
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)
        
        # Check batch sizes
        train_batch = next(iter(train_loader))
        self.assertEqual(len(train_batch), 2)  # image, segmentation
        self.assertEqual(train_batch[0].shape[0], 2)  # batch size
        
        val_batch = next(iter(val_loader))
        self.assertEqual(val_batch[0].shape[0], 2)  # batch size
        
        test_batch = next(iter(test_loader))
        self.assertEqual(test_batch[0].shape[0], 2)  # batch size

    def test_datamodule_without_seed(self):
        """Test that DataModule works without seed (None)"""
        dm = DiskDataModule(
            image_size=(50, 50),
            disk_max_num=3,
            labeled_disks=2,
            train_items=2,
            valid_items=2,
            test_items=2,
            batch_size=1,
            num_workers=1,
            seed=None,
        )
        
        dm.setup("fit")
        
        # Check that datasets have None seed
        self.assertIsNone(dm.ds_train.seed)
        self.assertIsNone(dm.ds_valid.seed)
        self.assertIsNone(dm.ds_test.seed)
        
        # Should still work normally
        img, seg = dm.ds_train[0]
        self.assertEqual(img.shape, (1, 50, 50))
        self.assertEqual(seg.shape, (50, 50))


if __name__ == "__main__":
    unittest.main()