import unittest
import numpy as np
from src.confusion_matrix import ConfusionMatrix


class TestConfusionMatrixNumpy(unittest.TestCase):
    """Tests for numpy-based ConfusionMatrix"""
    
    def setUp(self):
        self.confusion_matrix = ConfusionMatrix(iou_threshold=0.25, num_classes=4)
        self.target = np.zeros((1, 100, 100), dtype=np.int64)
        self.target[0, :50, :50] = 0
        self.target[0, :50, 50:] = 1
        self.target[0, 50:, :50] = 2
        self.target[0, 50:, 50:] = 3

        self.preds = self.target.copy()

    def test_perfect_score(self):
        result = self.confusion_matrix(self.target, self.target)
        expected = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
        np.testing.assert_array_equal(result, expected)

    def test_false_positive(self):
        target_modified = self.target.copy()
        target_modified[0, 50:, 50:] = 0
        result = self.confusion_matrix(self.preds, target_modified)
        expected = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]])
        np.testing.assert_array_equal(result, expected)

    def test_false_negative(self):
        preds_modified = self.preds.copy()
        preds_modified[0, :, :50] = 0
        result = self.confusion_matrix(preds_modified, self.target)
        expected = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]])
        np.testing.assert_array_equal(result, expected)

    def test_true_negative(self):
        preds_modified = self.preds.copy()
        target_modified = self.target.copy()
        preds_modified[0, 50:, 50:] = 0
        target_modified[0, 50:, 50:] = 0
        result = self.confusion_matrix(preds_modified, target_modified)
        expected = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        np.testing.assert_array_equal(result, expected)

    def test_different_iou_thresholds(self):
        """Test confusion matrix with different IoU thresholds"""
        # Create test data with partial overlap
        target = np.zeros((1, 100, 100), dtype=np.int64)
        target[0, 20:60, 20:60] = 1  # 40x40 = 1600 pixels
        
        preds = np.zeros((1, 100, 100), dtype=np.int64)
        preds[0, 30:70, 30:70] = 1  # 40x40 = 1600 pixels, offset by 10
        # IoU ≈ 0.39
        
        # Should be TP with low threshold
        cm_low = ConfusionMatrix(iou_threshold=0.1, num_classes=2)
        result_low = cm_low(preds, target)
        self.assertEqual(result_low[1, 0], 1)  # TP
        
        # Should be FN with high threshold
        cm_high = ConfusionMatrix(iou_threshold=0.9, num_classes=2)
        result_high = cm_high(preds, target)
        self.assertEqual(result_high[1, 2], 1)  # FN

    def test_2d_inputs(self):
        """Test that 2D inputs are handled correctly"""
        cm = ConfusionMatrix(iou_threshold=0.5, num_classes=2)
        
        target_2d = np.zeros((50, 50), dtype=np.int64)
        target_2d[10:30, 10:30] = 1
        
        preds_2d = target_2d.copy()
        
        result = cm(preds_2d, target_2d)
        self.assertEqual(result[1, 0], 1)  # Should be TP

    def test_state_accumulation(self):
        """Test that state accumulates correctly"""
        cm = ConfusionMatrix(iou_threshold=0.5, num_classes=2)
        
        target1 = np.zeros((50, 50), dtype=np.int64)
        target1[10:30, 10:30] = 1
        preds1 = target1.copy()
        
        target2 = np.zeros((50, 50), dtype=np.int64)
        target2[20:40, 20:40] = 1
        preds2 = target2.copy()
        
        # Update twice
        cm.update(preds1, target1)
        cm.update(preds2, target2)
        result = cm.compute()
        
        # Should have 2 TPs for class 1
        self.assertEqual(result[1, 0], 2)
        
        # Reset and check
        cm.reset()
        result_after_reset = cm.compute()
        self.assertEqual(result_after_reset.sum(), 0)

    def test_numerical_stability(self):
        """Test numerical stability with edge cases"""
        cm = ConfusionMatrix(iou_threshold=0.5, num_classes=2)
        
        # Test with zero union (both empty)
        target = np.zeros((50, 50), dtype=np.int64)
        preds = np.zeros((50, 50), dtype=np.int64)
        
        # Should not crash
        result = cm(preds, target)
        self.assertIsInstance(result, np.ndarray)

    def test_multiclass_scenario(self):
        """Test with multiple classes"""
        cm = ConfusionMatrix(iou_threshold=0.5, num_classes=4)
        
        target = np.zeros((2, 100, 100), dtype=np.int64)
        preds = np.zeros((2, 100, 100), dtype=np.int64)
        
        # Batch 1: Perfect matches
        target[0, :50, :50] = 1
        target[0, 50:, :50] = 2
        preds[0, :50, :50] = 1
        preds[0, 50:, :50] = 2
        
        # Batch 2: Some mismatches
        target[1, :50, 50:] = 3
        preds[1, :50, 50:] = 1  # Wrong class
        
        result = cm(preds, target)
        
        # Check some expected values
        self.assertEqual(result[1, 0], 1)  # TP for class 1 (from batch 1)
        self.assertEqual(result[2, 0], 1)  # TP for class 2 (from batch 1)
        self.assertEqual(result[3, 2], 1)  # FN for class 3 (from batch 2)

    def test_single_pixel_case(self):
        """Test with single pixel predictions"""
        cm = ConfusionMatrix(iou_threshold=0.5, num_classes=2)
        
        target = np.zeros((10, 10), dtype=np.int64)
        target[5, 5] = 1
        
        # Perfect match
        preds = target.copy()
        result = cm(preds, target)
        self.assertEqual(result[1, 0], 1)  # TP
        
        # No overlap
        preds_miss = np.zeros((10, 10), dtype=np.int64)
        preds_miss[5, 6] = 1
        result_miss = cm(preds_miss, target)
        self.assertEqual(result_miss[1, 2], 1)  # FN


if __name__ == "__main__":
    unittest.main()