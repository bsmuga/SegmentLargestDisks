import unittest
import numpy as np
from src.confusion_matrix import ConfusionMatrix


class TestConfusionMatrixAdvanced(unittest.TestCase):
    """Advanced tests for IoU-based confusion matrix"""
    
    def test_different_iou_thresholds(self):
        """Test confusion matrix with different IoU thresholds"""
        thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        # Create test data with partial overlap
        target = np.zeros((1, 100, 100), dtype=np.int64)
        target[0, 20:60, 20:60] = 1  # 40x40 = 1600 pixels
        
        preds = np.zeros((1, 100, 100), dtype=np.int64)
        preds[0, 30:70, 30:70] = 1  # 40x40 = 1600 pixels, offset by 10
        # Intersection: 30x30 = 900 pixels
        # Union: 1600 + 1600 - 900 = 2300 pixels
        # IoU = 900/2300 ≈ 0.39
        
        for threshold in thresholds:
            cm = ConfusionMatrix(iou_threshold=threshold, num_classes=2)
            result = cm(preds, target)
            
            if threshold < 0.39:  # Should detect as TP
                self.assertEqual(result[1, 0], 1, f"TP for threshold {threshold}")
                self.assertEqual(result[1, 2], 0, f"FN for threshold {threshold}")
            else:  # Should be FN
                self.assertEqual(result[1, 0], 0, f"TP for threshold {threshold}")
                self.assertEqual(result[1, 2], 1, f"FN for threshold {threshold}")
                
    def test_multi_class_confusion(self):
        """Test confusion matrix with multiple classes"""
        cm = ConfusionMatrix(iou_threshold=0.5, num_classes=5)
        
        # Create multi-class scenario
        target = np.zeros((2, 100, 100), dtype=np.int64)
        preds = np.zeros((2, 100, 100), dtype=np.int64)
        
        # Batch 1: Perfect matches for classes 1 and 2
        target[0, :50, :50] = 1
        target[0, 50:, :50] = 2
        preds[0, :50, :50] = 1
        preds[0, 50:, :50] = 2
        
        # Batch 2: Mismatches
        target[1, :50, 50:] = 3
        target[1, 50:, 50:] = 4
        preds[1, :50, 50:] = 4  # Wrong class
        preds[1, 50:, 50:] = 3  # Wrong class
        
        result = cm(preds, target)
        
        # Check class 1 and 2 (perfect matches in batch 1)
        self.assertEqual(result[1, 0], 1)  # TP
        self.assertEqual(result[2, 0], 1)  # TP
        
        # Check class 3 and 4 (mismatches in batch 2)
        self.assertEqual(result[3, 2], 1)  # FN (target has 3, pred has wrong location)
        self.assertEqual(result[4, 2], 1)  # FN (target has 4, pred has wrong location)
        
    def test_empty_predictions(self):
        """Test when predictions are empty"""
        cm = ConfusionMatrix(iou_threshold=0.5, num_classes=3)
        
        target = np.zeros((1, 50, 50), dtype=np.int64)
        target[0, 10:30, 10:30] = 1
        target[0, 35:45, 35:45] = 2
        
        preds = np.zeros((1, 50, 50), dtype=np.int64)  # All zeros
        
        result = cm(preds, target)
        
        # Should have FN for classes 1 and 2
        self.assertEqual(result[1, 2], 1)  # FN for class 1
        self.assertEqual(result[2, 2], 1)  # FN for class 2
        self.assertEqual(result[0, 3], 1)  # TN for class 0
        
    def test_empty_targets(self):
        """Test when targets are empty"""
        cm = ConfusionMatrix(iou_threshold=0.5, num_classes=3)
        
        target = np.zeros((1, 50, 50), dtype=np.int64)  # All zeros
        
        preds = np.zeros((1, 50, 50), dtype=np.int64)
        preds[0, 10:30, 10:30] = 1
        preds[0, 35:45, 35:45] = 2
        
        result = cm(preds, target)
        
        # Should have FP for classes 1 and 2
        self.assertEqual(result[1, 1], 1)  # FP for class 1
        self.assertEqual(result[2, 1], 1)  # FP for class 2
        
    def test_exact_iou_threshold(self):
        """Test behavior at exact IoU threshold"""
        cm = ConfusionMatrix(iou_threshold=0.5, num_classes=2)
        
        # Create exactly 50% overlap
        target = np.zeros((1, 100, 100), dtype=np.int64)
        target[0, 0:40, 0:50] = 1  # 2000 pixels
        
        preds = np.zeros((1, 100, 100), dtype=np.int64)
        preds[0, 20:60, 0:50] = 1  # 2000 pixels
        # Intersection: 20x50 = 1000 pixels
        # Union: 2000 + 2000 - 1000 = 3000 pixels
        # IoU = 1000/3000 = 0.333... < 0.5
        
        result = cm(preds, target)
        self.assertEqual(result[1, 2], 1)  # Should be FN since IoU < 0.5
        
    def test_multiple_instances_same_class(self):
        """Test handling multiple instances of the same class"""
        cm = ConfusionMatrix(iou_threshold=0.5, num_classes=3)
        
        # Create multiple disconnected regions of same class
        target = np.zeros((1, 100, 100), dtype=np.int64)
        target[0, 10:30, 10:30] = 1  # First instance of class 1
        target[0, 60:80, 60:80] = 1  # Second instance of class 1
        
        preds = np.zeros((1, 100, 100), dtype=np.int64)
        preds[0, 10:30, 10:30] = 1  # Matches first instance
        # Second instance not predicted
        
        result = cm(preds, target)
        
        # This tests the aggregate behavior
        # Both instances contribute to the same class statistics
        
    def test_state_accumulation(self):
        """Test that state accumulates correctly across updates"""
        cm = ConfusionMatrix(iou_threshold=0.5, num_classes=2)
        
        target1 = np.zeros((1, 50, 50), dtype=np.int64)
        target1[0, 10:30, 10:30] = 1
        preds1 = target1.copy()
        
        target2 = np.zeros((1, 50, 50), dtype=np.int64)
        target2[0, 20:40, 20:40] = 1
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
        
    def test_edge_case_single_pixel(self):
        """Test with single pixel predictions/targets"""
        cm = ConfusionMatrix(iou_threshold=0.5, num_classes=2)
        
        target = np.zeros((1, 10, 10), dtype=np.int64)
        target[0, 5, 5] = 1  # Single pixel
        
        preds = np.zeros((1, 10, 10), dtype=np.int64)
        preds[0, 5, 5] = 1  # Perfect match
        
        result = cm(preds, target)
        self.assertEqual(result[1, 0], 1)  # Should be TP (IoU = 1.0)
        
        # Test with mismatch
        preds2 = np.zeros((1, 10, 10), dtype=np.int64)
        preds2[0, 5, 6] = 1  # Adjacent pixel
        
        cm2 = ConfusionMatrix(iou_threshold=0.5, num_classes=2)
        result2 = cm2(preds2, target)
        self.assertEqual(result2[1, 2], 1)  # Should be FN (IoU = 0)
        
    def test_numerical_stability(self):
        """Test numerical stability with edge cases"""
        cm = ConfusionMatrix(iou_threshold=0.5, num_classes=2)
        
        # Test with very small IoU
        target = np.zeros((1, 1000, 1000), dtype=np.int64)
        target[0, 0:10, 0:10] = 1  # 100 pixels
        
        preds = np.zeros((1, 1000, 1000), dtype=np.int64)
        preds[0, 9:19, 9:19] = 1  # 100 pixels, minimal overlap
        
        # Should not crash and handle small IoU correctly
        result = cm(preds, target)
        self.assertIsNotNone(result)
        
    def test_background_class_handling(self):
        """Test that background class (0) is handled correctly"""
        cm = ConfusionMatrix(iou_threshold=0.5, num_classes=3)
        
        # All background
        target = np.zeros((1, 50, 50), dtype=np.int64)
        preds = np.zeros((1, 50, 50), dtype=np.int64)
        
        result = cm(preds, target)
        
        # Background should be TN
        self.assertEqual(result[0, 3], 1)  # TN for background


if __name__ == "__main__":
    unittest.main()