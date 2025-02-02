import unittest

import torch

from src.confusion_matrix import ConfusionMatrix


class TestConfusionMatric(unittest.TestCase):
    def setUp(self):
        self.confusion_matrix = ConfusionMatrix(iou_threshold=0.25, num_classes=4)
        self.target = torch.zeros((1, 100, 100))
        self.target[0, :50, :50] = 0
        self.target[0, :50, 50:] = 1
        self.target[0, 50:, :50] = 2
        self.target[0, 50:, 50:] = 3

        self.preds = torch.clone(self.target)

    def test_perfect_score(self):
        result = self.confusion_matrix(self.target, self.target)
        assert torch.all(
            result
            == torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
        )

    def test_false_positive(self):
        self.target[0, 50:, 50:] = 0
        result = self.confusion_matrix(self.preds, self.target)
        assert torch.all(
            result
            == torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]])
        )

    def test_false_negative(self):
        self.preds[0, :, :50] = 0
        result = self.confusion_matrix(
            self.preds,
            self.target,
        )
        assert torch.all(
            result
            == torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]])
        )

    def test_true_negative(self):
        self.preds[0, 50:, 50:] = 0
        self.target[0, 50:, 50:] = 0
        result = self.confusion_matrix(self.preds, self.target)
        assert torch.all(
            result
            == torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        )

    def test_too_small_iou(self):
        self.preds[0, :50, :90] = 0
        self.preds[0, :90, :50] = 0
        result = self.confusion_matrix(self.preds, self.target)
        assert torch.all(
            result
            == torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0]])
        )

    def test_if_batch_works(self):
        self.confusion_matrix.update(self.preds, self.target)
        self.confusion_matrix.update(self.preds, self.target)
        result = self.confusion_matrix.compute()

        batched_result = self.confusion_matrix(
            torch.concat([self.preds, self.preds]),
            torch.concat([self.target, self.target]),
        )

        assert torch.all(
            result
            == torch.tensor([[2, 0, 0, 0], [2, 0, 0, 0], [2, 0, 0, 0], [2, 0, 0, 0]])
        )
        assert torch.all(result == batched_result)
