import unittest

import torch

from deep_circle_counter.segmentation_module import SegmentationModule


class TestLoops(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        classes = 3
        cls.segmentation_module = SegmentationModule(
            arch="unet",
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=1,
            classes=classes,
        )
        cls.batch = (
            torch.randint(0, classes, (3, 1, 32, 32), dtype=torch.float),
            torch.randint(0, classes, (3, 32, 32)),
        )
        cls.batch_num = 0

    def test_training_loop(self):
        self.segmentation_module.training_step(self.batch, self.batch_num)

    def test_validation_loop(self):
        self.segmentation_module.validation_step(self.batch, self.batch_num)

    def test_testing_loop(self):
        self.segmentation_module.test_step(self.batch, self.batch_num)
