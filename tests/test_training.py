import unittest

import torch
import yaml


from deep_circle_counter.main import main


class TestTraining(unittest.TestCase):

    @unittest.skipIf(torch.cuda.is_available() is False, "No cuda device to test")
    def test_main(self):
        with open("tests/unet.yml", "r") as f:
            config = yaml.safe_load(f)
        main(config)
