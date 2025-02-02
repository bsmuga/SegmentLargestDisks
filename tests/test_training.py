import unittest

import torch
import yaml

from src.main import main


class TestTraining(unittest.TestCase):

    @unittest.skipIf(torch.cuda.is_available() is False, "No cuda device to test")
    def test_main(self) -> None:
        with open("tests/unet.yml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        main(config)
