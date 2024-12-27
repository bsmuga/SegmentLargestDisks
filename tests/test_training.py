import unittest

import yaml

from deep_circle_counter.main import main


class TestTraining(unittest.TestCase):
    def test_main(self):
        with open("tests/unet.yml", "r") as f:
            config = yaml.safe_load(f)
        main(config)
