import unittest

import numpy as np

from deep_circle_counter.dataset.generate_circles import generate_circles


class TestCircleGenerator(unittest.TestCase):
    def test_generate_circles(self):
        radiuses = [3, 5, 7]
        height, width = 40, 60

        image = np.zeros((height, width))

        circles = generate_circles(height, width, radiuses)
        yy, xx = np.meshgrid(range(height), range(width))

        for center_y, center_x, radius in circles:
            for y, x in zip(yy.flatten(), xx.flatten()):
                if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2:
                    image[y, x] += 1

        self.assertCountEqual(np.unique(image), [0.0, 1.0])
        self.assertEqual(np.sum(image), 259.0)
 