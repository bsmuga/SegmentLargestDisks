import unittest

import numpy as np

from deep_circle_counter.dataset.generate_circles import generate_circles


class TestCircleGenerator(unittest.TestCase):
    def test_generate_circles(self):
        size = (40, 60)
        points = 3

        image = np.zeros((size[0], size[1]))

        circles = generate_circles(size, points)
        yy, xx = np.meshgrid(range(size[0]), range(size[1]))

        for circle in circles:
            for y, x in zip(yy.flatten(), xx.flatten()):
                if (x - circle.x) ** 2 + (y - circle.y) ** 2 <= circle.r**2:
                    image[y, x] += 1

        self.assertCountEqual(np.unique(image), [0.0, 1.0]) 