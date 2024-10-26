import unittest

import numpy as np

from deep_circle_counter.dataset.generate_circles import generate_circles


class TestCircleGenerator(unittest.TestCase):
    def test_generate_circles(self):
        size = (200, 300)
        points = 5

        image = np.zeros((size[0], size[1]))
        circles = generate_circles(size, points)

        for i in range(points - 1):
            for j in range(i + 1, points):
                c_1 = circles[i]
                c_2 = circles[j]
                assert (c_1.x - c_2.x) ** 2 + (c_1.y - c_2.y) ** 2 > (c_1.r + c_2.r)**2

        self.assertCountEqual(np.unique(image), [0.0, 1.0])
