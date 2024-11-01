import unittest

import numpy as np

from deep_circle_counter.dataset.generate_circles import generate_circles


class TestCircleGenerator(unittest.TestCase):
    def test_generate_circles(self):
        circles = generate_circles((200, 300), 5)

        for i in range(len(circles) - 1):
            for j in range(i + 1, len(circles)):
                c_1 = circles[i]
                c_2 = circles[j]
                assert (c_1.x - c_2.x) ** 2 + (c_1.y - c_2.y) ** 2 > (
                    c_1.r + c_2.r
                ) ** 2
