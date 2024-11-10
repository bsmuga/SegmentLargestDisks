import unittest

from deep_circle_counter.dataset import CircleDataset


class TestCircleGenerator(unittest.TestCase):
    def test_generate_circles(self):
        circles = CircleDataset.generate_circles((200, 300), 5)

        for i in range(len(circles) - 1):
            for j in range(i + 1, len(circles)):
                c_1 = circles[i]
                c_2 = circles[j]
                assert (c_1.x - c_2.x) ** 2 + (c_1.y - c_2.y) ** 2 > (
                    c_1.r + c_2.r
                ) ** 2

    def test_dataset(self):
        dataset = CircleDataset((200, 300), 10, 20, 1)
        image, segmentation = dataset[0]
        assert image.max() == 1.
        assert segmentation.max() == 20
