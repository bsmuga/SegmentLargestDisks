import unittest

from src.data.dataset import DisksDataset


class TestDisksGenerator(unittest.TestCase):
    def test_generate_disks(self):
        disks = DisksDataset.generate_disks((200, 300), 5)

        for i in range(len(disks) - 1):
            for j in range(i + 1, len(disks)):
                c_1 = disks[i]
                c_2 = disks[j]
                assert (c_1.x - c_2.x) ** 2 + (c_1.y - c_2.y) ** 2 > (
                    c_1.r + c_2.r
                ) ** 2

    def test_dataset(self):
        dataset = DisksDataset((200, 300), 10, 20, 1)
        image, segmentation = next(iter(dataset))
        assert image.max() == 1.0
        assert segmentation.max() == 20
