from dataclasses import dataclass

import numpy as np
import torch
from sklearn.neighbors import KDTree
from torch.utils.data import Dataset


@dataclass
class Disk:
    x: int
    y: int
    r: int


class DisksDataset(Dataset):
    """Class that generate synthetic dataset
    with images with non overlapping disks.

    Parameters
    ----------
    image_size : tuple[int, int]
        size of the image
    disk_max_num : int
        maximal number of disks present in image
        not necessary maximum number is achieved
    labeled_disks : int
        Number of segmented disks, where the disks
        are segmented from largest to smallest
    items : int
        Number of images.
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        disk_max_num: int,
        labeled_disks: int,
        items: int,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.disk_max_num = disk_max_num
        self.labeled_disks = labeled_disks
        self.items = items

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        disks = self.generate_disks(self.image_size, self.disk_max_num)
        disks = sorted(disks, key=lambda disk: -disk.r)
        disc_num = len(disks)
        image = self.disks2img(self.image_size, disks, [1] * disc_num)
        labels = list(
            range(self.labeled_disks, max(-1, self.labeled_disks - disc_num - 1), -1)
        )
        segmentation = self.disks2img(self.image_size, disks, labels)
        return torch.from_numpy(image[None, ...]).to(torch.float), torch.from_numpy(
            segmentation
        ).to(torch.int64)

    def __len__(self) -> int:
        return self.items

    @staticmethod
    def generate_disks(size: tuple[int, int], num_points: int) -> list[Disk]:
        rng = np.random.default_rng()
        centers = np.asarray(
            [
                [x, y]
                for (x, y) in zip(
                    sorted(rng.integers(0, size[0], num_points)),
                    rng.integers(0, size[1], num_points),
                )
            ]
        )

        lazy_pairwise_distances = np.full((num_points, num_points), np.nan)
        tree = KDTree(centers, leaf_size=2)
        for i in range(num_points - 1):
            delta_x = centers[i][0] - centers[i + 1][0]
            delta_y = centers[i][1] - centers[i + 1][1]

            indices, distances = tree.query_radius(
                [centers[i, :]],
                r=2 * ((delta_x) ** 2 + (delta_y) ** 2) ** 0.5,
                # factor 2 to eliminate disks collisions
                return_distance=True,
            )

            for index, distance in zip(indices[0], distances[0]):
                if i != index:
                    lazy_pairwise_distances[i, index] = distance
                    lazy_pairwise_distances[index, i] = distance

        disks = list()
        for i, center in enumerate(centers):
            r = np.nanmin(
                [
                    *lazy_pairwise_distances[i, :],
                    center[0],
                    center[1],
                    size[0] - center[0],
                    size[1] - center[1],
                ]
            )
            lazy_pairwise_distances[:, i] -= r
            r = np.floor(r)
            if r > 0:
                disks.append(Disk(center[0], center[1], np.floor(r)))
        return disks

    @staticmethod
    def disks2img(
        size: tuple[int, int], disks: list[Disk], per_disk_values: list[int]
    ) -> np.ndarray:
        image = np.zeros(size).T
        xx, yy = np.meshgrid(range(size[0]), range(size[1]), indexing="ij")

        for disk, value in zip(disks, per_disk_values):
            for x, y in zip(xx.flatten(), yy.flatten()):
                if (x - disk.x) ** 2 + (y - disk.y) ** 2 < disk.r**2:
                    image[y, x] += value
        return image
