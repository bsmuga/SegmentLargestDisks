from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import KDTree


@dataclass
class Disk:
    x: int
    y: int
    r: int


class DisksDataset:
    """Class that generate synthetic dataset
    with images with non overlapping disks.

    Parameters
    ----------
    image_size : tuple[int, int]
        size of the image
    disk_max_num : int
        maximal number of disks present in image
        not necessary maximum number has to be present
    labeled_disks : int
        Number of segmented disks, where the disks
        are segmented by label from largest to smallest
    items : int
        Number of images.
    seed : int, optional
        Random seed for reproducibility
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        disk_max_num: int,
        labeled_disks: int,
        items: int,
        seed: int = None,
    ) -> None:
        self.image_size = image_size
        self.disk_max_num = disk_max_num
        self.labeled_disks = labeled_disks
        self.items = items
        self.seed = seed

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        if self.seed is not None:
            rng = np.random.default_rng(self.seed + idx)
        else:
            rng = np.random.default_rng()

        disks = self.generate_disks(self.image_size, self.disk_max_num, rng)
        disks = sorted(disks, key=lambda disk: disk.r, reverse=True)
        disc_num = len(disks)
        image = self.disks2img(self.image_size, disks, [1] * disc_num)

        labels = []
        for i in range(disc_num):
            if i < self.labeled_disks:
                labels.append(i + 1)
            else:
                labels.append(0)

        segmentation = self.disks2img(self.image_size, disks, labels)
        return image, segmentation

    def __len__(self) -> int:
        return self.items

    @staticmethod
    def generate_disks(
        size: tuple[int, int], num_points: int, rng: np.random.Generator
    ) -> list[Disk]:
        centers = np.asarray(
            [
                [x, y]
                for (x, y) in zip(
                    rng.integers(0, size[0], num_points),
                    rng.integers(0, size[1], num_points),
                )
            ]
        )
        centers = np.unique(centers, axis=0)
        centers = centers[np.argsort(centers[:, 0])]

        disks_num = len(centers)
        lazy_pairwise_distances = np.full((disks_num, disks_num), np.nan)
        
        tree = KDTree(centers, leaf_size=2)
        for i in range(disks_num - 1):
            delta_x = centers[i][0] - centers[i + 1][0]
            delta_y = centers[i][1] - centers[i + 1][1]

            indices, distances = tree.query_radius(
                [centers[i, :]],
                r=((delta_x) ** 2 + (delta_y) ** 2) ** 0.5,
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
                disks.append(Disk(center[0], center[1], r))
        return disks

    @staticmethod
    def disks2img(
        size: tuple[int, int], disks: list[Disk], per_disk_values: list[int]
    ) -> np.ndarray:
        image = np.zeros((size[1], size[0]))

        y_grid, x_grid = np.ogrid[:size[1], :size[0]]

        for disk, value in zip(disks, per_disk_values):
            mask = (x_grid - disk.x) ** 2 + (y_grid - disk.y) ** 2 < disk.r**2
            image[mask] = value

        return image
