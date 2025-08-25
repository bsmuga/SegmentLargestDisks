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
        """Initializes the DisksDataset.

        Args:
            image_size (tuple[int, int]): The size (width, height) of the images.
            disk_max_num (int): The maximum number of disks to attempt to place.
            labeled_disks (int): The number of largest disks to label in the segmentation.
            items (int): The total number of image-segmentation pairs in the dataset.
            seed (int, optional): A random seed for reproducibility. Defaults to None.
        """
        self.image_size = image_size
        self.disk_max_num = disk_max_num
        self.labeled_disks = labeled_disks
        self.items = items
        self.seed = seed

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Generates a single data sample, an image and its segmentation mask.

        Args:
            idx (int): The index of the item to generate.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                - The binary image with all generated disks drawn on it.
                - The segmentation mask, where the N largest disks are labeled
                  from 1 to N, and other disks are labeled 0.
        """
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
        """Returns the total number of items in the dataset.

        Returns:
            int: The total number of items specified during initialization.
        """
        return self.items

    @staticmethod
    def generate_disks(
        size: tuple[int, int], num_points: int, rng: np.random.Generator
    ) -> list[Disk]:
        """Generates a list of non-overlapping disks for a given image size.

        This method uses a greedy approach. It first generates a number of
        random candidate center points. It then iterates through these points,
        calculating the maximum possible radius for a disk at that center
        such that it does not collide with the image boundaries or other
        nearby disks. A random radius up to this maximum is chosen.

        Args:
            size (tuple[int, int]): The (width, height) of the canvas.
            num_points (int): The number of candidate center points to generate.
            rng (np.random.Generator): The random number generator to use.

        Returns:
            list[Disk]: A list of the generated Disk objects.
        """
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
        distances = DisksDataset._compute_distance_to_nearest_neighboours(centers)

        disks: list[Disk] = list()
        for i, center in enumerate(centers):
            r_max = np.nanmin(
                [
                    *distances[i, :],
                    center[0],
                    center[1],
                    size[0] - center[0],
                    size[1] - center[1],
                ]
            )
            if r_max > 1:
                r = np.random.randint(1, np.floor(r_max) + 1)
                disks.append(Disk(center[0], center[1], r))
        return disks

    @staticmethod
    def _compute_distance_to_nearest_neighboours(points: np.ndarray) -> np.ndarray:
        """Computes a sparse distance matrix for a set of points.

        Uses a KDTree to find all neighbors for each point within a dynamic
        radius. The radius is determined by the distance to the next point
        in the array, which assumes the points are sorted.

        Args:
            points (np.ndarray): A 2D array of shape (N, 2) containing the
                x, y coordinates of N points.

        Returns:
            np.ndarray: An (N, N) array where array[i, j] contains the
                Euclidean distance between point i and point j if they are
                close neighbors, and np.nan otherwise.
        """
        disks_num = len(points)
        distances = np.full((disks_num, disks_num), np.nan)
        
        tree = KDTree(points, leaf_size=2)
        for i in range(disks_num - 1):
            delta_x = points[i][0] - points[i + 1][0]
            delta_y = points[i][1] - points[i + 1][1]

            indices, distances_vec = tree.query_radius(
                [points[i, :]],
                r=((delta_x) ** 2 + (delta_y) ** 2) ** 0.5,
                return_distance=True,
            )

            for index, distance in zip(indices[0], distances_vec[0]):
                if i != index:
                    distances[i, index] = distance
                    distances[index, i] = distance
        return distances

    @staticmethod
    def disks2img(
        size: tuple[int, int], disks: list[Disk], per_disk_values: list[int]
    ) -> np.ndarray:
        """Renders a list of disks onto a 2D numpy array.

        Args:
            size (tuple[int, int]): The (width, height) of the image to create.
            disks (list[Disk]): The disks to draw on the image.
            per_disk_values (list[int]): The pixel value to use for each
                corresponding disk in the list.

        Returns:
            np.ndarray: A 2D array representing the image with disks drawn on it.
        """
        image = np.zeros((size[1], size[0]))

        y_grid, x_grid = np.ogrid[:size[1], :size[0]]

        for disk, value in zip(disks, per_disk_values):
            mask = (x_grid - disk.x) ** 2 + (y_grid - disk.y) ** 2 < disk.r**2
            image[mask] = value

        return image
