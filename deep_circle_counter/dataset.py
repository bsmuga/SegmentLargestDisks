from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from sklearn.neighbors import KDTree
from torch.utils.data import IterableDataset


@dataclass
class Circle:
    x: int
    y: int
    r: int


class CircleDataset(IterableDataset):
    def __init__(
        self,
        image_size: tuple[int, int],
        circles_max_num: int,
        labels: int,
        items: int,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.max_circles = circles_max_num
        self.labels = labels
        self.items = items

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for _ in range(self.items):
            circles = self.generate_circles(self.image_size, self.max_circles)
            circles = sorted(circles, key=lambda circle: -circle.r)

            num_all_circles = len(circles)
            image = self.circles2img(self.image_size, circles, [1] * num_all_circles)

            labels = list(
                range(self.labels, max(-1, self.labels - num_all_circles - 1), -1)
            )
            segmentation = self.circles2img(self.image_size, circles, labels)

            yield torch.from_numpy(image).to(torch.float), torch.from_numpy(
                segmentation
            )

    @staticmethod
    def generate_circles(size: tuple[int, int], num_points: int) -> list[Circle]:
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
                r=((delta_x) ** 2 + (delta_y) ** 2) ** 0.5,
                return_distance=True,
            )

            for index, distance in zip(indices[0], distances[0]):
                if i != index:
                    lazy_pairwise_distances[i, index] = distance
                    lazy_pairwise_distances[index, i] = distance

        circles = list()
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
                circles.append(Circle(center[0], center[1], np.floor(r)))
        return circles

    @staticmethod
    def circles2img(
        size: tuple[int, int], circles: list[Circle], per_circle_values: list[int]
    ) -> np.ndarray:
        image = np.zeros(size).T
        xx, yy = np.meshgrid(range(size[0]), range(size[1]), indexing="ij")

        for circle, value in zip(circles, per_circle_values):
            for x, y in zip(xx.flatten(), yy.flatten()):
                if (x - circle.x) ** 2 + (y - circle.y) ** 2 < circle.r**2:
                    image[y, x] += value
        return image
