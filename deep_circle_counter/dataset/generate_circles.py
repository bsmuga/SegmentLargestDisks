from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import KDTree


@dataclass
class Circle:
    x: int
    y: int
    r: int


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
