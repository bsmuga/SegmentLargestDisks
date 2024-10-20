import random
from dataclasses import dataclass

import numpy as np


@dataclass
class Circle:
    x: int
    y: int
    r: int


def generate_circles(size: tuple[int, int], points: int) -> list[Circle]:
    x_centers = sorted(random.choices(range(size[0]), k=points))
    y_centers = random.choices(range(size[1]), k=points)

    distances = np.concat(
        [
            np.full((points, points), np.inf),
            np.array([[x, y] for (x, y) in zip(x_centers, y_centers)]),
        ],
        axis=1,
    )

    radiuses = np.zeros(points, dtype=int)
    for i in range(points - 1):
        j = i + 1

        delta = x_centers[j] - x_centers[i]
        dist_ij = (
            (x_centers[j] - x_centers[i]) ** 2 + (y_centers[j] - y_centers[i]) ** 2
        ) ** 0.5
        distances[i, j] = distances[j, i] = dist_ij

        while (dist_ij <= delta) and (j < points):
            j += 1
            delta = x_centers[j] - x_centers[i]
            dist_ij = (
                (x_centers[j] - x_centers[i]) ** 2 + (y_centers[j] - y_centers[i]) ** 2
            ) ** 0.5
            distances[i, j] = distances[j, i] = dist_ij
        
        maximal_radius = int(np.min(distances[i, :]))
        if maximal_radius == 0:
            radiuses[i] = 0
        else:
            radiuses[i] = maximal_radius
            while j >= 0:
                distances[j, i] -= radiuses[i]
                j -= 1

    return [Circle(x, y, r) for (x, y, r) in zip(x_centers, y_centers, radiuses)]

if __name__ == "__main__":
    print(generate_circles((100, 100), 10))