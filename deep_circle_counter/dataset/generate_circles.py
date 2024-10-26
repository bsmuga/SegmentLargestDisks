import random
from dataclasses import dataclass

import numpy as np

@dataclass
class Point:
    x: int
    y: int

@dataclass
class Circle:
    x: int
    y: int
    r: int


def generate_circles(size: tuple[int, int], num_points: int) -> list[Circle]:
    centers_x = random.choices(range(size[0]), k=num_points)
    centers_y = random.choices(range(size[1]), k=num_points)

    points = [Point(x, y) for (x, y) in zip(centers_x, centers_y)]
    points = sorted(points, key=lambda point: (point.x, point.y))

    borders_x = [np.min([point.x, size[0] - point.x]) for point in points]
    borders_y = [np.min([point.y, size[1] - point.y]) for point in points]

    distances = np.concat(
        [
            np.full((num_points, num_points), np.nan),
            np.array([[x, y] for (x, y) in zip(borders_x, borders_y)]),
        ],
        axis=1,
    )

    radiuses = np.zeros(num_points, dtype=int)
    for i in range(num_points - 1):
        j = i + 1

        delta_x = points[j].x - points[i].x
        delta_y = points[j].y - points[i].y

        distance = ((delta_x) ** 2 + (delta_y) ** 2) ** 0.5
        distances[i, j] = distances[j, i] = distance

        delta_next = points[j].x - points[j-1].x
        while (distance < delta_x or delta_next == 0) and (j < num_points):
            j += 1
            delta_x = points[j].x - points[i].x
            delta_y = points[j].y - points[i].y
            
            distance = ((delta_x) ** 2 + (delta_y) ** 2) ** 0.5
            distances[i, j] = distances[j, i] = distance
            delta_next = points[j].x - points[j-1].x


        maximal_radius = int(np.nanmin(distances[i, :]))

        radiuses[i] = maximal_radius
        for k in range(j, i, -1):
            distances[k, i] -= radiuses[i]

    return [Circle(point.x, point.y, r) for (point, r) in zip(points, radiuses)]
