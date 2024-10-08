import random

import pandas as pd


def _distribute(num_items: int, num_bins: int) -> list[int]:
    distribution = [0] * num_bins
    for _ in range(num_items):
        distribution[random.randrange(num_bins)] += 1
    return distribution


def _generate_centers(radiuses: list[int], offsets: list[int]) -> list[int]:
    centers = [offsets[0] + radiuses[0]]
    for i in range(len(radiuses) - 1):
        centers.append(centers[-1] + radiuses[i] + offsets[i + 1] + radiuses[i + 1])
    return centers


def generate_circles(
    height: int, width: int, radiuses: list[int]
) -> list[tuple[int, int, int]]:
    diameter_sum = 2 * sum(radiuses)
    num_circles = len(radiuses)
    assert diameter_sum < min(height, width)

    radiuses = random.sample(radiuses, k=len(radiuses))

    offsets_x_distributed = _distribute(width - diameter_sum, num_circles + 1)[:-1]
    offsets_y_distributed = _distribute(height - diameter_sum, num_circles + 1)[:-1]

    centers_x = _generate_centers(radiuses, offsets_x_distributed)
    centers_y = _generate_centers(radiuses, offsets_y_distributed)

    return [circle for circle in zip(centers_y, centers_x, radiuses)]
