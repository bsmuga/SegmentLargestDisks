import matplotlib.pyplot as plt
import numpy as np

from deep_circle_counter.dataset.generate_circles import generate_circles


def plot_circles(size: tuple[int, int], max_number: int) -> plt.Figure:
    circles = generate_circles(size, max_number)
    xx, yy = np.meshgrid(range(size[0]), range(size[1]), indexing="ij")

    image = np.zeros(size).T
    for circle in circles:
        for x, y in zip(xx.flatten(), yy.flatten()):
            if (x - circle.x) ** 2 + (y - circle.y) ** 2 < circle.r**2:
                image[y, x] += 1

    fig, ax = plt.subplots()
    cax = ax.imshow(image)
    fig.colorbar(cax)
    return fig


if __name__ == "__main__":
    fig = plot_circles((300, 300), 10)
    plt.show(block=True)
