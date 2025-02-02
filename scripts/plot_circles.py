import matplotlib.pyplot as plt
import numpy as np

from src.data.dataset import DisksDataset


def plot_disks(size: tuple[int, int], num_points: int) -> plt.Figure:
    disks = DisksDataset.generate_disks(size, num_points)
    xx, yy = np.meshgrid(range(size[0]), range(size[1]), indexing="ij")

    image = np.zeros(size).T
    for disk in disks:
        for x, y in zip(xx.flatten(), yy.flatten()):
            if (x - disk.x) ** 2 + (y - disk.y) ** 2 < disk.r**2:
                image[y, x] += 1

    fig, ax = plt.subplots()
    cax = ax.imshow(image)
    fig.colorbar(cax)
    return fig


if __name__ == "__main__":
    fig = plot_disks((300, 200), 5)
    plt.show(block=True)
