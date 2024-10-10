from dataclasses import dataclass
import random

from sklearn.metrics.pairwise import pairwise_distances_argmin_min

@dataclass
class Circle:
    radius: int
    x_center: int
    y_center: int 

def generate_circle(size: tuple[int, int], number: int) -> list[Circle]:
    centers = set()
    while len(centers) == number:
        centers.add((random.randint(1, size[0]-1), random.randint(1, size[1]-1)))
    
    circles = list()
    for center in list(centers):
        X = [[center[0], 0], [0, center[1]]] + 
        radius = pairwise_distances_argmin_min()
        circles.append()