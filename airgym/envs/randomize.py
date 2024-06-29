import random

import numpy as np


class Random(object):
    def __init__(self, seed=None):
        self.seed = seed if seed is not None else random.randrange(1, 10000)
        random.seed(self.seed)
        self.mode = 'Drone'

    def random_position(self):
        x = random.randrange(-10, 10)
        y = random.randrange(-10, 10)
        z = random.randrange(-40, 5)
        self.position = np.array([x, y, z])
        return self.position

    def random_destination(self, position):
        distance = 10000
        while distance < 30 or distance > 80:
            x = random.randrange(
                position[0]-40, position[0]+40
            )
            y = random.randrange(
                position[1]-40, position[1]+40
            )
            z = random.randrange(
                position[2]-10, position[2]+30)
            self.destination = np.array([x, y, z])
            distance = np.linalg.norm((position - self.destination))
            print(distance)
            print(self.destination)
        return self.destination
