import numpy as np

class Node:

    def __init__(self):
        self.freq  = 1.0
        self.phase = 0.0
        self.x     = 1.0
        self.y     = 0.0
        self.plot  = False

    def tick(self, i):
        self.y = self.x * np.sin(
            np.pi * 2 * self.freq * i + self.phase
        )

