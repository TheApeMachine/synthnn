import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class SynthNN:

    def __init__(self):
        with open('./data/common-sense.txt') as f:
            content = f.readlines()

        self.neurons  = np.array([0.0, 1.0, 2.0])
        self.fig      = plt.figure()
        self.ax1      = self.fig.add_subplot(1, 1, 1)
        self.data     = [list(x.strip()) for x in content]
        self.data     = [item for sublist in self.data for item in sublist]

    def plot(self):
        ani = animation.FuncAnimation(self.fig, self.animate, interval=1)
        plt.show()

    def animate(self, i):
        frame  = ord(self.data[(i - 1) % len(self.data)]) / 100.0
        Fs     = 64
        f      = 1
        sample = 64
        x      = np.arange(sample)
        y      = np.sin(i + 2 * np.pi * frame * x / Fs)

        self.ax1.clear()
        self.ax1.plot(x, y)

        self.fig.suptitle('NO CURRENT THOUGHT', fontsize=10, fontweight='bold')
