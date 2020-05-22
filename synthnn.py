import numpy as np
import matplotlib.animation as animation
from plotter import Plotter, Axis
from layer import Layer
import random

class Data:

    def __init__(self):
        # We are going to train numbers to letters,
        # so we can just use the index of the array here
        # as the key of the label.
        self.labels = [
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'
        ]

        # Give me a bunch of random numbers between 0 and 9.
        self.train = [random.randint(0, 9) for _ in range(1000)]
        self.test  = [random.randint(0, 9) for _ in range(100)]

class SynthNN:

    def __init__(self):
        # Get the data.
        self.data = Data()

        # Setup a plotter we can use to visualize output.
        self.plotter = Plotter()

        # Make sure you scale in some way from the size of the
        # input vector, to the size of the label vector.
        self.layers = [
            Layer(1),
            Layer(5),
            Layer(10)
        ]

        # Add a couple of axes on the plotter so we can debug
        # individual node states.
        #for layer in self.layers:
        #    for node in layer.nodes:
        #        ax = Axis(self.plotter, node)
        #        ax.plot()

        ax = Axis(self.plotter, self.layers[0].nodes[0])
        ax.plot()

    def run(self):
        ani = animation.FuncAnimation(
            self.plotter.fig, self.animate, interval=1
        )

        self.plotter.show()

    def animate(self, i):
        if len(self.data.train) > 0:
            train = self.data.train.pop()

            for lidx in range(len(self.layers)):
                if lidx == 0:
                    self.layers[lidx].nodes[0].x = train
                else:
                    self.layers[lidx].feed(self.layers[lidx - 1])

                self.layers[lidx].tick(i)

            val   = 0
            cnode = 0
            count = 0

            for node in self.layers[-1].nodes:
                print(node.y)
                if node.y > val:
                    cnode = count
                    val   = node.y

                count += 1

            if train != cnode:
                # Error in prediction.
                pass

            self.plotter.tick(
                [train, self.data.labels[cnode]]
            )

        #if i > 0:
            #quit()
