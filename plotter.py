import matplotlib.pyplot as plt

class Plotter:

    def __init__(self):
        self.fig  = plt.figure()
        self.axes = []
        self.cidx = 0

    def add_axis(self, node):
        self.cidx += 1
        self.axes.append({
            "plot": self.fig.add_subplot(
                1, 1, 1
            ),
            "node": node
        })

    def show(self):
        plt.show()

    def tick(self, outputs):
        for ax in self.axes:
            # ax["plot"].clear()
            ax["plot"].plot(ax["node"].x, ax["node"].y)

        self.fig.suptitle(
            outputs,
            fontsize=10,
            fontweight='bold'
        )

class Axis:

    def __init__(self, plotter, node):
        self.plotter = plotter
        self.node    = node

    def plot(self):
        self.plotter.add_axis(self.node)

