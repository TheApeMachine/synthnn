from node import Node

class Layer:

    def __init__(self, size):
        self.nodes = [Node() for _ in range(size)]

    def feed(self, prev_layer):
        for pnode in prev_layer.nodes:
            for node in self.nodes:
                node.x = node.x * pnode.x

    def tick(self, i):
        for node in self.nodes:
            node.tick(i)
