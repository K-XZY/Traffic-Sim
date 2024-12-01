import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx  # For graph algorithms and layouts
import numpy as np
from visualize import visualize_network, node_coloring, edge_coloring
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import ScalarMappable
from serializer import SandboxSerializer
from sandbox import Sandbox

class Node:
    def __init__(self, id, x=0.0, y=0.0):
        self.id = id
        self.x = float(x) # in meters
        self.y = float(y)
        self.incoming_edges = []
        self.outgoing_edges = []

class Edge:
    def __init__(self, id, start_node, end_node, width=10.0):
        self.id = id
        self.start_node = start_node
        self.end_node = end_node
        self.width = float(width)
        self.cars = [] # a car consumes 10 pixels
        # Add this edge to the nodes' edge lists
        start_node.outgoing_edges.append(self)
        end_node.incoming_edges.append(self)

class Car:
    def __init__(self, id, edge, position=0.0):
        self.id = id
        self.edge = edge
        self.position = float(position)
        edge.cars.append(self)

class Network:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, id, x=0.0, y=0.0):
        if id in self.nodes:
            raise ValueError(f"Node ID {id} already exists.")
        node = Node(id, x, y)
        self.nodes[id] = node
        return node

    def add_edge(self, id, start_node_id, end_node_id, length=1.0):
        if id in self.edges:
            raise ValueError(f"Edge ID {id} already exists.")
        start_node = self.nodes[start_node_id]
        end_node = self.nodes[end_node_id]
        # Ensure no duplicate edges
        if self.edge_exists(start_node_id, end_node_id):
            return None
        edge = Edge(id, start_node, end_node, length)
        self.edges[id] = edge
        return edge

    def edge_exists(self, start_node_id, end_node_id):
        return any(
            edge.start_node.id == start_node_id and edge.end_node.id == end_node_id
            for edge in self.edges.values()
        )


def circle():
    network = Network()
    network.add_node(1,500,500)
    network.add_node(2,500,1000)
    network.add_node(3,1000,1000)
    network.add_node(4,1000,500)

    network.add_edge(1, 1, 2)
    network.add_edge(2, 2, 3)
    network.add_edge(3,3,4)
    network.add_edge(4,4,1)

    # visualize_network(network)
    return network

def test_energy_create():

# Create sandbox instance
    sandbox = Sandbox(2500, 2500, alpha = 0.5, beta = 0.2)

# Add energy zones
    sandbox.create_energy_zone(1500, 1050, 1000, 900)
    sandbox.create_energy_zone(600, 850, 1000, 900)
# Create a network (using your existing Network class)
    network = Network()
    network.add_node(1, 500, 500)
    network.add_node(2, 800,1000)
    network.add_node(3,1700,1800)

    network.add_edge(1,1,2)
    network.add_edge(2,1,3)
    network.add_edge(3,2,3)
    network.add_edge(4,3,1)

# Add the network to the sandbox
    sandbox.add_network(network)
    SandboxSerializer.save_sandbox(sandbox, "my_sandbox_1.json")


def test_energy_create2():

# Create sandbox instance
    sandbox = Sandbox(2500, 2500, alpha = 0.5, beta = 0.2)

    sandbox.create_energy_zone(1500, 1050, 0.1, 10)
# Create a network (using your existing Network class)
    network = Network()
    network.add_node(1, 500, 500)
    network.add_node(2, 500, 1000)
    network.add_node(3,1000, 500)
    network.add_node(4,1000, 1000)

    network.add_edge(1,1,2)
    network.add_edge(2,1,3)
    network.add_edge(3,2,4)
    network.add_edge(4,3,4)


# Add the network to the sandbox
    sandbox.add_network(network)
    SandboxSerializer.save_sandbox(sandbox, "my_sandbox_2.json")

def test_energy_load():
    sandbox = SandboxSerializer.load_sandbox("my_sandbox_2.json", Sandbox)
    sandbox.visualize_energy()
    sandbox.plot_heatmap()

# Example usage
if __name__ == "__main__":
    #test_energy_create2()
    test_energy_load()
