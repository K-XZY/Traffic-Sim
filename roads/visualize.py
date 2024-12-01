import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx  # For graph algorithms and layouts
import numpy as np

def visualize_network(network,rad=0.5):
    """
    Visualizes the network with nodes plotted according to their x, y coordinates,
    scaling them proportionally to fit the entire diagram with padding.

    Args:
        network (Network): The network to visualize.

    Returns:
        None
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx

    # Create a directed graph using NetworkX for easier visualization
    G = nx.DiGraph()

    # Add nodes and edges to the NetworkX graph
    for node_id, node in network.nodes.items():
        G.add_node(node_id, pos=(node.x, node.y))

    for edge_id, edge in network.edges.items():
        G.add_edge(edge.start_node.id, edge.end_node.id, id=edge_id)

    # Get positions from nodes
    pos = {node_id: (node.x, node.y) for node_id, node in network.nodes.items()}

    # Node coloring using a greedy algorithm
    node_colors = node_coloring(G)

    # Edge coloring using a greedy algorithm
    edge_colors = edge_coloring(G)

    # Create a 1080p plot
    plt.figure(figsize=(19.2, 10.8), dpi=100)

    # Get min and max of x and y for scaling and padding
    x_values = [position[0] for position in pos.values()]
    y_values = [position[1] for position in pos.values()]
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)
    x_range = max_x - min_x if max_x > min_x else 1
    y_range = max_y - min_y if max_y > min_y else 1
    padding = 0.5  # 5% padding on each side

    # Adjust plot limits with padding
    plt.xlim(min_x - x_range * padding, max_x + x_range * padding)
    plt.ylim(min_y - y_range * padding, max_y + y_range * padding)

    # For maintaining aspect ratio
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    # Draw nodes
    node_color_list = [node_colors[node_id] for node_id in G.nodes()]
    nx.draw_networkx_nodes(
        G, pos, node_size=200, node_color=node_color_list, cmap=plt.cm.tab20, ax=ax
    )

    # Draw edges with curvature and arrows
    for (u, v, data) in G.edges(data=True):
        rad = rad  # Adjust for curvature
        arrowprops = dict(
            arrowstyle='-|>',
            mutation_scale=10,  # Control the size of the arrow
            color=edge_colors.get((u, v), 'black'),
            shrinkA=10,
            shrinkB=10
        )
        edge_path = mpatches.FancyArrowPatch(
            posA=pos[u],
            posB=pos[v],
            connectionstyle=f'arc3,rad={rad}',
            **arrowprops
        )
        ax.add_patch(edge_path)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)

    # Set axis off and adjust layout
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def node_coloring(G):
    """
    Performs greedy coloring of nodes.

    Args:
        G (networkx.Graph): The graph.

    Returns:
        dict: Node colors mapping.
    """
    colors = nx.coloring.greedy_color(G, strategy='largest_first')
    # Map color indices to actual colors
    color_map = {}
    unique_colors = list(set(colors.values()))
    colormap = plt.cm.get_cmap('tab20', len(unique_colors))
    for node, color_idx in colors.items():
        color_map[node] = 'red' #colormap.colors[color_idx]
    return color_map

def edge_coloring(G):
    """
    Performs greedy coloring of edges.

    Args:
        G (networkx.Graph): The graph.

    Returns:
        dict: Edge colors mapping.
    """
    colors = nx.coloring.greedy_color(nx.line_graph(G), strategy='largest_first')
    # Map edge tuples to colors
    color_map = {}
    unique_colors = list(set(colors.values()))
    colormap = plt.cm.get_cmap('tab20b', len(unique_colors))
    for edge, color_idx in colors.items():
        u, v = edge
        color_map[(u, v)] = 'black' #colormap.colors[color_idx]
    return color_map
