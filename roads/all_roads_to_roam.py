
import numpy as np
import matplotlib.pyplot as plt
from sandbox import Sandbox
from road import Network
from serializer import SandboxSerializer
import logging

from visualize import visualize_network 

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_and_save_circle():
    """Generate network and sandbox, then save to file."""
    logger.info("Generating network and sandbox...")
    
    # Create a network
    network = Network()
    # Create nodes in a square pattern
    network.add_node(1, 400, 400)   # Bottom-left
    network.add_node(2, 400, 1600)  # Top-left
    network.add_node(3, 1600, 1600) # Top-right
    network.add_node(4, 1600, 400)  # Bottom-right
    
    # Connect nodes in a circle
    network.add_edge(1, 1, 2)
    network.add_edge(2, 2, 3)
    network.add_edge(3, 3, 4)
    network.add_edge(4, 4, 1)
    
    visualize_network(network)
    # Create sandbox and energy bump
    sandbox = Sandbox(2000, 2000, alpha=0.3, beta=0.2)
    
    # Add an energy zone at the center
    sandbox.create_energy_zone(1000, 1000, 2000, 1800)
    
    # Add network and generate paths
    sandbox.add_network(network)
    sandbox.visualize_energy()
    # Generate path_map
    sandbox.update_path_map()
    
    # Plot the path map
    logger.info("Plotting path map...")
    plt.figure(figsize=(10, 8))
    plt.imshow(sandbox.path_map.T, origin='lower', cmap='hot')
    plt.colorbar(label='Road Presence')
    plt.title('Road Path Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    # Save the sandbox
    logger.info("Saving sandbox to file...")
    SandboxSerializer.save_sandbox(sandbox, "circle_data2.json")
    logger.info("Save completed!")

def generate_and_save_turns():
    """Generate network and sandbox, then save to file."""
    logger.info("Generating network and sandbox...")
    
    # Create a network
    network = Network()
    # Create nodes in a square pattern
    network.add_node(1, 500, 500)   # Bottom-left
    network.add_node(2, 500, 1500)  # Top-left
    network.add_node(3, 1500, 1500) # Top-right
    network.add_node(4, 1500, 500)  # Bottom-right
    
    # Connect nodes in a circle
    network.add_edge(1, 1, 2)
    network.add_edge(2, 2, 3)
    network.add_edge(3, 3, 4)
    network.add_edge(4, 4, 1)
    
    visualize_network(network)

    # Create sandbox and energy bump
    sandbox = Sandbox(2000, 2000, alpha=0.3, beta=0.2)
    
    # Add an energy zone at the center
    sandbox.create_energy_zone(1300, 1300, 2000, 300)
    sandbox.create_energy_zone(800, 800, 2000, 500)
    sandbox.create_energy_zone(800, 1300, 2000, 400)
    sandbox.create_energy_zone(1300, 800, 2000, 400)
    # Add network and generate paths
    sandbox.add_network(network)
    sandbox.visualize_energy()
    # Generate path_map
    sandbox.update_path_map()
    
    # Plot the path map
    logger.info("Plotting path map...")
    plt.figure(figsize=(10, 8))
    plt.imshow(sandbox.path_map.T, origin='lower', cmap='hot')
    plt.colorbar(label='Road Presence')
    plt.title('Road Path Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    # Save the sandbox
    logger.info("Saving sandbox to file...")
    SandboxSerializer.save_sandbox(sandbox, "turns_data.json")
    logger.info("Save completed!")


def generate_and_save_butterfly():
    """Generate network and sandbox, then save to file."""
    logger.info("Generating network and sandbox...")
    
    # Create a network
    network = Network()
    # Create nodes in a square pattern
    network.add_node(1, 1000, 300)   # Bottom-left
    network.add_node(2, 300, 1700)  # Top-left
    network.add_node(3, 1700, 1700) # Top-right
    network.add_node(4, 1000, 1500)  # Bottom-right
    
    # Connect nodes in a circle
    network.add_edge(1, 2, 1)
    network.add_edge(2, 3, 1)
    network.add_edge(3, 1, 4)
    network.add_edge(4, 4, 2)
    network.add_edge(5, 4, 3)
    visualize_network(network)
    
    # Create sandbox and energy bump
    sandbox = Sandbox(2000, 2000, alpha=0.3, beta=0.2)
    
    # Add an energy zone at the center
    sandbox.create_energy_zone(800, 1000, 2000, 600)
    sandbox.create_energy_zone(1200, 1000, 2000, 600)
    
    # Add network and generate paths
    sandbox.add_network(network)
    sandbox.visualize_energy()
    # Generate path_map
    sandbox.update_path_map()
    
    # Plot the path map
    logger.info("Plotting path map...")
    plt.figure(figsize=(10, 8))
    plt.imshow(sandbox.path_map.T, origin='lower', cmap='hot')
    plt.colorbar(label='Road Presence')
    plt.title('Road Path Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    # Save the sandbox
    logger.info("Saving sandbox to file...")
    SandboxSerializer.save_sandbox(sandbox, "butterfly_data.json")
    logger.info("Save completed!")

def load_and_visualize():
    """Load sandbox from file and show visualizations."""
    logger.info("Loading sandbox from file...")
    
    # Load the sandbox
    sandbox = SandboxSerializer.load_sandbox("network_data.json", Sandbox)
    logger.info("Load completed!")
    
    # Plot the path map
    logger.info("Plotting path map...")
    plt.figure(figsize=(10, 8))
    plt.imshow(sandbox.path_map.T, origin='lower', cmap='hot')
    plt.colorbar(label='Road Presence')
    plt.title('Road Path Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    
    # Show the 3D energy visualization
    logger.info("Showing energy visualization...")

def generate_and_save_network():
    """Generate network and sandbox, then save to file."""
    logger.info("Generating network and sandbox...")
    
    # Create a network
    network = Network()
    # Create nodes in a square pattern
    # create a grid from 400, 1200, 1600 
    for i in range(3):
        for j in range(3):
            network.add_node(i*3+j+1,i*400+400, j*400+400)

    # connect them 
    network.add_edge(1, 1,2)
    network.add_edge(2, 2,3)
    network.add_edge(3, 3,6)
    network.add_edge(4, 5,2)
    network.add_edge(5, 4,1)
    network.add_edge(6, 4,5)
    network.add_edge(7, 5,6)
    network.add_edge(8, 7,4)
    network.add_edge(9, 5,8)
    network.add_edge(10, 6,9)
    network.add_edge(11, 9,8)
    network.add_edge(12, 8,7)
    # Create sandbox and energy bump
    sandbox = Sandbox(2000, 2000, alpha=0.3, beta=0.2)

    
    # Add an energy zone at the center
    sandbox.create_energy_zone(1000,1000, 1000, 800)
    
    # Add network and generate paths
    sandbox.add_network(network)
    sandbox.visualize_energy()
    # Generate path_map
    sandbox.update_path_map()
    
    # Plot the path map
    logger.info("Plotting path map...")
    plt.figure(figsize=(10, 8))
    plt.imshow(sandbox.path_map.T, origin='lower', cmap='hot')
    plt.colorbar(label='Road Presence')
    plt.title('Road Path Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    # Save the sandbox
    logger.info("Saving sandbox to file...")
    SandboxSerializer.save_sandbox(sandbox, "turns_data.json")
    logger.info("Save completed!")

if __name__ == "__main__":
    # To generate and save new data, uncomment the next line:
    generate_and_save_butterfly()
    # generate_and_save_turns()
    # generate_and_save_circle()
    # generate_and_save_network()
    
