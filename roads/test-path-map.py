import numpy as np
import matplotlib.pyplot as plt
from sandbox import Sandbox
from road import Network
from serializer import SandboxSerializer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_and_save():
    """Generate network and sandbox, then save to file."""
    logger.info("Generating network and sandbox...")
    
    # Create a network
    network = Network()
    # Create nodes in a square pattern
    network.add_node(1, 800, 800)   # Bottom-left
    network.add_node(2, 800, 1200)  # Top-left
    network.add_node(3, 1200, 1200) # Top-right
    network.add_node(4, 1200, 800)  # Bottom-right
    
    # Connect nodes in a circle
    network.add_edge(1, 1, 2)
    network.add_edge(2, 2, 3)
    network.add_edge(3, 3, 4)
    network.add_edge(4, 4, 1)
    
    # Create sandbox and energy bump
    sandbox = Sandbox(2000, 2000, alpha=0.3, beta=0.2)
    
    # Add an energy zone at the center
    sandbox.create_energy_zone(1000, 1000, 2000, 400)
    
    # Add network and generate paths
    sandbox.add_network(network)
    
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
    SandboxSerializer.save_sandbox(sandbox, "network_data.json")
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
    sandbox.visualize_energy()

if __name__ == "__main__":
    # To generate and save new data, uncomment the next line:
    generate_and_save()
    
    # Load and show visualizations
    load_and_visualize()
