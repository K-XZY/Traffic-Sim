import numpy as np
from PIL import Image, ImageChops
import io
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
import networkx as nx
from visualize import visualize_network
from road import Network
import os
import gc

def matrix_to_network(matrix):
    """
    Convert an adjacency matrix to a Network object.
    """
    n = len(matrix)
    network = Network()
    
    # Calculate node positions in a circle
    radius = 500
    center_x = 500
    center_y = 500
    
    # Add nodes
    for i in range(n):
        angle = 2 * np.pi * i / n
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        network.add_node(i+1, x, y)
    
    # Add edges based on matrix
    edge_id = 1
    for i in range(n):
        for j in range(n):
            if matrix[i][j] == 1:
                network.add_edge(edge_id, i+1, j+1)
                edge_id += 1
                
    return network

def save_network_plot(network, filename):
    """
    Save the network visualization to a file without displaying.
    """
    fig = plt.figure(figsize=(8, 8))
    visualize_network(network, rad=0.3)
    plt.savefig(filename, bbox_inches='tight', facecolor='white')
    plt.close('all')
    gc.collect()

def create_transition_frame(frame1, frame2, alpha=0.3):
    """
    Create a transition frame by blending two frames.
    The previous frame will appear as a dim trail.
    """
    # Convert frames to RGBA if they aren't already
    if frame1.mode != 'RGBA':
        frame1 = frame1.convert('RGBA')
    if frame2.mode != 'RGBA':
        frame2 = frame2.convert('RGBA')
    
    # Create a dimmed version of frame1
    dim_factor = int(255 * (1 - alpha))
    dim_frame = frame1.copy()
    dim_frame.putalpha(dim_factor)
    
    # Composite the frames
    composite = Image.alpha_composite(frame2, dim_frame)
    return composite

def generate_random_networks(n, base_output_dir='GIFS', num_samples=100):
    """
    Generate random network samples and create a GIF with transition effects.
    """
    # Create output directory
    os.makedirs(base_output_dir, exist_ok=True)
    temp_dir = os.path.join(base_output_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"Generating {num_samples} random {n}x{n} networks...")
    
    try:
        # Generate random networks
        frames = []
        prev_frame = None
        
        for i in range(num_samples):
            # Generate random adjacency matrix
            matrix = np.random.randint(0, 2, size=(n, n))
            
            # Convert to network and save visualization
            network = matrix_to_network(matrix)
            frame_filename = os.path.join(temp_dir, f'frame_{i:05d}.png')
            save_network_plot(network, frame_filename)
            
            if i % 10 == 0:
                print(f"Generated {i}/{num_samples} networks...")
            
            # Load the frame
            current_frame = Image.open(frame_filename).convert('RGBA')
            
            if prev_frame is not None:
                # Create transition frame
                transition_frame = create_transition_frame(prev_frame, current_frame)
                frames.append(transition_frame)
            
            frames.append(current_frame)
            prev_frame = current_frame.copy()
        
        # Save as GIF
        output_filename = os.path.join(base_output_dir, f'n{n}_graph.gif')
        print(f"Creating GIF: {output_filename}")
        
        # Save with transition effects
        frames[0].save(
            output_filename,
            save_all=True,
            append_images=frames[1:],
            duration=200,  # 200ms per frame
            loop=0
        )
        
    finally:
        # Cleanup
        print("Cleaning up temporary files...")
        for filename in os.listdir(temp_dir):
            try:
                os.remove(os.path.join(temp_dir, filename))
            except OSError as e:
                print(f"Error removing file: {e}")
        
        try:
            os.rmdir(temp_dir)
        except OSError as e:
            print(f"Error removing temporary directory: {e}")
        
        plt.close('all')
        gc.collect()
    
    print(f"GIF generation complete: {output_filename}")
    return output_filename

def main():
    # Generate GIFs for different n values
    for n in range(6, 8):  # Generate for n=2 to n=5
        gif_path = generate_random_networks(n)
        print(f"Generated GIF for n={n}: {gif_path}")

if __name__ == "__main__":
    main()
