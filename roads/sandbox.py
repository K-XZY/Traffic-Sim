
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Point:
    """Represents a 2D point with x, y coordinates."""
    x: float
    y: float

    def to_tuple(self) -> Tuple[int, int]:
        """Convert point to integer tuple coordinates."""
        return (int(round(self.x)), int(round(self.y)))

class TerrainGenerator:
    """Handles the generation and modification of energy terrain."""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.terrain = np.zeros((width, height), dtype=float)
    

    def add_energy_zone(self, center: Point, height: float, radius: float) -> None:
        """Add a bell-shaped energy zone to the terrain."""
        x_coords, y_coords = np.meshgrid(
            np.linspace(0, self.width - 1, self.width),
            np.linspace(0, self.height - 1, self.height),
            indexing='ij'
        )
        
        distance = np.sqrt((x_coords - center.x) ** 2 + (y_coords - center.y) ** 2)
        energy_bump = height * np.exp(-distance**2 / (2 * (radius / 3) ** 2))
        energy_bump[distance > radius] = 0
        
        self.terrain += energy_bump

class GradientPathFinder:
    """PathFinder that finds paths considering both 3D distances and terrain heights."""
    
    def __init__(self, terrain: np.ndarray, num_points: int = 50, 
                 alpha: float = 0.3,  # Weight for terrain height cost
                 beta: float = 0.2):   # Weight for terrain gradient cost
        """
        Initialize the pathfinder with terrain data and cost weights.
        
        Args:
            terrain: 2D array of terrain heights
            num_points: Number of points in the path
            alpha: Weight for terrain height cost
            beta: Weight for terrain gradient cost
        """
        self.terrain = terrain
        self.width, self.height = terrain.shape
        self.num_points = num_points
        self.alpha = alpha
        self.beta = beta
        
        # Create interpolation function for smooth terrain
        x = np.arange(0, self.width)
        y = np.arange(0, self.height)
        self.terrain_interp = RectBivariateSpline(x, y, terrain)
        
        # Compute terrain gradients using numpy
        self.grad_y, self.grad_x = np.gradient(terrain)
    
    def _get_height(self, x: float, y: float) -> float:
        """Get interpolated height at any point."""
        return float(self.terrain_interp(x, y, grid=False))
    
    def _get_gradient(self, x: float, y: float) -> Tuple[float, float]:
        """Get terrain gradient at any point through bilinear interpolation."""
        x_int, y_int = int(x), int(y)
        x_frac, y_frac = x - x_int, y - y_int
        
        # Ensure we don't go out of bounds
        x_int = min(max(x_int, 0), self.width - 2)
        y_int = min(max(y_int, 0), self.height - 2)
        
        # Bilinear interpolation for both gradient components
        dx00 = self.grad_x[x_int, y_int]
        dx01 = self.grad_x[x_int, y_int + 1]
        dx10 = self.grad_x[x_int + 1, y_int]
        dx11 = self.grad_x[x_int + 1, y_int + 1]
        
        dy00 = self.grad_y[x_int, y_int]
        dy01 = self.grad_y[x_int, y_int + 1]
        dy10 = self.grad_y[x_int + 1, y_int]
        dy11 = self.grad_y[x_int + 1, y_int + 1]
        
        dx = (dx00 * (1 - x_frac) * (1 - y_frac) +
              dx10 * x_frac * (1 - y_frac) +
              dx01 * (1 - x_frac) * y_frac +
              dx11 * x_frac * y_frac)
        
        dy = (dy00 * (1 - x_frac) * (1 - y_frac) +
              dy10 * x_frac * (1 - y_frac) +
              dy01 * (1 - x_frac) * y_frac +
              dy11 * x_frac * y_frac)
        
        return dx, dy
    
    def _path_length(self, path_coords: np.ndarray) -> float:
        """Calculate the total 3D length of the path."""
        points = path_coords.reshape(-1, 2)
        heights = np.array([self._get_height(x, y) for x, y in points])
        points_3d = np.column_stack((points, heights))
        diffs = np.diff(points_3d, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        return np.sum(segment_lengths)
    
    def _terrain_cost(self, path_coords: np.ndarray) -> float:
        """Calculate cost based on terrain heights along path."""
        points = path_coords.reshape(-1, 2)
        # Get heights and normalize by terrain max height
        heights = np.array([self._get_height(x, y) for x, y in points])
        height_cost = np.mean(heights) / np.max(self.terrain)
        return height_cost
    
    def _gradient_cost(self, path_coords: np.ndarray) -> float:
        """Calculate cost based on terrain gradients along path."""
        points = path_coords.reshape(-1, 2)
        gradients = np.array([self._get_gradient(x, y) for x, y in points])
        gradient_magnitudes = np.sqrt(np.sum(gradients**2, axis=1))
        gradient_cost = np.mean(gradient_magnitudes)
        return gradient_cost
    
    def _total_cost(self, path_coords: np.ndarray) -> float:
        """Calculate total cost combining path length and terrain costs."""
        # Get the basic path length cost
        length_cost = self._path_length(path_coords)
        
        # Get terrain height cost
        height_cost = self._terrain_cost(path_coords)
        
        # Get terrain gradient cost
        gradient_cost = self._gradient_cost(path_coords)
        
        # Combine costs with weights
        total_cost = (length_cost + 
                     self.alpha * height_cost + 
                     self.beta * gradient_cost)
        
        return total_cost
    
    def find_path(self, start: Point, end: Point) -> List[Point]:
        """Find path between start and end points using gradient descent."""
        # Create initial straight-line path
        t = np.linspace(0, 1, self.num_points)
        initial_x = start.x + (end.x - start.x) * t
        initial_y = start.y + (end.y - start.y) * t
        initial_path = np.column_stack((initial_x, initial_y)).flatten()
        
        # Define constraints to keep points within bounds
        bounds = [(0, self.width-1) if i % 2 == 0 else (0, self.height-1) 
                 for i in range(len(initial_path))]
        
        # Fix start and end points
        def fix_endpoints(path):
            path[0] = start.x
            path[1] = start.y
            path[-2] = end.x
            path[-1] = end.y
            return path
        
        # Optimization function
        def objective(path_coords):
            fixed_path = fix_endpoints(path_coords.copy())
            return self._total_cost(fixed_path)
        
        # Perform optimization with increased iterations
        result = minimize(
            objective,
            initial_path,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 200,  # Increased iterations
                'ftol': 1e-6     # Tighter tolerance
            }
        )
        
        # Convert optimized path back to Points
        optimized_path = fix_endpoints(result.x)
        return [Point(float(x), float(y)) 
                for x, y in optimized_path.reshape(-1, 2)]


class TerrainVisualizer:
    """Handles all visualization operations."""
    
    def __init__(self, terrain: np.ndarray):
        self.terrain = terrain
        self.width, self.height = terrain.shape
        
    def plot_3d_surface(self, paths: List[List[Point]], nodes: List[Point],
                       opacity: float = 0.2, path_height: float = 1, 
                       node_height: float = 2) -> None:
        """Create 3D visualization of terrain with wireframe and paths and nodes."""
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid for wireframe plot
        x, y = np.meshgrid(
            np.arange(self.width),
            np.arange(self.height),
            indexing='ij'
        )
        
        # Plot terrain wireframe
        ax.plot_wireframe(
            x, y, self.terrain,
            color='gray',
            alpha=opacity,
            rstride=50,
            cstride=50
        )
        
        # Plot paths
        for path in paths:
            xs = [p.x for p in path]
            ys = [p.y for p in path]
            zs = [self.terrain[int(p.x), int(p.y)] + path_height for p in path]
            ax.plot(xs, ys, zs, color='blue', linewidth=2, alpha=0.8)
        
        # Plot nodes
        if nodes:
            for node in nodes:
                z = self.terrain[int(node.x), int(node.y)] + node_height
                ax.scatter(node.x, node.y, z, s=100, c='red', alpha=1.0)
        
        # Setup plot
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_zlim(0, np.max(self.terrain) + 5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Energy')
        
        # Add color bar
        mappable = plt.cm.ScalarMappable(cmap='viridis')
        mappable.set_array(self.terrain)
        plt.colorbar(mappable, ax=ax, label='Energy Level')
        
        plt.title('Energy Terrain Wireframe with Paths')
        plt.tight_layout()
        plt.show()
        
    def plot_heatmap(self, paths: List[List[Point]], nodes: List[Point]) -> None:
        """Create 2D heatmap visualization with correct coordinate alignment."""
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Plot heatmap with correct orientation
        heatmap = ax.imshow(
            self.terrain.T,
            origin='lower',
            extent=[0, self.width, 0, self.height],
            aspect='equal',
            cmap='hot'
        )
        
        # Plot paths
        for path in paths:
            xs = [p.x for p in path]
            ys = [p.y for p in path]
            ax.plot(xs, ys, color='blue', linewidth=2, alpha=0.8)
        
        # Plot nodes
        if nodes:
            node_xs = [node.x for node in nodes]
            node_ys = [node.y for node in nodes]
            ax.scatter(node_xs, node_ys, color='red', s=50)
        
        plt.colorbar(heatmap, label='Energy Level')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Energy Terrain Heatmap')
        
        plt.tight_layout()
        plt.show()

class NetworkAdapter:
    """Handles conversion between Network objects and Sandbox data structures."""
    
    @staticmethod
    def extract_nodes(network: Any) -> List[Point]:
        """Extract nodes from a Network object into Point objects."""
        nodes = []
        for node_id, node in network.nodes.items():
            nodes.append(Point(float(node.x), float(node.y)))
        return nodes
    
    @staticmethod
    def extract_edges(network: Any) -> List[Tuple[Point, Point]]:
        """Extract edges from a Network object into pairs of Points."""
        edges = []
        for edge_id, edge in network.edges.items():
            start = Point(float(edge.start_node.x), float(edge.start_node.y))
            end = Point(float(edge.end_node.x), float(edge.end_node.y))
            edges.append((start, end))
        return edges

class Sandbox:
    """Main class that coordinates terrain generation, pathfinding, and visualization."""
    
    def __init__(self, width: int, height: int, 
                 alpha: float = 0.3, beta: float = 0.2):
        self.terrain_generator = TerrainGenerator(width, height)
        self.path_finder = GradientPathFinder(
            self.terrain_generator.terrain,
            alpha=alpha,
            beta=beta
        )
        self.visualizer = TerrainVisualizer(self.terrain_generator.terrain)
        self.nodes: List[Point] = []
        self.paths: List[List[Point]] = []
        self.network_edges: List[Tuple[Point, Point]] = []
        self.network_adapter = NetworkAdapter()
        
    def create_energy_zone(self, x: float, y: float, height: float, radius: float) -> None:
        """Create an energy zone in the terrain."""
        self.terrain_generator.add_energy_zone(Point(x, y), height, radius)
        # Update pathfinder and visualizer references
        self.path_finder = GradientPathFinder(self.terrain_generator.terrain)
        self.visualizer = TerrainVisualizer(self.terrain_generator.terrain)
        
    def add_node(self, x: float, y: float) -> None:
        """Add a node to the sandbox."""
        self.nodes.append(Point(x, y))
        
    def add_network(self, network: Any) -> None:
        """Add a network's nodes and edges to the sandbox."""
        network_nodes = self.network_adapter.extract_nodes(network)
        self.nodes.extend(network_nodes)
        
        network_edges = self.network_adapter.extract_edges(network)
        self.network_edges.extend(network_edges)
        
        for start, end in network_edges:
            self.make_path((start.x, start.y), (end.x, end.y))
            
    def make_path(self, start: Tuple[float, float], end: Tuple[float, float], 
                  w_d: float = 1.0, w_e: float = 1.0) -> List[Point]:
        """Create a path between two points."""
        path = self.path_finder.find_path(Point(*start), Point(*end))
        if path:
            self.paths.append(path)
        return path
        
    def visualize_energy(self, opacity: float = 0.2, show_nodes: bool = True,
                        path_height: float = 1, node_height: float = 2) -> None:
        """Create 3D visualization of the terrain."""
        self.visualizer.plot_3d_surface(
            self.paths,
            self.nodes if show_nodes else [],
            opacity,
            path_height,
            node_height
        )
        
    def plot_heatmap(self, show_nodes: bool = True) -> None:
        """Create 2D heatmap visualization."""
        self.visualizer.plot_heatmap(
            self.paths,
            self.nodes if show_nodes else []
        )
        
    def clear_paths(self) -> None:
        """Clear all paths while maintaining nodes and network structure."""
        self.paths = []
        
    def regenerate_network_paths(self, w_d: float = 1.0, w_e: float = 1.0) -> None:
        """Regenerate all network edge paths with new parameters."""
        self.clear_paths()
        for start, end in self.network_edges:
            self.make_path((start.x, start.y), (end.x, end.y), w_d, w_e)
