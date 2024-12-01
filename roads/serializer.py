import json
import numpy as np
from typing import Dict, List, Any, Union
from dataclasses import dataclass

@dataclass
class Point:
    """Represents a 2D point with x, y coordinates."""
    x: float
    y: float

class SandboxSerializer:
    """Handles serialization/deserialization of Sandbox data to/from JSON."""
    
    @staticmethod
    def serialize_numpy(arr: np.ndarray) -> List[List[float]]:
        """Convert numpy array to nested list for JSON serialization."""
        return arr.tolist()
    
    @staticmethod
    def deserialize_numpy(data: List[List[float]]) -> np.ndarray:
        """Convert nested list back to numpy array."""
        return np.array(data)
    
    @staticmethod
    def serialize_point(point: Point) -> Dict[str, float]:
        """Convert Point to dictionary."""
        return {'x': float(point.x), 'y': float(point.y)}
    
    @staticmethod
    def deserialize_point(data: Dict[str, float]) -> Point:
        """Convert dictionary to Point."""
        return Point(x=data['x'], y=data['y'])
    
    @staticmethod
    def serialize_energy_zone(zone: Dict[str, Any]) -> Dict[str, Any]:
        """Convert energy zone parameters to dictionary."""
        return {
            'center': SandboxSerializer.serialize_point(zone['center']),
            'height': float(zone['height']),
            'radius': float(zone['radius'])
        }
    
    @staticmethod
    def deserialize_energy_zone(data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert dictionary to energy zone parameters."""
        return {
            'center': SandboxSerializer.deserialize_point(data['center']),
            'height': float(data['height']),
            'radius': float(data['radius'])
        }
    
    @staticmethod
    def save_sandbox(sandbox: Any, filename: str) -> None:
        """
        Save sandbox data to JSON file.
        
        Args:
            sandbox: Sandbox instance to save
            filename: Path to save the JSON file
        """
        data = {
            # Basic parameters
            'width': sandbox.terrain_generator.width,
            'height': sandbox.terrain_generator.height,
            'alpha': sandbox.path_finder.alpha,
            'beta': sandbox.path_finder.beta,
            
            # Terrain data
            'terrain': SandboxSerializer.serialize_numpy(sandbox.terrain_generator.terrain),
            
            # Energy zones
            'energy_zones': [
                SandboxSerializer.serialize_energy_zone(zone) 
                for zone in getattr(sandbox.terrain_generator, 'energy_zones', [])
            ],
            
            # Nodes and paths
            'nodes': [SandboxSerializer.serialize_point(node) for node in sandbox.nodes],
            'paths': [[SandboxSerializer.serialize_point(point) for point in path] 
                     for path in sandbox.paths],
            
            # Network edges
            'network_edges': [
                (SandboxSerializer.serialize_point(start), 
                 SandboxSerializer.serialize_point(end))
                for start, end in sandbox.network_edges
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def load_sandbox(filename: str, sandbox_class: Any) -> Any:
        """
        Load sandbox data from JSON file.
        
        Args:
            filename: Path to the JSON file
            sandbox_class: The Sandbox class to instantiate
            
        Returns:
            Instantiated Sandbox with loaded data
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Create sandbox with basic parameters
        sandbox = sandbox_class(
            width=data['width'],
            height=data['height'],
            alpha=data['alpha'],
            beta=data['beta']
        )
        
        # Set terrain data
        sandbox.terrain_generator.terrain = SandboxSerializer.deserialize_numpy(data['terrain'])
        
        # Add energy zones
        if not hasattr(sandbox.terrain_generator, 'energy_zones'):
            sandbox.terrain_generator.energy_zones = []
        for zone_data in data['energy_zones']:
            zone = SandboxSerializer.deserialize_energy_zone(zone_data)
            sandbox.terrain_generator.energy_zones.append(zone)
        
        # Set nodes
        sandbox.nodes = [SandboxSerializer.deserialize_point(node_data) 
                        for node_data in data['nodes']]
        
        # Set paths
        sandbox.paths = [[SandboxSerializer.deserialize_point(point_data) 
                         for point_data in path_data]
                        for path_data in data['paths']]
        
        # Set network edges
        sandbox.network_edges = [
            (SandboxSerializer.deserialize_point(start_data),
             SandboxSerializer.deserialize_point(end_data))
            for start_data, end_data in data['network_edges']
        ]
        
        # Update pathfinder and visualizer with new terrain
        sandbox.path_finder = sandbox.path_finder.__class__(
            sandbox.terrain_generator.terrain,
            alpha=data['alpha'],
            beta=data['beta']
        )
        sandbox.visualizer = sandbox.visualizer.__class__(sandbox.terrain_generator.terrain)
        
        return sandbox
