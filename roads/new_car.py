import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from PIL import Image
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from sandbox import Sandbox
from serializer import SandboxSerializer
from math import atan2, degrees  # Add these imports
import scipy
import cv2
from visualize import visualize_car_speed

@dataclass
class Position:
    """Represents a 2D position with x, y coordinates."""
    x: int
    y: int
    
    def to_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)

@dataclass
class Point:
    """Represents a 2D point with x, y coordinates."""
    x: float
    y: float

    def to_tuple(self) -> Tuple[int, int]:
        """Convert point to integer tuple coordinates."""
        return (int(round(self.x)), int(round(self.y)))

class Car:
    """Represents a car in the simulation with movement and decision-making capabilities."""
    
    def __init__(self, position: Position, direction: str, max_vel: int, max_acc: int):
        """Initialize a car with its properties."""
        self.pos = position
        self.dir = direction
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.velocity = max_vel
        self.stuck_counter = 0
        self.decision_state = "Free Driving"
        self.nearby_car = False
        self.inactivity_counter = 0.0
        self.detection_radius = 100  # Radius to look for other cars
        self.list_velocity = [self.velocity]

        # Define possible movement directions with angles
        self.directions = {
            "North": [[-1, 1], [0, 1], [1, 1]],
            "Northeast": [[0, 1], [1, 1], [1, 0]],
            "East": [[1, 1], [1, 0], [1, -1]],
            "Southeast": [[1, 0], [1, -1], [0, -1]],
            "South": [[1, -1], [0, -1], [-1, -1]],
            "Southwest": [[0, -1], [-1, -1], [-1, 0]],
            "West": [[-1, -1], [-1, 0], [-1, 1]],
            "Northwest": [[-1, 0], [-1, 1], [0, 1]],
        }
        
        # Direction to angle mapping (in degrees)
        self.direction_angles = {
            "North": 90,
            "Northeast": 45,
            "East": 0,
            "Southeast": 315,
            "South": 270,
            "Southwest": 225,
            "West": 180,
            "Northwest": 135
        }

    def store_velocity(self) -> list:
        """Store the velocity of the car."""
        self.list_velocity.append(self.velocity)
        return self.list_velocity

    def set_velocity(self) -> None:
        """Determine velocity based on current decision state."""
        acc_coef = 1

        if self.decision_state == "Free Driving":
            # acc = random.randint(-self.max_acc, self.max_acc)
            acc = random.randint(0,self.max_acc)
            if (self.velocity + acc) >= self.max_vel:
                velocity = self.max_vel
            elif (self.velocity + acc) <= 1:
                velocity = 2
            else:
                velocity = self.velocity + acc
            self.velocity = max(1, min(velocity, self.max_vel))
        
        elif self.decision_state == "Approaching":
            acc = random.randint(-self.max_acc, 0)
            velocity = max(0, self.velocity + acc)
            self.velocity = max(0, min(velocity, self.max_vel))

    def check_nearby(self, cars: List['Car'], min_dist: float) -> bool:
        """
        Check for nearby cars within minimum distance and in the direction of travel.
        Uses the same directional logic as get_cars_in_front() but with distance threshold.
        """
        if len(cars) == 1:
            self.nearby_car = False
            return False

        current_angle = self.direction_angles[self.dir]
        
        for car in cars:
            if car is self:
                continue
                
            # Calculate relative position
            dx = car.pos.x - self.pos.x
            dy = car.pos.y - self.pos.y
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance > min_dist:
                continue
                
            # Calculate angle to other car
            angle_to_car = degrees(atan2(dy, dx)) % 360
            
            # Calculate angle difference
            angle_diff = (angle_to_car - current_angle) % 360
            
            # Consider cars within a 120-degree cone in front and within min_dist
            if angle_diff < 60 or angle_diff > 300:
                self.nearby_car = True
                return True

        self.nearby_car = False
        return False

    def distance_to(self, other_car: 'Car') -> float:
        """Calculate Euclidean distance to another car."""
        return np.sqrt((other_car.pos.x - self.pos.x)**2 + 
                      (other_car.pos.y - self.pos.y)**2)

    def perceive(self, grid: np.ndarray, position: Position, direction: str, 
                all_cars: List['Car']) -> Tuple[Position, str]:
        """Determine next position based on current state and surroundings."""
        self.set_velocity()
        current_pos = position
        
        current_dir = direction
        
        # Iterate based on velocity
        blocked = False
        for _ in range(self.velocity):
            # Get possible next positions
            perceived_points = [
                [current_pos.x + d[0], current_pos.y + d[1]] 
                for d in self.directions[current_dir]
            ]
            
            # Filter valid points
            valid_points = [
                p for p in perceived_points 
                if 0 <= p[1] < grid.shape[0] and 
                   0 <= p[0] < grid.shape[1] and 
                   grid[p[1]][p[0]] == 1
            ] # select road

            for p in valid_points:
                # look if any other cars if there
                for car in all_cars:
                    if car.pos.x == p[0] and car.pos.y == p[1]:
                        blocked = True
        
            if blocked:
                break


            
            # If no valid points, stay in current position
            if not valid_points:
                # try a new direction 
                new_dir = random.choice(list(self.directions.keys()))
                continue
            
            # Choose next position
            chosen_point = random.choice(valid_points)
            new_pos = Position(chosen_point[0], chosen_point[1])
            
            # Update direction based on movement
            new_dir = self.update_direction(current_pos, new_pos)
            if new_dir == "invalid entry":
                break
                
            current_pos = new_pos
            current_dir = new_dir
            
        return current_pos, current_dir

    def update_direction(self, current: Position, chosen: Position) -> str:
        """Update direction based on movement from current to chosen position."""
        dx = chosen.x - current.x
        dy = chosen.y - current.y

        direction_map = {
            (0, 1): "North",
            (1, 1): "Northeast",
            (1, 0): "East",
            (1, -1): "Southeast",
            (0, -1): "South",
            (-1, -1): "Southwest",
            (-1, 0): "West",
            (-1, 1): "Northwest"
        }
        
        return direction_map.get((dx, dy), "invalid entry")

class Simulation:
    """Manages the car simulation and visualization."""
    
    def __init__(self, grid: np.ndarray, network_edges: List[Tuple[Point, Point]], num_cars: int = 10):
        self.grid = grid
        self.network_edges = network_edges
        self.car_colors = self._generate_car_colors(num_cars)
        self.cars = self._initialize_cars(num_cars)
        self.image_folder = 'car_images'
        os.makedirs(self.image_folder, exist_ok=True)
        self.visualization_grid = self._create_thick_path_map(grid)

    def get_cars(self) -> List[Car]:
        return self.cars

    def _create_thick_path_map(self, grid: np.ndarray) -> np.ndarray:
        """Create a thickened version of the path map for visualization."""
        thick_grid = np.zeros_like(grid)
        road_positions = np.where(grid == 1)
        
        # For each road pixel, fill a 7x7 square (3 pixels in each direction)
        for y, x in zip(road_positions[0], road_positions[1]):
            y_start = max(0, y - 2)
            y_end = min(grid.shape[0], y + 3)
            x_start = max(0, x - 2)
            x_end = min(grid.shape[1], x + 3)
            thick_grid[y_start:y_end, x_start:x_end] = 1
            
        return thick_grid

    def _generate_car_colors(self, num_cars: int) -> List[Tuple[float, float, float]]:
        """Generate distinct colors for each car using HSV color space."""
        colors = []
        for i in range(num_cars):
            # Use HSV to generate evenly spaced colors
            hue = i / num_cars
            # Convert HSV to RGB (using full saturation and value)
            rgb = plt.cm.hsv(hue)[:3]  # Get RGB values (exclude alpha)
            colors.append(rgb)
        return colors



    def _point_to_line_distance(self, point: Position, start: Point, end: Point) -> float:
        """Calculate the distance from a point to a line segment."""
        # Convert Point to numpy arrays for vector operations
        p = np.array([point.x, point.y])
        a = np.array([start.x, start.y])
        b = np.array([end.x, end.y])
        
        # Calculate the line segment vector
        line_vec = b - a
        line_length = np.linalg.norm(line_vec)
        if line_length == 0:
            return np.linalg.norm(p - a)
        
        # Calculate relative position of projection
        t = max(0, min(1, np.dot(p - a, line_vec) / (line_length * line_length)))
        
        # Calculate closest point on line segment
        projection = a + t * line_vec
        
        # Return distance to closest point
        return np.linalg.norm(p - projection)

    def _find_closest_edge(self, pos: Position) -> Tuple[Point, Point]:
        """Find the closest network edge to a given position."""
        min_distance = float('inf')
        closest_edge = None
        
        for start, end in self.network_edges:  # network_edges from Sandbox
            distance = self._point_to_line_distance(pos, start, end)
            if distance < min_distance:
                min_distance = distance
                closest_edge = (start, end)
        
        return closest_edge

    def _get_direction_from_edge(self, start: Point, end: Point) -> str:
        """Get the direction based on edge orientation using cosine similarity."""
        # Calculate edge direction vector
        dx = end.x - start.x
        dy = end.y - start.y
        edge_vector = np.array([dx, dy])
        edge_vector = edge_vector / np.linalg.norm(edge_vector)
        
        # Standard directions with normalized vectors
        directions = {
            "North": np.array([0, 1]),
            "Northeast": np.array([1, 1]) / np.sqrt(2),
            "East": np.array([1, 0]),
            "Southeast": np.array([1, -1]) / np.sqrt(2),
            "South": np.array([0, -1]),
            "Southwest": np.array([-1, -1]) / np.sqrt(2),
            "West": np.array([-1, 0]),
            "Northwest": np.array([-1, 1]) / np.sqrt(2)
        }
        
        # Calculate cosine similarity with each direction
        similarities = {
            dir_name: np.dot(edge_vector, dir_vec)
            for dir_name, dir_vec in directions.items()
        }
        
        # Return direction with highest similarity
        return max(similarities.items(), key=lambda x: x[1])[0]

    def _initialize_cars(self, num_cars: int) -> List[Car]:
        """Initialize cars on network edges with appropriate directions."""
        cars = []
        valid_positions = np.where(self.grid == 1)
        positions = list(zip(valid_positions[1], valid_positions[0]))  # x, y coordinates
        
        for i in range(num_cars):
            # Get random position on road
            x, y = random.choice(positions)
            pos = Position(x, y)
            
            # Find closest edge and its direction
            start, end = self._find_closest_edge(pos)
            direction = self._get_direction_from_edge(start, end)
            
            # Create car with determined direction
            speed_max = random.randint(5, 30)
            acc_max = random.randint(int(0.1*speed_max)+1, int(0.5*speed_max))
            car = Car(pos, direction, speed_max, acc_max)
            car.color_idx = i
            cars.append(car)
        
        # special_car = Car(pos, direction, 3,1)
        # cars.append(special_car)
        return cars

    def check_blocking_info(self, car: Car, new_pos: Position) -> Tuple[bool, Optional[Position]]:
        """
        Check if there's any car blocking the path to the new position using actual road coordinates.
        
        Args:
            car: The car being checked
            new_pos: The intended new position
        
        Returns:
            Tuple[bool, Optional[Position]]: 
                - Boolean indicating if path is blocked
                - Position right after blocking car (if blocked), None otherwise
        """
        # 1. Get current edge
        start, end = self._find_closest_edge(car.pos)
        
        # 2. Find the road coordinates between current and new position
        # First, find the path that contains these points
        matching_path = None
        for path in self.network_edges:
            if (self._point_to_line_distance(car.pos, path[0], path[1]) < 5 and 
                self._point_to_line_distance(new_pos, path[0], path[1]) < 5):
                matching_path = path
                break
        
        if not matching_path:
            return False, None  # No valid path found
            
        # Generate path points using road coordinates
        points_to_check = self._generate_path_points(car.pos, new_pos, matching_path)
        
        # 3. Check for cars on these points
        blocking_car = None
        blocking_point_idx = None
        
        for i, point in enumerate(points_to_check):
            for other_car in self.cars:
                if other_car is car:  # Skip self
                    continue
                    
                # Check if other car is on same path segment
                if self._point_to_line_distance(other_car.pos, start, end) <= 5:
                    dist = np.sqrt((point.x - other_car.pos.x)**2 + 
                                 (point.y - other_car.pos.y)**2)
                    if dist < 10:  # Car width/length threshold
                        blocking_car = other_car
                        blocking_point_idx = i
                        break
            
            if blocking_car:
                break
        
        # 4 & 5. Return blocking status and safe position
        if blocking_car:
            # If blocked, return the last safe position (point before blocking car)
            if blocking_point_idx > 0:
                safe_pos = points_to_check[blocking_point_idx - 1]
            else:
                safe_pos = car.pos  # Stay in current position if blocked immediately
            return True, safe_pos
        
        return False, None

    def _generate_path_points(self, start_pos: Position, end_pos: Position, 
                             path: Tuple[Point, Point]) -> List[Position]:
        """
        Generate a list of points along the actual road path between start and end positions.
        
        Args:
            start_pos: Starting position
            end_pos: Ending position
            path: Tuple of start and end Points defining the road segment
        
        Returns:
            List[Position]: List of positions along the road
        """
        path_start, path_end = path
        
        # Project points onto path line segment
        def project_point_to_line(point: Position, line_start: Point, line_end: Point) -> Position:
            p = np.array([point.x, point.y])
            a = np.array([line_start.x, line_start.y])
            b = np.array([line_end.x, line_end.y])
            
            # Get vector from a to b
            ab = b - a
            # Get vector from a to p
            ap = p - a
            
            # Calculate the projection
            t = np.dot(ap, ab) / np.dot(ab, ab)
            t = max(0, min(1, t))  # Clamp to [0,1]
            
            projected = a + t * ab
            return Position(int(projected[0]), int(projected[1]))
        
        # Project start and end positions onto path
        proj_start = project_point_to_line(start_pos, path_start, path_end)
        proj_end = project_point_to_line(end_pos, path_start, path_end)
        
        # Get direction vector
        dx = proj_end.x - proj_start.x
        dy = proj_end.y - proj_start.y
        distance = np.sqrt(dx**2 + dy**2)
        
        # Create points along the path
        num_points = max(10, int(distance / 5))  # One point every 5 pixels, minimum 10 points
        points = []
        
        for i in range(num_points):
            t = i / (num_points - 1)
            x = proj_start.x + t * dx
            y = proj_start.y + t * dy
            points.append(Position(int(x), int(y)))
        
        return points

    def update_grid(self, stuck_threshold: int = 3) -> np.ndarray:
        """Update simulation state for one time step with collision prevention."""
        # Initialize grid with -1 for background
        current_grid = np.full_like(self.grid, -1, dtype=float)
        
        # Use visualization grid for road layout
        current_grid[self.visualization_grid == 1] = -2  # Road pixels
        
        for car_idx, car in enumerate(self.cars):
            prev_pos = car.pos

            # Check for nearby cars and update decision state
            nearby_car = car.check_nearby(self.cars, 10)
            car.decision_state = "Approaching" if nearby_car else "Free Driving"

            # Find the road edge the car is on and get its direction
            start, end = self._find_closest_edge(car.pos)
            edge_direction = self._get_direction_from_edge(start, end)
            
            # Enforce edge direction before movement
            if car.dir != edge_direction:
                car.dir = edge_direction

            # First, get the proposed new position from car's perception
            proposed_pos, proposed_direction = car.perceive(self.grid, car.pos, car.dir, self.cars)
            
            if proposed_direction == edge_direction:
                car.pos = proposed_pos
                car.dir = proposed_direction
            else:
                # with a probability, well calibrate diections
                if random.randint(0,10)>=5:
                    car.dir = edge_direction
                    car.pos = proposed_pos
                else:
                    car.pos = proposed_pos
                    car.dir = proposed_direction
            # Reset stuck counter as car moved successfully
            car.stuck_counter = 0

            # Update stuck status
            if prev_pos.to_tuple() == car.pos.to_tuple():
                car.stuck_counter += 1
            else:
                car.stuck_counter = 0

            # Handle persistent stuck situations
            if car.stuck_counter >= stuck_threshold:
                car.stuck_counter = 0  # Reset counter but maintain direction
                # Could add additional stuck handling here if needed

            # Visualize car on grid (create a 7x7 square for each car)
            y, x = car.pos.y, car.pos.x
            for dy in range(-7, 8):
                for dx in range(-7, 8):
                    new_y, new_x = y + dy, x + dx
                    if (0 <= new_y < current_grid.shape[0] and 
                        0 <= new_x < current_grid.shape[1]):
                        current_grid[new_y, new_x] = car_idx

            # Store car velocity for tracking
            car.store_velocity()

        return current_grid

    def create_frame_image(self, grid: np.ndarray) -> np.ndarray:
        """Create frame image without saving to disk."""
        # Create figure with fixed pixel dimensions
        width_pixels = 1000
        height_pixels = 1000
        dpi = 100
        figsize = (width_pixels/dpi, height_pixels/dpi)
        
        fig = plt.figure(figsize=figsize, dpi=dpi)
        
        # Create a custom colormap
        all_colors = ['black', 'gray'] + [plt.cm.hsv(i / len(self.cars))[:3] for i in range(len(self.cars))]
        cmap = ListedColormap(all_colors)
        
        # Create a normalized version of the grid
        normalized_grid = grid.copy()
        normalized_grid[self.visualization_grid == 1] = 1
        normalized_grid[self.visualization_grid == 0] = 0
        normalized_grid[grid >= 0] = grid[grid >= 0] + 2
        
        plt.imshow(normalized_grid, cmap=cmap, interpolation='nearest')
        plt.axis('off')
        
        # Set layout to tight to remove margins
        plt.tight_layout(pad=0)
        
        # Convert plot to image
        fig.canvas.draw()
        
        # Get the RGBA buffer from the figure
        buf = np.asarray(fig.canvas.buffer_rgba())
        
        # Convert RGBA to RGB
        image = buf[:, :, :3]
        
        plt.close()
        
        return image

    def save_frame(self, grid: np.ndarray, frame_number: int) -> str:
        """Save a single frame of the simulation."""
        plt.figure(figsize=(10, 10))
        
        # Create a custom colormap that includes background, road, and car colors
        all_colors = ['black', 'gray'] + [plt.cm.hsv(i / len(self.cars))[:3] for i in range(len(self.cars))]
        cmap = ListedColormap(all_colors)
        
        # Create a normalized version of the grid for proper color mapping
        normalized_grid = grid.copy()
        
        # Use the pre-computed thick path map for roads
        normalized_grid[self.visualization_grid == 1] = 1  # Road
        normalized_grid[self.visualization_grid == 0] = 0  # Background
        
        # Car indices will start from 2
        normalized_grid[grid >= 0] = grid[grid >= 0] + 2
        
        plt.imshow(normalized_grid, cmap=cmap, interpolation='nearest')
        plt.axis('off')
        
        path = os.path.join(self.image_folder, f'frame_{frame_number:04d}.png')
        plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
        
        return path

    def save_as_gif(self, num_frames: int, name = 'sim') -> str:
        """Save simulation as GIF for shorter sequences."""
        image_files = []
        
        for i in range(num_frames):
            current_grid = self.update_grid()
            frame_path = self.save_frame(current_grid, i)
            image_files.append(frame_path)
            print(f"Processing frame {i+1}/{num_frames}")
            
        # Create GIF
        images = [Image.open(image) for image in image_files]
        gif_path = os.path.join(self.image_folder, f'{name}.gif')
        
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=100,
            loop=0
        )

        # Clean up individual frame files
        for file in image_files:
            os.remove(file)
        
        return gif_path

    def save_as_video(self, num_frames: int, name='simulation') -> str:
        """Save simulation as MP4 video for longer sequences."""
        # Get frame size from first frame
        test_frame = self.update_grid()
        test_img = self.create_frame_image(test_frame)
        height, width = test_img.shape[:2]

        # Initialize video writer
        video_path = os.path.join(self.image_folder, f'{name}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 10, (width, height))

        try:
            for i in range(num_frames):
                current_grid = self.update_grid()
                # Convert frame to image without saving to disk
                frame = self.create_frame_image(current_grid)
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
                print(f"Processing frame {i+1}/{num_frames}")
        
        finally:
            out.release()

        return video_path

    def save_simulation(self, num_frames: int = 200, name='simulation') -> str:
        """Save simulation as either GIF or MP4 based on number of frames."""
        if num_frames < 10:
            return self.save_as_gif(num_frames, name)
        else:
            return self.save_as_video(num_frames, name)


def main():
    # Load sandbox data
    sb = SandboxSerializer.load_sandbox('./butterfly_data.json', Sandbox)
    grid = np.array(sb.path_map)
    
    # Create and run simulation with network edges
    sim = Simulation(grid, sb.network_edges, num_cars=300)
    output_path = sim.save_simulation(num_frames=30, name='sim_new_curved_1_not_all_same')
    print(f"Simulation completed. Animation saved to: {output_path}")

    cars = sim.get_cars()
    visualize_car_speed(cars)

if __name__ == "__main__":
    main()
