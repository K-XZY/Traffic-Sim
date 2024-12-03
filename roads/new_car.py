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

@dataclass
class Position:
    """Represents a 2D position with x, y coordinates."""
    x: int
    y: int
    
    def to_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)


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
        self.detection_radius = 50  # Radius to look for other cars

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

    def get_cars_in_front(self, cars: List['Car']) -> List['Car']:
        """Identify cars that are in front of the current car within detection radius."""
        cars_in_front = []
        current_angle = self.direction_angles[self.dir]
        
        for other_car in cars:
            if other_car is self:
                continue
                
            # Calculate relative position
            dx = other_car.pos.x - self.pos.x
            dy = other_car.pos.y - self.pos.y
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance > self.detection_radius:
                continue
                
            # Calculate angle to other car
            angle_to_car = degrees(atan2(dy, dx)) % 360
            
            # Calculate angle difference
            angle_diff = (angle_to_car - current_angle) % 360
            
            # Consider cars within a 120-degree cone in front
            if angle_diff < 60 or angle_diff > 300:
                cars_in_front.append(other_car)
        
        return cars_in_front

    def get_majority_direction(self, cars_in_front: List['Car']) -> str:
        """Determine the most common direction among cars in front."""
        if not cars_in_front:
            return self.dir
            
        direction_counts = {}
        for car in cars_in_front:
            direction_counts[car.dir] = direction_counts.get(car.dir, 0) + 1
            
        # If there are directions to choose from, pick the most common
        if direction_counts:
            return max(direction_counts.items(), key=lambda x: x[1])[0]
            
        return self.dir

    def set_velocity(self) -> None:
        """Determine velocity based on current decision state."""
        acc_coef = 1

        if self.decision_state == "Free Driving":
            acc = random.randint(-self.max_acc, self.max_acc)
            if (self.velocity + acc) >= self.max_vel:
                velocity = self.max_vel
            elif (self.velocity + acc) <= 1:
                velocity = 2
            else:
                velocity = self.velocity + acc
            self.velocity = max(1, min(velocity, self.max_vel))
        
        elif self.decision_state == "Approaching":
            acc = random.randint(-self.max_acc, 0)
            velocity = max(1, self.velocity + acc)
            self.velocity = max(1, min(velocity, self.max_vel))

    def check_nearby(self, cars: List['Car'], min_dist: float) -> bool:
        """Check for nearby cars within a minimum distance."""
        if len(cars) == 1:
            self.nearby_car = False
            return False
        
        for car in cars:
            if car is not self:
                dist = self.distance_to(car)
                if dist < min_dist:
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
        
        # Get cars in front and their majority direction
        cars_in_front = self.get_cars_in_front(all_cars)
        majority_dir = self.get_majority_direction(cars_in_front)
        
        # Update direction based on majority
        if cars_in_front:
            current_dir = majority_dir
        else:
            current_dir = direction
        
        # Iterate based on velocity
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
            ]
            
            # If no valid points, stay in current position
            if not valid_points:
                break
            
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
    
    def __init__(self, grid: np.ndarray, num_cars: int = 10):
        """Initialize simulation with grid and number of cars."""
        self.grid = grid
        # Generate unique colors for each car
        self.car_colors = self._generate_car_colors(num_cars)
        self.cars = self._initialize_cars(num_cars)
        self.image_folder = 'car_images'
        os.makedirs(self.image_folder, exist_ok=True)
        
        # Create thickened path map for visualization
        self.visualization_grid = self._create_thick_path_map(grid)

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

    def _initialize_cars(self, num_cars: int) -> List[Car]:
        """Initialize cars at random valid positions on the grid."""
        cars = []
        valid_positions = np.where(self.grid == 1)
        positions = list(zip(valid_positions[1], valid_positions[0]))  # x, y coordinates
        
        for i in range(num_cars):
            x, y = random.choice(positions)
            direction = random.choice(list(Car(Position(0, 0), "", 0, 0).directions.keys()))
            car = Car(Position(x, y), direction, 5, 2)
            # Store the car's color index
            car.color_idx = i
            cars.append(car)
            
        return cars

    def update_grid(self, stuck_threshold: int = 3) -> np.ndarray:
        """Update simulation state for one time step."""
        # Initialize grid with -1 for background
        current_grid = np.full_like(self.grid, -1, dtype=float)
        
        # Use visualization grid for road layout
        current_grid[self.visualization_grid == 1] = -2  # Road pixels
        
        for car_idx, car in enumerate(self.cars):
            prev_pos = car.pos

            nearby_car = car.check_nearby(self.cars, 10)
            car.decision_state = "Approaching" if nearby_car else "Free Driving"

            # Pass the list of all cars to perceive
            pos, direction = car.perceive(self.grid, car.pos, car.dir, self.cars)
            car.pos = pos
            car.dir = direction

            if prev_pos.to_tuple() == car.pos.to_tuple():
                car.stuck_counter += 1
            else:
                car.stuck_counter = 0

            if car.stuck_counter >= stuck_threshold:
                car.dir = random.choice(list(car.directions.keys()))
                car.stuck_counter = 0

            # Create a 7x7 square for each car
            y, x = car.pos.y, car.pos.x
            
            # Define the range for the 7x7 square
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    new_y, new_x = y + dy, x + dx
                    
                    # Check boundaries
                    if (0 <= new_y < current_grid.shape[0] and 
                        0 <= new_x < current_grid.shape[1]):
                        # Set car color index for this pixel
                        current_grid[new_y, new_x] = car_idx

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

    def save_as_gif(self, num_frames: int) -> str:
        """Save simulation as GIF for shorter sequences."""
        image_files = []
        
        for i in range(num_frames):
            current_grid = self.update_grid()
            frame_path = self.save_frame(current_grid, i)
            image_files.append(frame_path)
            print(f"Processing frame {i+1}/{num_frames}")
            
        # Create GIF
        images = [Image.open(image) for image in image_files]
        gif_path = os.path.join(self.image_folder, 'simulation.gif')
        
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
        if num_frames <= 50:
            return self.save_as_gif(num_frames)
        else:
            return self.save_as_video(num_frames, name)

def main():
    """Main function to run the simulation."""
    # Load sandbox data
    sb = SandboxSerializer.load_sandbox('circle_data2.json', Sandbox)
    grid = np.array(sb.path_map)
    
    # Create and run simulation
    sim = Simulation(grid, num_cars=400)
    output_path = sim.save_simulation(num_frames=1000, name = 'sim3')
    print(f"Simulation completed. Animation saved to: {output_path}")

if __name__ == "__main__":
    main()
