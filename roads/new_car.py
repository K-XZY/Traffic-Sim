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

        # Define possible movement directions
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

    def perceive(self, grid: np.ndarray, position: Position, 
                direction: str) -> Tuple[Position, str]:
        """Determine next position based on current state and surroundings."""
        self.set_velocity()
        current_pos = position
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
            
            # Update direction
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

        # Define possible movement directions
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

    def perceive(self, grid: np.ndarray, position: Position, 
                direction: str) -> Tuple[Position, str]:
        """Determine next position based on current state and surroundings."""
        self.set_velocity()
        current_pos = position
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
            
            # Update direction
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
        # Set road pixels to -2
        current_grid[self.grid == 1] = -2

        for car_idx, car in enumerate(self.cars):
            prev_pos = car.pos

            nearby_car = car.check_nearby(self.cars, 10)
            car.decision_state = "Approaching" if nearby_car else "Free Driving"

            pos, direction = car.perceive(self.grid, car.pos, car.dir)
            car.pos = pos
            car.dir = direction

            if prev_pos.to_tuple() == car.pos.to_tuple():
                car.stuck_counter += 1
            else:
                car.stuck_counter = 0

            if car.stuck_counter >= stuck_threshold:
                car.dir = random.choice(list(car.directions.keys()))
                car.stuck_counter = 0

            # Create a 5x5 square for each car
            y, x = car.pos.y, car.pos.x
            
            # Define the range for the 5x5 square
            for dy in range(-2, 3):  # -2, -1, 0, 1, 2
                for dx in range(-2, 3):  # -2, -1, 0, 1, 2
                    new_y, new_x = y + dy, x + dx
                    
                    # Check boundaries
                    if (0 <= new_y < current_grid.shape[0] and 
                        0 <= new_x < current_grid.shape[1]):
                        # Set car color index for this pixel
                        current_grid[new_y, new_x] = car.color_idx

        return current_grid

    def save_frame(self, grid: np.ndarray, frame_number: int) -> str:
        """Save a single frame of the simulation."""
        plt.figure(figsize=(10, 10))
        
        # Create a custom colormap that includes background, road, and car colors
        all_colors = ['black', 'white'] + [plt.cm.hsv(i / len(self.cars))[:3] for i in range(len(self.cars))]
        cmap = ListedColormap(all_colors)
        
        # Create a normalized version of the grid for proper color mapping
        normalized_grid = grid.copy()
        normalized_grid[grid == -1] = 0  # Background
        normalized_grid[grid == -2] = 1  # Road
        # Car indices will start from 2
        normalized_grid[grid >= 0] += 2
        
        plt.imshow(normalized_grid, cmap=cmap, interpolation='nearest')
        plt.axis('off')
        
        path = os.path.join(self.image_folder, f'frame_{frame_number:04d}.png')
        plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()
        
        return path

    def run_simulation(self, num_frames: int = 200) -> str:
        """Run the simulation and create animation."""
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
        
        return gif_path

def main():
    """Main function to run the simulation."""
    # Load sandbox data
    sb = SandboxSerializer.load_sandbox('circle_data2.json', Sandbox)
    grid = np.array(sb.path_map)
    
    # Create and run simulation
    sim = Simulation(grid, num_cars=10)
    gif_path = sim.run_simulation(num_frames=30)
    print(f"Simulation completed. Animation saved to: {gif_path}")

if __name__ == "__main__":
    main()
