import matplotlib.pyplot as plt
import random
import numpy as np
import all_roads_to_roam
import sandbox
from sandbox import Sandbox
from serializer import SandboxSerializer


class Car():

    def __init__(self, initial_pos, initial_dir, maxVel, maxAcc):

        self.pos = initial_pos
        self.dir = initial_dir
        self.maxVel = maxVel
        self.maxAcc = maxAcc
        self.prevVel = 0.0
        self.velocity = 0.0

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

        self.inactivityCtr = 0.0

    # Function to determine path
    def percieve(self, map, position, direction, iter=0):

        if(iter == self.velocity):
            return position, direction

        percievedPoints = [[position[0] + d[0], position[1] + d[1]] for d in self.directions[direction]]
        validPoints = [p for p in percievedPoints if self.map[p[1]][p[0]] == 1]

        if len(validPoints) == 0:
            return position, direction

        chosenPoint = random.choice(validPoints)
        direction = self.updateDirection(position, chosenPoint)
        iter = iter + 1

        return self.percieve(map, chosenPoint, direction, iter)

    # Function to update direction based on the last direction
    def updateDirection(self, point, chosenPoint):

        dx = chosenPoint[0] - point[0]
        dy = chosenPoint[1] - point[1]

        if dx == 0 and dy == 1:
            return "North"
        elif dx == 1 and dy == 1:
            return "Northeast"
        elif dx == 1 and dy == 0:
            return "East"
        elif dx == 1 and dy == -1:
            return "Southeast"
        elif dx == 0 and dy == -1:
            return "South"
        elif dx == -1 and dy == -1:
            return "Southwest"
        elif dx == -1 and dy == 0:
            return "West"
        elif dx == -1 and dy == 1:
            return "Northwest"
        
        return "invalid entry"

###########  
# Test code
###########

# Load the map
sb = SandboxSerializer.load_sandbox('circle_data.json', Sandbox)
grid = sb.path_map

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 10))  # Adjust figure size to fit your needs

# Display the grid using imshow (1 is white, 0 is black)
ax.imshow(grid, cmap='gray', interpolation='none')

# Get the grid dimensions
height, width = grid.shape

# Initialize the car at the top-left corner or any other valid position
car = Car([100, 100], "East", 3, 2)

# Set the aspect ratio to be equal
ax.set_aspect('equal')

# Optional: Remove axis labels to avoid clutter
ax.axis('off')

# Show the initial plot (without the car)
plt.draw()

# Simulate the movement of the car
for _ in range(100):  # Running 100 iterations for the example
    # Update car position
    new_pos, new_dir = car.percieve(grid, car.pos, car.dir)

    # Clear the old position of the car
    grid[car.pos[1], car.pos[0]] = 1  # Set the previous position back to 1 (free space)

    # Place the car at the new position
    grid[new_pos[1], new_pos[0]] = 2  # Set the new position to 2 (car)

    # Update car's position and direction
    car.pos = new_pos
    car.dir = new_dir

    # Display the updated grid
    ax.imshow(grid, cmap='gray', interpolation='none')
    plt.draw()  # Redraw the figure

    # Pause to simulate frame by frame animation
    plt.pause(0.2)  # Delay for animation effect (200 ms per frame)

# Show the final plot
plt.show()
