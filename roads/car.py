import numpy as np
import random

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

#############
# Test code #
#############

"""
import matplotlib.pyplot as plt
import all_roads_to_roam
import sandbox
import time
from sandbox import Sandbox
from serializer import SandboxSerializer

sb = SandboxSerializer.load_sandbox('circle_data.json',Sandbox)
grid = np.array(sb.path_map)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 10))  # Adjust figure size to fit your needs

# Display the grid using imshow (1 is white, 0 is black)
ax.imshow(-grid, cmap='gray', interpolation='none')

i = 0
j = 0
current = 0
rows, cols = grid.shape
for i in range(rows):
    for j in range(cols):
        current = grid[i, j]
        if current != 0:
            break

car = Car([i, j], "East", 3, 2)

# Update function for animation
def update(frame):

    # Create a copy of the grid for drawing
    grid_copy = grid.copy()

    # Get the car's current position
    car_pos = car.pos
    car_x, car_y = car_pos[0], car_pos[1]

    # Ensure the surrounding pixels (3x3) stay within grid bounds
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            x = car_x + dx
            y = car_y + dy

            # Check if the pixel is within grid bounds
            if 0 <= x < rows and 0 <= y < cols:
                grid_copy[x, y] = 0  # Set to blue (will show as 0 in black and white)

    # Set the car's pixel to blue as well
    grid_copy[car_x, car_y] = 0  # Set to blue (0)

    # Display the updated grid
    ax.imshow(grid_copy, cmap='Blues', interpolation='none')
    
    # Move the car (update the car's position)
    car.pos, car.dir = car.percieve(grid, [car_x, car_y], car.dir, iter=0)


# Set up the animation loop
from matplotlib.animation import FuncAnimation

ani = FuncAnimation(fig, update, frames=100, interval=200, repeat=False)

plt.show()  # Show the final plot when the loop ends

"""

################
# Test Code v2 #
################

import all_roads_to_roam
import sandbox
import time
from sandbox import Sandbox
from serializer import SandboxSerializer
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

sb = SandboxSerializer.load_sandbox('circle_data.json',Sandbox)
grid = np.array(sb.path_map)

def findOne(grid):
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 1:
                return i, j


i, j = findOne(grid)
car = Car([i, j], "Southeast", 3, 2)

def updateGrid(grid, car: Car):

    car.pos, car.dir = car.percieve(grid, car.pos, car.dir, iter=0)

    currentGrid = grid
    currentGrid[car.pos[0], car.pos[1]] = 2

    for dx in range(-3, 4):
        for dy in range(-3, 4):
            x = car.pos[1] + dx
            y = car.pos[0] + dy

            currentGrid[x, y] = 2

    return currentGrid

def saveGrid(grid, path):
    
    cmap = ListedColormap(['white', 'black', 'red'])
    plt.imshow(grid, cmap=cmap, interpolation='none')
    #plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close

for i in range(10):
    currentGrid = updateGrid(grid, car)
    print(car.pos)
    path = os.path.join('car_images', f'grid_image{i}.png')
    saveGrid(currentGrid, path)