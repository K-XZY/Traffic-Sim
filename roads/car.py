import numpy as np
import random

class Car():

    def __init__(self, initial_pos, initial_dir, maxVel, maxAcc):

        self.pos = initial_pos
        self.dir = initial_dir
        self.maxVel = maxVel
        self.maxAcc = maxAcc
        self.velocity = maxVel
        self.stuckCtr = 0
        self.decisionState = "Free Driving"
        self.nearbyCar = False
        self.inactivityCtr = 0.0
        self.maxIter = 100

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

    # Function to determine velocity based on decision state
    def setVelocity(self):

        accCoef = 1

        # If there are no cars impacting behaviour
        if self.decisionState == "Free Driving":

            low = int(self.maxAcc * -accCoef)
            high = int(self.maxAcc * accCoef)

            if high < low:
                high, low = low, high

            acc = random.randint(low, high)
            if ((self.velocity+acc) == self.maxVel):
                velocity = self.maxVel
            elif((self.velocity+acc) <= 1):
                velocity = 2
            else:
                velocity = self.velocity + acc
            self.velocity = np.random.randint(1, velocity)
        
        # If there is a car in front 
        elif self.decisionState == "Approaching":

            low = int(self.maxAcc * -accCoef)
            high = 0

            if high < low:
                high, low = low, high

            acc = random.randint(low, high)
            if ((self.velocity+acc) == 0):
                velocity = 0.1
            else:
                velocity = self.velocity + acc
            self.velocity = np.random.randint(0, velocity)

        """
        elif self.decisionState == "Following":
            return self.position
        elif self.decisionState == "Breaking":
            return self.position
        """
        
    # Function to check for nearby cars
    def checkNearby(self, cars, minDist):

        if len(cars) == 1:
            self.nearbyCar = False
            return self.nearbyCar
        
        for car in cars:
            if car is not self:
                dist = self.distanceTo(car)
                if dist < minDist:
                    self.nearbyCar = True
                    return self.nearbyCar
        self.nearbyCar = False
        return self.nearbyCar

    # Function to check proximity to other cars
    def distanceTo(self, otherCar):
        x1, y1 = self.pos
        x2, y2 = otherCar.pos
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Function to determine path
    def percieve(self, map, position, direction, iter=0):

        self.setVelocity()

        if iter >= self.maxIter:
            return position, direction

        if(iter == self.velocity):
            return position, direction
        
        percievedPoints = [[position[0] + d[0], position[1] + d[1]] for d in self.directions[direction]]
        validPoints = [p for p in percievedPoints if map[p[1]][p[0]] == 1]

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

sb = SandboxSerializer.load_sandbox('circle_data2.json',Sandbox)
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
from PIL import Image

sb = SandboxSerializer.load_sandbox('circle_data2.json',Sandbox)
paths = sb.paths
paths = [point for path in paths for point in path]
grid = np.array(sb.path_map)
cars = []

def findOne(grid):
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 1:
                return i, j


i, j = findOne(grid)

car = Car([i, j], "Southeast", 5, 2)
cars.append(car)

for i in range(9):
    point = random.choice(paths)
    cars.append(Car([int(point.x), int(point.y)], 
                    random.choice(list(car.directions.keys())),
                    5, 
                    2))

def updateGrid(grid, stuckThreshold, cars):

    currentGrid = grid.copy()

    for car in cars:
        prevPos = car.pos

        nearbyCar = car.checkNearby(cars, 10)
        if(nearbyCar):
            car.decisionState = "Approaching"
        else:
            car.decisionState = "Free Driving"

        pos, dir = car.percieve(grid, car.pos, car.dir, iter=0)
        car.pos = pos
        car.dir = dir

        if prevPos == car.pos:
            car.stuckCtr += 1
        else:
            car.stuckCtr = 0

        if car.stuckCtr >= stuckThreshold:
            car.dir = random.choice(list(car.directions.keys()))
            car.stuckCtr = 0

        currentGrid[car.pos[0], car.pos[1]] = 2

        for dx in range(-10, 11):
            for dy in range(-10, 11):
                x = car.pos[1] + dx
                y = car.pos[0] + dy

                currentGrid[y, x] = 2

    return currentGrid, cars

def saveGrid(grid, path):
    
    # cmap = ListedColormap(['white', 'black', 'red'])
    plt.figure(figsize=(5 ,5))
    plt.imshow(grid, cmap='hot', interpolation='none')
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()

currentGrid, cars = updateGrid(grid, 3, cars)

imageFolder = 'car_images'
imageFiles = []

for i in range(20):
    currentGrid, cars = updateGrid(grid, 3, cars)
    print(car.pos)
    path = os.path.join('car_images', f'grid_image{i}.png')
    saveGrid(currentGrid, path)
    imageFiles.append(path)

images = [Image.open(image) for image in imageFiles]
gif_path = os.path.join('car_images', 'car_animation.gif')
images[0].save(
    gif_path, 
    save_all=True, 
    append_images=images[1:],
    duration=100,
    loop=1
)