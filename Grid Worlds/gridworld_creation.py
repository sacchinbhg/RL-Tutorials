import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from queue import Queue

ratio = 0.45

def is_path_clear(grid, start, goal):
    parent = {start: None}
    queue = Queue()
    queue.put(start)

    while not queue.empty():
        x, y = queue.get()
        if (x, y) == goal:
            path = []
            while (x, y) != start:
                path.append((x, y))
                x, y = parent[(x, y)]
            path.append(start)  # Add the start position to the path
            return True, path[::-1]
        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 100 and 0 <= ny < 100 and grid[nx][ny] != 1 and (nx, ny) not in parent:
                parent[(nx, ny)] = (x, y)
                queue.put((nx, ny))
    return False, []

def add_obstacles(grid, obstacle_ratio):
    n_obstacles = int(grid.size * obstacle_ratio)
    while n_obstacles > 0:
        x, y = random.randint(0, 99), random.randint(0, 99)
        if grid[x][y] == 0 and (x, y) != (0, 0) and (x, y) != (50, 50):
            grid[x][y] = 1
            n_obstacles -= 1
    return grid

def create_grid_world(size=100, obstacle_ratio=ratio):
    grid = np.zeros((size, size), dtype=int)
    center = size // 2
    grid[center][center] = 2

    while True:
        grid_with_obstacles = add_obstacles(np.copy(grid), obstacle_ratio)
        path_exists, _ = is_path_clear(grid_with_obstacles, (0, 0), (50, 50))
        if path_exists:
            break

    return grid_with_obstacles

def save_grid(grid, filename="/home/sacchin/Desktop/dnt/RL Tutorials/"+str(ratio)+"grid_world.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(grid, f)

def plot_grid(grid, path):
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap='hot', interpolation='nearest')
    for (x, y) in path:
        plt.scatter(y, x, c='blue')
    plt.colorbar()
    plt.show()

grid_world = create_grid_world()
path_exists, path = is_path_clear(grid_world, (0, 0), (50, 50))
if path_exists:
    plot_grid(grid_world, path)
    save_grid(grid_world)
else:
    print("No path found")
