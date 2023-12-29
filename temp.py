import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_grid_from_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def plot_grid_with_numbers(grid):
    plt.figure(figsize=(12, 12))
    plt.imshow(grid, cmap='viridis')
    for (j, i), value in np.ndenumerate(grid):
        plt.text(i, j, int(value), ha='center', va='center', color='white')
    plt.colorbar()
    plt.show()

def plot_grid_as_heatmap(grid):
    plt.figure(figsize=(12, 12))
    heatmap = plt.imshow(grid, cmap='hot')
    plt.colorbar(heatmap)
    plt.show()

# Load the grid world from the pickle file
filename = '/home/sacchin/Desktop/dnt/RL Tutorials/Grid Worlds/0.45grid_world.pkl'  # Replace with your pickle file name
grid_world = load_grid_from_pickle(filename)

# # Plot the grid with numbers
# plot_grid_with_numbers(grid_world)

# Plot the grid as a heatmap
plot_grid_as_heatmap(grid_world)