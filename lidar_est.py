import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from main import *

data_analyzer = DataAnalyzer('3D_Map/australia1.xyz', '3D_Map/australia1.png')
grid_vec = data_analyzer.grid_data
print(grid_vec.shape)
xyz_data = data_analyzer.xyz_data
np.random.seed(42)
lidar_data = pd.DataFrame({'X': xyz_data[:, 0], 'Y': xyz_data[:, 1], 'Z': xyz_data[:, 2]})
x_coords = xyz_data[:, 0]
y_coords = xyz_data[:, 1]
z_values = xyz_data[:, 2]
num_points = np.max(lidar_data['X'] + 1) * np.max(lidar_data['Y'] + 1)

grid_resolution = 100

from PIL import Image
import numpy as np


def extract_image_edges(image_path, x_extent, y_extent):
    image = Image.open(image_path)

    width, height = image.size

    x_min, x_max = x_extent
    y_min, y_max = y_extent

    x_edges = np.linspace(x_min, x_max, width + 1)
    y_edges = np.linspace(y_min, y_max, height + 1)

    return x_edges, y_edges


image_path = '3D_Map/australia1.png'

x_extent = (0, 255)
y_extent = (0, 255)
x_edges, y_edges = extract_image_edges(image_path, x_extent, y_extent)

print("X Edges:", x_edges)
print("Y Edges:", y_edges)

hist, x_edges, y_edges = np.histogram2d(x_coords, y_coords, bins=[x_edges, y_edges], weights=z_values)
counts, _, _ = np.histogram2d(x_coords, y_coords, bins=[x_edges, y_edges])

mean_elevation = np.divide(hist, counts, where=counts > 0)

mean_elevation = np.nan_to_num(mean_elevation, nan=0)

# plt.figure(figsize=(8, 6))
# plt.imshow(mean_elevation, extent=(0, 100, 0, 100), origin='lower', cmap='viridis')
# plt.colorbar(label='Mean Elevation (z)')
# plt.title("Rasterized LiDAR Data (Elevation)")
# plt.xlabel("X Coordinate")
# plt.ylabel("Y Coordinate")
# plt.show()

x_coords = np.linspace(0, 904, mean_elevation.shape[1])
y_coords = np.linspace(0, 1899, mean_elevation.shape[0])
x_grid, y_grid = np.meshgrid(x_coords, y_coords)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# surf = ax.plot_surface(x_grid, y_grid, mean_elevation, cmap='viridis', edgecolor='none')
#
# ax.set_xlabel('X Coordinate')
# ax.set_ylabel('Y Coordinate')
# ax.set_zlabel('Elevation')
#
# fig.colorbar(surf, shrink=0.5, aspect=5)
#
# plt.show()

# Load the original image
img = plt.imread('3D_Map/australia1.png')

# Create a 3D surface plot with texture
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(x_grid, y_grid, mean_elevation, cmap='viridis', edgecolor='none')

# Add texture to the 3D surface plot
ax.imshow(img, extent=(0, 904, 0, 1899), alpha=0.5)

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Elevation')

plt.show()