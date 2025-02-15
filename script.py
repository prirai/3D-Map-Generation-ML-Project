import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
# Load the XYZ data
xyz_data = np.loadtxt('3D_Map/australia1.xyz')

# Create a 2D grid with dimensions imgheight x imgwidth
imgheight = int(np.max(xyz_data[:, 0])) + 1
imgwidth = int(np.max(xyz_data[:, 1])) + 1
grid = np.zeros((imgheight, imgwidth))

# Assign Z values to pixels
# for x, y, z in xyz_data:
#     grid[int(x), int(y)] = z
#
# # Normalize or scale Z values (optional)
# grid_normalized = (grid - np.min(grid)) / (np.max(grid) - np.min(grid)) * 255
#
# # Create a 3D scatter plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(xyz_data[:, 0], xyz_data[:, 1], xyz_data[:, 2])
#
# # Show the plot
# plt.show()

# plt.hist(xyz_data[:, 2], bins=50)
# plt.xlabel('Z value')
# plt.ylabel('Frequency')
# plt.title('Histogram of Z values')
# plt.show()

# corr_xy = np.corrcoef(xyz_data[:, 0], xyz_data[:, 1])[0, 1]
# corr_xz = np.corrcoef(xyz_data[:, 0], xyz_data[:, 2])[0, 1]
# corr_yz = np.corrcoef(xyz_data[:, 1], xyz_data[:, 2])[0, 1]
#
# print("Correlation coefficients:")
# print(f"XY: {corr_xy:.4f}")
# print(f"XZ: {corr_xz:.4f}")
# print(f"YZ: {corr_yz:.4f}")

# imgheight = int(np.max(xyz_data[:, 0])) + 1
# imgwidth = int(np.max(xyz_data[:, 1])) + 1
# depth_map = np.zeros((imgheight, imgwidth))
#
# # Assign Z values to pixels
# for x, y, z in xyz_data:
#     depth_map[int(x), int(y)] = z
#
# # Visualize the depth map
# plt.imshow(depth_map, cmap='viridis')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Depth Map')
# plt.colorbar(label='Depth (Z)')
# plt.show()

plt.hist(xyz_data[:, 2], bins=50)
plt.xlabel('Depth (Z)')
plt.ylabel('Frequency')
plt.title('Histogram of Depth Values')
plt.show()