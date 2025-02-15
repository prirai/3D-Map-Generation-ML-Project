import pandas as pd

from main import *

data_analyzer = DataAnalyzer('3D_Map/australia1.xyz', '3D_Map/australia1.png')
# data_analyzer.plot_data()
grid_vec = data_analyzer.grid_data
print(grid_vec.shape)
xyz_data = data_analyzer.xyz_data
df = pd.DataFrame({'X': xyz_data[:, 0], 'Y': xyz_data[:, 1], 'Z': xyz_data[:, 2]})
xyz_points = df.to_numpy()
import matplotlib.pyplot as plt
# plt.imshow(grid_vec, cmap='gray')
# plt.show()

import open3d as o3d

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz_points)

# Estimate the normals for the point cloud
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Perform Poisson Surface Reconstruction
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

# Visualize the reconstructed mesh
# o3d.visualization.draw_geometries([mesh])
o3d.io.write_triangle_mesh("reconstructed_mesh.ply", mesh)