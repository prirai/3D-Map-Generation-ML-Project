#%%
from main import *
data_analyzer = DataAnalyzer('3D_Map/australia1.xyz', '3D_Map/australia1.png')
data_analyzer.plot_data()

#%%
import numpy as np

xyz_data = data_analyzer.xyz_data
image_data = np.array(data_analyzer.image_data)

#%%

from scipy.sparse.linalg import svds

def compute_triangulation_matrix(xyz_points, image_points):
    # Create the DLT matrix
    A = np.zeros((2 * len(xyz_points), 12))
    for i, (xyz, image) in enumerate(zip(xyz_points, image_points)):
        x, y, z = xyz
        u, v = image
        A[2 * i] = [x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u]
        A[2 * i + 1] = [0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v]

    # Compute the SVD of the DLT matrix
    # U, S, Vt = np.linalg.svd(A)
    U, S, Vt = svds(A, k=A.shape)

    # Extract the triangulation matrix from the SVD
    triangulation_matrix = Vt[-1].reshape(3, 4)

    return triangulation_matrix
import pandas as pd
df = pd.DataFrame({'X': xyz_data[:, 0], 'Y': xyz_data[:, 1], 'Z': xyz_data[:, 2]})
# print(np.array(df[['X', 'Y']]))
xy_data = np.array(df[['X', 'Y']])
z_data = np.array(df[['Z']])
#%%
triangulation_matrix = compute_triangulation_matrix(xyz_data, xy_data)

#%%

# Compute the reprojection error
def compute_reprojection_error(triangulation_matrix, xyz_points, image_points):
    # Reproject the XYZ points onto the image plane
    reprojected_points = np.dot(triangulation_matrix, xyz_points)

    # Compute the reprojection error
    reprojection_error = np.linalg.norm(reprojected_points - image_points, axis=1)

    return reprojection_error


reprojection_error = compute_reprojection_error(triangulation_matrix, xyz_data, image_data)


# Estimate alpha
def estimate_alpha(reprojection_error):
    alpha = np.mean(reprojection_error)
    return alpha


alpha = estimate_alpha(reprojection_error)
print(f'Estimated alpha: {alpha}')