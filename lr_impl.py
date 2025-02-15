#%%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from main import *

basepath = '3D_Map'
casepath = 'japan1'
# print(f'3D_Map/{casepath}1.png', f'3D_Map/{casepath}1.png')
data_analyzer = DataAnalyzer(f'3D_Map/{casepath}.xyz', f'3D_Map/{casepath}.png')
image_vec = np.array(data_analyzer.image_data.convert('L')).flatten()

xyz_data = data_analyzer.xyz_data
df = pd.DataFrame({'X': xyz_data[:, 0], 'Y': xyz_data[:, 1], 'Z': xyz_data[:, 2], 'gray_pixel':image_vec})

max_x = int(df['X'].max() + 1)
max_y = int(df['Y'].max() + 1)

print(max_x, max_y)
xyz_data = np.zeros((max_x, max_y, 3))
gray_pixel_data = np.zeros((max_x, max_y))

for index, row in df.iterrows():
    x, y, z, gray_pixel = int(row['X']), int(row['Y']), row['Z'], row['gray_pixel']
    xyz_data[x, y] = [x, y, z]
    gray_pixel_data[x, y] = gray_pixel

# Divide the data into 16x16 patches
patch_size = 16
num_patches_x = int(np.ceil(max_x / patch_size))
num_patches_y = int(np.ceil(max_y / patch_size))

xyz_patches = np.zeros((num_patches_x * num_patches_y, patch_size, patch_size, 3))
gray_pixel_patches = np.zeros((num_patches_x * num_patches_y, patch_size, patch_size))

patch_index = 0
for i in range(num_patches_x):
    for j in range(num_patches_y):
        xyz_patch = xyz_data[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
        gray_pixel_patch = gray_pixel_data[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]

        # Pad the patch with zeros if necessary
        if xyz_patch.shape[0] < patch_size or xyz_patch.shape[1] < patch_size:
            padded_xyz_patch = np.zeros((patch_size, patch_size, 3))
            padded_gray_pixel_patch = np.zeros((patch_size, patch_size))

            padded_xyz_patch[:xyz_patch.shape[0], :xyz_patch.shape[1]] = xyz_patch
            padded_gray_pixel_patch[:gray_pixel_patch.shape[0], :gray_pixel_patch.shape[1]] = gray_pixel_patch

            xyz_patches[patch_index] = padded_xyz_patch
            gray_pixel_patches[patch_index] = padded_gray_pixel_patch
        else:
            xyz_patches[patch_index] = xyz_patch
            gray_pixel_patches[patch_index] = gray_pixel_patch

        patch_index += 1

xyz_patches_1d = xyz_patches.reshape(-1, patch_size * patch_size * 3)
gray_pixel_patches_1d = gray_pixel_patches.reshape(-1, patch_size * patch_size)

num_patches = xyz_patches_1d.shape[0]
models = []
for i in range(num_patches):
    xyz_patch = xyz_patches_1d[i]
    gray_pixel_patch = gray_pixel_patches_1d[i]

    # Train a linear regression model on the patch
    model = LinearRegression()
    model.fit(xyz_patch.reshape(-1, 3), gray_pixel_patch)

    # Append the trained model to the list of models
    models.append(model)

# Test the models on the same data
predicted_gray_pixel_data = np.zeros((max_x, max_y))
for i in range(num_patches_x):
    for j in range(num_patches_y):
        patch_index = i * num_patches_y + j
        model = models[patch_index]
        xyz_patch = xyz_data[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
        xyz_patch_1d = xyz_patch.reshape(-1, 3)
        predicted_gray_pixel_patch = model.predict(xyz_patch_1d)

        # Reshape the predicted patch based on its actual size
        patch_height, patch_width = xyz_patch.shape[:2]
        predicted_gray_pixel_data[i * patch_size:i * patch_size + patch_height,
        j * patch_size:j * patch_size + patch_width] = predicted_gray_pixel_patch.reshape(patch_height, patch_width)

#%%
predicted_gray_pixel_data


#%%
# Visualize the predicted gray pixel data
plt.imshow(predicted_gray_pixel_data, cmap='gray')
plt.axis('off')
plt.savefig(f'{casepath}_lr_16_patches.png', dpi=1200, bbox_inches='tight')