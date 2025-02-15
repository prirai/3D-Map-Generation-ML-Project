#%%
import pandas as pd

from main import *

data_analyzer = DataAnalyzer('3D_Map/australia1.xyz', '3D_Map/australia1.png')
# data_analyzer.plot_data()
grid_vec = data_analyzer.grid_data.flatten()
image_vec = np.array(data_analyzer.image_data.convert('L')).flatten()
print(grid_vec.shape, image_vec.shape)

#%%
import pandas as pd
xyz_data = data_analyzer.xyz_data
df = pd.DataFrame({'X': xyz_data[:, 0], 'Y': xyz_data[:, 1], 'Z': xyz_data[:, 2]})
print(df)

#%%
df['img_gray'] = image_vec
df['Z'].value_counts()

#%%
plt.scatter(df['X'], df['Z'], c=df['img_gray'])
plt.show()

#%%
plt.scatter(df['Z'], df['img_gray'], c=df['img_gray'], cmap='gray', s=1)
plt.show()

#%%
pt = {}
for z, p in zip(df['Z'], df['img_gray']):
    pt[z] = p

df['mapped'] = df['Z'].map(pt)

#%%
mapped_array = df['mapped'].to_numpy()
reshaped_array = mapped_array.reshape(904, 1899)

# Plot the array using plt.imshow()
plt.imshow(reshaped_array, cmap='gray')
plt.show()

#%%
df['Z_X'] = np.where(df['X'] > 0, df['Z'] / df['X'], 0)
df['Z_Y'] = np.where(df['Y'] > 0, df['Z'] / df['Y'], 0)
df['X_Y'] = np.where(df['Y'] > 0, df['X'] / df['Y'], 0)
df['Z_X_Y'] = np.where((df['Y'] > 0) & (df['X'] > 0), df['Z'] / (df['X'] * df['Y']), 0)
df['Z_normalized'] = (df['Z'] - df['Z'].min()) / (df['Z'].max() - df['Z'].min())
df['X_normalized'] = (df['X'] - df['X'].min()) / (df['X'].max() - df['X'].min())
df['Y_normalized'] = (df['Y'] - df['Y'].min()) / (df['Y'].max() - df['Y'].min())
df['geom_dist'] = np.sqrt(np.sqrt(df['X_normalized']**2 + df['Y_normalized']**2)**2 + df['Z_normalized']**2)
df

#%%

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error  # You'll likely want MSE or another standard regression metric

# Assuming 'df' is your DataFrame and 'target_column' is the name of your target variable
X = df.drop('img_gray', axis=1)
y = df['img_gray']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Adjust test_size as needed

def ssim_metric(y_true, y_pred):
    """Calculates the SSIM between y_true and y_pred.

    Reshapes the input arrays to 2D if they are 1D for SSIM calculation.
    Handles potential edge cases where y_true and y_pred have different shapes
    or are single-valued.

    Args:
        y_true: Ground truth values (numpy array or list).
        y_pred: Predicted values (numpy array or list).

    Returns:
        A tuple: ('SSIM', ssim_score, True).
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)  # Reshape to 2D if it's 1D

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    if y_true.shape != y_pred.shape:
        if y_true.size == 1 and y_pred.size > 1:
            y_true = np.repeat(y_true, y_pred.shape[0]).reshape(-1, 1)  # expand dimension of y_true to match y_pred
        elif y_true.size > 1 and y_pred.size == 1:
            y_pred = np.repeat(y_pred, y_true.shape[0]).reshape(-1, 1)  # expand dimension of y_pred to match y_true
        elif y_true.size == 1 and y_pred.size == 1:  # handle case when both become 0, after train and eval set are made
            y_pred = y_pred.reshape(-1, 1)  # expand to 2D
            y_true = y_true.reshape(-1, 1)
        else:
            raise ValueError("Input arrays must have compatible shapes for SSIM calculation.")

    ssim_score = ssim(y_true, y_pred, multichannel=False,
                      data_range=y_true.max() - y_true.min())  # multichannel=False for grayscale

    return 'SSIM', ssim_score, True

model = CatBoostRegressor(eval_metric=ssim_metric, loss_function='RMSE', random_seed=42) # RMSE is a standard regression loss

model.fit(X_train, y_train, eval_set=(X_test, y_test), cat_features=[], verbose=100)  # Empty list for categorical features

y_pred = model.predict(X_test)

ssim_result = ssim_metric(y_test, y_pred)
print(f"SSIM: {ssim_result[1]}")

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")