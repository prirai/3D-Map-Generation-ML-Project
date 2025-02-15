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

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error

# Assuming df is your DataFrame, and you have columns 'X', 'Y', and 'img_gray'

# Preparing the data
df['X'] = df['X'].astype(int)
df['Y'] = df['Y'].astype(int)

X = df.drop(['img_gray'], axis=1)
y = df['img_gray']

# K-Fold cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation
fold_predictions = np.zeros_like(y)  # Array to store the final predictions for each fold
model = CatBoostRegressor(eval_metric='RMSE', loss_function='RMSE', random_seed=42)

# Performing K-Fold cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Training fold {fold + 1}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Train the model
    model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=[0, 1], verbose=10)

    # Get predictions for the validation set
    fold_preds = model.predict(X_val)

    # Insert the fold predictions into the correct place in the array (using val_idx)
    fold_predictions[val_idx] = fold_preds

# Now fold_predictions contains the aggregated predictions (i.e., the predictions for each sample from each fold)
#%%
# fold_predictions
fold_grid = np.reshape(fold_predictions, (904, 1899))
plt.imshow(fold_grid, cmap='gray')
plt.show()

#%%
import pickle
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
#%%
print(model.feature_names_)
model.feature_importances_

#%%
import pickle
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_val, y_val) # rmse
print(result)

#%%

from skimage.metrics import structural_similarity as ssim
import numpy as np

# Load the original image
img1 = plt.imread('3D_Map/australia1.png')

# Load the generated 3D map image
img2 = plt.imread('generated_3d_map.png')

# Convert images to grayscale
img1_gray = np.dot(img1[...,:3], [0.2989, 0.5870, 0.1140])
img2_gray = np.dot(img2[...,:3], [0.2989, 0.5870, 0.1140])

# Compute SSIM score
ssim_score = ssim(img1_gray, img2_gray)

print("SSIM Score:", ssim_score)

#%%
# Reshape the final predictions and ground truth for SSIM comparison
final_predictions_reshaped = fold_predictions.reshape(1, -1)
y_test_reshaped = y.values.reshape(1, -1)

# Calculate SSIM between the aggregated predictions and the ground truth
ssim_result = ssim(y_test_reshaped, final_predictions_reshaped, data_range=1,
                   win_size=3)  # Adjust win_size if necessary
print(f"SSIM: {ssim_result}")

# Calculate Mean Squared Error for regression evaluation
mse = mean_squared_error(y, fold_predictions)
print(f"Mean Squared Error: {mse}")
