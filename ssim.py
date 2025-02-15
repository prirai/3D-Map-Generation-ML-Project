#%%

import pickle
from main import *

data_analyzer = DataAnalyzer('3D_Map/australia1.xyz', '3D_Map/australia1.png')
grid_vec = data_analyzer.grid_data.flatten()
image_vec = data_analyzer.image_data.convert('L')
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

import pandas as pd
xyz_data = data_analyzer.xyz_data
df = pd.DataFrame({'X': xyz_data[:, 0], 'Y': xyz_data[:, 1], 'Z': xyz_data[:, 2]})
df['img_gray'] = np.array(image_vec).flatten()
df['Z_X'] = np.where(df['X'] > 0, df['Z'] / df['X'], 0)
df['Z_Y'] = np.where(df['Y'] > 0, df['Z'] / df['Y'], 0)
df['X_Y'] = np.where(df['Y'] > 0, df['X'] / df['Y'], 0)
df['Z_X_Y'] = np.where((df['Y'] > 0) & (df['X'] > 0), df['Z'] / (df['X'] * df['Y']), 0)
df['Z_normalized'] = (df['Z'] - df['Z'].min()) / (df['Z'].max() - df['Z'].min())
df['X_normalized'] = (df['X'] - df['X'].min()) / (df['X'].max() - df['X'].min())
df['Y_normalized'] = (df['Y'] - df['Y'].min()) / (df['Y'].max() - df['Y'].min())
df['geom_dist'] = np.sqrt(np.sqrt(df['X_normalized']**2 + df['Y_normalized']**2)**2 + df['Z_normalized']**2)

df['X'] = df['X'].astype(int)
df['Y'] = df['Y'].astype(int)

X = df.drop(['img_gray'], axis=1)
y = df['img_gray']

#%%

result = loaded_model.score(X, y)
print(result)

#%%
preds = loaded_model.predict(X)
preds.shape

#%%
from skimage.metrics import structural_similarity as ssim
img_mat = np.array(image_vec, dtype=np.float32)

preds_reshaped = np.reshape(preds, (img_mat.shape[0], img_mat.shape[1]))
# print(img_mat.shape)
ssim_score = ssim(img_mat, preds_reshaped, data_range=1.0)

print("SSIM Score:", ssim_score)