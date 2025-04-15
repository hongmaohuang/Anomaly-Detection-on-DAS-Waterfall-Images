from sklearn.decomposition import FastICA, PCA
import pickle
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path
import os
import config
import shutil
from sklearn.preprocessing import StandardScaler

if os.path.exists(config.PCA_ICA_FOLDER):
    shutil.rmtree(config.PCA_ICA_FOLDER)
os.makedirs(config.PCA_ICA_FOLDER)

# Load data from file
with np.load(f"{Path(config.SCATTERING_COEFFICIENTS_FOLDER)}/scattering_coefficients.npz", allow_pickle=True) as data:
    order_1 = data["order_1"]
    order_2 = data["order_2"]
    distance = data["distance"]

# Reshape and stack scattering coefficients of all orders
order_1 = order_1.reshape(order_1.shape[0], -1)
order_2 = order_2.reshape(order_2.shape[0], -1)
scattering_coefficients = np.hstack((order_1, order_2))
scattering = np.where(scattering_coefficients <= 0, 1e-10, scattering_coefficients)

# transform into log
scattering_coefficients = np.log10(scattering)


# 5. Normalize
scattering = StandardScaler().fit_transform(scattering_coefficients)

# print info about shape
n_distance, n_coeff = scattering_coefficients.shape

# Apply PCA
pca_model = PCA(n_components=config.PCA_COMPONENTS, whiten=True)
pca_features = pca_model.fit_transform(scattering_coefficients)

# Plot the result from PCA
# Normalize features for display
features_normalized = pca_features / np.abs(pca_features).max(axis=0)

# Figure and axes instance
fig, axes = plt.subplots(1,3,figsize=(15,5),dpi=300)
axes[0].plot(np.cumsum(pca_model.explained_variance_ratio_/sum(pca_model.explained_variance_ratio_)),'.-')
axes[0].set_ylabel("cumsum variance")
axes[0].set_xlabel("Feature index")
axes[0].set_title("Information content\nof each feature")

# Plot features in distance
axes[1].plot(distance, features_normalized + np.arange(features_normalized.shape[1]), rasterized=True)
axes[1].set_xlim(distance.min(),distance.max())
axes[1].set_ylabel("distance index")
axes[1].set_xlabel("Distance")
axes[1].set_title("Features in Distance")

# Plot the first two principal components
mappable = axes[2].scatter(pca_features[:,0],pca_features[:,1],s=1,c=distance)
axes[2].set_ylabel("Feature 1")
axes[2].set_xlabel("Feature 0")
axes[2].set_title("First two principal components")

cbar_ax = plt.colorbar(mappable)
cbar_ax.ax.set_ylim(distance.min(),distance.max())

# Show
plt.savefig(f"{config.PCA_ICA_FOLDER}/pca.png", dpi=300)

# Apply FastICA
# If the PCA analysis showed us that only a few components 
# contain relevant information and we can use that 
# knowledge to set the number of components from 
# PCA's n_components to a lower number for 
# the ICA analysis. 
#

ica_model = FastICA(n_components=config.ICA_COMPONENTS, whiten="unit-variance", random_state=42)
ica_features = ica_model.fit_transform(scattering_coefficients)

# Save the features
np.savez(
    f"{config.PCA_ICA_FOLDER}/independent_components.npz",
    features=ica_features,
    distance=distance,
)

# Save the dimension reduction model
with open(f"{config.PCA_ICA_FOLDER}/dimension_model.pickle", "wb") as pickle_file:
    pickle.dump(
        ica_model,
        pickle_file,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

# Normalize features for display
features_normalized = ica_features / np.abs(ica_features).max(axis=0)

# Plot the result from ICA
fig, axes = plt.subplots(1,2,figsize=(10,5),dpi=300)

# Plot features in Distance
axes[0].plot(distance, features_normalized + np.arange(ica_features.shape[1]), rasterized=True)
axes[0].set_xlim(distance.min(),distance.max())
axes[0].set_ylabel("Feature index")
axes[0].set_xlabel("Distance")
axes[0].set_title("Features in Distance")

# Plot the first two principal components
mappable = axes[1].scatter(ica_features[:,0],ica_features[:,1],s=1,c=distance)
axes[1].set_ylabel("Feature 1")
axes[1].set_xlabel("Feature 2")

cbar_ax = plt.colorbar(mappable)
cbar_ax.ax.set_ylim(distance.min(),distance.max())

# Show
plt.savefig(f"{config.PCA_ICA_FOLDER}/ica.png", dpi=300)
