# %%
# Dimensionality reduction
#
from sklearn.decomposition import FastICA, PCA
import pickle
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import numpy as np
plt.rcParams["date.converter"] = "concise"
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)

OUT_DIR_scattering = Path("../../Outputs/scattering_coefficients")
OUT_DIR_pca = Path("../../Outputs/pca_ica")
OUT_DIR_pca.mkdir(parents=True, exist_ok=True)



# Load data from file
with np.load(f"{OUT_DIR_scattering}/scattering_coefficients.npz", allow_pickle=True) as data:
    order_1 = data["order_1"]
    order_2 = data["order_2"]
    distance = data["distance"]

# Reshape and stack scattering coefficients of all orders
order_1 = order_1.reshape(order_1.shape[0], -1)
order_2 = order_2.reshape(order_2.shape[0], -1)
scattering_coefficients = np.hstack((order_1, order_2))

# transform into log
scattering_coefficients = np.log10(scattering_coefficients)

# print info about shape
n_distance, n_coeff = scattering_coefficients.shape
print("Collected {} samples of {} dimensions each.".format(n_distance, n_coeff))

# Apply PCA
#
pca_model = PCA(n_components=10, whiten=True)
pca_features = pca_model.fit_transform(scattering_coefficients)

# Plot the result from PCA
# Normalize features for display
features_normalized = pca_features / np.abs(pca_features).max(axis=0)
myFmt = mdates.DateFormatter('%HH:%MM')

# Figure and axes instance
fig, axes = plt.subplots(1,3,figsize=(15,5),dpi=200)

# Plot the cumulative sum of the explained variance ratio of the principal components
axes[0].plot(np.cumsum(pca_model.explained_variance_ratio_/sum(pca_model.explained_variance_ratio_)),'.-')
axes[0].set_ylabel("cumsum variance")
axes[0].set_xlabel("Feature index")
axes[0].set_title("Information content\nof each feature")

# Plot features in time
axes[1].plot(distance, features_normalized + np.arange(features_normalized.shape[1]), rasterized=True)
axes[1].set_xlim(distance.min(),distance.max())
axes[1].xaxis.set_major_formatter(myFmt)
axes[1].xaxis.set_major_locator(mdates.HourLocator(interval=6))
axes[1].set_ylabel("Feature index")
axes[1].set_xlabel("UTC Time of 02/03/2025")
axes[1].set_title("Features in time")

# Plot the first two principal components
mappable = axes[2].scatter(pca_features[:,0],pca_features[:,1],s=1,c=distance)
axes[2].set_ylabel("Feature 1")
axes[2].set_xlabel("Feature 0")
axes[2].set_title("First two principal components")

cbar_ax = plt.colorbar(mappable)
cbar_ax.ax.set_ylim(distance.min(),distance.max())
cbar_ax.ax.yaxis.set_major_formatter(myFmt)
cbar_ax.ax.yaxis.set_major_locator(mdates.HourLocator(interval=6))

# Show
plt.show()

# Apply FastICA
# If the PCA analysis showed us that only a few components 
# contain relevant information and we can use that 
# knowledge to set the number of components from 
# PCA's n_components to a lower number for 
# the ICA analysis. 
#

ica_model = FastICA(n_components=3, whiten="unit-variance", random_state=42)
ica_features = ica_model.fit_transform(scattering_coefficients)

# Save the features
np.savez(
    f"{OUT_DIR_pca}/independent_components.npz",
    features=ica_features,
    distance=distance,
)

# Save the dimension reduction model
with open(f"{OUT_DIR_pca}/dimension_model.pickle", "wb") as pickle_file:
    pickle.dump(
        ica_model,
        pickle_file,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

# Normalize features for display
features_normalized = ica_features / np.abs(ica_features).max(axis=0)
myFmt = mdates.DateFormatter('%HH:%MM')

# Plot the result from ICA
fig, axes = plt.subplots(1,2,figsize=(10,5),dpi=200)

# Plot features in time
axes[0].plot(distance, features_normalized + np.arange(ica_features.shape[1]), rasterized=True)
axes[0].set_xlim(distance.min(),distance.max())
axes[0].xaxis.set_major_formatter(myFmt)
axes[0].xaxis.set_major_locator(mdates.HourLocator(interval=6))
axes[0].set_ylabel("Feature index")
axes[0].set_xlabel("UTC Time of 02/03/2025")

axes[0].set_title("Features in time")


# Plot the first two principal components
mappable = axes[1].scatter(ica_features[:,0],ica_features[:,1],s=1,c=distance)
axes[1].set_ylabel("Feature 1")
axes[1].set_xlabel("Feature 2")

cbar_ax = plt.colorbar(mappable)
cbar_ax.ax.set_ylim(distance.min(),distance.max())
cbar_ax.ax.yaxis.set_major_formatter(myFmt)
cbar_ax.ax.yaxis.set_major_locator(mdates.HourLocator(interval=6))

# Show
plt.show()
