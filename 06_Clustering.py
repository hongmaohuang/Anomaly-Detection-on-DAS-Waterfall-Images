import numpy as np
import matplotlib.pylab as plt
from pathlib import Path
import os
import shutil
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import obspy
import glob
import config
import sys

if os.path.exists(config.CLUSTERING_RESULTS_FOLDER):
    shutil.rmtree(config.CLUSTERING_RESULTS_FOLDER)
os.makedirs(config.CLUSTERING_RESULTS_FOLDER)

# load features & distance
with np.load(f"{Path(config.PCA_ICA_FOLDER)}/independent_components.npz", allow_pickle=True) as data:
    features = data["features"]
    distance = data["distance"]

# clustering method
method = config.CLUSTER_METHOD.lower()  # 'kmeans', 'gmm', 'dbscan', or 'agglomerative'
print(f"\nClustering method: {method}\n")
print("Start Clustering!")
#print(f"Number of samples: {features.shape[0]}")
#print(f"Feature dimension: {features.shape[1]}\n")

# select model
if method == "kmeans":
    model = KMeans(
        n_clusters=config.KMEANS_CLUSTERS,
        n_init="auto",
        random_state=4
    )
elif method == "gmm":
    model = GaussianMixture(
        n_components=config.GMM_N_COMPONENTS,
        covariance_type=config.GMM_COVARIANCE_TYPE,
        random_state=4
    )
elif method == "dbscan":
    model = DBSCAN(
        eps=config.DBSCAN_EPS,
        min_samples=config.DBSCAN_MIN_SAMPLES
    )
elif method == "agglomerative":
    model = AgglomerativeClustering(
        n_clusters=config.AGG_N_CLUSTERS,
        linkage=config.AGG_LINKAGE
    )
else:
    raise ValueError(f"Unknown clustering method: {config.CLUSTER_METHOD}")

# fit & predict
if method in ("kmeans", "dbscan", "agglomerative"):
    model.fit(features)
    if method == "kmeans":
        predictions = model.predict(features)
    else:  # DBSCAN or AgglomerativeClustering
        predictions = model.labels_
else:  # GMM
    model.fit(features)
    predictions = model.predict(features)

out_path = Path(config.CLUSTERING_RESULTS_FOLDER) / f"cluster_labels.dat"
combined = np.column_stack((distance, predictions))
header = "dist clusters"
np.savetxt(out_path, combined, fmt="%.3f %d", header=header, comments="")