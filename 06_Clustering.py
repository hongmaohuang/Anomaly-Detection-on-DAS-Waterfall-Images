import numpy as np
from pathlib import Path
import os
import shutil
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import config

# recreate clustering results folder
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

model = None
predictions = None

# determine model
if method == "kmeans":
    model = KMeans(
        n_clusters=config.KMEANS_CLUSTERS,
        n_init="auto",
        random_state=4
    )
elif method == "gmm":
    select_method = getattr(config, "SELECT_CLUSTERS_WITH", "none").lower()
    if select_method in ("aic", "bic"):
        start, end = config.GMM_COMPONENT_RANGE
        best_score = np.inf
        best_gmm = None
        for n in range(start, end + 1):
            print(f"Trying GMM with {n} components...")
            gmm = GaussianMixture(
                n_components=n,
                covariance_type=config.GMM_COVARIANCE_TYPE,
                random_state=4,
            )
            gmm.fit(features)
            score = gmm.aic(features) if select_method == "aic" else gmm.bic(features)
            if score < best_score:
                best_score = score
                best_gmm = gmm
        model = best_gmm
        print(f"Best number of components determined by {select_method.upper()}: {model.n_components}")
    else:
        model = GaussianMixture(
            n_components=config.GMM_N_COMPONENTS,
            covariance_type=config.GMM_COVARIANCE_TYPE,
            random_state=4,
        )
elif method == "dbscan":
    model = DBSCAN(
        eps=config.DBSCAN_EPS,
        min_samples=config.DBSCAN_MIN_SAMPLES,
    )
elif method == "agglomerative":
    model = AgglomerativeClustering(
        n_clusters=config.AGG_N_CLUSTERS,
        linkage=config.AGG_LINKAGE,
    )
else:
    raise ValueError(f"Unknown clustering method: {config.CLUSTER_METHOD}")

# fit & predict
if method in ("kmeans", "dbscan", "agglomerative"):
    model.fit(features)
    if method == "kmeans":
        predictions = model.predict(features)
    else:
        predictions = model.labels_
else:  # GMM
    model.fit(features)
    predictions = model.predict(features)

# save results
out_path = Path(config.CLUSTERING_RESULTS_FOLDER) / "cluster_labels.dat"
combined = np.column_stack((distance, predictions))
header = "dist clusters"
np.savetxt(out_path, combined, fmt="%.3f %d", header=header, comments="")
