import numpy as np
import matplotlib.pylab as plt
from pathlib import Path
import os
import shutil
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import obspy
import config
import glob

#if os.path.exists(config.CLUSTERING_RESULTS_FOLDER):
#    shutil.rmtree(config.CLUSTERING_RESULTS_FOLDER)
#os.makedirs(config.CLUSTERING_RESULTS_FOLDER)

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

# determine number of clusters & mapping
if method == "kmeans":
    n_clusters = config.KMEANS_CLUSTERS
elif method == "gmm":
    n_clusters = config.GMM_N_COMPONENTS
elif method == "agglomerative":
    n_clusters = config.AGG_N_CLUSTERS
else:  # dbscan: ignore noise label -1
    unique_labels = sorted(set(predictions) - {-1})
    n_clusters = len(unique_labels)

label_to_index = {
    lab: idx
    for idx, lab in enumerate(
        range(n_clusters)
        if method in ("kmeans", "gmm", "agglomerative")
        else unique_labels
    )
}

# one-hot encoding
one_hot = np.zeros((len(distance), n_clusters))
for i, lab in enumerate(predictions):
    if lab in label_to_index:
        one_hot[i, label_to_index[lab]] = 1

# plotting
# 1. Load waterfall npz
npz_folder = config.WATERFALL_NPZ_FOLDER
npz_files  = sorted(glob.glob(os.path.join(npz_folder, '*.npz')))
waterfall = np.load(npz_files[0])['waterfall']   # shape: (time_pix, chan_pix)
waterfall = waterfall.mean(axis=2)

# waterfall extent（
n_time, n_chan     = waterfall.shape
total_sec         = config.DURATION_WATERFALL * 60
x_min, x_max      = distance.min(), distance.max()
extent = [x_min, x_max, 0, total_sec]

# DAS waterfall
fig, ax_wf = plt.subplots(figsize=(12, 6))
im = ax_wf.imshow(
    waterfall,
    cmap='jet',
    aspect='auto',
    extent=extent,
    origin='lower',
    alpha=0.6,
)
ax_wf.set_xlabel("Distance (km)")
ax_wf.set_ylabel("Time (sec)")
ax_wf.set_ylim(total_sec, 0)

# cluster Results
ax_cl = ax_wf.twinx()
colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
for i in range(n_clusters):
    dr = np.convolve(
        one_hot[:, i],
        np.ones(config.SMOOTH_KERNEL),
        mode="same"
    ) / config.SMOOTH_KERNEL

    baseline = i
    ax_cl.fill_between(
        distance,
        baseline,
        baseline + dr,
        color=colors[i],
        alpha=0.85,
        label=f"Cluster {i}"
    )

ax_cl.set_ylabel("Cluster index")
ax_cl.set_ylim(-0.5, n_clusters-0.5)
ax_cl.set_yticks(range(n_clusters))
ax_cl.set_yticklabels([f"{i}" for i in range(n_clusters)])
ax_cl.legend(loc="upper right", title="Clusters")

plt.tight_layout()


# Extract the date and time from the input filename
input_files = os.listdir(config.DAS_WATERFALL_PATH)
input_file = input_files[0]  # i.e. "Waterfall_RMS_20250331_020545.png"
file_parts = input_file.split('_')
date_part = file_parts[2]  # i.e. "20250331"
time_part = file_parts[3].split('.')[0]  # i.e. "020545"
output_filename = f"clustering_result_{date_part}_{time_part}_{config.DURATION_WATERFALL}_min.png"
output_path = os.path.join(config.CLUSTERING_RESULTS_FOLDER, output_filename)

# Save the clustering results
plt.savefig(
    output_path,
    dpi=300
)
