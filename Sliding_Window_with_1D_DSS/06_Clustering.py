# %%
# Clustering
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
import numpy as np
import obspy
from pathlib import Path
import os
import config
import shutil

if os.path.exists(config.CLUSTERING_RESULTS_FOLDER):
    shutil.rmtree(config.CLUSTERING_RESULTS_FOLDER)
os.makedirs(config.CLUSTERING_RESULTS_FOLDER)

with np.load(f"{Path(config.PCA_ICA_FOLDER)}/independent_components.npz", allow_pickle=True) as data:
    features = data["features"]
    distance = data["distance"]

# Perform clustering
model = KMeans(n_clusters=config.N_CLUSTERS, n_init="auto", random_state=4)
model.fit(features)

# Predict cluster for each sample
predictions = model.predict(features)

# Convert predictions to one-hot encoding
one_hot = np.zeros((len(distance), config.N_CLUSTERS + 1))
one_hot[np.arange(len(distance)), predictions] = 1

fig, ax = plt.subplots(figsize=(12, 6))

colors = plt.cm.viridis(np.linspace(0, 1, config.N_CLUSTERS))
spacing = 2.0 

for i in range(config.N_CLUSTERS):
    detection_rate = np.convolve(one_hot[:, i], np.ones(config.SMOOTH_KERNEL), mode="same") / config.SMOOTH_KERNEL
    baseline = i * spacing
    ax.fill_between(distance, baseline, detection_rate + baseline, color=colors[i], alpha=0.7, label=f"Cluster {i}")

ax.set_xlim(distance[0], distance[-1])
ax.set_xlabel("Distance")
ax.set_ylabel("Cluster index")
ax.set_yticks([i * spacing for i in range(config.N_CLUSTERS)])
ax.set_yticklabels([f"Cluster {i}" for i in range(config.N_CLUSTERS)])
ax.legend(loc='lower right', title="Clusters")
plt.tight_layout()
plt.savefig(f"{Path(config.CLUSTERING_RESULTS_FOLDER)}/clustering_result.png", dpi=300)