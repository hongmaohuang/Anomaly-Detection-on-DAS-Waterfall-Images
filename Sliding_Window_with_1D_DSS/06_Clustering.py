# %%
# Clustering
from sklearn.cluster import KMeans
from scipy import signal
import pickle
import matplotlib.pylab as plt
import numpy as np
import obspy
plt.rcParams["date.converter"] = "concise"
import matplotlib.dates as mdates
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)

OUT_DIR_pca = Path("../../Outputs/pca_ica")

with np.load(f"{OUT_DIR_pca}/independent_components.npz", allow_pickle=True) as data:
    features = data["features"]
    distance = data["distance"]

N_CLUSTERS = 4

# Perform clustering
model = KMeans(n_clusters=N_CLUSTERS, n_init="auto", random_state=4)
model.fit(features)

# Predict cluster for each sample
predictions = model.predict(features)

SMOOTH_KERNEL = 20

# Convert predictions to one-hot encoding
one_hot = np.zeros((len(distance), N_CLUSTERS + 1))
one_hot[np.arange(len(distance)), predictions] = 1

fig, ax = plt.subplots(figsize=(12, 6))

colors = plt.cm.viridis(np.linspace(0, 1, N_CLUSTERS))
spacing = 2.0 

for i in range(N_CLUSTERS):
    detection_rate = np.convolve(one_hot[:, i], np.ones(SMOOTH_KERNEL), mode="same") / SMOOTH_KERNEL
    baseline = i * spacing
    ax.fill_between(distance, baseline, detection_rate + baseline, color=colors[i], alpha=0.7, label=f"Cluster {i}")

ax.set_xlim(distance[0], distance[-1])

ax.set_xlabel("Distance")
ax.set_ylabel("Cluster index")
ax.set_yticks([i * spacing for i in range(N_CLUSTERS)])
ax.set_yticklabels([f"Cluster {i}" for i in range(N_CLUSTERS)])
ax.legend(title="Clusters")
plt.tight_layout()
plt.show()