# %%
from pathlib import Path
import config
import numpy as np
import matplotlib.pyplot as plt
import os

with np.load(f"{Path(config.PCA_ICA_FOLDER)}/independent_components.npz", allow_pickle=True) as data:
    features = data["features"]
    distance = data["distance"]

method = config.CLUSTER_METHOD.lower()  

in_path = Path(config.CLUSTERING_RESULTS_FOLDER) / f"cluster_labels.dat"
predictions = np.loadtxt(in_path, usecols=1, dtype=int, skiprows=1)

fig, ax = plt.subplots(figsize=(12, 5))
unique_labels = np.unique(predictions)
cmap = plt.cm.get_cmap("tab20", len(unique_labels)) 

for idx, lab in enumerate(unique_labels):
    mask = predictions == lab
    ax.scatter(
        distance[mask],           
        np.zeros(mask.sum()),
        s=12,
        color=cmap(idx),
        label=f"Cluster {lab}"
    )
ax.set_yticks([])                

files = sorted(os.listdir(config.DAS_WATERFALL_PATH))  
total = len(files)

for n, input_file in enumerate(files, start=1):
    print(f"[{n}/{total}] for clusters only")
    start_by = config.TOTAL_DISTANCE_KM * (n - 1)
    end_by   = config.TOTAL_DISTANCE_KM *  n
    ax.set_xlim(start_by, end_by)
    parts = input_file.split('_')
    date_part = parts[2]                 # e.g. 20250331
    time_part = parts[3].split('.')[0]   # e.g. 020545
    ax.set_title(f"{date_part} {time_part}")
    ax.set_xticks([start_by, end_by])
    ax.set_xticklabels(['0', f'{config.TOTAL_DISTANCE_KM}'])
    
    output_filename = f"clustering_result_{date_part}_{time_part}.png"
    output_path = os.path.join(config.CLUSTERING_RESULTS_FOLDER, output_filename)
    ax.set_xlabel("Distance (km)")
    ax.legend(loc="lower right")
    fig.savefig(output_path, dpi=300)

plt.close(fig)

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
npz_folder = Path(config.WATERFALL_NPZ_FOLDER)
npz_files  = sorted(npz_folder.glob("*.npz"))
waterfalls = []
for fp in npz_files:
    arr = np.load(fp)["waterfall"].mean(axis=2)   
    waterfalls.append(arr)
big_waterfall = np.concatenate(waterfalls, axis=1)

# waterfall extent（
n_time, n_chan     = big_waterfall.shape
total_sec         = config.DURATION_WATERFALL * 60
x_min, x_max      = distance.min(), distance.max()
extent = [x_min, x_max, 0, total_sec]

fig, ax_wf = plt.subplots(figsize=(12, 6))
im = ax_wf.imshow(
    big_waterfall,
    cmap="gray",
    aspect="auto",
    extent=[distance.min(), distance.max(), 0, total_sec],
    origin="lower",
    alpha=0.6,
)
ax_wf.set_xlabel("Distance (km)")
ax_wf.set_ylabel("Time (sec)")
ax_wf.set_ylim(total_sec, 0)

ax_cl = ax_wf.twinx()                        
colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))

for i in range(n_clusters):
    idx = one_hot[:, i].astype(bool)        
    ax_cl.scatter(
        distance[idx],                       
        np.zeros(idx.sum()),               
        color=colors[i],
        s=10,
        label=f"Cluster {i}",
        zorder=2
    )

ax_cl.set_ylim(-0.5, 0.5)                    
ax_cl.set_yticks([])                         
ax_cl.set_ylabel("")                         
ax_cl.legend(loc="upper right", title="Clusters")

plt.tight_layout()

files = sorted(os.listdir(config.DAS_WATERFALL_PATH))
total = len(files)

for n, input_file in enumerate(files, start=1):
    print(f"[{n}/{total}] for waterfall overlapped")
    start_by = config.TOTAL_DISTANCE_KM * (n - 1)
    end_by   = config.TOTAL_DISTANCE_KM * n
    ax_wf.set_xlim(start_by, end_by)
    ax_wf.set_xticks([start_by, end_by])
    ax_wf.set_xticklabels(['0', f'{config.TOTAL_DISTANCE_KM}'])

    parts = input_file.split('_')
    date_part, time_part = parts[2], parts[3].split('.')[0]
    ax_wf.set_title(f"{date_part} {time_part}")

    output_filename = f"[with_wtf]clustering_result_{date_part}_{time_part}.png"
    output_path = os.path.join(config.CLUSTERING_RESULTS_FOLDER, output_filename)
    fig.savefig(output_path, dpi=300)

plt.close(fig)


''' 
#One-hot images
# DAS waterfall
fig, ax_wf = plt.subplots(figsize=(12, 6))
im = ax_wf.imshow(
    big_waterfall,
    cmap='gray',
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

first_start_by = 0                          # = config.TOTAL_DISTANCE_KM * (1-1)
first_end_by   = config.TOTAL_DISTANCE_KM   # = config.TOTAL_DISTANCE_KM * 1

files = sorted(os.listdir(config.DAS_WATERFALL_PATH))  
total = len(files)

for n, input_file in enumerate(files, start=1):
    print(f"[{n}/{total}]")
    start_by = config.TOTAL_DISTANCE_KM * (n - 1)
    end_by   = config.TOTAL_DISTANCE_KM *  n
    ax_wf.set_xlim(start_by, end_by)

    parts = input_file.split('_')
    date_part = parts[2]                 # e.g. 20250331
    time_part = parts[3].split('.')[0]   # e.g. 020545
    ax_wf.set_title(f"{date_part} {time_part}")

    ax_wf.set_xticks([start_by, end_by])
    ax_wf.set_xticklabels(['0', f'{config.TOTAL_DISTANCE_KM}'])
    
    output_filename = f"clustering_result_{date_part}_{time_part}.png"
    output_path = os.path.join(config.CLUSTERING_RESULTS_FOLDER, output_filename)
    fig.savefig(output_path, dpi=300)

plt.close(fig)
'''