import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import config

count_dir = Path(config.VISUALIZATION_FOLDER) / "cluster_counts"
count_dir.mkdir(parents=True, exist_ok=True)

label_path = Path(config.CLUSTERING_RESULTS_FOLDER) / "cluster_labels.dat"
label_data = np.loadtxt(label_path, skiprows=1)
distances = label_data[:, 0]
labels = label_data[:, 1].astype(int)
unique_clusters = np.unique(labels)

npz_files = sorted(Path(config.WATERFALL_NPZ_FOLDER).glob("*.npz"))
num_files = len(npz_files)
file_times = [
    datetime.strptime(p.stem.split('_')[2] + p.stem.split('_')[3], "%Y%m%d%H%M%S")
    for p in npz_files
]

target_dist = config.OCCURRENCES_LOC
dist_per_file = config.TOTAL_DISTANCE_KM
group_size = config.ACCUMULATIONS_PER_FILE

indices = []
for i in range(num_files):
    global_target = target_dist + i * dist_per_file
    idx = int(np.argmin(np.abs(distances - global_target)))
    indices.append(idx)

groups = [indices[i:i + group_size] for i in range(0, len(indices), group_size)]
group_times = [file_times[i] for i in range(0, len(file_times), group_size)]
time_vals = mdates.date2num(group_times)

for clust in unique_clusters:
    counts = [np.sum(labels[idxs] == clust) for idxs in groups]
    plt.figure(figsize=(8, 4))
    plt.plot(time_vals, counts, marker='o', color='gray', markersize=2)
    ax = plt.gca()
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.title(f"Cluster {clust} occurrences at {target_dist} km")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(count_dir / f"cluster{clust}_counts.png", dpi=300)
    plt.close()

    accumulated_counts = np.cumsum(counts)
    plt.figure(figsize=(8, 4))
    plt.plot(time_vals, accumulated_counts, marker='o', color='gray', markersize=2)
    ax = plt.gca()
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xlabel("Time")
    plt.ylabel("Accumulated Count")
    plt.title(f"Accumulated occurrences of Cluster {clust} at {target_dist} km")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(count_dir / f"cluster{clust}_accumulated_counts.png", dpi=300)
    plt.close()

'''
# Events With waterfall images
from pathlib import Path
import os 
import numpy as np
import matplotlib.pyplot as plt
os.chdir(Path(__file__).resolve().parent)
import config
import shutil
from matplotlib.colors import ListedColormap

if os.path.exists(config.VISUALIZATION_FOLDER):
    shutil.rmtree(config.VISUALIZATION_FOLDER)
os.makedirs(config.VISUALIZATION_FOLDER)

files_per_group = 5      
step_file       = 2     

waterfall_key   = "waterfall"  
label_path = Path(config.CLUSTERING_RESULTS_FOLDER) / "cluster_labels.dat"
print(f"Loading labels from {label_path} …")
predictions = np.loadtxt(label_path, usecols=1, dtype=int, skiprows=1)

npz_folder = Path(config.WATERFALL_NPZ_FOLDER)
npz_files  = sorted(npz_folder.glob("*.npz"))[::step_file]
print(f"Found {len(npz_files)} npz files after sampling (step={step_file}).")

total_files = len(npz_files)
groups = [npz_files[i:i + files_per_group] for i in range(0, total_files, files_per_group)]
print(f"Split into {len(groups)} groups (files_per_group={files_per_group}).")

n_labels = np.unique(predictions).max() + 1
cmap = plt.cm.get_cmap("tab20", n_labels)
label_cmap = ListedColormap(cmap.colors[:n_labels])

for g_idx, group in enumerate(groups, 1):
    waterfalls = [np.load(fp)["waterfall"].mean(axis=2) for fp in reversed(group)]
    stacked    = np.vstack(waterfalls)
    x_min, x_max      = 0, config.TOTAL_DISTANCE_KM
    sec_per_file      = config.DURATION_WATERFALL * 60
    total_sec         = len(group) * sec_per_file
    num_seg           = len(group)
    seg_len           = len(predictions) // num_seg
    x_vals            = np.linspace(x_min, x_max, seg_len, endpoint=False)

    for cluster_num in range(n_labels):
        print(f'Cluster: {cluster_num}')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(
            stacked,
            cmap="gray",
            aspect="auto",
            extent=[x_min, x_max, 0, total_sec],
            origin="lower",
        )

        for seg_idx in range(num_seg):
            print(f'NO. {seg_idx} waterfall images')
            start, end = seg_idx * seg_len, (seg_idx + 1) * seg_len
            print(start)
            print(end)
            idx = np.where(predictions[start:end] == cluster_num)[0]
            y_center = total_sec - (seg_idx + 0.5) * sec_per_file
            print(y_center)
            ax.scatter(
                x_vals[idx],                        
                np.full(idx.shape, y_center),       
                c="k",                             
                s=1,
                marker="s",
                linewidths=0,
            )
        ax.set_yticks(np.arange(0, total_sec + 60, 60))
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Time (s)")
        ax.set_ylim(total_sec, 0)                  
        out_png = Path(config.VISUALIZATION_FOLDER) / f"{str(group[-1]).split('/')[-1].split('_')[2]+'_'+str(group[-1]).split('/')[-1].split('_')[3]}_cluster{cluster_num}.png"
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
''' 
