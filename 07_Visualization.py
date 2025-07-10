import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import cKDTree
from skimage import exposure
import config
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap, BoundaryNorm

##### Cluster Counts Visualization #####

out_dir = Path(config.VISUALIZATION_FOLDER) / "cluster_counts"
out_dir.mkdir(parents=True, exist_ok=True)
label_path = Path(config.CLUSTERING_RESULTS_FOLDER) / "cluster_labels.dat"
data = np.loadtxt(label_path, skiprows=1)
distances = data[:, 0]
labels = data[:, 1].astype(int)
clusters = np.unique(labels)

npz_files = sorted(Path(config.WATERFALL_NPZ_FOLDER).glob("*.npz"))
file_times = [datetime.strptime(p.stem.split('_')[2] + p.stem.split('_')[3], "%Y%m%d%H%M%S") for p in npz_files]
time_vals = mdates.date2num(file_times)

target = config.OCCURRENCES_LOC
step = config.TOTAL_DISTANCE_KM
group_size = config.ACCUMULATIONS_PER_FILE
indices = [int(np.argmin(np.abs(distances - (target + i * step)))) for i in range(len(npz_files))]
groups = [indices[i:i+group_size] for i in range(0, len(indices), group_size)]
group_times = [
    time_vals[min(i + group_size - 1, len(time_vals) - 1)]
    for i in range(0, len(time_vals), group_size)
]

for cl in clusters:
    counts = [np.sum(labels[idxs] == cl) for idxs in groups]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(group_times, counts, marker='o', markersize=2)
    #ax.plot(time_vals[:len(counts)], counts, marker='o', markersize=2)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1000))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    ax.set(xlabel="Time", ylabel="Count", title=f"Cluster {cl} occurrences at {target} km")
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(out_dir / f"cluster{cl}_counts.png", dpi=300)
    plt.close(fig)

    acc = np.cumsum(counts)
    fig, ax = plt.subplots(figsize=(8, 4))
    #ax.plot(time_vals[:len(acc)], acc, marker='o', markersize=2)
    ax.plot(group_times, acc, marker='o', markersize=2)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1000))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    ax.set(xlabel="Time", ylabel="Accumulated Count", title=f"Accumulated occurrences of Cluster {cl} at {target} km")
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(out_dir / f"cluster{cl}_accumulated_counts.png", dpi=300)
    plt.close(fig)


##### Waterfall Visualization with Clustering Results #####

out_dir = Path(config.VISUALIZATION_FOLDER) / "cluster_blackdots"
out_dir.mkdir(parents=True, exist_ok=True)

label_path = Path(config.CLUSTERING_RESULTS_FOLDER) / "cluster_labels.dat"
preds = np.loadtxt(label_path, usecols=1, dtype=int, skiprows=1)

step_file = config.IMAGE_STEPS
files_per_group = config.IMAGE_PER_GROUP
npz_files = sorted(Path(config.WATERFALL_NPZ_FOLDER).glob("*.npz"))[::step_file]
groups = [npz_files[i:i+files_per_group] for i in range(0, len(npz_files), files_per_group)]

n_labels = preds.max() + 1
preds_per_file = len(preds) // len(npz_files)
pred_offset = 0  

jet = cm.get_cmap("jet", 256)(np.arange(256))[:, :3]
tree = cKDTree(jet)
dmin, dmax = config.ORIGINAL_DMIN, config.ORIGINAL_DMAX

for group in groups:
    processed = []
    for fp in reversed(group):
        with np.load(fp) as data:
            wf_rgb = data["waterfall"][..., :3].astype(np.uint8)
            h, w, _ = wf_rgb.shape
            lut_idx = tree.query(wf_rgb.reshape(-1, 3), k=1)[1]
            lut_idx = lut_idx.reshape(h, w).astype(np.uint8)
            wf_db = dmin + lut_idx.astype(np.float32)*(dmax - dmin)/256.0
            processed.append(wf_db)

    stacked = np.vstack(processed)

    count = len(group) * preds_per_file
    block = preds[pred_offset : pred_offset + count]
    pred_offset += count

    overlay = block.reshape(len(group), preds_per_file)
    overlay = overlay[::-1, :]
    factor = stacked.shape[0] // overlay.shape[0]
    overlay_img = np.repeat(overlay, 1, axis=0)

    total_time = len(group) * config.DURATION_WATERFALL * 60
    total_dist = config.TOTAL_DISTANCE_KM
    H, W = overlay_img.shape
    def idx_to_xy(rows, cols):
        x = (cols + 0.5) * (total_dist / W)
        y = (rows + 0.5) * (total_time / H)
        return x, y

    stem = group[-1].stem.split('_')
    endtime = f"{stem[2]}_{stem[3]}"

    for cl in range(n_labels):
        xs, ys= np.where(overlay_img == cl)
        if ys.size == 0:
            continue

        xs, ys = idx_to_xy(xs, ys)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(stacked, cmap="jet", aspect="auto", vmin=dmin, vmax=dmax,
                  extent=[0, total_dist, total_time, 0])
        ax.scatter(xs, ys, s=10, marker='.', color='k',
                   alpha=0.5, linewidths=0, edgecolors='none')
        ax.set(xlabel="Distance (km)",
               ylabel="Relative Time (s)",
               ylim=(total_time, 0),
               title=f"Cluster {cl} before {endtime}")
        plt.tight_layout()

        out_path = out_dir / f"{endtime}_cluster{cl}.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    #break 

