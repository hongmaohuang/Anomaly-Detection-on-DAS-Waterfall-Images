import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.spatial import cKDTree
from skimage import exposure
from skimage.filters import gaussian, gabor
import config

##### Cluster Counts Visualization #####
if config.VISUALIZATION_METHOD == "cluster_counts":
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
if config.VISUALIZATION_METHOD == "waterfall_clusters":
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
        # print the progress in %
        progress = (npz_files.index(group[-1]) + 1) / len(npz_files) * 100
        print(f"Processing group {npz_files.index(group[-1]) + 1}/{len(npz_files)} ({progress:.2f}%)")
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

##### Features visualization with clusters distribution##### 
#if config.VISUALIZATION_METHOD == "features_clusters":

out_dir = Path(config.VISUALIZATION_FOLDER) / "features_clusters"
out_dir.mkdir(parents=True, exist_ok=True)

label_path = Path(config.CLUSTERING_RESULTS_FOLDER) / "cluster_labels.dat"
pred = pd.read_csv(label_path, sep=' ')

npz_files = sorted(Path(config.WATERFALL_NPZ_FOLDER).glob("*.npz"))
jet = cm.get_cmap("jet", 256)(np.arange(256))[:, :3]
tree = cKDTree(jet)
dmin, dmax = config.ORIGINAL_DMIN, config.ORIGINAL_DMAX

wf_rgb = np.load(npz_files[0])["waterfall"][..., :3].astype(np.uint8)
h, w, _ = wf_rgb.shape
lut_idx = tree.query(wf_rgb.reshape(-1, 3), k=1)[1].reshape(h, w).astype(np.uint8)
wf_db = dmin + lut_idx.astype(np.float32)*(dmax - dmin)/256.0
rms_img = wf_db

thetas = np.linspace(0, np.pi, 8, endpoint=False)
responses = [np.hypot(*gabor(rms_img, frequency=0.05, theta=th)) for th in thetas]
responses = np.stack(responses, axis=0)
sloped_idxs = [np.argmin(np.abs(thetas - np.pi/4)), np.argmin(np.abs(thetas - 3*np.pi/4))]
gabor_sloped = np.max(responses[sloped_idxs], axis=0)

H, W = rms_img.shape
x_vals = np.linspace(0, config.TOTAL_DISTANCE_KM, W)
total_time = config.DURATION_WATERFALL * 60
y_center = total_time / 2
dt_sec = config.DURATION_WATERFALL * 60 / H

# Feature 1: RMS STD
result_std = np.array([
    rms_img[:, i:i+config.WINDOW_SIZE_CHANNEL].std()
    for i in range(0, W - config.WINDOW_SIZE_CHANNEL + 1, config.STEP_CHANNEL)
], dtype=np.float16)

x_mid_std = np.array([
    x_vals[i:i+config.WINDOW_SIZE_CHANNEL].mean()
    for i in range(0, W - config.WINDOW_SIZE_CHANNEL + 1, config.STEP_CHANNEL)
])
y_vals_std = y_center - result_std * 1  # amplitude_std = 1

# Feature 2: Gabor Mean
result_gabor = np.array([
    gabor_sloped[:, i:i+config.WINDOW_SIZE_CHANNEL].mean()
    for i in range(0, W - config.WINDOW_SIZE_CHANNEL + 1, config.STEP_CHANNEL)
], dtype=np.float16)

x_mid_gabor = np.array([
    x_vals[i:i+config.WINDOW_SIZE_CHANNEL].mean()
    for i in range(0, W - config.WINDOW_SIZE_CHANNEL + 1, config.STEP_CHANNEL)
])
y_vals_gabor = y_center - result_gabor * 5  # amplitude_gabor = 5

# Feature 3: Dominant Frequency
feat_dom_freq = []
x_mid_freq = []

for i in range(0, W - config.WINDOW_SIZE_CHANNEL + 1, config.STEP_CHANNEL):
    window = rms_img[:, i:i+config.WINDOW_SIZE_CHANNEL]
    freqs_per_col = []
    for k in range(window.shape[1]):
        col = window[:, k] - window[:, k].mean()
        fft_vals = np.fft.rfft(col)
        amp_spec = np.abs(fft_vals) / col.size
        freqs = np.fft.rfftfreq(col.size, d=dt_sec)
        if len(freqs) > 1:
            dominant_freq = freqs[np.argmax(amp_spec[1:]) + 1]
        else:
            dominant_freq = 0.0
        freqs_per_col.append(dominant_freq)
    feat_dom_freq.append(np.mean(freqs_per_col))
    x_mid_freq.append(x_vals[i:i+config.WINDOW_SIZE_CHANNEL].mean())

feat_dom_freq = np.array(feat_dom_freq, dtype=np.float32)
x_mid_freq = np.array(x_mid_freq, dtype=np.float32)
norm_freq = (feat_dom_freq - feat_dom_freq.min()) / (feat_dom_freq.max() - feat_dom_freq.min())
y_vals_freq = y_center - norm_freq * 5  # amplitude_freq = 5

# Plotting
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

axs[0].imshow(rms_img, cmap="jet", aspect="auto", vmin=dmin, vmax=dmax,
              extent=[0, config.TOTAL_DISTANCE_KM, total_time, 0])
axs[0].plot(x_mid_std, y_vals_std, color='white', linewidth=2)
axs[0].set(xlabel="Distance (km)", ylabel="Relative Time (s)",
           ylim=(total_time, 0), title="RMS with STD")

axs[1].imshow(gabor_sloped, cmap='gray', aspect="auto",
              extent=[0, config.TOTAL_DISTANCE_KM, total_time, 0])
axs[1].plot(x_mid_gabor, y_vals_gabor, color='white', linewidth=2)
axs[1].set(xlabel="Distance (km)", ylabel="Relative Time (s)",
           ylim=(total_time, 0), title="Gabor Sloped with Mean")

axs[2].imshow(rms_img, cmap="jet", aspect="auto", vmin=dmin, vmax=dmax,
              extent=[0, config.TOTAL_DISTANCE_KM, total_time, 0])
axs[2].plot(x_mid_freq, y_vals_freq, color='white', linewidth=2)
axs[2].set(xlabel="Distance (km)", ylabel="Relative Time (s)",
           ylim=(total_time, 0), title="RMS Dominant Frequency")

max_dist = x_vals[-1]
idx_cut = np.searchsorted(pred.dist, max_dist, side="right")

axs[3].scatter(
    x=pred.dist[:idx_cut],
    y=np.full(idx_cut, 30),
    c=pred.clusters[:idx_cut],
    cmap='tab10', s=500, marker='|'
)
axs[3].set(xlabel="Distance (km)", ylabel="", title="Cluster Labels Distribution",
           ylim=(total_time, 0))
unique_clusters = np.unique(pred.clusters[:idx_cut])

legend_patches = [
    mpatches.Patch(color=plt.cm.tab10(clust_id % 10), label=f"Cluster {clust_id}")
    for clust_id in unique_clusters
]

axs[3].legend(
    handles=legend_patches,
    loc="lower center",           
    bbox_to_anchor=(0.5, 0.05),  
    ncol=len(unique_clusters),
    frameon=False,
    fontsize=8,
    title_fontsize=9
)

plt.tight_layout()