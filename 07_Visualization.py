import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap, BoundaryNorm
from datetime import datetime
import config

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

files_per_group = 5
step_file = 2
preds = np.loadtxt(label_path, usecols=1, dtype=int, skiprows=1)
npz_files = sorted(Path(config.WATERFALL_NPZ_FOLDER).glob("*.npz"))[::step_file]
groups = [npz_files[i:i+files_per_group] for i in range(0, len(npz_files), files_per_group)]
n_labels = preds.max() + 1
cmap = plt.cm.get_cmap("tab20", n_labels)
label_cmap = ListedColormap(cmap.colors[:n_labels])
preds_per_file = len(preds) // len(npz_files)
idx = 0

for group in groups:
    print('Processing group:', group[0].stem, 'to', group[-1].stem)
    waterfalls = [np.load(fp)["waterfall"].mean(axis=2) for fp in reversed(group)]
    stacked = np.vstack(waterfalls)
    block = preds[idx: idx + len(group) * preds_per_file]
    idx += len(group) * preds_per_file
    overlay = block.reshape(len(group), preds_per_file)
    factor = stacked.shape[0] // overlay.shape[0]
    overlay_img = np.repeat(overlay, factor, axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(
        stacked,
        cmap="gray",
        aspect="auto",
        extent=[0, config.TOTAL_DISTANCE_KM, 0, len(group) * config.DURATION_WATERFALL * 60],
        origin="lower",
    )

    norm = BoundaryNorm(np.arange(n_labels+1), ncolors=n_labels)
    im = ax.imshow(
        overlay_img,
        cmap=label_cmap,
        norm=norm,
        alpha=0.7,
        aspect="auto",
        extent=[0, config.TOTAL_DISTANCE_KM, 0, len(group) * config.DURATION_WATERFALL * 60],
        origin="lower",
        interpolation="nearest",
    )

    cbar = fig.colorbar(im, ax=ax, boundaries=np.arange(n_labels+1))
    cbar.set_ticks(np.arange(n_labels) + 0.5)
    cbar.set_ticklabels(np.arange(n_labels))
    cbar.set_label("Cluster Label", rotation=270, labelpad=15)

    ax.set(
        ylim=(len(group) * config.DURATION_WATERFALL * 60, 0),
        xlabel="Distance (km)",
        ylabel="Time (s)",
    )
    ts = group[-1].stem.split('_')[2] + '_' + group[-1].stem.split('_')[3]
    out = Path(config.VISUALIZATION_FOLDER) / f"{ts}_clusters.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)