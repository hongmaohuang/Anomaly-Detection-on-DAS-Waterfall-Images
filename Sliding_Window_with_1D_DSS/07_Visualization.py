# %%
from pathlib import Path
import os 
os.chdir(Path(__file__).resolve().parent)
import numpy as np
import matplotlib.pyplot as plt
import config
import shutil

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

for g_idx, group in enumerate(groups, 1):
    if not group:
        continue
    print(f"➤ Processing group {g_idx}/{len(groups)} with {len(group)} files …")
    waterfalls = [np.load(fp)[waterfall_key].mean(axis=2) for fp in reversed(group)]
    stacked = np.vstack(waterfalls)

    x_min, x_max = 0, config.TOTAL_DISTANCE_KM
    sec_per_file = config.DURATION_WATERFALL * 60
    total_sec    = len(group) * sec_per_file

    num_seg = len(group)
    seg_len = len(predictions) // num_seg
    x_vals  = np.linspace(x_min, x_max, seg_len, endpoint=False)
    step    = x_vals[1] - x_vals[0]

    unique_labels = np.unique(predictions)
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))

    for idx_lab, lab in enumerate(unique_labels):
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.imshow(
            stacked,
            cmap="gray",
            aspect="auto",
            extent=[x_min, x_max, 0, total_sec],
            origin="lower",
        )

        for seg in range(num_seg):
            start, end = seg * seg_len, (seg + 1) * seg_len
            preds_seg = predictions[start:end] == lab
            idx_true  = np.where(preds_seg)[0]
            splits = np.split(idx_true, np.where(np.diff(idx_true) != 1)[0] + 1)
            y_center = total_sec - (seg + 0.5) * sec_per_file
            for blk in splits:
                x0 = x_vals[blk[0]]
                x1 = min(x_max, x_vals[blk[-1]] + 0.05)
                ax.hlines(y=y_center, xmin=x0, xmax=x1,
                          color=cmap(idx_lab), linewidth=10)

        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Relative Time (min)")
        yticks = np.arange(0, total_sec + 1, 60)
        ax.set_yticks(yticks)
        ax.set_yticklabels((yticks / 60).astype(int))
        ax.set_ylim(total_sec, 0)

        last_stem = group[-1].stem
        date_time = "_".join(last_stem.split("_")[2:4])
        ax.set_title(f"Cluster {lab} before {date_time}")
        plt.tight_layout()

        out_name = f"Cluster_{lab}_before_{date_time}.png"
        out_path = Path(config.VISUALIZATION_FOLDER) / out_name
        plt.savefig(out_path, dpi=300)
        plt.close(fig)

print("All groups processed. Done! 🎉")
