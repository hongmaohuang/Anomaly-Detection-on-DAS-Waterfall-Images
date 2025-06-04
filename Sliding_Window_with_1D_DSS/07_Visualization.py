# %%
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
                s=5,
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
