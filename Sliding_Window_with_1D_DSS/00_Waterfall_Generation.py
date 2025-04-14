# %%
# Simulate waterfall images from OptoDAS data in RMS and save as both PNG and NPZ

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for safe plotting in scripts
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from PIL import Image
import xdas
from datetime import timezone
os.chdir(Path(__file__).resolve().parent)

# === Config ===
data_directory = "../../Inputs/DAS_data/20250331/waveforms"
output_directory = "../../Outputs/Waterfall_Images"
output_npz_subfolder = "waterfall_npz"

files_per_plot = 6
figure_size = (12, 6)
dpi = 300
window_seconds = 1
colormap = 'jet'
use_auto_range = True

scale_factor = 1.0213001907746815 / 9112677.96164918
invert_sign = True
engine_type = "asn"
# ==============

Path(output_directory).mkdir(parents=True, exist_ok=True)
all_files = sorted(glob.glob(os.path.join(data_directory, "*.hdf5")))
total_batches = len(all_files) // files_per_plot

for i in range(total_batches):
    batch = all_files[i * files_per_plot : (i + 1) * files_per_plot]
    if not batch:
        continue

    date_str = os.path.basename(os.path.dirname(data_directory))
    file_prefix = os.path.basename(batch[0]).split('.')[0]
    output_path = os.path.join(output_directory, f"Waterfall_1_min_{date_str}_{file_prefix}.png")

    if os.path.exists(output_path):
        print(f"[{i+1}/{total_batches}] Skipped (already exists): {output_path}")
        continue

    data = xdas.open_mfdataarray(batch, engine=engine_type)
    strain_rate_data = data * scale_factor
    if invert_sign:
        strain_rate_data *= -1

    times = strain_rate_data.coords['time'].values.astype('datetime64[ms]')
    dt = (times[1] - times[0]) / np.timedelta64(1, 's')
    samples_per_window = int(window_seconds / dt)
    n_windows = strain_rate_data.shape[0] // samples_per_window

    rms_blocks = [
        np.sqrt(np.mean(strain_rate_data[j*samples_per_window:(j+1)*samples_per_window, :].values**2, axis=0))
        for j in range(n_windows)
    ]
    rms_array = np.stack(rms_blocks)

    t0 = times[0]
    new_times = [
        t0 + np.timedelta64(int((j + 0.5) * samples_per_window * dt * 1000), 'ms')
        for j in range(n_windows)
    ]
    distance_vals = strain_rate_data.coords['distance'].values.astype(float)
    db_rms = 20 * np.log10(np.clip(rms_array, 1e-20, None))

    vmin, vmax = np.percentile(db_rms, [1, 99]) if use_auto_range else (None, None)
    print(f"[{i+1}/{total_batches}] Processing {os.path.basename(batch[0])} - {os.path.basename(batch[-1])}")
    print(f"Start Time: {times[0]} | End Time: {times[-1]} | dB Range: {vmin:.1f} ~ {vmax:.1f}")

    fig, ax = plt.subplots(figsize=figure_size)
    time_vals = pd.to_datetime(new_times).astype(np.int64) / 1e6

    im = ax.imshow(
        db_rms,
        extent=[
            distance_vals[0] / 1000,
            distance_vals[-1] / 1000,
            time_vals[-1],
            time_vals[0],
        ],
        aspect='auto',
        cmap=colormap,
        interpolation='none',
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Time")
    plt.axis('off')
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# Save as NPZ
npz_output_dir = os.path.join(output_directory, output_npz_subfolder)
Path(npz_output_dir).mkdir(parents=True, exist_ok=True)

for fname in os.listdir(output_directory):
    if not fname.lower().endswith(".png"):
        continue
    img_path = os.path.join(output_directory, fname)
    image = Image.open(img_path)
    img_array = np.array(image)

    parts = fname.split("_")
    date_str = parts[3]          
    time_str = parts[4].split(".")[0]  
    npz_name = f"Waterfall_RMS_{date_str}_{time_str}_utc.npz"

    np.savez(os.path.join(npz_output_dir, npz_name), waterfall=img_array)