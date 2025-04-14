# %%
# Simulate waterfall images from OptoDAS data in RMS
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for safe plotting in script
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import xdas
from pathlib import Path
import matplotlib.gridspec as gridspec
import pandas as pd
os.chdir(Path(__file__).resolve().parent)

# ==============================
# User Parameters
# ==============================
data_directory = "../../Inputs/DAS_data/20250331/waveforms"
output_directory = "../../Outputs/Waterfall_Images"
files_per_plot = 6            # Number of HDF5 files per image (e.g., 6 x 10s = 60s)
figure_size = (12, 6)         # Figure size
dpi = 300                     # Output image resolution

# RMS plot parameters
window_seconds = 1
use_auto_range = True
colormap = 'jet'
# ==============================

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

    data = xdas.open_mfdataarray(batch, engine="asn")
    strain_rate_data = (data * 1.0213001907746815 / 9112677.96164918) * -1

    t_start = strain_rate_data.coords['time'][0].values
    t_end = strain_rate_data.coords['time'][-1].values
    print(f"[{i+1}/{total_batches}] Processing {os.path.basename(batch[0])} - {os.path.basename(batch[-1])}")
    print(f"Start Time: {t_start}")
    print(f"End Time:   {t_end}")

    times = strain_rate_data.coords['time'].values.astype('datetime64[ms]')
    dt = (times[1] - times[0]) / np.timedelta64(1, 's')
    samples_per_window = int(window_seconds / dt)
    n_windows = strain_rate_data.shape[0] // samples_per_window

    rms_blocks = []
    for j in range(n_windows):
        chunk = strain_rate_data[j*samples_per_window : (j+1)*samples_per_window, :]
        rms_chunk = np.sqrt(np.mean(chunk.values**2, axis=0))
        rms_blocks.append(rms_chunk)

    rms_array = np.stack(rms_blocks)
    t0 = times[0]
    new_times = [t0 + np.timedelta64(int((j + 0.5) * samples_per_window * dt * 1000), 'ms') for j in range(n_windows)]
    distance_vals = strain_rate_data.coords['distance'].values.astype(float)

    db_rms = 20 * np.log10(np.clip(rms_array, 1e-20, None))


    vmin, vmax = np.percentile(db_rms, [1, 99])
    print(f"[Auto vmin/vmax] {vmin:.1f} ~ {vmax:.1f} dB")


    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111)
    time_vals = pd.to_datetime(new_times).astype(np.int64) / 1e6

    im = ax.imshow(
        db_rms,
        extent=[
            distance_vals[0]/1000,
            distance_vals[-1]/1000,
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

