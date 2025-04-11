# %%
# %%
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import xdas

# ==============================
# User Parameters
# ==============================
data_directory = "./DAS_Detection/DAS_data/20250331/waveforms"
output_directory = "./Waterfall_Images"
files_per_plot = 6            # Number of HDF5 files per image (e.g., 6 x 10s = 60s)
vmin, vmax = -450, 450        # Color limits for waterfall plot
colormap = 'seismic'          # Colormap for imshow
figure_size = (12, 6)         # Figure size
dpi = 300                     # Output image resolution
# ==============================

os.makedirs(output_directory, exist_ok=True)
all_files = sorted(glob.glob(os.path.join(data_directory, "*.hdf5")))
total_batches = len(all_files) // files_per_plot

for i in range(total_batches):
    batch = all_files[i * files_per_plot : (i + 1) * files_per_plot]
    if not batch:
        continue

    data = xdas.open_mfdataarray(batch, engine="asn")
    t_start = data.coords['time'][0].values
    t_end = data.coords['time'][-1].values
    duration_sec = (np.datetime64(t_end) - np.datetime64(t_start)) / np.timedelta64(1, 's')

    print(f"[{i+1}/{total_batches}] Processing {os.path.basename(batch[0])} - {os.path.basename(batch[-1])}")
    print(f"Start Time: {t_start}")
    print(f"End Time:   {t_end}")

    # Create figure
    fig = plt.figure(figsize=figure_size)
    plt.imshow(
        data.T,
        extent=[
            data.coords['distance'][0].values.astype(float),
            data.coords['distance'][-1].values.astype(float),
            data.coords['time'][-1].values.astype('datetime64[ms]').astype(float),
            data.coords['time'][0].values.astype('datetime64[ms]').astype(float),
        ],
        aspect='auto',
        cmap=colormap,
        interpolation='none',
        vmin=vmin,
        vmax=vmax,
    )
    plt.axis('off')
    plt.tight_layout()

    # Output image path
    date_str = os.path.basename(os.path.dirname(data_directory))
    file_prefix = os.path.basename(batch[0]).split('.')[0]
    output_path = os.path.join(output_directory, f"Waterfall_1_min_{date_str}_{file_prefix}.png")
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
