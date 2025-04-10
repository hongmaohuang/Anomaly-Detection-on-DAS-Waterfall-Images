# %% 
import matplotlib.pyplot as plt
import pandas as pd 
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
import numpy as np
import matplotlib
import glob
from datetime import datetime
import os
import xdas
from pathlib import Path
# Read the CSV with clustering results
df = pd.read_csv('waveform_cluster_results.csv')

# ==== CUSTOMIZABLE PARAMETERS ====
# Data settings
DATA_DATE = "20250331"
TARGET_IDX = 0
COLORMAP_WATERFALL = "gray"

# Font settings (optional)
FONT_PATH = './Helvetica.ttf'
TITLE_FONT_PATH = './Helvetica_Bold.ttf'
FONT_SIZE = 12
TITLE_FONT_SIZE = 18
DATA_DIR = Path('./DAS_data/20250331/waveforms')

waveform_files = list(DATA_DIR.glob('*.hdf5'))
data = xdas.open_mfdataarray(str(waveform_files[0]), engine="asn")
max_distance = np.max(data.coords['distance'].values)

# Constants
TOTAL_DURATION_SEC = 17 * 60
# depends on your DAS waterfall image
TOTAL_DISTANCE_KM = max_distance/1000

# ==== Setup Fonts ====
fm.fontManager.addfont(FONT_PATH)
title_font = FontProperties(fname=TITLE_FONT_PATH, size=TITLE_FONT_SIZE, weight='bold')
matplotlib.rcParams['font.family'] = 'Helvetica'
matplotlib.rcParams['font.size'] = FONT_SIZE

# ==== Load DAS File ====
file_pattern = f'./DAS_data/{DATA_DATE}/waterfall_npz/*.npz'
file_list = sorted(glob.glob(file_pattern))
if TARGET_IDX >= len(file_list):
    raise IndexError(f"TARGET_IDX exceeds available files ({len(file_list)})")
file = file_list[TARGET_IDX]
basename = os.path.basename(file)
# Extract DAS start time from filename (assumes filename parts contain date info)
dt_str = basename.split('_')[2] + basename.split('_')[3]
dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
print(f"Processing DAS data starting at: {dt.strftime('%Y-%m-%d %H:%M:%S')}")

# ==== Load Image Data ====
data = np.load(file)
img_gray = data["waterfall"].mean(axis=2)  # Convert RGB image to grayscale

# ==== Plotting ====
fig, ax = plt.subplots(figsize=(14, 5))
ax.imshow(img_gray, aspect='auto', cmap=COLORMAP_WATERFALL,
          extent=[0, TOTAL_DISTANCE_KM, TOTAL_DURATION_SEC, 0])
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Time (sec)")

# ==== Overlay Classification Points on DAS Waterfall ====
# "Distance (m)" is provided in meters in the CSV, so convert it to kilometers.
# Calculate the time offset from the CSV 'starttime' relative to the DAS start time.
unique_clusters = sorted(df['cluster'].unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

for clust, color in zip(unique_clusters, colors):
    subset = df[df['cluster'] == clust]
    # Convert 'Distance (m)' from meters to kilometers
    x_points = subset['Distance (m)'].astype(float) / 1000.0
    # Convert 'starttime' to datetime and then remove timezone info to ensure tz-naive
    start_times = pd.to_datetime(subset['starttime']).dt.tz_localize(None)
    # Compute offsets in seconds relative to DAS start time
    offsets = (start_times - dt).dt.total_seconds()
    # Plot each classification point as a circle marker
    ax.scatter(x_points, offsets, marker='o', color=color, edgecolor='black',
               s=80, label=f"Cluster {clust}")

ax.legend()
ax.set_title("DAS Waterfall with Classification Points")
plt.tight_layout()
plt.show()
