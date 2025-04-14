# %%
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from sklearn.ensemble import IsolationForest
from kymatio import Scattering2D
from datetime import datetime
from matplotlib.font_manager import FontProperties
import matplotlib
import matplotlib.font_manager as fm

# ======= User Parameters =======
data_date = "20250331"
target_idx = 0
channel_group_size = 2
j_scattering = 1
L_scattering = 8
contamination_rate = 0.05
anomaly_threshold = 0.11
vmax_features = 5
colormap_waterfall = "jet"
colormap_scattering_matrix = "viridis"


# ======= Font (Optional) =======
fm.fontManager.addfont('/home/hmhuang/Work/BY_thesis/Helvetica.ttf')
title_font = FontProperties(fname='/home/hmhuang/Work/BY_thesis/Helvetica_Bold.ttf', size=18, weight='bold')
matplotlib.rcParams['font.family'] = 'Helvetica'
matplotlib.rcParams['font.size'] = 12

# ======= Constants =======
total_duration_sec = 17 * 60
total_distance_km = 28.086

# ======= Load File =======
file_pattern = f'/home/hmhuang/Work/Research_Assistant/AI_Chulin_20240721/Scatseisnet_2D/DAS_data/{data_date}/waterfall/*.npz'
file_list = sorted(glob.glob(file_pattern))
if target_idx >= len(file_list):
    raise IndexError(f"target_idx exceeds available files ({len(file_list)})")

file = file_list[target_idx]
basename = os.path.basename(file)
dt_str = basename.split('_')[2] + basename.split('_')[3]
dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
print(f"Processing {dt.strftime('%Y-%m-%d %H:%M:%S')}")

# ======= Load DAS Waterfall Image =======
data = np.load(file)
img_gray = data["waterfall"].mean(axis=2)  # Convert RGB to grayscale
DAS_data = img_gray
num_time_samples, num_channels = DAS_data.shape

# ======= Create Channel Groups =======
channel_groups = [
    range(i, i + channel_group_size)
    for i in range(0, num_channels - channel_group_size + 1, channel_group_size)
]

features_all = []


for i, group in enumerate(channel_groups):
    print(f"Processing group {i+1}/{len(channel_groups)} ...")
    DAS_group = DAS_data[:, group]
    h, w = DAS_group.shape
    if min(h, w) < 2 ** j_scattering:
        continue

    scattering = Scattering2D(J=j_scattering, shape=DAS_group.shape, L=L_scattering)
    scat = scattering(DAS_group)

    DSM_group = np.stack([
        scat.mean(axis=1),
        scat.std(axis=1),
        scat.max(axis=1)
    ], axis=1)

    features_all.append(DSM_group.flatten())

# ======= Stack Feature Matrix =======
DSM_features = np.stack(features_all, axis=0)
np.savez("DSM_features_temp.npz", features=DSM_features)

# %%
# ======= Anomaly Detection =======
channel_features = DSM_features
model = IsolationForest(contamination=contamination_rate, random_state=42)
model.fit(channel_features)
scores = model.decision_function(channel_features)
anomalous_idx = np.where((scores < -anomaly_threshold) | (scores > anomaly_threshold))[0]

# ======= X-Axis Location of Groups =======
num_valid_groups = len(DSM_features)
group_width_km = total_distance_km / len(channel_groups)
group_centers_km = np.arange(num_valid_groups) * group_width_km + group_width_km / 2

# ======= Plot =======
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                                    gridspec_kw={'height_ratios': [1, 3, 2]})

# --- Anomaly Score ---
ax1.bar(group_centers_km, scores, width=group_width_km * 0.9, color='gray', label="Anomaly Score")
ax1.scatter(group_centers_km[anomalous_idx], scores[anomalous_idx],
            color='red', zorder=5, label="Detected Anomaly")
ax1.set_ylabel("Score")
ax1.set_title("Anomaly Detection")
ax1.grid(True)
ax1.legend()

# --- Scattering Feature Heatmap ---
ax2.imshow(DSM_features.T, aspect='auto', cmap=colormap_scattering_matrix,
           extent=[group_centers_km[0], group_centers_km[-1], DSM_features.shape[1], 0],
           vmin=0, vmax=vmax_features)
for ch in anomalous_idx:
    ax2.axvline(x=group_centers_km[ch], color='red', linestyle='--', alpha=0.8)
ax2.set_ylabel("Feature Index")
ax2.set_title("Scattering Features")

# --- DAS Waterfall ---
ax3.imshow(img_gray, aspect='auto', cmap=colormap_waterfall,
           extent=[0, total_distance_km, total_duration_sec, 0])
ax3.set_xlabel("Distance (km)")
ax3.set_ylabel("Time (sec)")
ax3.set_title("Original DAS Waterfall")

plt.tight_layout()
plt.show()