# %%
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from datetime import datetime
from sklearn.ensemble import IsolationForest
from skimage.measure import label, regionprops

# ==== CUSTOMIZABLE PARAMETERS ====
# Data settings
DATA_DATE = "20250331"
TARGET_IDX = 1
CONTAMINATION_RATE = 0.03
COLORMAP_WATERFALL = "jet"

# Font settings (optional)
FONT_PATH = './Helvetica.ttf'
TITLE_FONT_PATH = './Helvetica_Bold.ttf'
FONT_SIZE = 12
TITLE_FONT_SIZE = 18

# Constants
TOTAL_DURATION_SEC = 17 * 60
TOTAL_DISTANCE_KM = 28.086

# Moving window parameters
WINDOW_SIZE_TIME = 10      # Window height along time axis (pixels)
WINDOW_SIZE_CHANNEL = 3    # Window width along distance axis (pixels)
STEP_TIME = 5              # Step size along time axis (pixels)
STEP_CHANNEL = 1           # Step size along distance axis (pixels)

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
dt_str = basename.split('_')[2] + basename.split('_')[3]
dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
print(f"Processing {dt.strftime('%Y-%m-%d %H:%M:%S')}")

# ==== Load Image Data ====
data = np.load(file)
img_gray = data["waterfall"].mean(axis=2)  # Convert RGB to grayscale
DAS_data = img_gray
num_time_samples, num_channels = DAS_data.shape

time_per_sample = TOTAL_DURATION_SEC / num_time_samples
distance_per_channel = TOTAL_DISTANCE_KM / num_channels

# ==== Moving Window Feature Extraction ====
features_list = []
window_centers_time = []
window_centers_distance = []

num_windows_time = (num_time_samples - WINDOW_SIZE_TIME) // STEP_TIME + 1
num_windows_channel = (num_channels - WINDOW_SIZE_CHANNEL) // STEP_CHANNEL + 1
total_windows = num_windows_time * num_windows_channel

counter = 0
for i in range(0, num_time_samples - WINDOW_SIZE_TIME + 1, STEP_TIME):
    for j in range(0, num_channels - WINDOW_SIZE_CHANNEL + 1, STEP_CHANNEL):
        window = DAS_data[i:i+WINDOW_SIZE_TIME, j:j+WINDOW_SIZE_CHANNEL]
        feat_mean = window.mean()
        feat_std = window.std()
        feat_max = window.max()
        sobel_x = cv2.Sobel(window, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(window, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_feature = sobel_mag.mean()
        laplacian_feature = cv2.Laplacian(window, cv2.CV_64F).var()

        feature_vector = np.array([
            feat_mean, 
            feat_std, 
            feat_max, 
            sobel_feature, 
            laplacian_feature
        ])
        features_list.append(feature_vector)

        center_time = (i + WINDOW_SIZE_TIME / 2) * time_per_sample
        center_distance = (j + WINDOW_SIZE_CHANNEL / 2) * distance_per_channel
        window_centers_time.append(center_time)
        window_centers_distance.append(center_distance)

        counter += 1
        if counter % 100 == 0 or counter == total_windows:
            print(f"Processed {counter}/{total_windows} windows")

features_array = np.vstack(features_list)

# Save extracted features (optional)
np.savez("DSM_features_moving_window.npz", features=features_array,
         window_centers_time=window_centers_time,
         window_centers_distance=window_centers_distance)

# ==== Anomaly Detection ====
model = IsolationForest(contamination=CONTAMINATION_RATE, random_state=42)
model.fit(features_array)
scores = model.decision_function(features_array)
anomalous_idx = np.where(scores < 0.03)[0]

# ==== Create 2D Anomaly Map ====
anomaly_map = scores.reshape(num_windows_time, num_windows_channel)

binary_mask = np.zeros(anomaly_map.shape, dtype=int)
binary_mask[anomaly_map < 0.03] = 1

labels = label(binary_mask, connectivity=2)
filtered_anomalies = np.zeros_like(binary_mask)
for region in regionprops(labels):
    minr, minc, maxr, maxc = region.bbox
    if (maxc - minc > 3) or (maxr - minr > 3):
        continue
    filtered_anomalies[labels == region.label] = 1

anomalous_idx = np.where(filtered_anomalies.flatten() == 1)[0]

time_min = min(window_centers_time) - (WINDOW_SIZE_TIME / 2) * time_per_sample
time_max = max(window_centers_time) + (WINDOW_SIZE_TIME / 2) * time_per_sample
distance_min = min(window_centers_distance) - (WINDOW_SIZE_CHANNEL / 2) * distance_per_channel
distance_max = max(window_centers_distance) + (WINDOW_SIZE_CHANNEL / 2) * distance_per_channel

# ==== Plotting ====
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

im = ax1.imshow(anomaly_map, aspect='auto', cmap='gray',
                extent=[distance_min, distance_max, time_max, time_min],
                vmax=0.16, vmin=0)
ax1.set_ylabel("Time (sec)")

anomaly_distances = np.array([window_centers_distance[i] for i in anomalous_idx])
anomaly_times = np.array([window_centers_time[i] for i in anomalous_idx])
ax1.scatter(anomaly_distances, anomaly_times, color='red', edgecolors='k')

ax2.imshow(img_gray, aspect='auto', cmap=COLORMAP_WATERFALL,
           extent=[0, TOTAL_DISTANCE_KM, TOTAL_DURATION_SEC, 0])
ax2.set_xlabel("Distance (km)")
ax2.set_ylabel("Time (sec)")
ax2.set_title("Original DAS Waterfall")

plt.tight_layout()
plt.show()

# ==== Write Anomaly Log ====
with open('anomaly_points.log', 'w') as f:
    f.write("Anomaly Points Log (Distance_km, Time_sec)\n")
    f.write("=" * 40 + "\n")
    for d, t in zip(anomaly_distances, anomaly_times):
        f.write(f"{d:.4f}, {t:.2f}\n")

# %%
