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
from sklearn.neighbors import LocalOutlierFactor
from skimage.measure import label, regionprops
import xdas
from pathlib import Path

# ==== CUSTOMIZABLE PARAMETERS ====
# Data settings
DATA_DATE = "20250331"
TARGET_IDX = 0
CONTAMINATION_RATE = 0.03
COLORMAP_WATERFALL = "jet"

# Font settings (optional)
FONT_PATH = './Helvetica.ttf'
TITLE_FONT_PATH = './Helvetica_Bold.ttf'
FONT_SIZE = 12
TITLE_FONT_SIZE = 18
DURATION_WATERFALL = 17 
# (in minutes)
DATA_DIR = Path(f'./DAS_data/{DATA_DATE}/waveforms')



# Moving window parameters
WINDOW_SIZE_TIME = 10      # Window height along time axis (pixels)
WINDOW_SIZE_CHANNEL = 3    # Window width along distance axis (pixels)
STEP_TIME = 5              # Step size along time axis (pixels)
STEP_CHANNEL = 1           # Step size along distance axis (pixels)


# ==== Duration and distance of DAS waterfall ====
waveform_files = list(DATA_DIR.glob('*.hdf5'))
data = xdas.open_mfdataarray(str(waveform_files[0]), engine="asn")
max_distance = np.max(data.coords['distance'].values)
TOTAL_DURATION_SEC = DURATION_WATERFALL * 60
TOTAL_DISTANCE_KM = max_distance/1000

# ==== Setup Fonts ====
fm.fontManager.addfont(FONT_PATH)
title_font = FontProperties(fname=TITLE_FONT_PATH, size=TITLE_FONT_SIZE, weight='bold')
matplotlib.rcParams['font.family'] = 'Helvetica'
matplotlib.rcParams['font.size'] = FONT_SIZE

# ==== Load DAS File ====
waterfall_npz = f'./DAS_data/{DATA_DATE}/waterfall_npz/*.npz'
file_list = sorted(glob.glob(waterfall_npz))
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
        hist, _ = np.histogram(window, bins=32, density=True)
        hist = hist[hist > 0]  # 避免 log(0)
        entropy_feature = -np.sum(hist * np.log2(hist))
        laplacian_feature = cv2.Laplacian(window, cv2.CV_64F).var()

        feature_vector = np.array([
            feat_mean, 
            feat_std, 
            feat_max, 
            entropy_feature, 
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
#np.savez("DSM_features_moving_window.npz", features=features_array,
#         window_centers_time=window_centers_time,
#         window_centers_distance=window_centers_distance)

# ==== Anomaly Detection using LOF ====
lof = LocalOutlierFactor(n_neighbors=150, novelty=False)
y_pred = lof.fit_predict(features_array)
lof_scores = -lof.negative_outlier_factor_  

# ==== Create 2D Anomaly Map ====
anomaly_map = lof_scores.reshape(num_windows_time, num_windows_channel)
threshold = np.percentile(lof_scores, 99)  
anomalous_idx = np.where(lof_scores > threshold)[0]

# ==== Write Anomaly Log ====
anomaly_distances = np.array([window_centers_distance[i] for i in anomalous_idx])
anomaly_times = np.array([window_centers_time[i] for i in anomalous_idx])
filename = os.path.basename(file)
parts = filename.split('_')
log_name = f'./DAS_data/{DATA_DATE}/anomaly_points_{parts[2]}_{parts[3].split(".")[0]}_anomaly.log'
with open(log_name, 'w') as f:
    f.write("Anomaly Points Log (Distance_km, Time_sec)\n")
    f.write(f"Starting Time: {dt.strftime('%Y%m%d_%H%M%S')}\n")
    f.write("=" * 40 + "\n")
    for d, t in zip(anomaly_distances, anomaly_times):
        f.write(f"{d:.4f}, {t:.2f}\n")

# ==== Plotting ====
time_min = min(window_centers_time) - (WINDOW_SIZE_TIME / 2) * time_per_sample
time_max = max(window_centers_time) + (WINDOW_SIZE_TIME / 2) * time_per_sample
distance_min = min(window_centers_distance) - (WINDOW_SIZE_CHANNEL / 2) * distance_per_channel
distance_max = max(window_centers_distance) + (WINDOW_SIZE_CHANNEL / 2) * distance_per_channel

fig = plt.figure(figsize=(14, 5))
plt.imshow(img_gray, aspect='auto', cmap=COLORMAP_WATERFALL,
           extent=[0, TOTAL_DISTANCE_KM, TOTAL_DURATION_SEC, 0])
plt.scatter(anomaly_distances, anomaly_times, color='red', edgecolors='k')
plt.xlabel("Distance (km)")
plt.ylabel("Time (sec)")
plt.title("Original DAS Waterfall")
plt.tight_layout()
plt.show()

# %%
