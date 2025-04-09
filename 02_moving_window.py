# %%
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from sklearn.ensemble import IsolationForest
from datetime import datetime
from matplotlib.font_manager import FontProperties
import matplotlib
import matplotlib.font_manager as fm
import cv2
from skimage.measure import label, regionprops

# ======= User Parameters =======
data_date = "20250331"
target_idx = 1
contamination_rate = 0.03
colormap_waterfall = "jet"

# ======= Font (Optional) =======
fm.fontManager.addfont('./Helvetica.ttf')
title_font = FontProperties(fname='./Helvetica_Bold.ttf', size=18, weight='bold')
matplotlib.rcParams['font.family'] = 'Helvetica'
matplotlib.rcParams['font.size'] = 12

# ======= Constants =======
total_duration_sec = 17 * 60
total_distance_km = 28.086

# ======= Load File =======
file_pattern = f'./DAS_data/{data_date}/waterfall_npz/*.npz'
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
img_gray = data["waterfall"].mean(axis=2)  # RGB 轉為灰階
DAS_data = img_gray
num_time_samples, num_channels = DAS_data.shape

# ======= Moving Window 參數 =======
# 視窗大小（以 pixel 為單位），視窗大小應根據影像解析度與實際需求調整
window_size_time = 10      # 時間軸視窗高度
window_size_channel = 3   # 距離（頻道）軸視窗寬度
# 移動步長（可重疊，數值越小重疊越多）
step_time = 5    # 時間軸步長
step_channel = 1           # 距離軸步長

# 每個 pixel 對應的時間與距離
time_per_sample = total_duration_sec / num_time_samples
distance_per_channel = total_distance_km / num_channels

# ======= 移動視窗特徵擷取 =======
features_list = []
window_centers_time = []
window_centers_distance = []

# 計算總共的視窗數
num_windows_time = (num_time_samples - window_size_time) // step_time + 1
num_windows_channel = (num_channels - window_size_channel) // step_channel + 1
total_windows = num_windows_time * num_windows_channel

counter = 0  # 計數器，追蹤已處理的視窗數

# 雙層迴圈計算moving window特徵
for i in range(0, num_time_samples - window_size_time + 1, step_time):
    for j in range(0, num_channels - window_size_channel + 1, step_channel):
        window = DAS_data[i:i+window_size_time, j:j+window_size_channel]

        # 基礎統計特徵
        feat_mean = window.mean()
        feat_std = window.std()
        feat_max = window.max()

        # Gradient特徵 (Sobel平均值與Laplacian變異數)
        sobel_x = cv2.Sobel(window, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(window, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_feature = sobel_mag.mean()

        laplacian_feature = cv2.Laplacian(window, cv2.CV_64F).var()

        # 組合所有特徵成feature_vector
        feature_vector = np.array([
            feat_mean, 
            feat_std, 
            feat_max, 
            sobel_feature, 
            laplacian_feature
        ])
        features_list.append(feature_vector)

        center_time = (i + window_size_time / 2) * time_per_sample
        center_distance = (j + window_size_channel / 2) * distance_per_channel
        window_centers_time.append(center_time)
        window_centers_distance.append(center_distance)


        # 更新處理進度
        counter += 1
        # 每處理 100 個視窗或最後一個視窗時印出進度
        if counter % 100 == 0 or counter == total_windows:
            print(f"Processed {counter}/{total_windows} windows")


# 將所有視窗的特徵整合成矩陣，每一列為一個視窗的特徵
features_array = np.vstack(features_list)

# 儲存特徵（可選擇性儲存）
np.savez("DSM_features_moving_window.npz", features=features_array,
         window_centers_time=window_centers_time,
         window_centers_distance=window_centers_distance)
# %%
# ======= 利用 IsolationForest 進行異常偵測 =======
model = IsolationForest(contamination=contamination_rate, random_state=42)
model.fit(features_array)
scores = model.decision_function(features_array)
anomalous_idx = np.where((scores < 0.03))[0]

# ======= 將視窗排列成 2D 格狀，便於重建異常分數影像 =======
num_windows_time = (num_time_samples - window_size_time) // step_time + 1
num_windows_channel = (num_channels - window_size_channel) // step_channel + 1
anomaly_map = scores.reshape(num_windows_time, num_windows_channel)

# 建立 binary mask (0: 正常, 1: 異常)
binary_mask = np.zeros(anomaly_map.shape, dtype=int)
binary_mask[anomaly_map < 0.03] = 1  # 閾值可自行調整

# 標記連通區域
labels = label(binary_mask, connectivity=2)

# 去除連續區域過大的事件
filtered_anomalies = np.zeros_like(binary_mask)
for region in regionprops(labels):
    minr, minc, maxr, maxc = region.bbox
    # 如果區域過大（例如橫跨距離方向或時間方向），直接略過
    if (maxc - minc > 3) or (maxr - minr > 3):  # 此閾值可調整
        continue
    else:
        filtered_anomalies[labels == region.label] = 1

# 重新獲得真正 anomalous_idx 的座標
anomalous_idx = np.where(filtered_anomalies.flatten() == 1)[0]

# 計算異常影像的 x 與 y 軸 extent 值
# 以各視窗中心座標來估算，這裡假設影像的時間與距離分別獨立排列
time_min = min(window_centers_time) - (window_size_time/2)*time_per_sample
time_max = max(window_centers_time) + (window_size_time/2)*time_per_sample
distance_min = min(window_centers_distance) - (window_size_channel/2)*distance_per_channel
distance_max = max(window_centers_distance) + (window_size_channel/2)*distance_per_channel


# ======= 畫圖 =======
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

# 畫出 anomaly heatmap，保留時間與距離的空間位置
im = ax1.imshow(anomaly_map, aspect='auto', cmap='gray',
                extent=[distance_min, distance_max, time_max, time_min],vmax=0.16, vmin=0)
#ax1.set_xlabel("Distance (km)")
ax1.set_ylabel("Time (sec)")
#ax1.set_title("Anomaly Detection Heatmap (Moving Window)")
#fig.colorbar(im, ax=ax1, label="Anomaly Score")

# 在 heatmap 上標註偵測到異常的視窗中心
# 將異常 marker 的中心座標轉換成 NumPy 陣列
anomaly_distances = np.array([window_centers_distance[i] for i in anomalous_idx])
anomaly_times = np.array([window_centers_time[i] for i in anomalous_idx])

# 一次性繪製所有 marker
ax1.scatter(anomaly_distances, anomaly_times, color='red', edgecolors='k')

# 畫出原始 DAS 水瀑影像供參考
ax2.imshow(img_gray, aspect='auto', cmap=colormap_waterfall,
           extent=[0, total_distance_km, total_duration_sec, 0])
ax2.set_xlabel("Distance (km)")
ax2.set_ylabel("Time (sec)")
ax2.set_title("Original DAS Waterfall")

plt.tight_layout()
plt.show()

log_filename = 'anomaly_points.log'
with open(log_filename, 'w') as f:
    f.write("Anomaly Points Log (Distance_km, Time_sec)\n")
    f.write("="*40 + "\n")
    for d, t in zip(anomaly_distances, anomaly_times):
        f.write(f"{d:.4f}, {t:.2f}\n")

# %%

