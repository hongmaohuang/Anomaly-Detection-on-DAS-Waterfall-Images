import os
import glob
import numpy as np
from datetime import datetime
import config 
import shutil
import matplotlib.pyplot as plt
if os.path.exists(config.FEATURES_FOLDER):
    shutil.rmtree(config.FEATURES_FOLDER)
os.makedirs(config.FEATURES_FOLDER)

# ==== Load All Waterfall Images ====
file_list = sorted(glob.glob(f'{config.WATERFALL_NPZ_FOLDER}/*.npz'))
feat_1_all = []
feat_2_all = []
feat_3_all = []
for idx, file in enumerate(file_list):
    print(f"[{idx}/{len(file_list)}]")
    basename = os.path.splitext(os.path.basename(file))[0]
    dt_str = basename.split('_')[2] + basename.split('_')[3]
    dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
    #print(f"\nProcessing {dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
    data = np.load(file)
    img_gray = data["waterfall"].mean(axis=2)
    DAS_data = img_gray
    num_time_samples, num_channels = DAS_data.shape
    time_per_sample = config.TOTAL_DURATION_SEC / num_time_samples
    distance_per_channel = config.TOTAL_DISTANCE_KM / num_channels
    if file == file_list[0]:
        print(f"Time per Sample: {time_per_sample:.3f} sec")
        print(f"Distance per Channel: {distance_per_channel*1000:.5f} m")
        print(f"Total Duration: {config.TOTAL_DURATION_SEC:.2f} sec")
        print(f"Total Distance: {config.TOTAL_DISTANCE_KM:.2f} km\n")
        print(f"Extracting features every {config.STEP_CHANNEL} pixels with a window size of {config.WINDOW_SIZE_CHANNEL} pixels\n")
    
    for j in range(0, num_channels - config.WINDOW_SIZE_CHANNEL + 1, config.STEP_CHANNEL):
        window = DAS_data[:, j:j+config.WINDOW_SIZE_CHANNEL]
        feat_1 = np.median(window)
        feat_2 = window.std()
        feat_3 = np.percentile(window, 75)
        feat_1_all.append(feat_1)
        feat_2_all.append(feat_2)
        feat_3_all.append(feat_3)

feature_vector = np.array([feat_1_all, feat_2_all, feat_3_all])
#center_distance = (j + config.WINDOW_SIZE_CHANNEL / 2) * distance_per_channel
#window_centers_distance.append(center_distance)
features_array = np.vstack(feature_vector)
base_no_ext = os.path.splitext(os.path.basename(file_list[0]))[0]
final_no_ext = os.path.splitext(os.path.basename(file_list[-1]))[0]
base_parts = base_no_ext.split('_')
base_parts_final = final_no_ext.split('_')
output_path = os.path.join(
    config.FEATURES_FOLDER,
    f"features_{base_parts[2]}_{base_parts[3]}-{base_parts_final[2]}_{base_parts_final[3]}.npz"
)
np.savez(output_path, features=features_array)
