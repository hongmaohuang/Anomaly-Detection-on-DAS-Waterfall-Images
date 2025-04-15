import os
import glob
import numpy as np
from datetime import datetime
import xdas
from pathlib import Path
import config 
os.makedirs(config.FEATURES_FOLDER, exist_ok=True)

# ==== Load All Waterfall Images ====
file_list = sorted(glob.glob(f'{config.WATERFALL_NPZ_FOLDER}/*.npz'))

# ==== Duration and distance ====
waveform_files = list(Path(config.DAS_WAVEFORM_PATH).glob('*.hdf5'))
data = xdas.open_mfdataarray(str(waveform_files[0]), engine="asn")
max_distance = np.max(data.coords['distance'].values)
TOTAL_DURATION_SEC = config.DURATION_WATERFALL * 60
TOTAL_DISTANCE_KM = max_distance / 1000

for file in file_list:
    basename = os.path.basename(file)
    dt_str = basename.split('_')[2] + basename.split('_')[3]
    dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
    print(f"Processing {dt.strftime('%Y-%m-%d %H:%M:%S')}")
    data = np.load(file)
    img_gray = data["waterfall"].mean(axis=2)  
    DAS_data = img_gray
    num_time_samples, num_channels = DAS_data.shape
    time_per_sample = TOTAL_DURATION_SEC / num_time_samples
    distance_per_channel = TOTAL_DISTANCE_KM / num_channels
    if file == file_list[0]:
        print(f"Time per Sample: {time_per_sample:.3f} sec")
        print(f"Distance per Channel: {distance_per_channel:.2f} km")
        print(f"Total Duration: {TOTAL_DURATION_SEC:.2f} sec")
        print(f"Total Distance: {TOTAL_DISTANCE_KM:.2f} km")

    # ==== Extract Features across Distance ====
    features_list = []
    window_centers_distance = []
    num_windows_channel = (num_channels - config.WINDOW_SIZE_CHANNEL) // config.STEP_CHANNEL + 1
    for j in range(0, num_channels - config.WINDOW_SIZE_CHANNEL + 1, config.STEP_CHANNEL):
        window = DAS_data[:, j:j+config.WINDOW_SIZE_CHANNEL]
        feat_mean = window.mean()
        feat_std = window.std()
        feat_max = window.max()
        feature_vector = np.array([feat_mean, feat_std, feat_max])
        features_list.append(feature_vector)
        center_distance = (j + config.WINDOW_SIZE_CHANNEL / 2) * distance_per_channel
        window_centers_distance.append(center_distance)
    features_array = np.vstack(features_list)
    output_path = os.path.join(config.FEATURES_FOLDER, f"features_{basename.split('_')[2]+'_'+basename.split('_')[3]}.npz")
    np.savez(output_path, features=features_array)
