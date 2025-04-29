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

for file in file_list:
    basename = os.path.splitext(os.path.basename(file))[0]
    dt_str = basename.split('_')[2] + basename.split('_')[3]
    dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
    print(f"\nProcessing {dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
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

    # ==== Extract Features across Distance ====
    features_list = []
    window_centers_distance = []
    print(f"Extracting features every {config.STEP_CHANNEL} pixels with a window size of {config.WINDOW_SIZE_CHANNEL} pixels\n")
    
    for j in range(0, num_channels - config.WINDOW_SIZE_CHANNEL + 1, config.STEP_CHANNEL):
        window = DAS_data[:, j:j+config.WINDOW_SIZE_CHANNEL]
        mean_time_series = np.mean(window, axis=1)
        #plt.plot(mean_time_series)
        #plt.show()
        #break
        fft_result = np.fft.fft(mean_time_series)
        fft_freq = np.fft.fftfreq(len(mean_time_series))  
        half = len(fft_result) // 2
        fft_result = fft_result[:half]
        fft_freq = fft_freq[:half]
        #print(np.abs(fft_result))
        #plt.figure(figsize=(10, 6))
        #plt.plot(fft_freq, np.abs(fft_result))  
        #plt.xlabel('Frequency')
        #plt.ylabel('Amplitude')
        #plt.title('FFT of Mean Per Row')
        #plt.grid(True)
        #plt.tight_layout()
        #plt.show()
        #feat_1 = np.percentile(np.abs(fft_result), 75) 
        feat_1 = np.median(window)
        feat_2 = window.std()
        feat_3 = np.percentile(window, 75)
        feature_vector = np.array([feat_1, feat_2, feat_3])
        features_list.append(feature_vector)
        center_distance = (j + config.WINDOW_SIZE_CHANNEL / 2) * distance_per_channel
        window_centers_distance.append(center_distance)
    features_array = np.vstack(features_list)
    output_path = os.path.join(config.FEATURES_FOLDER, f"features_{basename.split('_')[2]+'_'+basename.split('_')[3]}.npz")
    np.savez(output_path, features=features_array)
