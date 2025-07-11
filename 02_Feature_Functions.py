import os
import glob
import numpy as np
from datetime import datetime
import config 
import shutil
import matplotlib.cm as cm, matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from skimage import exposure
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.filters import gaussian, gabor

#if os.path.exists(config.FEATURES_FOLDER):
#    shutil.rmtree(config.FEATURES_FOLDER)
#os.makedirs(config.FEATURES_FOLDER)
os.makedirs(config.FEATURES_FOLDER, exist_ok=True)

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
    out_single = os.path.join(config.FEATURES_FOLDER, f"features_{dt_str}.npz")
    if os.path.exists(out_single):
        print(f"Skipping existing {out_single}")
        saved = np.load(out_single)["features"]
        feat_1_all.extend(saved[0])
        feat_2_all.extend(saved[1])
        feat_3_all.extend(saved[2])
        continue
    #print(f"\nProcessing {dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
    data = np.load(file)
    #img_gray = data["waterfall"].mean(axis=2)
    wf_rgba = data["waterfall"]
    wf_rgb  = wf_rgba[..., :3].astype(np.uint8)  
    jet = cm.get_cmap("jet", 256)(np.arange(256))[:, :3]
    idx_img = cKDTree(jet).query(wf_rgb.reshape(-1,3), k=1)[1]\
                        .reshape(wf_rgb.shape[:2]).astype(np.uint8)
    dmin, dmax = config.ORIGINAL_DMIN, config.ORIGINAL_DMAX
    wf_db = dmin + idx_img.astype(np.float32)*(dmax-dmin)/255.0
    rms_img = wf_db
    num_time_samples, num_channels = rms_img.shape
    dt_sec = config.DURATION_WATERFALL * 60 / num_time_samples         
    dx_m   = config.TOTAL_DISTANCE_KM * 1000 / num_channels
    dI_dt, dI_dx = np.gradient(rms_img, dt_sec, dx_m)
    ''' 
    eps = 1e-6
    slope_img = dI_dt / (dI_dx + eps)
    aspect_rad = np.arctan2(dI_dt, dI_dx)
    aspect_deg_img = (np.degrees(aspect_rad) + 360) % 360
    asp_slo_prod_img = aspect_deg_img * slope_img
    '''
    time_per_sample = config.TOTAL_DURATION_SEC / num_time_samples
    distance_per_channel = config.TOTAL_DISTANCE_KM / num_channels
    
    thetas = np.linspace(0, np.pi, 8, endpoint=False)
    responses = []
    for th in thetas:
        real, imag = gabor(rms_img, frequency=0.05, theta=th)
        mag = np.hypot(real, imag)
        responses.append(mag)
    responses = np.stack(responses, axis=0)
    sloped_idxs = [np.argmin(np.abs(thetas - np.pi/4)),
                np.argmin(np.abs(thetas - 3*np.pi/4))]
    gabor_sloped = np.max(responses[sloped_idxs], axis=0)

    feat_1 = []
    feat_2 = []
    feat_3 = []
 
    if file == file_list[0]:
        print(f"Time per Sample: {time_per_sample:.3f} sec")
        print(f"Distance per Channel: {distance_per_channel*1000:.5f} m")
        print(f"Total Duration: {config.TOTAL_DURATION_SEC:.2f} sec")
        print(f"Total Distance: {config.TOTAL_DISTANCE_KM:.2f} km\n")
        print(f"Extracting features every {config.STEP_CHANNEL} pixels with a window size of {config.WINDOW_SIZE_CHANNEL} pixels\n")
    for j in range(0, num_channels - config.WINDOW_SIZE_CHANNEL + 1, config.STEP_CHANNEL):
        window_1 = rms_img[:, j:j+config.WINDOW_SIZE_CHANNEL]
        window_2 = gabor_sloped[:, j:j+config.WINDOW_SIZE_CHANNEL]
        feat_1_val = window_1.std()
        feat_2_val = window_2.mean()
        window_3 = rms_img[:, j:j+config.WINDOW_SIZE_CHANNEL].astype(np.float16)
        feat_3_in_one = []
        for k in range(window_3.shape[1]):
            col = window_3[:, k]
            col -= col.mean()
            fft_vals = np.fft.rfft(col)
            amp_spec = np.abs(fft_vals) / col.size
            freqs    = np.fft.rfftfreq(col.size, d=dt_sec)
            feat_3_in_one.append(freqs[np.argmax(amp_spec)])
        feat_3_val = np.mean(feat_3_in_one)
        ''' 
        window = window_1 #original use
        feat_1 = np.median(window) #original use
        feat_2 = window.std() #original use
        feat_3 = np.percentile(window, 75) #original use
        '''
        feat_1.append(feat_1_val)
        feat_2.append(feat_2_val)
        feat_3.append(feat_3_val)
        feat_1_all.append(feat_1_val)
        feat_2_all.append(feat_2_val)
        feat_3_all.append(feat_3_val)

    features_single = np.vstack([feat_1, feat_2, feat_3])
    np.savez(os.path.join(config.FEATURES_FOLDER, f"features_{dt_str}.npz"), features=features_single)
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
