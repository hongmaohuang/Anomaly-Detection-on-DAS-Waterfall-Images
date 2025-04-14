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
from pathlib import Path
import xdas
os.chdir(Path(__file__).resolve().parent)

# ==== CUSTOMIZABLE PARAMETERS ====
DATA_DATE = "20250331"
COLORMAP_WATERFALL = "jet"

# Font settings
FONT_PATH = '../../../Helvetica.ttf'
TITLE_FONT_PATH = '../../../Helvetica_Bold.ttf'
FONT_SIZE = 12
TITLE_FONT_SIZE = 18

DURATION_WATERFALL = 1  # in minutes
DATA_DIR = Path(f'../../Inputs/DAS_data/{DATA_DATE}/waveforms')
OUT_DIR = Path("../../Outputs/Features")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
WINDOW_SIZE_CHANNEL = 3
STEP_CHANNEL = 1

# ==== Duration and distance ====
waveform_files = list(DATA_DIR.glob('*.hdf5'))
data = xdas.open_mfdataarray(str(waveform_files[0]), engine="asn")
max_distance = np.max(data.coords['distance'].values)
TOTAL_DURATION_SEC = DURATION_WATERFALL * 60
TOTAL_DISTANCE_KM = max_distance / 1000

# ==== Setup Fonts ====
fm.fontManager.addfont(FONT_PATH)
title_font = FontProperties(fname=TITLE_FONT_PATH, size=TITLE_FONT_SIZE, weight='bold')
matplotlib.rcParams['font.family'] = 'Helvetica'
matplotlib.rcParams['font.size'] = FONT_SIZE

# ==== Load All Waterfall Images ====
waterfall_npz = f'../../Inputs/DAS_data/{DATA_DATE}/waterfall_npz/*.npz'
file_list = sorted(glob.glob(waterfall_npz))

for file in file_list:
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
    #print(f"Second per Pixel: {time_per_sample:.3f}")
    #print(f"KM per Pixel: {distance_per_channel:.2f}")
    #print(f"Window Size (Distance): {WINDOW_SIZE_CHANNEL * distance_per_channel:.2f} km")
    #print(f"Step Size (Distance): {STEP_CHANNEL * distance_per_channel:.2f} km")

    # ==== Extract Features across Distance ====
    features_list = []
    window_centers_distance = []

    num_windows_channel = (num_channels - WINDOW_SIZE_CHANNEL) // STEP_CHANNEL + 1
    for j in range(0, num_channels - WINDOW_SIZE_CHANNEL + 1, STEP_CHANNEL):
        window = DAS_data[:, j:j+WINDOW_SIZE_CHANNEL]

        feat_mean = window.mean()
        feat_std = window.std()
        feat_max = window.max()

        feature_vector = np.array([feat_mean, feat_std, feat_max])
        features_list.append(feature_vector)

        center_distance = (j + WINDOW_SIZE_CHANNEL / 2) * distance_per_channel
        window_centers_distance.append(center_distance)

    features_array = np.vstack(features_list)

    output_path = os.path.join(OUT_DIR, f"features_{basename.split('_')[2]+'_'+basename.split('_')[3]}.npz")
    np.savez(output_path, features=features_array)


