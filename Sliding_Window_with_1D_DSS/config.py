import os
import numpy as np
import xdas
from pathlib import Path
import shutil

# ==== Inputs ==== #
DATE = "20250331"
DAS_DATA_PATH = f'../../Inputs/{DATE}'
DAS_WATERFALL_PATH = os.path.join(DAS_DATA_PATH, "waterfall_images_1min")
DAS_WAVEFORM_PATH = os.path.join(DAS_DATA_PATH, "waveforms")

# ==== Outputs ==== #
OUTPUT_PATH = f'../../Outputs/{DATE}'
WATERFALL_NPZ_FOLDER = os.path.join(OUTPUT_PATH, "waterfall_npz")
FEATURES_FOLDER = os.path.join(OUTPUT_PATH, "features")
WAVELET_FOLDER = os.path.join(OUTPUT_PATH, "wavelets")
SCATTERING_COEFFICIENTS_FOLDER = os.path.join(OUTPUT_PATH, "scattering_coefficients")
PCA_ICA_FOLDER = os.path.join(OUTPUT_PATH, "pca_ica")
CLUSTERING_RESULTS_FOLDER = os.path.join(OUTPUT_PATH, "clustering_results")

# ==== Image Crops ==== #
CROP_OR_NOT = 'NO'  # 'YES' or 'NO'
CROP_TOP = 160
CROP_BOTTOM = 696
CROP_LEFT = 70
CROP_RIGHT = 1789

# ==== Waterfall Parameters ==== #
DURATION_WATERFALL = 1  # in minutes
waveform_files = list(Path(DAS_WAVEFORM_PATH).glob('*.hdf5'))
data = xdas.open_mfdataarray(str(waveform_files[0]), engine="asn") 
max_distance = np.max(data.coords['distance'].values)
# the waveform data is just for extracting the distance, no need to load the entire dataset
TOTAL_DISTANCE_KM = max_distance / 1000
TOTAL_DURATION_SEC = DURATION_WATERFALL * 60

# ==== Sliding Window Parameters ==== #
WINDOW_SIZE_CHANNEL = 3
STEP_CHANNEL = 1

# ==== Scattering Network Parameters ==== #
SEGMENT_DISTANCE = 0.04  # in km
SEGMENT_OVERLAP = 0.2  # as fraction (e.g., 0.2 = 20% overlap)
OCTAVES_1 = 8
RESOLUTION_1 = 6
QUALITY_1 = 1
OCTAVES_2 = 8
RESOLUTION_2 = 2
QUALITY_2 = 1

# ==== PCA and ICA ==== #
PCA_COMPONENTS = 10
ICA_COMPONENTS = 4 # Could be estimated using PCA Result 

# ==== KMeans Parameters ==== #
N_CLUSTERS = 6

# ==== Visualization Parameters ==== #
SMOOTH_KERNEL = 20
