import os
import numpy as np
import xdas
from pathlib import Path
import shutil

# ==== Inputs ==== #
DAS_DATA_PATH = f'/raid4/DAS_data/OptoDAS_HOPE/HOPE/'
DAS_WATERFALL_PATH = os.path.join(DAS_DATA_PATH, "screenshots_HOPB_20250429/Waterfall_RMS_202505*.png")
DAS_WAVEFORM_PATH = os.path.join(DAS_DATA_PATH, "20250518/proc")

# Optional time range for selecting waterfall images. Use the format
# "YYYYMMDD_HHMMSS". If either value is "None", the full dataset is used.
START_TIME = "20250515_000000"  # e.g. "20250515_000000"
END_TIME = "20250518_235959"    # e.g. "20250518_235959"
ORIGINAL_DMIN = -113.0  # Original minimum dB value in the waterfall images
ORIGINAL_DMAX = -35.0   # Original maximum dB value in the waterfall images

# ==== Outputs ==== #
OUTPUT_PATH = f'/home/hmhuang/Work/Hualien_DAS_Monitoring/Outputs/'
WATERFALL_NPZ_FOLDER = os.path.join(OUTPUT_PATH, "waterfall_npz")
FEATURES_FOLDER = os.path.join(OUTPUT_PATH, "features")
WAVELET_FOLDER = os.path.join(OUTPUT_PATH, "wavelets")
SCATTERING_COEFFICIENTS_FOLDER = os.path.join(OUTPUT_PATH, "scattering_coefficients")
PCA_ICA_FOLDER = os.path.join(OUTPUT_PATH, "pca_ica")
CLUSTERING_RESULTS_FOLDER = os.path.join(OUTPUT_PATH, "clustering_results")
VISUALIZATION_FOLDER = os.path.join(OUTPUT_PATH, "visualizations")

# ==== Image Crops ==== #
CROP_OR_NOT = 'YES'  # 'YES' or 'NO'
CROP_TOP = 160
CROP_BOTTOM = 696
CROP_LEFT = 70
CROP_RIGHT = 1789
# OptoDAS Default

# ==== Waterfall Parameters ==== #
DURATION_WATERFALL = 1  # in minutes
waveform_files = list(Path(DAS_WAVEFORM_PATH).glob('*.hdf5'))
data = xdas.open_mfdataarray(str(waveform_files[0]), engine="asn") 
max_distance = np.max(data.coords['distance'].values)
# the waveform data is just for extracting the distance, no need to load the entire dataset
TOTAL_DISTANCE_KM = max_distance / 1000
# If you have the extact distance, you can set it directly here
TOTAL_DURATION_SEC = DURATION_WATERFALL * 60

# ==== Sliding Window Parameters ==== #
WINDOW_SIZE_CHANNEL = 3
STEP_CHANNEL = 2

# ==== Scattering Network Parameters ==== #
SEGMENT_DISTANCE = 0.2  # in km
SEGMENT_OVERLAP = 0.2  # as fraction (e.g., 0.2 = 20% overlap)
OCTAVES_1 = 4
RESOLUTION_1 = 6
QUALITY_1 = 1
OCTAVES_2 = 2
RESOLUTION_2 = 4
QUALITY_2 = 1

# ==== PCA and ICA ==== #
RUN_PCA = "YES" # "YES" or "NO"
PCA_COMPONENTS = 5
ICA_COMPONENTS = 5 # Could be estimated using PCA Result 

# ==== Clustering Parameters ==== #
CLUSTER_METHOD = "gmm"   # "kmeans", "gmm", "dbscan", "agglomerative"
KMEANS_CLUSTERS = 5
GMM_N_COMPONENTS = 6
GMM_COVARIANCE_TYPE = "full"
SELECT_CLUSTERS_WITH = "none"  # "none", "aic", or "bic"
GMM_COMPONENT_RANGE = (2, 10)   # used when SELECT_CLUSTERS_WITH is "aic" or "bic"
DBSCAN_EPS = 0.6
DBSCAN_MIN_SAMPLES = 50
AGG_N_CLUSTERS = 5
AGG_LINKAGE = "single"  # "ward", "complete", "average", "single"

# ==== Visualization Parameters ==== #
OCCURRENCES_LOC = 13.8896
ACCUMULATIONS_PER_FILE = 4
IMAGE_STEPS = 2     # Extract images every n files
IMAGE_PER_GROUP = 5 # Number of images in each group



