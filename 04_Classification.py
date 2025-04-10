# %%
import glob
import re
import numpy as np
import scipy.stats as stats
from obspy import read
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd

DATE = '20250331'
# =================================
# Natural sorting function
# =================================
def natural_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

# =================================
# Read files and sort (all .mseed files)
# =================================
all_files = sorted(glob.glob('./Detected_anomalies/*.mseed'), key=natural_key)
print(f"Number of files read: {len(all_files)}")

# =================================
# Read waveforms and extract header information
# =================================
waveforms = []       # To store waveform data (numpy arrays)
header_info = []     # To store header info: starttime, endtime, station
for file in all_files:
    st = read(file)
    trace = st[0]  # Default: take the first trace
    waveform_data = trace.data
    waveforms.append(waveform_data)
    # Extract starttime, endtime, and station from trace.stats
    header_info.append({
        "starttime": trace.stats.starttime, 
        "endtime": trace.stats.endtime, 
        "station": trace.stats.station
    })

print(f"Total number of waveforms: {len(waveforms)}")

# =================================
# Define a function to compute signal entropy
# =================================
def compute_entropy(signal, bins=10):
    hist, _ = np.histogram(signal, bins=bins, density=True)
    # Filter out zeros to avoid log(0)
    hist = hist[hist > 0]
    return stats.entropy(hist)

# =================================
# Define a function to extract 13 features from a single waveform
# =================================
def extract_features(waveform, bins=8, sampling_rate=1.0):
    # (1) Normalize waveform (avoid division by zero by adding a small constant)
    norm_waveform = (waveform - np.mean(waveform)) / (np.std(waveform) + 1e-8)
    
    # (2) Time domain features: avg, max, min, std, kurtosis, entropy
    avg_time       = np.mean(norm_waveform)
    max_time       = np.max(norm_waveform)
    min_time       = np.min(norm_waveform)
    std_time       = np.std(norm_waveform)
    kurtosis_time  = stats.kurtosis(norm_waveform)
    entropy_time   = compute_entropy(norm_waveform, bins=bins)
    
    # (3) Frequency domain features: compute FFT and 6 statistics
    fft_vals = np.abs(np.fft.rfft(norm_waveform))
    freqs = np.fft.rfftfreq(len(norm_waveform), d=1.0/sampling_rate)
    avg_freq       = np.mean(fft_vals)
    max_freq       = np.max(fft_vals)
    min_freq       = np.min(fft_vals)
    std_freq       = np.std(fft_vals)
    kurtosis_freq  = stats.kurtosis(fft_vals)
    entropy_freq   = compute_entropy(fft_vals, bins=bins)
    
    # (4) Get the frequency with the highest power
    idx_peak = np.argmax(fft_vals)
    peak_frequency = freqs[idx_peak]
    
    # (5) Combine into a 13-dimensional feature vector
    features = [avg_time, max_time, min_time, std_time, kurtosis_time, entropy_time,
                avg_freq, max_freq, min_freq, std_freq, kurtosis_freq, entropy_freq,
                peak_frequency]
    return features

# =================================
# Build the feature matrix (each row contains 13 features for a waveform)
# =================================
feature_matrix = []
for wf in waveforms:
    features = extract_features(wf, bins=8, sampling_rate=1.0)
    feature_matrix.append(features)
feature_matrix = np.array(feature_matrix)
print("Feature matrix shape:", feature_matrix.shape)

# %%
# =================================
# Apply PCA to reduce to 2 dimensions (visualization is not performed here)
# =================================
pca = PCA(n_components=2)
features_2d = pca.fit_transform(feature_matrix)

# =================================
# KMeans clustering (default set to 3 clusters; adjust parameters as needed)
# =================================
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(feature_matrix)

# =================================
# Create an output table with: waveform index, starttime, endtime, station, and cluster label
# =================================
records = []
for i, header in enumerate(header_info):
    record = {
        "starttime": str(header["starttime"]),  # Convert to string
        "endtime": str(header["endtime"]),
        "Distance (m)": header["station"],
        "cluster": int(labels[i])
    }
    records.append(record)

df = pd.DataFrame(records)
csv_filename = f"./DAS_data/{DATE}/waveform_cluster_results.csv"
df.to_csv(csv_filename, index=False)
print(f"CSV file generated: {csv_filename}")


# %%