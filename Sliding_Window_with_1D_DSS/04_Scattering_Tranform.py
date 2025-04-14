# %%
# Apply the scattering network to the three-component seismic data
#

import pickle
import numpy as np
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
import obspy
from datetime import datetime, timedelta
import os
from pathlib import Path
from obspy import Trace, Stream, UTCDateTime

os.chdir(Path(__file__).resolve().parent)

#plt.rcParams["date.converter"] = "concise"
OUT_DIR = Path("../../Outputs/wavelets")
OUT_DIR_scattering = Path("../../Outputs/scattering_coefficients")
OUT_DIR_scattering.mkdir(parents=True, exist_ok=True)

FEATURES_DIR = Path("../../Outputs/Features")

network = pickle.load(open(f"{OUT_DIR}/scattering_network.pickle", "rb"))
channel_id = 0

print("Loading Data.....")
file = sorted(FEATURES_DIR.glob("features_*.npz"))
data = np.load(file[0])
features_array = data["features"] 
mean_values = features_array[:, 0]
std_values = features_array[:, 1]
max_values = features_array[:, 2]


# Fake the start time
starttime = UTCDateTime("2025-01-01T00:00:00")

header = {
    'npts': len(mean_values),
    'sampling_rate': network.sampling_rate,  # 每秒一個點，代表這些是 index-based 的序列
    'starttime': starttime
}

# 建立每條 trace
tr_mean = Trace(data=np.array(mean_values, dtype=np.float32), header=header.copy())
tr_std = Trace(data=np.array(std_values, dtype=np.float32), header=header.copy())
tr_max = Trace(data=np.array(max_values, dtype=np.float32), header=header.copy())

tr_mean.stats.channel = "MEAN"
tr_std.stats.channel = "STD"
tr_max.stats.channel = "MAX"

feature_stream = Stream(traces=[tr_mean, tr_std, tr_max])


# Extract segment length (from any layer)
segment_dist_km = network.bins / network.sampling_rate
print("each segment distance is: ", segment_dist_km)
overlap = 0.2

segment_len_samples = int(segment_dist_km * network.sampling_rate)  # segment 長度（以 sample 為單位）
step_samples = int(segment_len_samples * (1 - overlap))  # sliding 的步長（以 sample 為單位）
total_segments = (len(tr_mean.data) - segment_len_samples) // step_samples + 1

segments = []
distance_all = []

for i in range(total_segments):
    start_idx = i * step_samples
    end_idx = start_idx + segment_len_samples

    if end_idx > len(tr_mean.data):
        break

    seg = [trace.data[start_idx:end_idx] for trace in feature_stream]
    segments.append(np.array(seg))
    distance_all.append(i * segment_dist_km * (1 - overlap))
print("Running the scattering transform.....")
scattering_coefficients = network.transform(segments, reduce_type=np.median)

np.savez(
    f"{OUT_DIR_scattering}/scattering_coefficients.npz",
    order_1=scattering_coefficients[0],
    order_2=scattering_coefficients[1],
    distance=distance_all,
)
