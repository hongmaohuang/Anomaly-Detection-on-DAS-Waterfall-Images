import pickle
import numpy as np
import os
from pathlib import Path
from obspy import Trace, Stream, UTCDateTime
import config
os.makedirs(config.SCATTERING_COEFFICIENTS_FOLDER, exist_ok=True)

network = pickle.load(open(f"{config.WAVELET_FOLDER}/scattering_network.pickle", "rb"))
file = sorted(Path(config.FEATURES_FOLDER).glob("features_*.npz"))
data = np.load(file[0])
features_array = data["features"] 
mean_values = features_array[:, 0]
std_values = features_array[:, 1]
max_values = features_array[:, 2]

# Fake the start time
starttime = UTCDateTime("2025-01-01T00:00:00")
header = {
    'npts': len(mean_values),
    'sampling_rate': network.sampling_rate, 
    'starttime': starttime
}
# Traces
tr_mean = Trace(data=np.array(mean_values, dtype=np.float32), header=header.copy())
tr_std = Trace(data=np.array(std_values, dtype=np.float32), header=header.copy())
tr_max = Trace(data=np.array(max_values, dtype=np.float32), header=header.copy())
tr_mean.stats.channel = "MEAN"
tr_std.stats.channel = "STD"
tr_max.stats.channel = "MAX"
feature_stream = Stream(traces=[tr_mean, tr_std, tr_max])

# Extract segment length (from any layer)
segment_dist_km = network.bins / network.sampling_rate
segment_len_samples = int(segment_dist_km * network.sampling_rate)  # length of segment (samples)
step_samples = int(segment_len_samples * (1 - config.SEGMENT_OVERLAP))  # length of steps (samples)
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
    distance_all.append(i * segment_dist_km * (1 - config.SEGMENT_OVERLAP))

# Run the scattering transform
scattering_coefficients = network.transform(segments, reduce_type=np.median)
np.savez(
    f"{config.SCATTERING_COEFFICIENTS_FOLDER}/scattering_coefficients.npz",
    order_1=scattering_coefficients[0],
    order_2=scattering_coefficients[1],
    distance=distance_all,
)
