import pickle
import numpy as np
import os
from pathlib import Path
from obspy import Trace, Stream, UTCDateTime
import config
import shutil

if os.path.exists(config.SCATTERING_COEFFICIENTS_FOLDER):
    shutil.rmtree(config.SCATTERING_COEFFICIENTS_FOLDER)
os.makedirs(config.SCATTERING_COEFFICIENTS_FOLDER)

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
distance_per_sample = config.TOTAL_DISTANCE_KM / len(tr_mean.data)
segment_len_samples = network.bins
step_samples = int(segment_len_samples * (1 - config.SEGMENT_OVERLAP))

segments = []
distance_all = []
for i in range(0, len(tr_mean.data) - segment_len_samples + 1, step_samples):
    seg = [trace.data[i:i+segment_len_samples] for trace in feature_stream]
    segments.append(np.array(seg))
    distance_all.append(i * distance_per_sample)

print("========================== Confirm the distances ==========================")
print(f"Origianl Distance: {config.TOTAL_DISTANCE_KM} km")
print(f"Distance: {distance_all[-1]} km")

# Run the scattering transform
order_1, order_2 = network.transform(segments, reduce_type=np.median)

np.savez(
    f"{config.SCATTERING_COEFFICIENTS_FOLDER}/scattering_coefficients.npz",
    order_1=order_1,
    order_2=order_2,
    distance=distance_all,
)
