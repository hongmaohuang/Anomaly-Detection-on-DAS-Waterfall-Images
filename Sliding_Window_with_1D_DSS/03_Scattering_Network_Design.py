# %%
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scatseisnet import ScatteringNetwork
import xdas  # Make sure this package is installed

# Change to script directory
os.chdir(Path(__file__).resolve().parent)

# === Basic settings ===
DURATION_WATERFALL_MIN = 1
DATA_DATE = "20250331"
DATA_DIR = Path(f"../../Inputs/DAS_data/{DATA_DATE}/waveforms")
FEATURES_DIR = Path("../../Outputs/Features")
OUT_DIR = Path("../../Outputs/wavelets")
WINDOW_SIZE_CHANNEL = 3
STEP_CHANNEL = 1

# === Check if data exists ===
waveform_files = list(DATA_DIR.glob("*.hdf5"))
feature_files = sorted(FEATURES_DIR.glob("features_*.npz"))

# === Compute distance and duration ===
data = xdas.open_mfdataarray(str(waveform_files[0]), engine="asn")
TOTAL_DISTANCE_KM = np.max(data.coords['distance'].values) / 1000

# === Load features and calculate parameters ===
features = np.load(feature_files[0])["features"]
num_samples = features.shape[0]

segment_distance_km = 0.04
sampling_rate_per_km = num_samples / TOTAL_DISTANCE_KM
samples_per_segment = int(segment_distance_km * sampling_rate_per_km)

# === Create the Scattering Network ===
network = ScatteringNetwork(
    {"octaves": 8, "resolution": 6, "quality": 1},
    {"octaves": 8, "resolution": 2, "quality": 1},
    bins=samples_per_segment,
    sampling_rate=sampling_rate_per_km,
)
print(network)

# === Save the network ===
OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_DIR / "scattering_network.pickle", "wb") as f:
    pickle.dump(network, f, protocol=pickle.HIGHEST_PROTOCOL)

'''
# === Visualize each wavelet bank ===
for bank in network.banks:
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 5))

    for wavelet, spectrum, ratio in zip(bank.wavelets, bank.spectra, bank.ratios):
        ax[0].plot(bank.times, wavelet.real + ratio, "C0")
        ax[1].plot(bank.frequencies, np.log(np.abs(spectrum) + 1) + ratio, "C0")

    width_max = 3 * bank.widths.max()
    ax[0].set(xlabel="Time (s)", ylabel="Octaves (log2)", xlim=(-width_max, width_max))
    ax[0].grid()
    ax[1].set(xlabel="Frequency (Hz)", xscale="log")
    ax[1].grid()
    plt.tight_layout()
    plt.show()
'''
