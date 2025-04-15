import os
import pickle
import numpy as np
from pathlib import Path
from scatseisnet import ScatteringNetwork
import config
import shutil

if os.path.exists(config.WAVELET_FOLDER):
    shutil.rmtree(config.WAVELET_FOLDER)
os.makedirs(config.WAVELET_FOLDER)

# === Extract Distance and Duration Information ===
feature_files = sorted(Path(config.FEATURES_FOLDER).glob("features_*.npz"))
features = np.load(feature_files[0])["features"]
num_samples = features.shape[0]
sampling_rate_per_km = num_samples / config.TOTAL_DISTANCE_KM
samples_per_segment = max(1, round(config.SEGMENT_DISTANCE * sampling_rate_per_km))

# === Create the Scattering Network ===
network = ScatteringNetwork(
    {"octaves": config.OCTAVES_1, "resolution": config.RESOLUTION_1, "quality": config.QUALITY_1},
    {"octaves": config.OCTAVES_2, "resolution": config.RESOLUTION_2, "quality": config.QUALITY_2},
    bins=samples_per_segment,
    sampling_rate=sampling_rate_per_km,
)

with open(Path(config.WAVELET_FOLDER) / "scattering_network.pickle", "wb") as f:
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
