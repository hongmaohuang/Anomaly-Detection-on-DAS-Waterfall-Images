import os
import pickle
import numpy as np
from pathlib import Path
from scatseisnet import ScatteringNetwork
import config
import shutil
import sys

if os.path.exists(config.WAVELET_FOLDER):
    print(f"{config.WAVELET_FOLDER} is already created. Please confirm if that fits your project.")
    sys.exit(0)
os.makedirs(config.WAVELET_FOLDER)

# === Extract Distance and Duration Information ===
feature_files = sorted(Path(config.FEATURES_FOLDER).glob("features_*.npz"))
features = np.load(feature_files[0])["features"]
num_samples = features.shape[0]
sampling_rate_per_km = num_samples / config.TOTAL_DISTANCE_KM
samples_per_segment = max(1, round(config.SEGMENT_DISTANCE * sampling_rate_per_km))
# === Create the Scattering Network ===

print(f"You are using the scattering network with the following parameters:\n"
      f"  - Number of octaves (bank 1): {config.OCTAVES_1}\n"
      f"  - Resolution (bank 1): {config.RESOLUTION_1}\n"
      f"  - Quality (bank 1): {config.QUALITY_1}\n"
      f"  - Number of octaves (bank 2): {config.OCTAVES_2}\n"
      f"  - Resolution (bank 2): {config.RESOLUTION_2}\n"
      f"  - Quality (bank 2): {config.QUALITY_2}\n"
)

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
