# Anomaly Detection on DAS Waterfall Images

This repository contains a collection of Python scripts for detecting anomalous behaviour in Distributed Acoustic Sensing (DAS) waterfall images. The project processes a series of 1‑minute waterfall PNG images, extracts spectral features, applies a scattering transform, performs dimensionality reduction, clusters the data and visualizes the results.

## Repository Layout

- **01_Image_Clips.py** – Converts waterfall PNGs into cropped ``.npz`` files for faster access.
- **02_Feature_Functions.py** – Computes statistical and spectral features from the cropped images.
- **03_Scattering_Network_Design.py** – Builds a scattering network based on the feature data.
- **04_Scattering_Transform.py** – Applies the scattering transform to the features.
- **05_PCA.py** – Performs optional PCA and ICA for dimensionality reduction.
- **06_Clustering.py** – Clusters the transformed features using the method defined in ``config.py``.
- **07_Visualization.py** – Generates plots of cluster occurrences and accumulated counts.
- ``config.py`` – Central location for all input paths and processing parameters.
- ``useless/run.sh`` – Example shell script that runs the entire pipeline in sequence.

## Quickstart

Edit ``config.py`` to point to your input and output folders. The key variables for paths are defined near the top:

```python
DAS_DATA_PATH = f'../../Inputs/'
DAS_WATERFALL_PATH = os.path.join(DAS_DATA_PATH, "waterfall_images_1min")
DAS_WAVEFORM_PATH = os.path.join(DAS_DATA_PATH, "waveforms")
```

Then execute the provided shell script:

```bash
bash useless/run.sh
```

The script runs each processing stage in order:

```bash
python 01_Image_Clips.py
python 02_Feature_Functions.py
python 03_Scattering_Network_Design.py
python 04_Scattering_Transform.py
python 05_PCA.py
python 06_Clustering.py
python 07_Visualization.py
```

## Processing Steps

1. **Image Preparation** – ``01_Image_Clips.py`` loads the waterfall PNG images, optionally crops them and stores the result in ``npz`` format. Example logging from the script shows the process:

```python
print(f"Found {total} waterfall images. Starting processing...")
[...]
print("All images processed!\n")
```

2. **Feature Extraction** – ``02_Feature_Functions.py`` calculates features such as the standard deviation and dominant frequency for each sliding window:

```python
feat_1 = window_1.std(ddof=1)
feat_2 = window_2.mean()
feat_3_in_one.append(freqs[np.argmax(amp_spec)])
```

3. **Scattering Network** – ``03_Scattering_Network_Design.py`` constructs a scattering network using parameters from ``config.py``:

```python
network = ScatteringNetwork(
    {"octaves": config.OCTAVES_1, "resolution": config.RESOLUTION_1, "quality": config.QUALITY_1},
    {"octaves": config.OCTAVES_2, "resolution": config.RESOLUTION_2, "quality": config.QUALITY_2},
    bins=samples_per_segment,
    sampling_rate=sampling_rate_per_km,
)
```

4. **Scattering Transform** – ``04_Scattering_Transform.py`` applies the network to the feature stream and saves coefficients:

```python
order_1, order_2 = network.transform(segments, reduce_type=np.median)
np.savez(
    f"{config.SCATTERING_COEFFICIENTS_FOLDER}/scattering_coefficients.npz",
    order_1=order_1,
    order_2=order_2,
    distance=distance_all,
)
```

5. **Dimensionality Reduction** – ``05_PCA.py`` optionally runs PCA (if ``RUN_PCA="YES"``) and then performs ICA to obtain independent components.

6. **Clustering** – ``06_Clustering.py`` clusters the ICA features. The method is selected using ``CLUSTER_METHOD`` in ``config.py`` (``kmeans``, ``gmm``, ``dbscan`` or ``agglomerative``). For GMM, you can optionally determine the number of components automatically by setting ``SELECT_CLUSTERS_WITH`` to ``"aic"`` or ``"bic"`` and specifying ``GMM_COMPONENT_RANGE``.

7. **Visualization** – ``07_Visualization.py`` plots the number of occurrences for each cluster versus time and saves PNG images in ``visualizations/cluster_counts``.

## Dependencies

The scripts rely on several third‑party packages including:

- ``numpy``
- ``scipy``
- ``Pillow``
- ``matplotlib``
- ``scikit-learn``
- ``scikit-image``
- ``obspy``
- ``xdas``
- ``scatseisnet``

Install these packages in your Python environment before running the pipeline.

## Outputs

Processed files and results are written to the ``../../Outputs`` directory as configured in ``config.py``. Subfolders include
``waterfall_npz``, ``features``, ``wavelets``, ``scattering_coefficients``, ``pca_ica``, ``clustering_results`` and ``visualizations``.

## Additional Scripts

The ``useless/useless_codes`` directory contains experimental or archived code such as wavelet visualization and event-by-event processing. These scripts are not required for the main workflow but may serve as references.

---
