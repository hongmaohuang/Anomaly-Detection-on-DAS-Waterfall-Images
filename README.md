# DAS Data Processing and Analysis Project 🚀
# This README is generated using ChatGPT
This project is designed for processing Distributed Acoustic Sensing (DAS) waterfall imagery. It includes modules for image cropping, moving window feature extraction with anomaly detection, and an upcoming classification module that will further analyze waveform data based on detected anomalies.

---

## Table of Contents
- [Introduction](#introduction)
- [Modules](#modules)
  - [01_Image_Clips.py](#01_image_clipspy)
  - [02_moving_window.py](#02_moving_windowpy)
  - [Classification Module (Under Development)](#classification-module-under-development)
- [Installation Requirements](#installation-requirements)
- [Folder Structure](#folder-structure)
- [Usage](#usage)
- [Future Plans](#future-plans)
- [License](#license)

---

## Introduction

This repository contains a series of scripts aimed at processing DAS waterfall imagery. The workflow includes:

1. **Image Cropping**: Extracting regions of interest from raw PNG images.
2. **Feature Extraction & Anomaly Detection**: Using a moving window approach for feature extraction and applying an IsolationForest model to identify anomalies.
3. **Classification (Under Development)**: Leveraging detected anomalies from the log file to read corresponding waveform data and perform event classification with various algorithms.

---

## Modules

### 01_Image_Clips.py
- **Purpose**: 
  - Reads all PNG images from a designated folder.
  - Crops each image based on a pre-defined region (top, bottom, left, right).
  - Saves the cropped images as `.npz` files in a subfolder named `waterfall_npz`.
- **Key Features**:
  - Automatically creates the destination folder if it does not exist.
  - Designed to preprocess large volumes of image data for subsequent analysis.

### 02_moving_window.py
- **Purpose**:
  - Loads the cropped image data (`.npz` files) generated by `01_Image_Clips.py` and converts RGB images to grayscale.
  - Applies a moving window approach to extract features (mean, standard deviation, maximum value, Sobel gradient mean, and Laplacian variance) from the image.
  - Uses an IsolationForest model to perform anomaly detection on the extracted features.
  - Visualizes the anomaly scores as a heatmap and overlays detected anomalies on the original DAS waterfall image.
  - Logs the center coordinates (distance and time) of anomalous windows in `anomaly_points.log` for further analysis.
- **Key Features**:
  - Customizable window size and step parameters for fine-tuning the feature extraction.
  - Applies connected component labeling to filter out overly large anomaly regions.
  - Provides visual comparisons between the anomaly heatmap and the original image data.

### Classification Module (Under Development) 🚧
- **Objective**:
  - To read waveform data corresponding to the anomaly points recorded in `anomaly_points.log`.
  - Develop classification algorithms (using either traditional machine learning or deep learning approaches) to categorize different types of anomalous events.
- **Concept**:
  - **Feature Fusion**: Combine time-domain and frequency-domain features extracted from waveforms for improved classification accuracy.
  - **Multimodal Data Processing**: Future plans include integrating additional sensor data to perform cross-domain analysis.
  - **Modular Design**: Aims to be easily extensible for experimenting with various classification methods, working closely with the anomaly detection results from the moving window module.
  
> **Note**: The classification module is still in the development phase and is currently in the concept validation stage. Stay tuned for updates!

---

## Installation Requirements

Ensure you have Python 3.x installed. The following Python packages are required:
- numpy
- Pillow
- matplotlib
- scikit-learn
- opencv-python (cv2)
- scikit-image

You can install the dependencies using:

```bash
pip install numpy Pillow matplotlib scikit-learn opencv-python scikit-image
