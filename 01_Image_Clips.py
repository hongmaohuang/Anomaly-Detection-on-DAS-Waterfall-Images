import os
import numpy as np
from PIL import Image, UnidentifiedImageError
import shutil
import config
import glob
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
#from skimage.filters.rank import entropy
#from skimage.morphology import disk
#from skimage import exposure
#from scipy.spatial import cKDTree
#import matplotlib.cm as cm, matplotlib.pyplot as plt
#from pathlib import Path
#os.chdir(Path(__file__).resolve().parent)

if os.path.exists(config.WATERFALL_NPZ_FOLDER):
    shutil.rmtree(config.WATERFALL_NPZ_FOLDER)
os.makedirs(config.WATERFALL_NPZ_FOLDER)

print(f"\nYou are using: {config.DURATION_WATERFALL} minutes with {config.TOTAL_DISTANCE_KM} km of DAS waterfall image!\n")

#file_list = os.listdir(config.DAS_WATERFALL_PATH)
file_list = sorted(glob.glob(config.DAS_WATERFALL_PATH))

# Filter by configured time range if specified
if config.START_TIME and config.END_TIME:
    start_dt = datetime.strptime(config.START_TIME, "%Y%m%d_%H%M%S")
    end_dt = datetime.strptime(config.END_TIME, "%Y%m%d_%H%M%S")

    def _in_range(path):
        base = os.path.basename(path)
        parts = base.split("_")
        if len(parts) >= 4:
            dt = datetime.strptime(parts[2] + "_" + parts[3], "%Y%m%d_%H%M%S")
            return start_dt <= dt <= end_dt
        return False

    file_list = [f for f in file_list if _in_range(f)]

total = len(file_list)
print(f"Found {total} waterfall images. Starting processing...")
print(f"The data will be from {file_list[0]} to {file_list[-1]}\n")

if total:
    print(f"The data will be from {file_list[0]} to {file_list[-1]}\n")
else:
    print("No files found for the specified time range\n")
    
for idx, fname in enumerate(file_list, start=1):
    print(f"[{idx}/{total}] Processing {fname}")
    npz_name = os.path.splitext(os.path.basename(fname))[0] + ".npz"
    save_path = os.path.join(config.WATERFALL_NPZ_FOLDER, npz_name)
    if os.path.exists(save_path):
        print(f"  - Output exists, skipping: {save_path}\n")
        continue
    try:
        image = Image.open(fname)
    except (UnidentifiedImageError, OSError) as e:
        print(f"  - Failed to open image: {e}. Skipping.\n")
        continue
    img_array = np.array(image)

    # Print original image info
    print(f"  - Original shape: {img_array.shape}, dtype: {img_array.dtype}")

    # Crop if requested
    if config.CROP_OR_NOT == 'YES':
        cropped = img_array[
            config.CROP_TOP:config.CROP_BOTTOM,
            config.CROP_LEFT:config.CROP_RIGHT,
            :
        ]
        print(f"  - Cropped shape:  {cropped.shape}\n")
    else:
        cropped = img_array
        print("  - No cropping applied\n")

    # Print pixel-value statistics
    print("The original resolution will be: ")
    print(f"{config.DURATION_WATERFALL/cropped.shape[0]*60} sec per pixel")
    print(f"{config.TOTAL_DISTANCE_KM/cropped.shape[1]*1000} m per pixel")

    # Save as .npz
    np.savez(save_path, waterfall=cropped)
    print(f"  - Saved to: {save_path}\n")

print("All images processed!\n")

# ----- Data Availability Plot -----
npz_files = sorted(glob.glob(os.path.join(config.WATERFALL_NPZ_FOLDER, "*.npz")))
timestamps = []
for fp in npz_files:
    base = os.path.basename(fp)
    parts = base.split("_")
    if len(parts) >= 4:
        try:
            dt = datetime.strptime(parts[2] + parts[3], "%Y%m%d%H%M%S")
            timestamps.append(dt)
        except ValueError:
            continue

if timestamps:
    timestamps = sorted(timestamps)
    start_t = timestamps[0]
    end_t = timestamps[-1]
    expected = []
    cur = start_t
    while cur <= end_t:
        expected.append(cur)
        cur += timedelta(minutes=config.DURATION_WATERFALL)
    present = set(timestamps)
    availability = [1 if t in present else 0 for t in expected]

    fig, ax = plt.subplots(figsize=(12, 1.8))
    ax.scatter(timestamps, [1]*len(timestamps), color='royalblue', marker='|', s=80)

    ax.set_yticks([1])
    ax.set_yticklabels(["Available"])
    ax.set_ylim(0.9, 1.1)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    ax.set_title("Data Availability")
    fig.autofmt_xdate()
    fig.tight_layout()

    out_png = os.path.join(config.WATERFALL_NPZ_FOLDER, "data_availability.png")
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    print(f"Scatter data availability plot saved to: {out_png}\n")

''' 
# Testing for different transformation on waterfall images
target = "Waterfall_RMS_20250518_081111_utc.npz"
# This file is an example which includes mudslides on May 18, 2025
npz_files = [os.path.join(config.WATERFALL_NPZ_FOLDER, target)]

for f in npz_files:
    with np.load(f) as data:
        wf = data["waterfall"].mean(axis=2)
        wf_rgba = data["waterfall"]
    wf_rgb  = wf_rgba[..., :3].astype(np.uint8)  
    jet = cm.get_cmap("jet", 256)(np.arange(256))[:, :3]
    idx_img = cKDTree(jet).query(wf_rgb.reshape(-1,3), k=1)[1]\
                        .reshape(wf_rgb.shape[:2]).astype(np.uint8)
    dmin, dmax = -113.0, -35.0
    wf_db = dmin + idx_img.astype(np.float32)*(dmax-dmin)/256.0
    wf_db_u8 = exposure.rescale_intensity(
            wf_db, in_range=(dmin, dmax), out_range='uint8'
        ).astype(np.uint8)

    ny, nx = wf_db_u8.shape
    dt_sec = config.DURATION_WATERFALL * 60 / ny         
    dx_m   = config.TOTAL_DISTANCE_KM * 1000 / nx
    dI_dt, dI_dx = np.gradient(wf_db_u8, dt_sec, dx_m)
    eps = 1e-6
    slope = dI_dt / (dI_dx + eps)
    aspect_rad = np.arctan2(dI_dt, dI_dx)
    aspect_deg = np.degrees(aspect_rad)               
    aspect_deg = (aspect_deg + 360) % 360
    asp_slo_prod = aspect_deg * slope          
    ent_img = entropy(wf_db_u8, footprint=disk(1))    

    fig_w, ax_w = plt.subplots(figsize=(12, 4), dpi=300)
    im_w = ax_w.imshow(
        wf_db_u8,
        extent=[0, config.TOTAL_DISTANCE_KM, config.DURATION_WATERFALL, 0],
        aspect="auto",
        cmap="jet",   
    )
    fig_w.colorbar(im_w, ax=ax_w)
    fig_w.suptitle(f"RMS (0-255 rescaled)", fontsize=16, fontweight='bold')

    fig_s, ax_s = plt.subplots(figsize=(12, 4), dpi=300)
    im_s = ax_s.imshow(
        aspect_deg,
        extent=[0, config.TOTAL_DISTANCE_KM, config.DURATION_WATERFALL, 0],
        aspect="auto",
        cmap="jet",
    )
    fig_s.colorbar(im_s, ax=ax_s)
    fig_s.suptitle(f"ASPECT (cal. from rms)", fontsize=16, fontweight='bold')
    
    fig_e, ax_e = plt.subplots(figsize=(12, 4), dpi=300)
    im_e = ax_e.imshow(
        ent_img,
        extent=[0, config.TOTAL_DISTANCE_KM, config.DURATION_WATERFALL, 0],
        aspect="auto",
        cmap="jet"          
    )
    fig_e.colorbar(im_e, ax=ax_e)
    fig_e.suptitle(f"ENTROPY (cal. from rms)", fontsize=16, fontweight='bold')
    plt.show()
'''