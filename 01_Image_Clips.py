import os
import numpy as np
from PIL import Image
import shutil
import config
import glob
#from skimage.filters.rank import entropy
#from skimage.morphology import disk
#from skimage import exposure
#from scipy.spatial import cKDTree
#import matplotlib.cm as cm, matplotlib.pyplot as plt
#from pathlib import Path
#os.chdir(Path(__file__).resolve().parent)

# Remove and recreate output folder
if os.path.exists(config.WATERFALL_NPZ_FOLDER):
    shutil.rmtree(config.WATERFALL_NPZ_FOLDER)
os.makedirs(config.WATERFALL_NPZ_FOLDER)

print(f"\nYou are using: {config.DURATION_WATERFALL} minutes with {config.TOTAL_DISTANCE_KM} km of DAS waterfall image!\n")

#file_list = os.listdir(config.DAS_WATERFALL_PATH)
file_list = sorted(glob.glob(config.DAS_WATERFALL_PATH))
total = len(file_list)
print(f"Found {total} waterfall images. Starting processing...")
print(f"The data will be from {file_list[0]} to {file_list[-1]}\n")

for idx, fname in enumerate(file_list, start=1):
    print(f"[{idx}/{total}] Processing {fname}")
    img_path = os.path.join(config.DAS_WATERFALL_PATH, fname)
    image = Image.open(img_path)
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
    npz_name = os.path.splitext(fname)[0] + ".npz"
    save_path = os.path.join(config.WATERFALL_NPZ_FOLDER, npz_name)
    np.savez(save_path, waterfall=cropped)
    print(f"  - Saved to: {save_path}\n")
    

print("All images processed!\n")

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