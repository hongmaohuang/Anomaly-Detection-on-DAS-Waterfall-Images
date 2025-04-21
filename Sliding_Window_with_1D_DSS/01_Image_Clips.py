import os
import numpy as np
from PIL import Image
import config
import shutil

# Remove and recreate output folder
if os.path.exists(config.WATERFALL_NPZ_FOLDER):
    shutil.rmtree(config.WATERFALL_NPZ_FOLDER)
os.makedirs(config.WATERFALL_NPZ_FOLDER)

print(f"\nYou are using: {config.DURATION_WATERFALL} minutes with {config.TOTAL_DISTANCE_KM} km of DAS waterfall image!\n")

# List all input files
file_list = os.listdir(config.DAS_WATERFALL_PATH)
total = len(file_list)
print(f"Found {total} waterfall images. Starting processing...")

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

print("All images processed!\n")
