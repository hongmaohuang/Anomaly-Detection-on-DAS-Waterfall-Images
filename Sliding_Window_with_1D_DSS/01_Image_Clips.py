import os
import numpy as np
from PIL import Image
import config
import shutil

if os.path.exists(config.WATERFALL_NPZ_FOLDER):
    shutil.rmtree(config.WATERFALL_NPZ_FOLDER)
os.makedirs(config.WATERFALL_NPZ_FOLDER)

for fname in os.listdir(config.DAS_WATERFALL_PATH):
    print(f"Processing {fname}")
    img_path = os.path.join(config.DAS_WATERFALL_PATH, fname)
    image = Image.open(img_path)
    img_array = np.array(image)
    if config.CROP_OR_NOT == 'YES':
        cropped_img = img_array[config.CROP_TOP:config.CROP_BOTTOM, config.CROP_LEFT:config.CROP_RIGHT, :]
    else:
        cropped_img = img_array
    npz_name = os.path.splitext(fname)[0] + ".npz"
    np.savez(os.path.join(config.WATERFALL_NPZ_FOLDER, npz_name), waterfall=cropped_img)
