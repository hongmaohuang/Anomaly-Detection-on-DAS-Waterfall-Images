# %%
import os
import numpy as np
from PIL import Image

DATA_PATH = './DAS_data/20250331/'
OUTPUT_FOLDER = os.path.join(DATA_PATH, "waterfall_npz")
CROP_TOP = 160
CROP_BOTTOM = 696
CROP_LEFT = 70
CROP_RIGHT = 1789

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for fname in os.listdir(DATA_PATH):
    if fname.endswith(".png"):
        img_path = os.path.join(DATA_PATH, fname)
        image = Image.open(img_path)
        img_array = np.array(image)
        cropped_img = img_array[CROP_TOP:CROP_BOTTOM, CROP_LEFT:CROP_RIGHT, :]
        npz_name = os.path.splitext(fname)[0] + ".npz"
        np.savez(os.path.join(OUTPUT_FOLDER, npz_name), waterfall=cropped_img)
