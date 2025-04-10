# %%
import os
import numpy as np
from PIL import Image

DATE = "20250331"
DATA_PATH = f'./DAS_data/{DATE}'
WATERFALL_PATH = os.path.join(DATA_PATH, "waterfall_images")
OUTPUT_FOLDER = os.path.join(DATA_PATH, "waterfall_npz")
CROP_TOP = 160
CROP_BOTTOM = 696
CROP_LEFT = 70
CROP_RIGHT = 1789
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for fname in os.listdir(WATERFALL_PATH):
    img_path = os.path.join(WATERFALL_PATH, fname)
    image = Image.open(img_path)
    img_array = np.array(image)
    cropped_img = img_array[CROP_TOP:CROP_BOTTOM, CROP_LEFT:CROP_RIGHT, :]
    npz_name = os.path.splitext(fname)[0] + ".npz"
    np.savez(os.path.join(OUTPUT_FOLDER, npz_name), waterfall=cropped_img)
