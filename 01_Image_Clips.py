# %%
import os
import numpy as np
from PIL import Image

# 原圖檔資料夾
file_path = './DAS_data/20250331/'
save_folder = os.path.join(file_path, "waterfall_npz")

# 確保儲存資料夾存在
os.makedirs(save_folder, exist_ok=True)

# 裁切範圍
top = 160
bottom = 696
left = 70
right = 1789

# 逐張處理資料夾內所有 PNG 圖片
for fname in os.listdir(file_path):
    if fname.endswith(".png"):
        img_path = os.path.join(file_path, fname)
        image = Image.open(img_path)
        img_array = np.array(image)

        # 裁切
        waterfall_img = img_array[top:bottom, left:right, :]

        # 儲存為 .npz（副檔名改成 .npz）
        npz_name = os.path.splitext(fname)[0] + ".npz"
        np.savez(os.path.join(save_folder, npz_name), waterfall=waterfall_img)

print("Finish")
