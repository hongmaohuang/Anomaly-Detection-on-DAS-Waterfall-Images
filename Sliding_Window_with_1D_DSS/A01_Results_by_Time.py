import os
import shutil
import subprocess
import glob
import config

# Please make sure you also update the parameters in config.py !
all_files = glob.glob('/home/hmhuang/Work/Hualien_DAS_Monitoring/waterfall_images_1min/*.png')

def extract_datetime(file_path):
    basename = os.path.basename(file_path)
    parts = basename.split('_')  # i.e. ['Waterfall', 'RMS', '20250331', '020551', 'utc.png']
    date_part = parts[2]         # i.e. '20250331'
    time_part = parts[3]         # i.e. '020551'
    return int(date_part + time_part)  # i.e. 20250331020551（int）

files_sorted = sorted(all_files, key=extract_datetime)

for file_path in files_sorted:
    print(f"\nProcessing: {file_path}")
    date_part = os.path.basename(file_path).split('_')[2]
    time_part = os.path.basename(file_path).split('_')[3]
    output_filename = f"clustering_result_{date_part}_{time_part}_{config.DURATION_WATERFALL}_min.png"
    output_path = os.path.join(config.CLUSTERING_RESULTS_FOLDER, output_filename)

    if os.path.exists(output_path):
        print(f"Skipping {file_path} → already processed!")
        continue

    # Make sure only one file is in the input folder
    if os.path.exists(config.DAS_WATERFALL_PATH):
        shutil.rmtree(config.DAS_WATERFALL_PATH)
    os.makedirs(config.DAS_WATERFALL_PATH)

    # Copy files to input folder
    shutil.copy(file_path, config.DAS_WATERFALL_PATH)

    # script
    scripts_to_run = [
        '01_Image_Clips.py',
        '02_Feature_Functions.py',
        '03_Scattering_Network_Design.py',
        '04_Scattering_Transform.py',
        '05_PCA.py',
        '06_Clustering.py'
    ]

    for script in scripts_to_run:
        print(f"Running {script}")
        result = subprocess.run(['python', script], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running {script}: {result.stderr}")
            break  

print("\nAll files processed.")
