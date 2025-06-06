# run_pipeline.sh
#!/usr/bin/env bash

python 01_Image_Clips.py      
python 02_Feature_Functions.py 
python 03_Scattering_Network_Design.py 
python 04_Scattering_Transform.py 
python 05_PCA.py   
python 06_Clustering.py  
python 07_Visualization.py 
