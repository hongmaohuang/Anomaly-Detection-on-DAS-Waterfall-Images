# %%
# This is a useless code right now!
#
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)
import numpy as np
import matplotlib.pyplot as plt

data = np.load("/home/hmhuang/Work/Hualien_DAS_Monitoring/Outputs/clustering_results/clusters/clustering_result_20250331_020645_1_min_clusters.npz")
one_hot = data["one_hot"]   
plt.figure(figsize=(12, 4))

for i in range(one_hot.shape[1]):
    y = np.full(one_hot.shape[0], i)  
    x = np.arange(one_hot.shape[0])   
    mask = one_hot[:, i] > 0          
    plt.scatter(x[mask], y[mask], label=f"Cluster {i}", s=10)

plt.title("Cluster Memberships (Scatter View)")
plt.xlabel("Sample Index")
plt.ylabel("Cluster ID")
plt.yticks(range(one_hot.shape[1]))
plt.legend()
plt.tight_layout()
plt.show()