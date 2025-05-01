import matplotlib
import matplotlib.font_manager as fm
from matplotlib import rcParams
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import xdas
import re
import bisect
import numpy as np
import random
import config

cluster_you_want = 3
how_many_waveforms_in_plot = 30
h_plot = 6
w_plot = 5
scale_factor = 1.0213001907746815 / 9112677.96164918
how_long_time_in_plot = 3
Y_lim = (-0.00001, 0.00001)
file = '../../Outputs/clustering_results/clusters_dist_log/clustering_result_20250430_070758_1_min_clusters_distance.log'

engine_type = 'asn'
tables = pd.read_csv(file, sep='\t')
all_cluster3_loc = tables[tables.cluster_number==cluster_you_want].distance_km

basename = os.path.basename(file)
m_time = re.search(r'_(\d{6})_', basename)
m_win  = re.search(r'_(\d+)_min', basename)
event_str  = m_time.group(1)          
window_min = int(m_win.group(1))      

def hms_to_sec(hms: str) -> int:
    hh, mm, ss = int(hms[:2]), int(hms[2:4]), int(hms[4:6])
    return hh*3600 + mm*60 + ss

event_sec = hms_to_sec(event_str)
start_sec = event_sec                   
end_sec   = event_sec + window_min*60   

wave_files = sorted(glob.glob(os.path.join(
    config.DAS_WAVEFORM_PATH, '*.hdf5')))

files_with_sec = []
for fn in wave_files:
    stamp = os.path.splitext(os.path.basename(fn))[0]
    t = hms_to_sec(stamp)
    files_with_sec.append((fn, t))

t_secs = [t for _, t in files_with_sec]
i_start = bisect.bisect_left(t_secs, start_sec)
i_end   = bisect.bisect_right(t_secs, end_sec) - 1
i_first = i_start - 1 if i_start > 0 else i_start
i_first = max(i_first, 0)
selected = [files_with_sec[i][0] for i in range(i_first, i_end + 1)]
data = xdas.open_mfdataarray(selected, engine=engine_type)

cluster3_m = all_cluster3_loc.values * 1000  
idxs = sorted(random.sample(range(len(cluster3_m)), how_many_waveforms_in_plot))

# Font settings (optional)
FONT_PATH = '/home/hmhuang/Work/Helvetica.ttf'
TITLE_FONT_PATH = '/home/hmhuang/Work/Helvetica_Bold.ttf'
fm.fontManager.addfont(FONT_PATH)
fm.fontManager.addfont(TITLE_FONT_PATH)
matplotlib.rcParams['font.family'] = 'Helvetica'

fig, axes = plt.subplots(h_plot, w_plot, figsize=(15, 6), sharex=True, sharey=True)
axes = axes.flatten()
for ax, i in zip(axes, idxs):
    wave = data.sel(distance=cluster3_m[i], method="nearest")
    t0 = wave.coords["time"].values[0]             
    secs = (wave.coords["time"].values - t0) / np.timedelta64(1, "s")
    ax.plot(secs, wave.values*scale_factor, c= 'gray',linewidth=2)
    ax.set_xlim(0, how_long_time_in_plot)
    ax.text(
        0.95, 0.05,
        f"Dist. {cluster3_m[i]:.1f} m",
        ha='right', va='bottom',
        transform=ax.transAxes,
        fontsize=8
    )
    ax.tick_params(axis='both', direction='in', which='both')
    ax.set_ylim(Y_lim)
    for spine in ax.spines.values():
        spine.set_linewidth(1)
fig.suptitle(f"Random {how_many_waveforms_in_plot} Waveforms in Cluster {cluster_you_want}", fontsize=16)
fig.supylabel("Strain Rate", fontsize=12)
fig.subplots_adjust(left=0.08, top=0.92)  
fig.savefig(f'../../Outputs/{cluster_you_want}_random_{how_many_waveforms_in_plot}_waveforms.png', dpi=300, bbox_inches='tight')