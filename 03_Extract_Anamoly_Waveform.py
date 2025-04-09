# %%
import pandas as pd
import glob
import xdas 
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import obspy
import os
from pathlib import Path
from tqdm import tqdm

# Open and read header
with open('anomaly_points.log', 'r') as f:
    header_lines = [next(f) for _ in range(3)]
    starting_time_str = header_lines[1].split(":", 1)[1].strip()

# Read data
df = pd.read_csv('anomaly_points.log', skiprows=3, names=['Distance_km', 'Time_sec'])

# Verify input data
if df.empty:
    raise ValueError("No anomaly data found in log file")

# Load waveform files
data_dir = Path('./DAS_data/20250331/waveforms')
waveform_files = glob.glob(str(data_dir / '*.hdf5'))

# Process DAS data
ds_all = [xdas.open_mfdataarray(f, engine="asn") for f in tqdm(waveform_files, desc='讀取進度')]
ds_all_sorted = sorted(ds_all, key=lambda ds: ds.coords['time'].values[0])
ds_all_xr_sorted = [xr.DataArray(ds.values, 
                    coords={'time': ds.coords['time'], 'distance': ds.coords['distance']},
                    dims=['time', 'distance']) 
                    for ds in ds_all_sorted]
ds_concat_xr = xr.concat(ds_all_xr_sorted, dim='time')
ds_concat_xr.to_netcdf('ds_concat_xr_20250331.nc')

''' 
time = ds_concat_xr.time.values
distance = ds_concat_xr.distance.values
data = ds_concat_xr.values
plt.figure(figsize=(12, 6))
plt.pcolormesh(distance, time, data, shading='auto', cmap='seismic', vmax=200, vmin=-200)
plt.colorbar(label='Strain or Amplitude')
plt.xlabel('Distance (km)')
plt.ylabel('Time')
plt.gca().invert_yaxis()
'''


# %%
# Create output directory
output_dir = Path('./Detected_anomalies')
output_dir.mkdir(exist_ok=True)

starting_time = datetime.strptime(starting_time_str, '%Y%m%d_%H%M%S')

for order, (_, first_anomaly) in enumerate(df.iterrows()):
    print(f"Processing anomaly {order + 1} of {len(df)}")
    
    distance_km = first_anomaly['Distance_km']
    time_sec = first_anomaly['Time_sec']
    anomaly_time = starting_time + timedelta(seconds=time_sec)

    # Extract waveform
    waveform = ds_concat_xr.sel(distance=distance_km, method='nearest')
    time_values = waveform.time.values
    
    if len(time_values) < 2:
        print(f"Warning: Insufficient time points for anomaly {order + 1}")
        continue

    # Create obspy trace
    anomaly_waveform = obspy.Trace()
    time_diff = (time_values[1] - time_values[0]).astype('timedelta64[ns]').astype(float) / 1e9
    anomaly_waveform.stats.sampling_rate = 1 / time_diff
    anomaly_waveform.data = waveform.values
    anomaly_waveform.stats.starttime = obspy.UTCDateTime(str(time_values[0]))    
    anomaly_waveform.stats.network = 'DAS'

    # Trim and save
    anomaly_start = obspy.UTCDateTime(anomaly_time)
    anomaly_end = anomaly_start + 10
    
    #anomaly_waveform.trim(starttime=anomaly_start, endtime=anomaly_end)
    output_file = output_dir / f"anomaly_waveform_{order}.mseed"
    anomaly_waveform.write(str(output_file), format="MSEED")
