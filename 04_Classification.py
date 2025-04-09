# %%
import pandas as pd
import glob
import xdas 
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import obspy

# Open the file and read the first three header lines
with open('anomaly_points.log', 'r') as f:
    header_lines = [next(f) for _ in range(3)]
    # Extract the starting time from the second header line
    starting_time_str = header_lines[1].split(":", 1)[1].strip()

# Read the log file, skipping the header lines
df = pd.read_csv('anomaly_points.log', skiprows=3, names=['Distance_km', 'Time_sec'], header=None)

waveform_files = glob.glob('./DAS_data/20250331/waveforms/*.hdf5')

ds_all = []
for i in range(len(waveform_files)):
    ds = xdas.open_mfdataarray(waveform_files[i], engine="asn")
    ds_all.append(ds)

# Assuming ds_all is your list of xdas.DataArray objects
# Sort by the first time value
ds_all_sorted = sorted(ds_all, key=lambda ds: ds.coords['time'].values[0])

# Convert each xdas.DataArray to xarray.DataArray
ds_all_xr_sorted = [
    xr.DataArray(
        ds.values,
        coords={'time': ds.coords['time'], 'distance': ds.coords['distance']},
        dims=['time', 'distance']
    )
    for ds in ds_all_sorted
]

# Concatenate along the time dimension
ds_concat_xr = xr.concat(ds_all_xr_sorted, dim='time')

''' 
# 假設 ds_concat_xr 是你的 xarray.DataArray
time = ds_concat_xr.time.values
distance = ds_concat_xr.distance.values
data = ds_concat_xr.values

# 創建圖形
plt.figure(figsize=(12, 6))
plt.pcolormesh(distance, time, data, shading='auto', cmap='seismic')
plt.colorbar(label='Strain or Amplitude')
plt.xlabel('Distance (km)')
plt.ylabel('Time')

# reverse the y-axis
plt.gca().invert_yaxis()
'''
# Potentially有個問題是guage length太長，在抓最近距離的時候會抓到太遠的channel
for order in range(len(df)):
    print(f"Processing anomaly {order + 1} of {len(df)}")
    first_anomaly = df.iloc[order]
    distance_km = first_anomaly['Distance_km']
    time_sec = first_anomaly['Time_sec']

    # Calculate the actual time of the anomaly
    starting_time = datetime.strptime(starting_time_str, '%Y%m%d_%H%M%S')
    anomaly_time = starting_time + timedelta(seconds=time_sec)

    # Find the closest distance in ds_concat_xr
    closest_distance = ds_concat_xr.distance.sel(distance=distance_km, method='nearest')

    # Extract the waveform at this distance (time series)
    waveform = ds_concat_xr.sel(distance=closest_distance, method='nearest')

    # Extract time and data values for plotting
    time_values = waveform.time.values
    data_values = waveform.values

    # Save anomaly waveform as MSEED
    anomaly_waveform = obspy.Trace()
    time_diff = (time_values[1] - time_values[0]).astype('timedelta64[ns]').astype(float) / 1e9
    anomaly_waveform.stats.sampling_rate = 1 / time_diff
    anomaly_waveform.data = data_values
    anomaly_waveform.stats.starttime = obspy.UTCDateTime(time_values[0])

    # Convert anomaly_time to UTCDateTime and calculate end time
    anomaly_start = obspy.UTCDateTime(anomaly_time)
    anomaly_end = anomaly_start + 10  # Adds 10 seconds directly with UTCDateTime

    # Trim the waveform and save
    anomaly_waveform.trim(starttime=anomaly_start, endtime=anomaly_end)
    anomaly_waveform.write(f"./Detected_anamolies/anomaly_waveform_{order}.mseed", format="MSEED")