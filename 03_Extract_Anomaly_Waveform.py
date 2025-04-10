# %%
import pandas as pd
import glob
import xdas
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import obspy
from pathlib import Path
from tqdm import tqdm

# ===== Configurable Variables =====
LOG_FILE = 'anomaly_points.log'
DATA_DIR = Path('./DAS_data/20250331/waveforms')
OUTPUT_DIR = Path('./Detected_anomalies')
TIME_SPAN = 10  # Duration in seconds for trimming the anomaly waveform
DAS_NETWORK = 'DH'
#Only 2 charcters for network code on MiniSEED!
 
# NETCDF_FILENAME = 'ds_concat_xr_20250331.nc'  # Uncomment to save waveform data to netCDF

# ===== Read log file header and extract starting time =====
with open(LOG_FILE, 'r') as f:
    header_lines = [next(f) for _ in range(3)]
    starting_time_str = header_lines[1].split(":", 1)[1].strip()

# ===== Read anomaly data from log file =====
df = pd.read_csv(LOG_FILE, skiprows=3, names=['Distance_km', 'Time_sec'])
if df.empty:
    raise ValueError("No anomaly data found in the log file.")

# ===== Load waveform files =====
waveform_files = list(DATA_DIR.glob('*.hdf5'))
if not waveform_files:
    raise ValueError("No waveform files found in the specified directory.")

# ===== Process and sort waveform data =====
ds_all = [xdas.open_mfdataarray(str(f), engine="asn") for f in tqdm(waveform_files, desc='Loading waveform files')]
ds_all_sorted = sorted(ds_all, key=lambda ds: ds.coords['time'].values[0])
ds_all_xr_sorted = [
    xr.DataArray(ds.values,
                 coords={'time': ds.coords['time'], 'distance': ds.coords['distance']},
                 dims=['time', 'distance'])
    for ds in ds_all_sorted
]

ds_concat_xr = xr.concat(ds_all_xr_sorted, dim='time')
ds_concat_xr['distance'] = ds_concat_xr['distance'] / 1000

# ===== Optional: Save and load concatenated waveform data via netCDF =====
# ds_concat_xr.to_netcdf(NETCDF_FILENAME, engine='h5netcdf')
# ds_concat_xr = xr.open_dataset(NETCDF_FILENAME, engine='h5netcdf')

# ===== Create output directory if it does not exist =====
OUTPUT_DIR.mkdir(exist_ok=True)

# ===== Convert starting time string to a datetime object =====
starting_time = datetime.strptime(starting_time_str, '%Y%m%d_%H%M%S')

# %%
# ===== Process each anomaly =====
for i, (_, anomaly) in enumerate(df.iterrows()):
    print(f"Processing anomaly {i+1} of {len(df)}")
    
    distance_km = anomaly['Distance_km']
    time_sec = anomaly['Time_sec']
    anomaly_time = starting_time + timedelta(seconds=time_sec)

    # Select waveform based on the specified distance using nearest method
    waveform = ds_concat_xr.sel(distance=distance_km, method='nearest')
    time_values = waveform.time.values
    print(f"Specified distance: {distance_km} km, Nearest distance: {float(waveform.distance.values)} km")
    
    if len(time_values) < 2:
        print(f"Warning: Insufficient time points for anomaly {i+1}")
        continue

    # Create obspy Trace and calculate sampling rate based on time difference
    anomaly_trace = obspy.Trace()
    time_diff = (time_values[1] - time_values[0]).astype('timedelta64[ns]').astype(float) / 1e9
    anomaly_trace.stats.sampling_rate = 1 / time_diff
    anomaly_trace.data = np.array(waveform.values)
    anomaly_trace.stats.starttime = obspy.UTCDateTime(str(time_values[0]))
    anomaly_trace.stats.network = DAS_NETWORK
    anomaly_trace.stats.station = f"{int(float(waveform.distance.values)*1000):05d}"[:5]
    anomaly_trace.stats.location = 'm'
    # Trim the waveform to a defined TIME_SPAN starting at the anomaly time
    anomaly_start = obspy.UTCDateTime(anomaly_time)
    anomaly_end = anomaly_start + TIME_SPAN
    #print(f"Start time: {anomaly_start}, End time: {anomaly_end}")
    anomaly_trace.trim(starttime=anomaly_start, endtime=anomaly_end)

    # Save the trimmed anomaly waveform as a miniSEED file
    output_file = OUTPUT_DIR / f"anomaly_waveform_{i}.mseed"
    anomaly_trace.write(str(output_file), format="MSEED")

