# %%
import obspy
import glob

all_files = glob.glob('./Detected_anomalies/*.mseed')
st = obspy.read(all_files[10], format='MSEED')

st.plot()
