
import numpy as np
import scipy 
from scipy.signal import find_peaks, resample, ZoomFFT
from functions import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
import matplotlib.axes
import matplotlib.lines as lines
from matplotlib.patches import Ellipse
from math import pi
import bioread
from scipy.signal import periodogram
from scipy.signal import welch
from scipy.interpolate import interp1d
import seaborn as sns
from scipy.integrate import simpson
from scipy.signal import coherence


#Sample Script to visualize HRV analysis 


#Open ACQ File
ECG_source = "data/SAMPLE.acq"
file = bioread.read_file(ECG_source)
Channel_List=file.channels


#Pull ECG Data  and all Variables - Need to label channel correctly
ECG_Data = file.channels[0].raw_data
Time = file.channels[0].time_index
ECG_fs = len(ECG_Data)/max(Time)
x = ECG_Data

#Trim signals to any time we want (cutting the first x seconds)
# TrimmedECG = SignalTrimmer(ECG_Data, ECG_fs, 60)
# TrimmedBP = SignalTrimmer (BP_Data, BP_fs, 60)
# TrimmedECG_time = TimeTrimmer(Time, 60)
# TrimmedBP_time = TimeTrimmer(BP_Time, 60)

#Tag R Intervals and create Array of RR Interval Distances
peaks, _ = find_peaks(x, height = 0.8, threshold = None, distance = 100, prominence=(0.7,None), width=None, wlen=None, rel_height=None, plateau_size=None)
td_peaks = (peaks / ECG_fs)
RRDistance = distancefinder(td_peaks)
#convert to ms
RRDistance_ms = [element * 1000 for element in RRDistance]

# #Small Snippet of ECG Data for visualization purposes
# Define window around R-peaks
start_idx = 2
end_idx = start_idx + 7
start_sample = int(peaks[start_idx] - 0.5 * ECG_fs)
end_sample = int(peaks[end_idx] + 0.5 * ECG_fs)

# # Slice signal
ECG_segment = x[start_sample:end_sample]
Time_segment = Time[start_sample:end_sample]

# # Get relevant R-peaks
segment_peaks = [p for p in peaks if start_sample <= p <= end_sample]
segment_peaks_rel = [p - start_sample for p in segment_peaks]

# # Normalize time to start at 0
Time_segment_normalized = Time_segment - Time_segment[0]

# #UNLABELED ECG SEGMENT ---
plt.figure(figsize=(10, 4))
plt.plot(Time_segment_normalized, ECG_segment, label="ECG")
plt.title("Example ECG Segment (Unlabeled)")
plt.xlabel("Time (s)")
plt.ylabel("ECG (mV)")
plt.xlim(0, 7.36)  # force clean x-axis
plt.tight_layout()
plt.savefig("figures/ecg_segment_unlabeled.tif", dpi=600, format='tiff')

# #ANNOTATED ECG SEGMENT ---
plt.figure(figsize=(10, 4))
plt.plot(Time_segment_normalized, ECG_segment, label="ECG")
plt.plot(Time_segment_normalized[segment_peaks_rel], ECG_segment[segment_peaks_rel], "rx", label="R-peaks")

for i in range(len(segment_peaks_rel) - 1):
    rr = (segment_peaks_rel[i+1] - segment_peaks_rel[i]) / ECG_fs * 1000  # ms
    mid_time = (Time_segment_normalized[segment_peaks_rel[i]] + Time_segment_normalized[segment_peaks_rel[i+1]]) / 2
    plt.text(mid_time, max(ECG_segment)*0.8, f"{rr:.0f} ms", ha="center", fontsize=9)

plt.title("ECG Segment with Annotated R-R Intervals")
plt.xlabel("Beat Number")
plt.ylabel("ECG (mV)")
plt.xlim(0, 7.36)
plt.tight_layout()



# #RRI TACHOGRAM FROM ECG SEGMENT === Get RR intervals from the segment
segment_td_peaks = [Time[p] for p in segment_peaks]
segment_RR_intervals = np.diff(segment_td_peaks)  # in seconds
segment_RR_ms = segment_RR_intervals * 1000       # convert to ms

# Midpoints of each RR interval (for plotting against time)
segment_mid_times = [(segment_td_peaks[i] + segment_td_peaks[i+1]) / 2 for i in range(len(segment_RR_ms))]

# # Plot RRI
# Plot RRI (cleaned version)
plt.figure(figsize=(8, 3))
plt.plot(range(1, len(segment_RR_ms) + 1), segment_RR_ms, marker='o')
plt.title("R-R Interval (RRI) Tachogram")
plt.xlabel("Beat Number")
plt.ylabel("RRI (ms)")
plt.ylim(800, 1000)
plt.tight_layout()
plt.show()

