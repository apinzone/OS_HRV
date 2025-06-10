
import numpy as np
import scipy 
# import pandas as pd
from scipy.signal import find_peaks, resample, ZoomFFT
from functions import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
import matplotlib.axes
import matplotlib.lines as lines
# import pywt
from matplotlib.patches import Ellipse
from math import pi
import bioread
from scipy.signal import periodogram
from scipy.signal import welch
from scipy.interpolate import interp1d
import seaborn as sns
from scipy.integrate import simpson
from scipy.signal import coherence

#Open ACQ File
ECG_source = "data/REST.acq"
file = bioread.read_file(ECG_source)
Channel_List=file.channels

#Pull BP Data 
BP_Data = file.channels[0].raw_data
BP_Time = file.channels[0].time_index
BP_fs = len(BP_Data)/max(BP_Time)
BP = BP_Data
BP_peaks, _ = find_peaks(BP, height = 110, threshold = None, distance = 100, prominence=(5,None), width=None, wlen=None, rel_height=None, plateau_size=None)
td_BP_peaks = (BP_peaks/BP_fs)

#Pull ECG Data  and all Variables - Need to label channel correctly
ECG_Data = file.channels[1].raw_data
Time = file.channels[1].time_index
ECG_fs = len(ECG_Data)/max(Time)
x = ECG_Data

#Tag R Intervals and create Array of RR Interval Distances
peaks, _ = find_peaks(x, height = 0.8, threshold = None, distance = 100, prominence=(0.7,None), width=None, wlen=None, rel_height=None, plateau_size=None)
td_peaks = (peaks / ECG_fs)
RRDistance = distancefinder(td_peaks)
#convert to ms
RRDistance_ms = [element * 1000 for element in RRDistance]
PulseIntervalDistance = distancefinder(td_BP_peaks)
#Convert to ms
PI_ms = [element * 1000 for element in PulseIntervalDistance]
m = 2
r = 0.2 * np.std(RRDistance_ms)

#Start of All ECG Plots 
#Raw ECG
plt.figure()
plt.plot(Time,ECG_Data)
plt.xlabel("time (s)")
plt.ylabel("ECG (mV)")

#ECG with R intervals tagged
plt.figure()
plt.title("Raw ECG Signal with R-R Detected")
plt.plot(x)
plt.plot(peaks, x[peaks], "x")

#Time domain HRV Variables
Successive_time_diff=SuccessiveDiff(RRDistance_ms)
AvgDiff=np.average(Successive_time_diff)
SDNN=np.std(RRDistance_ms)
SDSD=np.std(Successive_time_diff)
NN50=NNCounter(Successive_time_diff, 50)
pNN50=(NN50/len(td_peaks))*100
RMSSD = np.sqrt(np.average(rms(Successive_time_diff)))
SD1 = np.sqrt(0.5*math.pow(SDSD,2))
SD2 = np.sqrt((2*math.pow(SDNN,2) - (0.5*math.pow(SDSD,2))))
S = math.pi * SD1 * SD2
Sampling_Time = max(td_peaks)
Num_Beats = len(RRDistance_ms)
HR = np.round(Num_Beats/(Sampling_Time/60),2)
sampen = SampEn(RRDistance_ms, m, r)

#Print Time Domain HRV Variables
print("n = " + str(Num_Beats) + " beats are included for analysis")
print("The total sampling time is " + str(Sampling_Time) + " seconds")
print("The average heart rate during the sampling time is = " + str(HR) + " BPM")
print("the mean difference between successive R-R intervals is = " + str(np.round(AvgDiff,3)) + " ms")
print("The mean R-R Interval duration is  " + str(np.round(np.average(RRDistance_ms),3)) + " ms")
print("pNN50 = " + str(np.round(pNN50,3)) + " %" )
print("RMSSD = " + str(np.round(RMSSD,3)) + " ms")
print("SDNN = " + str(np.round(SDNN,3)) + " ms")
print("SDSD = " + str(np.round(SDSD,3)) + " ms")
print("Sample Entropy = " + str(sampen))
print("SD1 = " + str(np.round(SD1,3)) + " ms")
print("SD2 = " + str(np.round(SD2,3)) + " ms")
print("SD1/SD2 = " + str(np.round((SD1/SD2),3)))
print("The area of the ellipse fitted over the Poincaré Plot (S) is " + str(np.round(S,3)) + " ms^2")

#Blood Pressure Information
Systolic_Array = BP[BP_peaks]
Avg_BP = np.round((np.average(Systolic_Array)),3)
SD_BP = np.round((np.std(Systolic_Array)),3)

#Print Blood Pressure Information
print("The average systolic blood pressure during the sampling time is " + str(Avg_BP) + " + - " + str(SD_BP) + " mmHg")
print(str(len(Systolic_Array)) + " pressure waves are included in the analysis")

#RRI / Tachogram
plt.figure()
#Need to remove last element of td_peaks in order for two arrays that we are plotting to match in size 
plt.plot(np.delete(td_peaks,-1), RRDistance_ms)
plt.title("RRI")
plt.xlabel("time (s)")
plt.ylabel("RRI (ms)")
plt.text(0, 600, 'n = ' + str (len(RRDistance_ms)), fontsize=10)
plt.text(0, 500, 'Mean = ' + str (np.round(np.average(RRDistance_ms),1)) + ' ms', fontsize=10)
plt.text(200, 500, 'σ2 = ' + str (np.round(np.var(RRDistance_ms),1)) + 'ms\u00b2', fontsize=10)  
plt.savefig("figures/ecg_tachogram_REAL.tif", dpi=600, format = 'tiff')

#Create axes for Poincare Plot
RRIplusOne = Poincare(RRDistance_ms)

# Poincaré Plot (RRI, RRI + 1)
EllipseCenterX = np.average(np.delete(RRDistance_ms,-1))
EllipseCenterY = np.average(RRIplusOne)
Center_coords = EllipseCenterX, EllipseCenterY
fig, ax = plt.subplots()
z = np.polyfit(np.delete(RRDistance_ms,-1), RRIplusOne, 1)
p = np.poly1d(z)
slope = z[0]
theta = np.degrees(np.arctan(slope))
theta_rad = np.radians(theta)
# Scatter + regression
ax.set_title("Poincaré Plot")
ax.scatter(np.delete(RRDistance_ms,-1), RRIplusOne)
ax.plot(np.delete(RRDistance_ms,-1), p(np.delete(RRDistance_ms,-1)), color="red")
# Ellipse
e = Ellipse(xy=Center_coords, width=SD2*2, height=SD1*2, angle=theta,
            edgecolor='black', facecolor='none')
ax.add_patch(e)
# === Axis lines ===
# SD2 (major axis)
x_sd2 = [EllipseCenterX, EllipseCenterX + SD2 * np.cos(theta_rad)]
y_sd2 = [EllipseCenterY, EllipseCenterY + SD2 * np.sin(theta_rad)]
ax.plot(x_sd2, y_sd2, 'b-', linewidth=2)
ax.text(x_sd2[1], y_sd2[1], 'SD2', color='blue', fontsize=9, weight = 'bold', ha='left', va='bottom')

# SD1 (minor axis)
x_sd1 = [EllipseCenterX, EllipseCenterX - SD1 * np.sin(theta_rad)]
y_sd1 = [EllipseCenterY, EllipseCenterY + SD1 * np.cos(theta_rad)]
ax.plot(x_sd1, y_sd1, 'g-', linewidth=2)
ax.text(x_sd1[1], y_sd1[1], 'SD1', color='green', fontsize=9, weight = 'bold', ha='left', va='bottom')

# === Clean SD1/SD2 value box in bottom-left ===
# Use axes limits to position dynamically
text_x = ax.get_xlim()[0] + 20
text_y = ax.get_ylim()[0] + 20

# Add black text in a white rounded box
ax.text(
    text_x, text_y,
    f'SD1 = {SD1:.1f} ms\nSD2 = {SD2:.1f} ms',
    fontsize=10, color='black', ha='left', va='bottom',
    bbox=dict(boxstyle="round,pad=0.4", facecolor='white', edgecolor='black', alpha=0.9)
)
# Labels and legend
ax.set_xlabel("RRI (ms)")
ax.set_ylabel("RRI + 1 (ms)")
plt.tight_layout()
plt.savefig("figures/poincare.tif", dpi=600, format = 'tiff')

#Start of BP Plots 
#Raw BP Data 
plt.figure()
plt.plot(BP_Time, BP_Data)
plt.xlabel("time (s)")
plt.ylabel("Finger Pressure (mmHg) ")

# #Systolic Tagged
plt.figure()
plt.plot(BP)
plt.plot(BP_peaks, BP[BP_peaks], "x")
plt.ylabel("Blood Pressure (mmHg)")
plt.title("Raw BP with Systolic Detected")

#plt.show()
interp_fs = 4
uniform_time = np.arange(0, max(td_peaks), 1/interp_fs)
rr_interp_func = interp1d(td_peaks[:-1], RRDistance, kind='cubic', fill_value="extrapolate")
rr_fft = rr_interp_func(uniform_time)

frequencies, psd = welch(rr_fft, fs = interp_fs, nperseg = 256)

#Band Power
lf_band = (frequencies >= 0.04) & (frequencies < 0.15)
hf_band = (frequencies >= 0.15) & (frequencies < 0.4)
vlf_band = (frequencies >= 0.003) & (frequencies < 0.04)

vlf_power = np.trapezoid(psd[vlf_band], frequencies[vlf_band]) * 1e6 
lf_power = np.trapezoid(psd[lf_band], frequencies[lf_band]) * 1e6 #convert to ms for reporting
hf_power = np.trapezoid(psd[hf_band], frequencies[hf_band]) * 1e6 #convert to ms for reporting
lf_hf_ratio = lf_power / hf_power
total_power = lf_power + hf_power

#nu
lf_nu = (lf_power/total_power)
hf_nu = (hf_power/total_power)

plt.figure(figsize=(10, 6))
plt.plot(frequencies, psd * 1e6)  # Convert to ms² for plotting
plt.fill_between(frequencies[lf_band], psd[lf_band] * 1e6, color='skyblue', alpha=0.5, label='LF Band')
plt.fill_between(frequencies[hf_band], psd[hf_band] * 1e6, color='salmon', alpha=0.5, label='HF Band')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (ms²/Hz)')
plt.title('HRV Frequency Domain Analysis')
plt.xlim(0,1.5)
plt.legend()
plt.tight_layout()
plt.savefig("figures/PSD.tif", dpi=600, format = 'tiff')

# Print the results
print(f"LF Power: {lf_power:.2f} ms²")
print(f"HF Power: {hf_power:.2f} ms²")
print(f"Total Power: {total_power:.2f} ms²")
print(f"LF/HF Ratio: {lf_hf_ratio:.2f}")

print(f"LF Power: {lf_nu:.2f} n.u.")
print(f"HF Power: {hf_nu:.2f} n.u.")

#plt.show()
uniform_time_bp = np.arange(0, max(td_BP_peaks), 1/interp_fs)
bp_interp_func = interp1d(td_BP_peaks, Systolic_Array, kind='cubic', fill_value="extrapolate")
bp_fft = bp_interp_func(uniform_time_bp)
frequencies_bp, psd_bp = welch(bp_fft, fs = interp_fs, nperseg = 256)

lf_power_BP = np.trapezoid(psd_bp[lf_band], frequencies_bp[lf_band]) 
hf_power_BP = np.trapezoid(psd_bp[hf_band], frequencies_bp[hf_band]) 

frequencies_coh, coherence_values = coherence(rr_fft, bp_fft, fs=interp_fs, nperseg=256)

# Calculate coherence in LF band
lf_coherence = np.mean(coherence_values[(frequencies_coh >= 0.04) & (frequencies_coh < 0.15)])

# Only calculate BRS if coherence is good
if lf_coherence > 0.5:
    brs_lf = np.sqrt(lf_power) / np.sqrt(lf_power_BP)
    print(f"BRS LF: {brs_lf:.3f} ms/mmHg (coherence OK)")
else:
    print(f"Low coherence ({lf_coherence:.3f}) - BRS may not be reliable")

plt.show()

