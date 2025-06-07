
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

# import EntropyHubs



#Open ACQ File
ECG_source = "data/IHG.acq"
file = bioread.read_file(ECG_source)
Channel_List=file.channels



#Pull BP Data 
BP_Data = file.channels[0].raw_data
BP_Time = file.channels[0].time_index
BP_fs = len(BP_Data)/max(BP_Time)
BP = BP_Data
BP_peaks, _ = find_peaks(BP, height = 50, threshold = None, distance = 100, prominence=(5,None), width=None, wlen=None, rel_height=None, plateau_size=None)
td_BP_peaks = (BP_peaks/BP_fs)

#Pull ECG Data  and all Variables
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
peaks, _ = find_peaks(x, height = 40, threshold = None, distance = 100, prominence=(0.7,None), width=None, wlen=None, rel_height=None, plateau_size=None)
td_peaks = (peaks / ECG_fs)
RRDistance = distancefinder(td_peaks)
#convert to ms
RRDistance_ms = [element * 1000 for element in RRDistance]

# #Tag Systolic BP Peaks Untrimmed and trimmed
BP_peaks, _ = find_peaks(BP, height = 50, threshold = None, distance = 100, prominence=(40,None), width=None, wlen=None, rel_height=None, plateau_size=None)
td_BP_peaks = (BP_peaks/BP_fs)
# Trimmed_BP_peaks, _ = find_peaks(TrimmedBP, height = 50, threshold = None, distance = 100, prominence=(40,None), width=None, wlen=None, rel_height=None, plateau_size=None)
# Trimmed_td_BP_peaks = (Trimmed_BP_peaks/BP_fs)
# Trimmed_Systolic_Array = TrimmedBP[Trimmed_BP_peaks]
#Obtain pulse interval (time difference between BP peaks)
PulseIntervalDistance = distancefinder(td_BP_peaks)
#Convert to ms
PI_ms = [element * 1000 for element in PulseIntervalDistance]
m = 2
r = 0.2 * np.std(RRDistance_ms)

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
#Create axes for Poincare Plot
RRIplusOne = Poincare(RRDistance_ms)

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
#Leave log transformations in in case we want them
# print("Ln RMSSD = " + str(np.log((RMSSD))))
# print("Ln SDNN = " + str(np.log(SDNN)))
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

#RRI / Tachogram
plt.figure()
#Need to remove last element of td_peaks in order for two arrays that we are plotting to match in size 
plt.plot(np.delete(td_peaks,-1), RRDistance_ms)
plt.title("RRI")
plt.xlabel("time (s)")
plt.ylabel("RRI (ms)")
plt.ylim(400,1100)
plt.text(0, 600, 'n = ' + str (len(RRDistance_ms)), fontsize=10)
plt.text(0, 500, 'Mean = ' + str (np.round(np.average(RRDistance_ms),1)) + ' ms', fontsize=10)
plt.text(200, 500, 'σ2 = ' + str (np.round(np.var(RRDistance_ms),1)) + 'ms\u00b2', fontsize=10)  
# plt.show()

#Poincare Plot (RRI, RRI + 1)
EllipseCenterX = np.average(np.delete(RRDistance_ms,-1))
EllipseCenterY = np.average(RRIplusOne)
Center_coords = EllipseCenterX,EllipseCenterY
fig = plt.figure()
ax=plt.axes()
#need to remove last element of array of RR Distances to make arrays we are plotting match
z = np.polyfit(np.delete(RRDistance_ms,-1), RRIplusOne, 1)
p = np.poly1d(z)
slope = z[0]
theta=np.degrees(np.arctan(slope))
plt.title("Poincaré Plot")
plt.scatter(np.delete(RRDistance_ms,-1), RRIplusOne)
#create ellipse parameters, xy coordinates for center, width of ellipse, height of ellipse, angle of ellipse, colors of outline and inside
e=Ellipse(xy=(Center_coords),width = SD2*2,height = SD1*2,angle = theta, edgecolor='black',facecolor='none')
matplotlib.axes.Axes.add_patch(ax,e)
plt.plot(np.delete(RRDistance_ms,-1), p(np.delete(RRDistance_ms,-1)), color="red")
plt.ylabel("RRI + 1 (ms)")
plt.xlabel("RRI (ms)")
plt.text(950, 750, 'SD1 = ' + str(np.round((SD1),1)) + " ms", fontsize=10)
plt.text(950, 700, 'SD2 = ' + str(np.round((SD2),1)) + "ms", fontsize=10)

#Start of BP Plots 
# #Raw BP Data 
plt.figure()
plt.plot(BP_Time, BP_Data)
plt.xlabel("time (s)")
plt.ylabel("Finger Pressure (mmHg) ")

#Systolic Tagged
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
plt.legend()
plt.grid(True)
plt.tight_layout()

# Print the results
print(f"LF Power: {lf_power:.2f} ms²")
print(f"HF Power: {hf_power:.2f} ms²")
print(f"Total Power: {total_power:.2f} ms²")
print(f"LF/HF Ratio: {lf_hf_ratio:.2f}")

print(f"LF Power: {lf_nu:.2f} n.u.")
print(f"HF Power: {hf_nu:.2f} n.u.")
# plt.show()