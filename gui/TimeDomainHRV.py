from statistics import stdev
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
from functions import *
import bioread
import defaults

class TimeDomainHRV():
    
    def __init__(self, input_data=None):
        # Init defaults
        self.input_file = defaults.ecg_file_path
        self.BP_height = defaults.BP_height
        self.BP_distance = defaults.BP_distance
        self.BP_prominence = defaults.BP_prominence
        self.ECG_height = defaults.ECG_height
        self.ECG_distance = defaults.ECG_distance
        self.ECG_prominence = defaults.ECG_prominence

        self.init_data()
    
    def run(self):
        self.init_data() 
        self.calculate_peaks()
        self.compute()
        self.print()
        self.graph()

    def init_data(self):
        self.file = bioread.read_file(self.input_file)
        channel_list = self.file.channels

        # Get BP data
        self.BP_data = channel_list[0].raw_data
        BP_time = channel_list[0].time_index
        self.BP_fs = len(self.BP_data)/max(BP_time)
        
        # Get ECG data
        self.ECG_data = channel_list[1].raw_data
        self.time = channel_list[1].time_index
        self.ECG_fs = len(self.ECG_data)/max(self.time)
    
    def calculate_peaks(self):
        # Calculate BP peaks
        self.BP_peaks, _ = find_peaks(self.BP_data, height = self.BP_height, 
            threshold = None, distance = self.BP_distance, prominence=(self.BP_prominence, None), 
            width=None, wlen=None, rel_height=None, plateau_size=None)
        # td_BP_peaks = (self.BP_peaks/self.BP_fs)

        # Calculate ECG peaks
        self.peaks, _ = find_peaks(self.ECG_data, height = self.ECG_height, 
            threshold = None, distance = self.ECG_distance, prominence=(self.ECG_prominence, None), 
            width=None, wlen=None, rel_height=None, plateau_size=None)
        self.td_peaks = (self.peaks / self.ECG_fs)

        RR_distance = distancefinder(self.td_peaks)
        self.RR_distance_ms = [element * 1000 for element in RR_distance]

    def compute(self):
        self.compute_time_domain_HRV_vars()
        self.compute_blood_pressure_info()

    def print(self):
        print(self.print_s())

    def print_s(self) -> str:
        s = ''
        s += "n = " + str(self.num_beats) + " beats are included for analysis\n"
        s += "The total sampling time is " + str(self.sampling_time) + " seconds\n"
        s += "The average heart rate during the sampling time is = " + str(self.HR) + " BPM\n"
        s += "the mean difference between successive R-R intervals is = " + str(np.round(np.average(self.sd), 3)) + " ms\n"
        s += "The mean R-R Interval duration is  " + str(np.round(np.average(self.RR_distance_ms),3)) + " ms\n"
        s += "pNN50 = " + str(np.round(self.pNN50, 3)) + " %\n" 
        s += "RMSSD = " + str(np.round(self.RMSSD, 3)) + " ms\n"
        s += "SDNN = " + str(np.round(self.SDNN, 3)) + " ms\n"
        s += "SDSD = " + str(np.round(self.SDSD, 3)) + " ms\n"
        s += "SD1 = " + str(np.round(self.SD1, 3)) + " ms\n"
        s += "SD2 = " + str(np.round(self.SD2, 3)) + " ms\n"
        s += "SD1/SD2 = " +str(np.round((self.SD1 / self.SD2), 3)) + "\n"
        s += "The area of the ellipse fitted over the Poincaré Plot (S) is " + str(np.round(self.S,3)) + " ms^2\n"
        s += "The average systolic blood pressure during the sampling time is " + str(self.avg_BP) + " + - " + str(self.SD_BP) + " mmHg\n"
        s += str(len(self.systolic_array)) + " pressure waves are included in the analysis\n"
        return s

    def graph(self):
        #graph 1: raw ECG
        plt.title("Raw ECG Signal")
        plt.plot(self.time, self.ECG_data)
        plt.xlabel("time (s)")
        plt.ylabel("ECG (mV)")

        # graph 2: ECG with R intervals tagged
        plt.figure()
        plt.title("Raw ECG Signal with R-R Detected")
        plt.plot(self.ECG_data)
        plt.plot(self.peaks, self.ECG_data[self.peaks], "x")
        plt.xlabel("time (s)")
        plt.ylabel("ECG (mV)")

        # graph 3: RRI
        plt.figure()
        plt.title("RRI")
        plt.plot(np.delete(self.td_peaks, -1), self.RR_distance_ms)
        plt.xlabel("time (s)")
        plt.ylabel("ECG (mV)") # This right?

        plt.show()

    def set_region(self, region):
        start = region[0]
        stop = region[1]
        # We refresh the time index so we dont lose data with repeated region selections
        self.init_data()
        start = find_nearest(self.time, start)
        stop = find_nearest(self.time, stop)
        start_index = np.where(self.time == start)
        stop_index = np.where(self.time == stop)
        self.time = self.time[start_index[0][0] : stop_index[0][0]]
        self.ECG_data = self.ECG_data[start_index[0][0] : stop_index[0][0]]
        # TODO: BP needs to be trimmed as well?

    def compute_time_domain_HRV_vars(self):
        self.SDNN = np.std(self.RR_distance_ms)
        self.sd = SuccessiveDiff(self.RR_distance_ms)
        self.SDSD = np.std(self.sd)
        self.NN50 = NNCounter(self.sd, 50)
        self.pNN50 = (self.NN50 / len(self.td_peaks)) * 100
        self.RMSSD = np.sqrt(np.average(rms(self.sd)))
        self.SD1 = np.sqrt(0.5 * math.pow(self.SDSD, 2))
        self.SD2 = np.sqrt(2 * math.pow(self.SDNN, 2) - (0.5 * math.pow(self.SDSD, 2)))
        self.S = math.pi * self.SD1 * self.SD2
        self.sampling_time = max(self.td_peaks)
        self.num_beats = len(self.RR_distance_ms)
        self.HR = np.round(self.num_beats / (self.sampling_time / 60), 2)

        # Create axes for Poincare Plot
        PPlot = np.delete(self.RR_distance_ms, -1)
        RRI_plus_one = Poincare(self.RR_distance_ms)

    def compute_blood_pressure_info(self):
        self.systolic_array = self.BP_data[self.BP_peaks]
        self.avg_BP = np.round(np.average(self.systolic_array), 3)
        self.SD_BP = np.round(np.std(self.systolic_array), 3)

    def set_input_file(self, name):
        self.input_file = name
    
    def set_BP_height(self, value):
        self.BP_height = value

    def set_BP_distance(self, value):
        self.BP_distance = value
    
    def set_BP_prominence(self, value):
        self.BP_prominence = value

    def set_ECG_height(self, value):
        self.ECG_height = value

    def set_ECG_distance(self, value):
        self.ECG_distance = value

    def set_ECG_prominence(self, value):
        self.ECG_prominence = value