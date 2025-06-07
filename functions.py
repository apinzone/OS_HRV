import math
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
from scipy import signal
from scipy import spatial #imports abridged
from scipy.stats import entropy
import scipy

#Root mean squared calculation
def rms(input):
     SquareArray = []
     for x in input:
             SquareArray.append(np.square(x))
     return SquareArray     

#Find distance between subsequent elements of an array  
def distancefinder(input):
    size=len(input)
    distanceArray = []
    for x in range(size-1):
            distanceArray.append(abs(input[x]-input[x+1]))
    return distanceArray

#Counts NN intervals over a given threshold to calculate PNN50 
def NNCounter(input,thresh):
    counter=0
    for x in input:
        if x>thresh:
            counter += 1
    return counter

def NNIndexer(input):
    Size1=len(input)
    Mean=np.mean(input)
    StDevArray=[]
    for x in input:
        StDevArray.append(np.sqrt(np.sum(np.absolute(x-np.mean(input)))**2)/Size1)  
    return StDevArray

def SuccessiveDiff(input):
    size=len(input)
    SDArray=[]
    for x in range(size-1):
            SDArray.append(abs(input[x]-input[x+1]))
    return SDArray
def RemoveOutliers(x, y, threshold):
    new_x = []
    new_y = []
    size = len(y)
    for i in range(size):
        if y[i] < threshold:
            new_x.append(x[i])
            new_y.append(y[i])
    return new_x, new_y

#Creating parameters for scatter plot of an element vs. the next element in an array
def Poincare(input):
    size=len(input)
    newArray=[]
    for x in range(size-1):
        newArray.append(input[x+1])
    return newArray

#Trimming? 
def TimeTrimmer(input,thresh):
    timeArray=[]
    for x in input:
        if x >= thresh:
            timeArray.append(x)
    return timeArray
def SignalTrimmer(input, fs, thresh):
    trimmedArray=[]
    counter=0
    for x in input:
        counter += 1
        timex=counter/fs
        if timex >=thresh:
            trimmedArray.append(x)
    return trimmedArray


def FindTimeIndex(time_list, time):
    for i in range(0, len(time_list)):
        if time_list[i] >= time:
            # As soon as we find the first occurance where time[i] is >= the time we want,
            # we can exit the function and return the index
            return i
    
    # If we never found the index, return 0
    print("[CutTime]: Could not find index!")
    return 0

#Counting up and down ramps in BP
def bpCount(BP_input, BP_thresh):

    class direction(Enum):
        nil = 1
        up = 2
        down = 3

    up_count = 0
    down_count = 0
    qualifying_count_num = 3
    temp_count = 0
    d1 = direction.nil
    size = len(BP_input)

    for i in range(1, size):
        if  BP_input[i] >= (BP_input[i-1] + BP_thresh): # up
            if d1 == direction.down:
                if temp_count >= qualifying_count_num:
                    down_count += 1
                temp_count = 0

            temp_count += 1
            d1 = direction.up
        elif BP_input[i] <= (BP_input[i-1] - BP_thresh): # down
            if d1 == direction.up:
                if temp_count >= qualifying_count_num:
                    up_count += 1
                temp_count = 0

            temp_count += 1
            d1 = direction.down
        else:
            if temp_count >= qualifying_count_num:
                if d1 == direction.up:
                    up_count += 1
                elif d1 == direction.down:
                    down_count += 1
                    
            temp_count = 0
            d1 = direction.nil

        # if i == size-1:
        #     if temp_count >= qualifying_count_num:
        #         if d1 == direction.up:
        #             up_count += 1
        #         elif d1 == direction.down:
        #             down_count += 1

    return up_count, down_count


#Counting up-up and down-down events in PI + BP 
def count(RR_input, BP_input, RR_thresh, BP_thresh):

    class direction(Enum):
        nil = 1
        up = 2
        down = 3

    up_count = 0
    down_count = 0
    qualifying_count_num = 3
    temp_count = 0
    d1 = direction.nil
    size = len(RR_input)

    if len(RR_input) != len(BP_input):
        print('[count]: lengths are not equal')

    for i in range(1, size):
        if RR_input[i] <= (RR_input[i-1] - RR_thresh) and BP_input[i] >= (BP_input[i-1] + BP_thresh): # up
            if d1 == direction.down:
                if temp_count >= qualifying_count_num:
                    down_count += 1
                temp_count = 0

            temp_count += 1
            d1 = direction.up
        elif RR_input[i] >= (RR_input[i-1] + RR_thresh) and BP_input[i] <= (BP_input[i-1] - BP_thresh): # down
            if d1 == direction.up:
                if temp_count >= qualifying_count_num:
                    up_count += 1
                temp_count = 0

            temp_count += 1
            d1 = direction.down
        else:
            if temp_count >= qualifying_count_num:
                if d1 == direction.up:
                    up_count += 1
                elif d1 == direction.down:
                    down_count += 1
                    
            temp_count = 0
            d1 = direction.nil

        # if i == size-1:
        #     if temp_count >= qualifying_count_num:
        #         if d1 == direction.up:
        #             up_count += 1
        #         elif d1 == direction.down:
        #             down_count += 1

    return up_count, down_count
 
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
        
import numpy as np
import matplotlib.pyplot as plt

#Resample tachogram at a lower sampling rate (250 hz)
def resample_tachogram(tachogram, original_sampling_rate, target_sampling_rate):
    original_length = len(tachogram)
    original_duration = original_length / original_sampling_rate
    target_length = int(original_duration * target_sampling_rate)
    target_duration = target_length / target_sampling_rate
    resampled_tachogram = signal.resample(tachogram, target_length)
    return resampled_tachogram, target_sampling_rate

def ApEn(U, m, r) -> float:
    """Approximate_entropy."""
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0)
            for x_i in x]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))
    N = len(U)
    return abs(_phi(m + 1) - _phi(m))   

def SampEn(U, m, r):
    """Sample Entropy of time series U with embedding dimension m and tolerance r."""
    U = np.array(U)
    N = len(U)

    def _phi(m):
        x = np.array([U[i:i + m] for i in range(N - m + 1)])
        C = np.sum([np.sum(np.max(np.abs(x - xi), axis=1) <= r) - 1 for xi in x])
        return C / (N - m + 1) / (N - m)

    return -np.log(_phi(m + 1) / _phi(m))


def OutlierRemove(xinput, yinput):
    thresh1 = np.average(yinput) + 2.5 * np.std(yinput) 
    thresh2 = np.average(yinput) - 2.5 * np.std(yinput)
    for x in yinput:
        if x >= thresh1 or x <= thresh2:
            index = yinput.index(x)
            new_y = np.delete(yinput,index)
            new_x = np.delete(xinput,index)
    return(new_x,new_y)
            
def bandpower(x, fs, fmin, fmax):
    f, Pxx = scipy.signal.periodogram(x, fs=fs)
    ind_min = scipy.argmax(f > fmin) - 1
    ind_max = scipy.argmax(f > fmax) - 1
    return scipy.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])  

def going_up(array, thresh):
    for i in range(0,len(array) - 1):
        if array[i] > array[i + 1] + thresh:
            return False
    return True

def going_down(array, thresh):
    for i in range(0,len(array) - 1):
        if array[i] < array[i + 1] + thresh:
            return False
    return True

def sequencing(PI, BP):
    window = 3 # indexes
    PI_thresh = 4 # ms
    BP_thresh = 1 # mmHg
    if len(PI) != len(BP):
        print('Ay, pass in arrays with the same size')
        return [0]
    
    PI_view_window = []
    BP_view_window = []
    output = [] # up event = +1, down event = -1

    for i in range(0, len(PI)):
        if len(PI_view_window) < window:
            PI_view_window.append(PI[i])
            BP_view_window.append(BP[i])
        else:
            if going_down(PI_view_window, PI_thresh) and going_up(BP_view_window, BP_thresh):
                output.append(1)
            elif going_up(PI_view_window, PI_thresh) and going_down(BP_view_window, BP_thresh):
                output.append(-1)
            else:
                # nothing happend (i guess)
                output.append(0)

            PI_view_window.pop(0)
            PI_view_window.append(PI[i])
            BP_view_window.pop(0)
            BP_view_window.append(BP[i])
    return output

