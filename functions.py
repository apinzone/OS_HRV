import math
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
from scipy import signal
from scipy import spatial #imports abridged
from scipy.stats import entropy
import scipy
from scipy.stats import linregress
from typing import List, Tuple, Dict

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

from scipy.stats import linregress
import numpy as np

#SEQUENCING
def find_sap_ramps(
    sbp: np.ndarray,
    min_len: int = 3,
    thresh: float = 0
) -> List[Tuple[int, int, str]]:
    ramps = []
    i = 0
    n = len(sbp)

    while i <= n - min_len:
        direction = None
        length = 1
        for j in range(i + 1, n):
            diff = sbp[j] - sbp[j - 1]

            if direction is None:
                if diff >= thresh:
                    direction = 'up'
                    length += 1
                elif diff <= -thresh:
                    direction = 'down'
                    length += 1
                else:
                    break
            else:
                if direction == 'up' and diff >= thresh:
                    length += 1
                elif direction == 'down' and diff <= -thresh:
                    length += 1
                else:
                    break

            if length >= min_len:
                ramps.append((i, j, direction))
                break
        i += 1
    return ramps

def full_sequence_brs(
    sbp: np.ndarray,
    pi: np.ndarray,
    min_len: int = 3,
    delay_range: Tuple[int, int] = (0, 4),
    r_threshold: float = 0.8,
    thresh_sbp: float = 0,
    thresh_pi: float = 0
) -> Dict[str, float]:
    best_results = {
        'BRS_mean': np.nan,
        'BEI': 0.0,
        'num_sequences': 0,
        'num_sbp_ramps': 0,
        'n_up': 0,
        'n_down': 0,
        'best_delay': -1
    }

    for d in range(delay_range[0], delay_range[1] + 1):
        ramps = find_sap_ramps(sbp, min_len=min_len, thresh=thresh_sbp)
        slopes = []
        n_ramps = len(ramps)
        n_sequences = 0
        n_up = 0
        n_down = 0

        for start, end, direction in ramps:
            if end + d >= len(pi):
                continue  # avoid indexing past end
            pi_ramp = pi[start + d:end + 1 + d]
            sbp_ramp = sbp[start:end + 1]

            if len(pi_ramp) != len(sbp_ramp):
                continue

            if np.any(np.abs(np.diff(pi_ramp)) < thresh_pi):
                continue  # fail PI threshold check


            slope, intercept, r_value, _, _ = linregress(sbp_ramp, pi_ramp)
            if abs(r_value) >= r_threshold:
                # Ensure slope is positive and directionally matched
                if (direction == 'up' and slope > 0) or (direction == 'down' and slope > 0):
                    slopes.append(slope)
                    n_sequences += 1
                    if direction == 'up':
                        n_up += 1
                    elif direction == 'down':
                        n_down += 1

        if n_sequences > best_results['num_sequences']:
            best_results.update({
                'BRS_mean': np.mean(slopes) if slopes else np.nan,
                'BEI': n_sequences / n_ramps if n_ramps > 0 else 0,
                'num_sequences': n_sequences,
                'num_sbp_ramps': n_ramps,
                'n_up': n_up,
                'n_down': n_down,
                'best_delay': d
            })

    return best_results

def plot_brs_sequences(
    sbp: np.ndarray,
    pi: np.ndarray,
    ramps: List[Tuple[int, int, str]],
    delay: int = 1,
    r_threshold: float = 0.8,
    thresh_pi: float = 0,
    max_plots: int = 10  # Limit to avoid clutter
):
    count = 0
    for start, end, direction in ramps:
        if end + delay >= len(pi):
            continue

        sbp_ramp = sbp[start:end + 1]
        pi_ramp = pi[start + delay:end + 1 + delay]

        if len(sbp_ramp) != len(pi_ramp):
            continue

        if np.any(np.abs(np.diff(pi_ramp)) < thresh_pi):
            continue

        slope, intercept, r_value, _, _ = linregress(sbp_ramp, pi_ramp)
        if abs(r_value) >= r_threshold:
            # Plot the valid sequence
            plt.figure(figsize=(5, 4))
            plt.plot(sbp_ramp, pi_ramp, 'o-', label=f'r = {r_value:.2f}, slope = {slope:.2f}')
            plt.plot(sbp_ramp, intercept + slope * np.array(sbp_ramp), 'r--')
            plt.xlabel('Systolic BP (mmHg)')
            plt.ylabel('Pulse Interval (ms)')
            plt.title(f'Sequence {count + 1} ({direction})')
            plt.legend()
            plt.tight_layout()
            plt.show()
            count += 1
        if count >= max_plots:
            break
