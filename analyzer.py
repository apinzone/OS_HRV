# analyzer_consistent.py
import numpy as np
import math
from scipy.signal import find_peaks, welch, coherence, csd
from scipy.interpolate import interp1d
import bioread
from functions import *  # All your existing functions

class CardiovascularAnalyzer:
    def __init__(self):
        self.ecg_data = {}
        self.bp_data = {}
        self.results = {}
        self.time_window = None  # NEW: Store time window
        
    def load_file(self, filepath):
        """Exactly matches your original file loading"""
        file = bioread.read_file(filepath)
        Channel_List = file.channels
        
        # BP Data (Channel 0) - exactly as in your code
        BP_Data = file.channels[0].raw_data
        BP_Time = file.channels[0].time_index
        BP_fs = len(BP_Data)/max(BP_Time)
        
        self.bp_data = {
            'raw': BP_Data,
            'time': BP_Time,
            'fs': BP_fs
        }
        
        # ECG Data (Channel 1) - exactly as in your code
        ECG_Data = file.channels[1].raw_data
        Time = file.channels[1].time_index
        ECG_fs = len(ECG_Data)/max(Time)
        
        self.ecg_data = {
            'raw': ECG_Data,
            'time': Time,
            'fs': ECG_fs
        }
        
    def set_time_window(self, start_time, end_time):
        """NEW: Set time window for analysis"""
        self.time_window = {
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time
        }
        
    def filter_data_by_time_window(self, data_array, time_array, start_time, end_time):
        """NEW: Filter data to only include points within the specified time window"""
        mask = (time_array >= start_time) & (time_array <= end_time)
        return data_array[mask], time_array[mask], mask

    def filter_peaks_by_time_window(self, peaks, time_array, start_time, end_time):
        """NEW: Filter peaks to only include those within the specified time window"""
        # Convert peak indices to times
        peak_times = np.array([time_array[p] for p in peaks if p < len(time_array)])
        
        # Find peaks within time window
        time_mask = (peak_times >= start_time) & (peak_times <= end_time)
        
        # Get the original peak indices that fall within the time window
        valid_peaks = [peaks[i] for i in range(len(peaks)) if i < len(time_mask) and time_mask[i]]
        
        return np.array(valid_peaks), time_mask
        
    def find_peaks(self):
        """Exactly matches your original peak detection"""
        # ECG peaks - your exact parameters
        x = self.ecg_data['raw']
        peaks, _ = find_peaks(x, height=0.8, threshold=None, distance=100, 
                             prominence=(0.7,None), width=None, wlen=None, 
                             rel_height=None, plateau_size=None)
        
        td_peaks = (peaks / self.ecg_data['fs'])
        RRDistance = distancefinder(td_peaks)  # Your function
        RRDistance_ms = [element * 1000 for element in RRDistance]  # Your conversion
        
        self.ecg_data.update({
            'peaks': peaks,
            'td_peaks': td_peaks,
            'rr_intervals': RRDistance_ms
        })
        
        # BP peaks - your exact parameters
        BP = self.bp_data['raw']
        BP_peaks, _ = find_peaks(BP, height=110, threshold=None, distance=100, 
                                prominence=(5,None), width=None, wlen=None, 
                                rel_height=None, plateau_size=None)
        
        td_BP_peaks = (BP_peaks/self.bp_data['fs'])
        Systolic_Array = BP[BP_peaks]  # Your variable name
        
        self.bp_data.update({
            'peaks': BP_peaks,
            'td_peaks': td_BP_peaks,
            'systolic': Systolic_Array
        })

    def find_peaks_with_params(self, ecg_height=0.8, ecg_distance=100, ecg_prominence=0.7,
                            bp_height=110, bp_distance=100, bp_prominence=5):
        """Find peaks with custom parameters"""
        
        # ECG peaks with custom parameters
        x = self.ecg_data['raw']
        peaks, _ = find_peaks(x, 
                            height=ecg_height, 
                            threshold=None, 
                            distance=ecg_distance, 
                            prominence=(ecg_prominence, None), 
                            width=None, 
                            wlen=None, 
                            rel_height=None, 
                            plateau_size=None)
        
        td_peaks = (peaks / self.ecg_data['fs'])
        RRDistance = distancefinder(td_peaks)
        RRDistance_ms = [element * 1000 for element in RRDistance]
        
        self.ecg_data.update({
            'peaks': peaks,
            'td_peaks': td_peaks,
            'rr_intervals': RRDistance_ms
        })
        
        # BP peaks with custom parameters
        BP = self.bp_data['raw']
        BP_peaks, _ = find_peaks(BP, 
                                height=bp_height, 
                                threshold=None, 
                                distance=bp_distance, 
                                prominence=(bp_prominence, None), 
                                width=None, 
                                wlen=None, 
                                rel_height=None, 
                                plateau_size=None)
        
        td_BP_peaks = (BP_peaks/self.bp_data['fs'])
        Systolic_Array = BP[BP_peaks]
        
        self.bp_data.update({
            'peaks': BP_peaks,
            'td_peaks': td_BP_peaks,
            'systolic': Systolic_Array
        })
        
        # Store the parameters used
        self.peak_detection_params = {
            'ecg_height': ecg_height,
            'ecg_distance': ecg_distance,
            'ecg_prominence': ecg_prominence,
            'bp_height': bp_height,
            'bp_distance': bp_distance,
            'bp_prominence': bp_prominence
        }

    def analyze_with_current_peaks(self, time_window=None):
        """NEW: Run analysis with current peak detection results and optional time window"""
        if time_window:
            self.set_time_window(time_window['start_time'], time_window['end_time'])
        
        self.calculate_time_domain()
        self.calculate_frequency_domain()
        self.calculate_brs_sequence()
        self.calculate_brs_spectral()
        
    def get_windowed_data(self):
        """NEW: Get ECG and BP data filtered by time window if set"""
        if self.time_window is None:
            # No time window set, return all data
            return {
                'ecg_peaks': self.ecg_data['peaks'],
                'ecg_td_peaks': self.ecg_data['td_peaks'],
                'ecg_rr_intervals': self.ecg_data['rr_intervals'],
                'bp_peaks': self.bp_data['peaks'],
                'bp_td_peaks': self.bp_data['td_peaks'],
                'bp_systolic': self.bp_data['systolic']
            }
        
        # Filter ECG data by time window
        start_time = self.time_window['start_time']
        end_time = self.time_window['end_time']
        
        # Filter ECG peaks
        ecg_peaks_windowed, _ = self.filter_peaks_by_time_window(
            self.ecg_data['peaks'], self.ecg_data['time'], start_time, end_time
        )
        
        # Recalculate td_peaks and RR intervals for windowed data
        if len(ecg_peaks_windowed) > 1:
            ecg_td_peaks_windowed = ecg_peaks_windowed / self.ecg_data['fs']
            ecg_rr_windowed = distancefinder(ecg_td_peaks_windowed)
            ecg_rr_windowed_ms = [element * 1000 for element in ecg_rr_windowed]
        else:
            ecg_td_peaks_windowed = np.array([])
            ecg_rr_windowed_ms = []
        
        # Filter BP peaks
        bp_peaks_windowed, _ = self.filter_peaks_by_time_window(
            self.bp_data['peaks'], self.bp_data['time'], start_time, end_time
        )
        
        # Recalculate BP data for windowed peaks
        if len(bp_peaks_windowed) > 1:
            bp_td_peaks_windowed = bp_peaks_windowed / self.bp_data['fs']
            bp_systolic_windowed = self.bp_data['raw'][bp_peaks_windowed]
        else:
            bp_td_peaks_windowed = np.array([])
            bp_systolic_windowed = np.array([])
        
        return {
            'ecg_peaks': ecg_peaks_windowed,
            'ecg_td_peaks': ecg_td_peaks_windowed,
            'ecg_rr_intervals': ecg_rr_windowed_ms,
            'bp_peaks': bp_peaks_windowed,
            'bp_td_peaks': bp_td_peaks_windowed,
            'bp_systolic': bp_systolic_windowed
        }
        
    def calculate_time_domain(self):
        """Updated to use windowed data"""
        windowed_data = self.get_windowed_data()
        RRDistance_ms = windowed_data['ecg_rr_intervals']
        td_peaks = windowed_data['ecg_td_peaks']
        
        if len(RRDistance_ms) < 2:
            # Not enough data for analysis
            self.results['time_domain'] = {
                'error': 'Insufficient RR intervals in selected time window',
                'num_beats': len(RRDistance_ms)
            }
            return
        
        # Your exact calculations
        Successive_time_diff = SuccessiveDiff(RRDistance_ms)  # Your function
        AvgDiff = np.average(Successive_time_diff)
        SDNN = np.std(RRDistance_ms)
        SDSD = np.std(Successive_time_diff)
        NN50 = NNCounter(Successive_time_diff, 50)  # Your function
        pNN50 = (NN50/len(td_peaks))*100 if len(td_peaks) > 0 else 0
        RMSSD = np.sqrt(np.average(rms(Successive_time_diff)))  # Your function
        SD1 = np.sqrt(0.5*math.pow(SDSD,2))
        SD2 = np.sqrt((2*math.pow(SDNN,2) - (0.5*math.pow(SDSD,2))))
        S = math.pi * SD1 * SD2
        
        if len(td_peaks) > 0:
            Sampling_Time = max(td_peaks) - min(td_peaks) if self.time_window else max(td_peaks)
        else:
            Sampling_Time = 0
            
        Num_Beats = len(RRDistance_ms)
        HR = np.round(Num_Beats/(Sampling_Time/60),2) if Sampling_Time > 0 else 0
        
        # Sample entropy with your exact parameters
        m = 2
        r = 0.2 * np.std(RRDistance_ms)
        sampen = SampEn(RRDistance_ms, m, r) if len(RRDistance_ms) > m else 0  # Your function
        
        self.results['time_domain'] = {
            'num_beats': Num_Beats,
            'sampling_time': Sampling_Time,
            'hr': HR,
            'avg_diff': AvgDiff,
            'mean_rr': np.average(RRDistance_ms),
            'rmssd': RMSSD,
            'sdnn': SDNN,
            'sdsd': SDSD,
            'pnn50': pNN50,
            'sd1': SD1,
            'sd2': SD2,
            'sd1_sd2_ratio': SD1/SD2 if SD2 > 0 else 0,
            'ellipse_area': S,
            'sample_entropy': sampen
        }
        
    def calculate_frequency_domain(self):
        """Updated to use windowed data"""
        windowed_data = self.get_windowed_data()
        td_peaks = windowed_data['ecg_td_peaks']
        RRDistance_ms = windowed_data['ecg_rr_intervals']
        
        if len(RRDistance_ms) < 10:  # Need minimum data for frequency analysis
            self.results['frequency_domain'] = {
                'error': 'Insufficient data for frequency domain analysis',
                'num_rr': len(RRDistance_ms)
            }
            return
        
        # Your exact interpolation
        interp_fs = 4
        uniform_time = np.arange(0, max(td_peaks), 1/interp_fs) if len(td_peaks) > 1 else np.array([])
        
        if len(uniform_time) < 10:
            self.results['frequency_domain'] = {
                'error': 'Time window too short for frequency analysis',
                'window_duration': max(td_peaks) - min(td_peaks) if len(td_peaks) > 1 else 0
            }
            return
            
        rr_interp_func = interp1d(td_peaks[:-1], RRDistance_ms, kind='cubic', fill_value="extrapolate")
        rr_fft = rr_interp_func(uniform_time)
        
        frequencies, psd = welch(rr_fft, fs=interp_fs, nperseg=min(256, len(rr_fft)//4))
        
        # Your exact band definitions
        lf_band = (frequencies >= 0.04) & (frequencies < 0.15)
        hf_band = (frequencies >= 0.15) & (frequencies < 0.4)
        vlf_band = (frequencies >= 0.003) & (frequencies < 0.04)
        
        # Your exact power calculations (no unit conversion here to match original)
        vlf_power = np.trapezoid(psd[vlf_band], frequencies[vlf_band]) if np.any(vlf_band) else 0
        lf_power = np.trapezoid(psd[lf_band], frequencies[lf_band]) if np.any(lf_band) else 0
        hf_power = np.trapezoid(psd[hf_band], frequencies[hf_band]) if np.any(hf_band) else 0
        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
        total_power = lf_power + hf_power
        
        # Normalized units
        lf_nu = (lf_power/total_power) if total_power > 0 else 0
        hf_nu = (hf_power/total_power) if total_power > 0 else 0
        
        self.results['frequency_domain'] = {
            'vlf_power': vlf_power,
            'lf_power': lf_power,
            'hf_power': hf_power,
            'total_power': total_power,
            'lf_hf_ratio': lf_hf_ratio,
            'lf_nu': lf_nu,
            'hf_nu': hf_nu,
            'frequencies': frequencies,
            'psd': psd,
            'rr_fft': rr_fft,  # Store for BRS analysis
            'uniform_time': uniform_time
        }
        
    def calculate_brs_sequence(self):
        """Updated to use windowed data"""
        windowed_data = self.get_windowed_data()
        Systolic_Array = windowed_data['bp_systolic']
        RRDistance_ms = windowed_data['ecg_rr_intervals']
        
        if len(Systolic_Array) < 3 or len(RRDistance_ms) < 3:
            self.results['brs_sequence'] = {
                'error': 'Insufficient data for BRS sequence analysis',
                'bp_beats': len(Systolic_Array),
                'ecg_beats': len(RRDistance_ms)
            }
            return
        
        # Your exact function call with exact parameters
        results = full_sequence_brs(
            sbp=Systolic_Array,
            pi=RRDistance_ms,
            min_len=3,
            delay_range=(0, 4),
            r_threshold=0.8,
            thresh_sbp=1,
            thresh_pi=4
        )
        
        self.results['brs_sequence'] = results
        
    def calculate_brs_spectral(self):
        """Updated to use windowed data"""
        windowed_data = self.get_windowed_data()
        td_BP_peaks = windowed_data['bp_td_peaks']
        Systolic_Array = windowed_data['bp_systolic']
        
        if len(Systolic_Array) < 10:
            self.results['brs_spectral'] = {
                'error': 'Insufficient BP data for spectral BRS analysis',
                'bp_beats': len(Systolic_Array)
            }
            return
        
        # Check if we have frequency domain results
        if 'frequency_domain' not in self.results or 'error' in self.results['frequency_domain']:
            self.results['brs_spectral'] = {
                'error': 'Frequency domain analysis required for spectral BRS',
                'dependency': 'frequency_domain'
            }
            return
        
        # Your exact interpolation for BP
        interp_fs = 4
        uniform_time_bp = np.arange(0, max(td_BP_peaks), 1/interp_fs) if len(td_BP_peaks) > 1 else np.array([])
        
        if len(uniform_time_bp) < 10:
            self.results['brs_spectral'] = {
                'error': 'BP time window too short for spectral analysis',
                'bp_duration': max(td_BP_peaks) - min(td_BP_peaks) if len(td_BP_peaks) > 1 else 0
            }
            return
            
        bp_interp_func = interp1d(td_BP_peaks, Systolic_Array, kind='cubic', fill_value="extrapolate")
        bp_fft = bp_interp_func(uniform_time_bp)
        frequencies_bp, psd_bp = welch(bp_fft, fs=interp_fs, nperseg=min(256, len(bp_fft)//4))
        
        # Get RR data from frequency domain results
        rr_fft = self.results['frequency_domain']['rr_fft']
        
        # Your exact band definitions
        lf_band = (frequencies_bp >= 0.04) & (frequencies_bp < 0.15)
        hf_band = (frequencies_bp >= 0.15) & (frequencies_bp < 0.4)
        
        # Your exact power calculations (no unit conversion to match original)
        lf_power_BP = np.trapezoid(psd_bp[lf_band], frequencies_bp[lf_band]) if np.any(lf_band) else 0
        hf_power_BP = np.trapezoid(psd_bp[hf_band], frequencies_bp[hf_band]) if np.any(hf_band) else 0
        
        # Your exact coherence calculation
        try:
            frequencies_coh, coherence_values = coherence(rr_fft, bp_fft, fs=interp_fs, nperseg=min(256, len(rr_fft)//4))
            lf_coherence = np.mean(coherence_values[(frequencies_coh >= 0.04) & (frequencies_coh < 0.15)])
            hf_coherence = np.mean(coherence_values[(frequencies_coh >= 0.15) & (frequencies_coh < 0.4)])
            
            # Your exact cross-spectral calculation
            frequencies_csd, csd_bp_rr = csd(bp_fft, rr_fft, fs=interp_fs, nperseg=min(256, len(bp_fft)//4))
            _, psd_bp_auto = welch(bp_fft, fs=interp_fs, nperseg=min(256, len(bp_fft)//4))
            transfer_gain = np.abs(csd_bp_rr)/psd_bp_auto
            
            # Your exact band definitions for transfer function
            lf_band_tf = (frequencies_csd >= 0.04) & (frequencies_csd < 0.15)
            hf_band_tf = (frequencies_csd >= 0.15) & (frequencies_csd < 0.4)
            brs_lf_tf = np.mean(transfer_gain[lf_band_tf]) if np.any(lf_band_tf) else 0
            brs_hf_tf = np.mean(transfer_gain[hf_band_tf]) if np.any(hf_band_tf) else 0
            
            self.results['brs_spectral'] = {
                'brs_lf_tf': brs_lf_tf,
                'brs_hf_tf': brs_hf_tf,
                'lf_coherence': lf_coherence,
                'hf_coherence': hf_coherence,
                'lf_power_bp': lf_power_BP,
                'hf_power_bp': hf_power_BP,
                'frequencies_coh': frequencies_coh,
                'coherence_values': coherence_values,
                'transfer_gain': transfer_gain,
                'frequencies_csd': frequencies_csd,
                'valid_lf': lf_coherence > 0.5,
                'valid_hf': hf_coherence > 0.5
            }
        except Exception as e:
            self.results['brs_spectral'] = {
                'error': f'Spectral BRS calculation failed: {str(e)}',
                'rr_length': len(rr_fft),
                'bp_length': len(bp_fft)
            }
        
    def analyze_all(self, time_window=None):
        """Run complete analysis exactly like your main.py, with optional time window"""
        if time_window:
            self.set_time_window(time_window['start_time'], time_window['end_time'])
            
        self.find_peaks()
        self.calculate_time_domain()
        self.calculate_frequency_domain()
        self.calculate_brs_sequence()
        self.calculate_brs_spectral()
        
    def get_summary(self):
        """Enhanced summary including all your original metrics, with time window info"""
        if not self.results:
            return "No analysis completed yet"
            
        summary = []
        
        # Time window information
        if self.time_window:
            tw = self.time_window
            summary.append("=== ANALYSIS WINDOW ===")
            summary.append(f"Time window: {tw['start_time']:.1f}s to {tw['end_time']:.1f}s")
            summary.append(f"Window duration: {tw['duration']:.1f} seconds ({tw['duration']/60:.1f} minutes)")
            summary.append("")
        
        # Time domain (exactly matching your original print statements)
        if 'time_domain' in self.results:
            td = self.results['time_domain']
            if 'error' in td:
                summary.append("=== TIME DOMAIN RESULTS ===")
                summary.append(f"ERROR: {td['error']}")
                summary.append("")
            else:
                summary.append("=== TIME DOMAIN RESULTS ===")
                summary.append(f"n = {td['num_beats']} beats are included for analysis")
                summary.append(f"The total sampling time is {td['sampling_time']:.3f} seconds")
                summary.append(f"The average heart rate during the sampling time is = {td['hr']} BPM")
                summary.append(f"the mean difference between successive R-R intervals is = {td['avg_diff']:.3f} ms")
                summary.append(f"The mean R-R Interval duration is {td['mean_rr']:.3f} ms")
                summary.append(f"pNN50 = {td['pnn50']:.3f} %")
                summary.append(f"RMSSD = {td['rmssd']:.3f} ms")
                summary.append(f"SDNN = {td['sdnn']:.3f} ms")
                summary.append(f"SDSD = {td['sdsd']:.3f} ms")
                summary.append(f"Sample Entropy = {td['sample_entropy']}")
                summary.append(f"SD1 = {td['sd1']:.3f} ms")
                summary.append(f"SD2 = {td['sd2']:.3f} ms")
                summary.append(f"SD1/SD2 = {td['sd1_sd2_ratio']:.3f}")
                summary.append(f"The area of the ellipse fitted over the Poincaré Plot (S) is {td['ellipse_area']:.3f} ms^2")
                summary.append("")
        
        # Blood pressure info
        windowed_data = self.get_windowed_data()
        if len(windowed_data['bp_systolic']) > 0:
            Avg_BP = np.round((np.average(windowed_data['bp_systolic'])),3)
            SD_BP = np.round((np.std(windowed_data['bp_systolic'])),3)
            summary.append("=== BLOOD PRESSURE ===")
            summary.append(f"The average systolic blood pressure during the sampling time is {Avg_BP} + - {SD_BP} mmHg")
            summary.append(f"{len(windowed_data['bp_systolic'])} pressure waves are included in the analysis")
            summary.append("")
        
        # Frequency domain (matching your original print format)
        if 'frequency_domain' in self.results:
            fd = self.results['frequency_domain']
            if 'error' in fd:
                summary.append("=== FREQUENCY DOMAIN ===")
                summary.append(f"ERROR: {fd['error']}")
                summary.append("")
            else:
                summary.append("=== FREQUENCY DOMAIN ===")
                summary.append(f"LF Power: {fd['lf_power']:.2f} ms²")
                summary.append(f"HF Power: {fd['hf_power']:.2f} ms²")
                summary.append(f"Total Power: {fd['total_power']:.2f} ms²")
                summary.append(f"LF/HF Ratio: {fd['lf_hf_ratio']:.2f}")
                summary.append(f"LF Power: {fd['lf_nu']:.2f} n.u.")
                summary.append(f"HF Power: {fd['hf_nu']:.2f} n.u.")
                summary.append("")
        
        # BRS sequence (matching your original print format)
        if 'brs_sequence' in self.results:
            brs = self.results['brs_sequence']
            if 'error' in brs:
                summary.append("=== BRS SEQUENCE METHOD ===")
                summary.append(f"ERROR: {brs['error']}")
                summary.append("")
            else:
                summary.append("=== BRS SEQUENCE METHOD ===")
                summary.append(f"BRS (mean): {brs['BRS_mean']:.2f} ms/mmHg")
                summary.append(f"BEI: {brs['BEI']:.2f}")
                summary.append(f"Valid BRS sequences: {brs['num_sequences']}")
                summary.append(f"Total SAP ramps: {brs['num_sbp_ramps']}")
                summary.append(f"Up BRS sequences: {brs['n_up']}")
                summary.append(f"Down BRS sequences: {brs['n_down']}")
                summary.append(f"Best delay: {brs['best_delay']} beats")
                summary.append("")
        
        # BRS spectral (matching your original print format)
        if 'brs_spectral' in self.results:
            brs_spec = self.results['brs_spectral']
            if 'error' in brs_spec:
                summary.append("=== BRS CROSS-SPECTRAL METHOD ===")
                summary.append(f"ERROR: {brs_spec['error']}")
                summary.append("")
            else:
                summary.append("=== BRS CROSS-SPECTRAL METHOD ===")
                if brs_spec['valid_lf']:
                    summary.append(f"Spectral BRS (Transfer Function, LF): {brs_spec['brs_lf_tf']:.3f} ms/mmHg (LF coherence OK)")
                else:
                    summary.append(f"Low coherence ({brs_spec['lf_coherence']:.3f}) – Spectral BRS not reliable")
                
                if brs_spec['valid_hf']:
                    summary.append(f"Spectral BRS (Transfer Function, HF): {brs_spec['brs_hf_tf']:.3f} ms/mmHg (HF coherence OK)")
                else:
                    summary.append(f"Low HF coherence ({brs_spec['hf_coherence']:.3f}) – HF BRS not reliable")
        
        return "\n".join(summary)