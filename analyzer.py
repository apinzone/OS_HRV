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

    # Add this method to your CardiovascularAnalyzer class
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

    def analyze_with_current_peaks(self):
        """Run analysis with current peak detection results"""
        self.calculate_time_domain()
        self.calculate_frequency_domain()
        self.calculate_brs_sequence()
        self.calculate_brs_spectral()
        
    def calculate_time_domain(self):
        """Exactly matches your original time domain calculations"""
        RRDistance_ms = self.ecg_data['rr_intervals']
        td_peaks = self.ecg_data['td_peaks']
        
        # Your exact calculations
        Successive_time_diff = SuccessiveDiff(RRDistance_ms)  # Your function
        AvgDiff = np.average(Successive_time_diff)
        SDNN = np.std(RRDistance_ms)
        SDSD = np.std(Successive_time_diff)
        NN50 = NNCounter(Successive_time_diff, 50)  # Your function
        pNN50 = (NN50/len(td_peaks))*100
        RMSSD = np.sqrt(np.average(rms(Successive_time_diff)))  # Your function
        SD1 = np.sqrt(0.5*math.pow(SDSD,2))
        SD2 = np.sqrt((2*math.pow(SDNN,2) - (0.5*math.pow(SDSD,2))))
        S = math.pi * SD1 * SD2
        Sampling_Time = max(td_peaks)
        Num_Beats = len(RRDistance_ms)
        HR = np.round(Num_Beats/(Sampling_Time/60),2)
        
        # Sample entropy with your exact parameters
        m = 2
        r = 0.2 * np.std(RRDistance_ms)
        sampen = SampEn(RRDistance_ms, m, r)  # Your function
        
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
            'sd1_sd2_ratio': SD1/SD2,
            'ellipse_area': S,
            'sample_entropy': sampen
        }
        
    def calculate_frequency_domain(self):
        """Exactly matches your original frequency domain code"""
        td_peaks = self.ecg_data['td_peaks']
        RRDistance_ms = self.ecg_data['rr_intervals']
        
        # Your exact interpolation
        interp_fs = 4
        uniform_time = np.arange(0, max(td_peaks), 1/interp_fs)
        rr_interp_func = interp1d(td_peaks[:-1], RRDistance_ms, kind='cubic', fill_value="extrapolate")
        rr_fft = rr_interp_func(uniform_time)
        
        frequencies, psd = welch(rr_fft, fs=interp_fs, nperseg=256)
        
        # Your exact band definitions
        lf_band = (frequencies >= 0.04) & (frequencies < 0.15)
        hf_band = (frequencies >= 0.15) & (frequencies < 0.4)
        vlf_band = (frequencies >= 0.003) & (frequencies < 0.04)
        
        # Your exact power calculations (no unit conversion here to match original)
        vlf_power = np.trapezoid(psd[vlf_band], frequencies[vlf_band])
        lf_power = np.trapezoid(psd[lf_band], frequencies[lf_band])
        hf_power = np.trapezoid(psd[hf_band], frequencies[hf_band])
        lf_hf_ratio = lf_power / hf_power
        total_power = lf_power + hf_power
        
        # Normalized units
        lf_nu = (lf_power/total_power)
        hf_nu = (hf_power/total_power)
        
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
        """Exactly matches your original sequence BRS"""
        Systolic_Array = self.bp_data['systolic']
        RRDistance_ms = self.ecg_data['rr_intervals']
        
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
        """Exactly matches your original cross-spectral BRS"""
        td_BP_peaks = self.bp_data['td_peaks']
        Systolic_Array = self.bp_data['systolic']
        
        # Your exact interpolation for BP
        interp_fs = 4
        uniform_time_bp = np.arange(0, max(td_BP_peaks), 1/interp_fs)
        bp_interp_func = interp1d(td_BP_peaks, Systolic_Array, kind='cubic', fill_value="extrapolate")
        bp_fft = bp_interp_func(uniform_time_bp)
        frequencies_bp, psd_bp = welch(bp_fft, fs=interp_fs, nperseg=256)
        
        # Get RR data from frequency domain results
        rr_fft = self.results['frequency_domain']['rr_fft']
        
        # Your exact band definitions
        lf_band = (frequencies_bp >= 0.04) & (frequencies_bp < 0.15)
        hf_band = (frequencies_bp >= 0.15) & (frequencies_bp < 0.4)
        
        # Your exact power calculations (no unit conversion to match original)
        lf_power_BP = np.trapezoid(psd_bp[lf_band], frequencies_bp[lf_band])
        hf_power_BP = np.trapezoid(psd_bp[hf_band], frequencies_bp[hf_band])
        
        # Your exact coherence calculation
        frequencies_coh, coherence_values = coherence(rr_fft, bp_fft, fs=interp_fs, nperseg=256)
        lf_coherence = np.mean(coherence_values[(frequencies_coh >= 0.04) & (frequencies_coh < 0.15)])
        hf_coherence = np.mean(coherence_values[(frequencies_coh >= 0.15) & (frequencies_coh < 0.4)])
        
        # Your exact cross-spectral calculation
        frequencies_csd, csd_bp_rr = csd(bp_fft, rr_fft, fs=interp_fs, nperseg=256)
        _, psd_bp_auto = welch(bp_fft, fs=interp_fs, nperseg=256)
        transfer_gain = np.abs(csd_bp_rr)/psd_bp_auto
        
        # Your exact band definitions for transfer function
        lf_band_tf = (frequencies_csd >= 0.04) & (frequencies_csd < 0.15)
        hf_band_tf = (frequencies_csd >= 0.15) & (frequencies_csd < 0.4)
        brs_lf_tf = np.mean(transfer_gain[lf_band_tf])
        brs_hf_tf = np.mean(transfer_gain[hf_band_tf])
        
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
        
    def analyze_all(self):
        """Run complete analysis exactly like your main.py"""
        self.find_peaks()
        self.calculate_time_domain()
        self.calculate_frequency_domain()
        self.calculate_brs_sequence()
        self.calculate_brs_spectral()
        
    def get_summary(self):
        """Enhanced summary including all your original metrics"""
        if not self.results:
            return "No analysis completed yet"
            
        summary = []
        
        # Time domain (exactly matching your original print statements)
        if 'time_domain' in self.results:
            td = self.results['time_domain']
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
        if hasattr(self, 'bp_data') and 'systolic' in self.bp_data:
            Avg_BP = np.round((np.average(self.bp_data['systolic'])),3)
            SD_BP = np.round((np.std(self.bp_data['systolic'])),3)
            summary.append("=== BLOOD PRESSURE ===")
            summary.append(f"The average systolic blood pressure during the sampling time is {Avg_BP} + - {SD_BP} mmHg")
            summary.append(f"{len(self.bp_data['systolic'])} pressure waves are included in the analysis")
            summary.append("")
        
        # Frequency domain (matching your original print format)
        if 'frequency_domain' in self.results:
            fd = self.results['frequency_domain']
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