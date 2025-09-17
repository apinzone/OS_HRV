# neurokit_validation_analysis.py
# Streamlined NeuroKit2 HRV analysis for validation studies

import neurokit2 as nk
import numpy as np
import pandas as pd
import pyedflib
import os
from pathlib import Path

class NeuroKitValidator:
    """
    NeuroKit2-based HRV analysis for validation studies
    """
    
    def __init__(self):
        self.sampling_rate = 256  # Will be updated when loading files
        
    def load_edf_file(self, filepath):
        """
        Load EDF file and extract ECG signal
        
        Parameters:
        filepath: Path to EDF file
        
        Returns:
        ecg_signal: ECG signal array
        sampling_rate: Sampling frequency
        """
        try:
            with pyedflib.EdfReader(filepath) as edf_file:
                # Get sampling rate
                self.sampling_rate = int(edf_file.getSampleFrequency(0))
                
                # Read ECG signal (assuming first channel is ECG)
                ecg_signal = edf_file.readSignal(0)
                
                return ecg_signal, self.sampling_rate
                
        except Exception as e:
            return None, None
    
    def analyze_hrv_with_neurokit(self, ecg_signal, sampling_rate):
        """
        Perform HRV analysis using NeuroKit2
        
        Parameters:
        ecg_signal: ECG signal array
        sampling_rate: Sampling frequency
        
        Returns:
        results: Dictionary with all HRV metrics
        """
        try:
            # Clean ECG signal
            ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
            
            # Find R-peaks
            _, r_peaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
            
            # Calculate time domain metrics
            time_domain = nk.hrv_time(r_peaks, sampling_rate=sampling_rate, show=False)
            
            # Calculate frequency domain metrics  
            frequency_domain = nk.hrv_frequency(r_peaks, sampling_rate=sampling_rate, show=False)
            
            # Calculate nonlinear metrics (for SD1, SD2)
            nonlinear = nk.hrv_nonlinear(r_peaks, sampling_rate=sampling_rate, show=False)
            
            # Extract specific metrics 
            results = self._extract_matching_metrics(
                r_peaks, time_domain, frequency_domain, nonlinear, sampling_rate
            )
            
            return results
            
        except Exception as e:
            return None
    
    def _extract_matching_metrics(self, r_peaks, time_domain, frequency_domain, nonlinear, sampling_rate):
        """
        Extract and calculate HRV metrics from NeuroKit2 results
        """
        # Calculate RR intervals manually for some metrics
        rr_times = r_peaks['ECG_R_Peaks'] / sampling_rate
        rr_intervals_sec = np.diff(rr_times)
        rr_intervals_ms = rr_intervals_sec * 1000
        
        # Basic counts and timing
        num_beats = len(r_peaks['ECG_R_Peaks'])
        total_time = rr_times[-1] - rr_times[0] if len(rr_times) > 1 else 0
        
        # Heart rate
        avg_hr = (num_beats - 1) / (total_time / 60) if total_time > 0 else 0
        mean_rr_ms = np.mean(rr_intervals_ms) if len(rr_intervals_ms) > 0 else 0
        
        # Time domain metrics
        rmssd = time_domain['HRV_RMSSD'].iloc[0] if 'HRV_RMSSD' in time_domain.columns else np.nan
        sdnn = time_domain['HRV_SDNN'].iloc[0] if 'HRV_SDNN' in time_domain.columns else np.nan
        pnn50 = time_domain['HRV_pNN50'].iloc[0] if 'HRV_pNN50' in time_domain.columns else np.nan
        
        # Nonlinear metrics (Poincaré)
        sd1 = nonlinear['HRV_SD1'].iloc[0] if 'HRV_SD1' in nonlinear.columns else np.nan
        sd2 = nonlinear['HRV_SD2'].iloc[0] if 'HRV_SD2' in nonlinear.columns else np.nan
        sd1_sd2_ratio = sd1 / sd2 if (sd2 > 0 and not np.isnan(sd1) and not np.isnan(sd2)) else np.nan
        
        # Frequency domain metrics - including VLF and TP
        vlf_power = frequency_domain['HRV_VLF'].iloc[0] if 'HRV_VLF' in frequency_domain.columns else np.nan
        lf_power = frequency_domain['HRV_LF'].iloc[0] if 'HRV_LF' in frequency_domain.columns else np.nan
        hf_power = frequency_domain['HRV_HF'].iloc[0] if 'HRV_HF' in frequency_domain.columns else np.nan
        total_power = frequency_domain['HRV_TP'].iloc[0] if 'HRV_TP' in frequency_domain.columns else np.nan
        
        # Calculate ratios
        lf_hf_ratio = lf_power / hf_power if (hf_power > 0 and not np.isnan(lf_power) and not np.isnan(hf_power)) else np.nan

        # Calculate normalized units (n.u.) - use LF+HF only for n.u. calculation (standard practice)
        lf_hf_total = lf_power + hf_power if (not np.isnan(lf_power) and not np.isnan(hf_power)) else np.nan
        lf_nu = (lf_power / lf_hf_total) * 100 if (lf_hf_total > 0 and not np.isnan(lf_hf_total)) else np.nan
        hf_nu = (hf_power / lf_hf_total) * 100 if (lf_hf_total > 0 and not np.isnan(lf_hf_total)) else np.nan
        
        # Compile results
        results = {
            # Basic metrics
            'num_beats': num_beats,
            'avg_hr_bpm': avg_hr,
            'mean_rr_ms': mean_rr_ms,
            'total_time_sec': total_time,
            
            # Time domain
            'rmssd_ms': rmssd,
            'sdnn_ms': sdnn,
            'pnn50_percent': pnn50,
            
            # Nonlinear (Poincaré)
            'sd1_ms': sd1,
            'sd2_ms': sd2,
            'sd1_sd2_ratio': sd1_sd2_ratio,
            
            # Frequency domain - complete spectrum
            'vlf_power_ms2': vlf_power,
            'lf_power_ms2': lf_power,
            'hf_power_ms2': hf_power,
            'total_power_ms2': total_power,
            'lf_hf_ratio': lf_hf_ratio,
            'lf_nu': lf_nu,
            'hf_nu': hf_nu,
        
            # Raw data for verification
            'r_peaks_indices': r_peaks['ECG_R_Peaks'].tolist(),
            'rr_intervals_ms': rr_intervals_ms.tolist() if len(rr_intervals_ms) > 0 else []
        }
        
        return results
    
    def analyze_single_file(self, filepath):
        """
        Analyze a single EDF file and return results
        
        Parameters:
        filepath: Path to EDF file
        
        Returns:
        results: Dictionary with HRV metrics
        """
        # Load EDF file
        ecg_signal, sampling_rate = self.load_edf_file(filepath)
        
        if ecg_signal is None:
            return None
        
        # Analyze with NeuroKit2
        results = self.analyze_hrv_with_neurokit(ecg_signal, sampling_rate)
        
        if results is None:
            return None
        
        return results
    
    def batch_process_directory(self, directory_path):
        """
        Batch process all EDF files in directory and save results to CSV
        
        Parameters:
        directory_path: Path to directory containing EDF files
        
        Returns:
        results_df: DataFrame with all results
        """
        print(f"Processing directory: {directory_path}")
        
        # Find all EDF files
        edf_files = [f for f in os.listdir(directory_path) if f.endswith('.edf')]
        edf_files.sort()
        
        print(f"Found {len(edf_files)} EDF files")
        
        if len(edf_files) == 0:
            return None
        
        results = []
        
        for i, edf_file in enumerate(edf_files, 1):
            print(f"[{i:3d}/{len(edf_files)}] Processing: {edf_file}")
            
            file_path = os.path.join(directory_path, edf_file)
            result = self.analyze_single_file(file_path)
            
            if result is not None:
                result['filename'] = edf_file
                result['file_path'] = file_path
                results.append(result)
        
        # Save to CSV
        if results:
            df = pd.DataFrame(results)
            df.to_csv("neurokit_hrv_batch_results.csv", index=False)
            print(f"\nResults saved to: neurokit_hrv_batch_results.csv")
            return df
        else:
            return None

def main():
    """
    Main function for validation analysis
    """
    validator = NeuroKitValidator()
    
    # Set directory path
    dataset_dir = r"C:\Users\Anthony\Desktop\peak_detector\data\validation_synthetic_ecg"
    
    if os.path.exists(dataset_dir):
        results_df = validator.batch_process_directory(dataset_dir)
        
        if results_df is not None:
            print(f"Processing complete!")
        else:
            print(f"Processing failed!")
    else:
        print(f"Directory {dataset_dir} not found!")

if __name__ == "__main__":
    main()