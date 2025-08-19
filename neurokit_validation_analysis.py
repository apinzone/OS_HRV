# neurokit_validation_analysis.py
# NeuroKit2 HRV analysis script for validation against your pipeline

import neurokit2 as nk
import numpy as np
import pandas as pd
import pyedflib
import os
from pathlib import Path

class NeuroKitValidator:
    """
    NeuroKit2-based HRV analysis for validation studies
    Matches the exact metrics from your pipeline
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
                
                print(f"Loaded {filepath}")
                print(f"  Duration: {len(ecg_signal)/self.sampling_rate:.1f} seconds")
                print(f"  Sampling rate: {self.sampling_rate} Hz")
                print(f"  Signal range: {np.min(ecg_signal):.3f} to {np.max(ecg_signal):.3f}")
                
                return ecg_signal, self.sampling_rate
                
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
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
            # Find R-peaks
            _, r_peaks = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)
            
            # Extract RR intervals
            rr_intervals = nk.hrv_time(r_peaks, sampling_rate=sampling_rate, show=False)
            
            # Calculate time domain metrics
            time_domain = nk.hrv_time(r_peaks, sampling_rate=sampling_rate, show=False)
            
            # Calculate frequency domain metrics  
            frequency_domain = nk.hrv_frequency(r_peaks, sampling_rate=sampling_rate, show=False)
            
            # Calculate nonlinear metrics (for SD1, SD2)
            nonlinear = nk.hrv_nonlinear(r_peaks, sampling_rate=sampling_rate, show=False)
            
            # Extract specific metrics to match your pipeline
            results = self._extract_matching_metrics(
                r_peaks, time_domain, frequency_domain, nonlinear, sampling_rate
            )
            
            print("NeuroKit2 analysis completed successfully")
            return results
            
        except Exception as e:
            print(f"NeuroKit2 analysis failed: {e}")
            return None
    
    def _extract_matching_metrics(self, r_peaks, time_domain, frequency_domain, nonlinear, sampling_rate):
        """
        Extract metrics that match your pipeline exactly
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
        
        
        # Frequency domain metrics (NO conversion)
        lf_power = frequency_domain['HRV_LF'].iloc[0] if 'HRV_LF' in frequency_domain.columns else np.nan
        hf_power = frequency_domain['HRV_HF'].iloc[0] if 'HRV_HF' in frequency_domain.columns else np.nan
        lf_hf_ratio = lf_power / hf_power if (hf_power > 0 and not np.isnan(lf_power) and not np.isnan(hf_power)) else np.nan

        # Calculate normalized units (n.u.)
        total_power = lf_power + hf_power if (not np.isnan(lf_power) and not np.isnan(hf_power)) else np.nan
        lf_nu = (lf_power / total_power) if (total_power > 0 and not np.isnan(total_power)) else np.nan
        hf_nu = (hf_power / total_power) if (total_power > 0 and not np.isnan(total_power)) else np.nan

        # DEBUG: Export raw data for comparison
        self._export_debug_data(r_peaks['ECG_R_Peaks'], rr_times, rr_intervals_ms, sampling_rate)
        
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
            
            # Frequency domain
            'lf_power_ms2': lf_power,
            'hf_power_ms2': hf_power,
            'lf_hf_ratio': lf_hf_ratio,
            'lf_nu': lf_nu,           # ADD THIS
            'hf_nu': hf_nu,           # ADD THIS
            'total_power': total_power, # ADD THIS
        
            # Raw data for verification
            'r_peaks_indices': r_peaks['ECG_R_Peaks'].tolist(),
            'rr_intervals_ms': rr_intervals_ms.tolist() if len(rr_intervals_ms) > 0 else []
        }
        
        return results
    
    def _export_debug_data(self, r_peaks_indices, rr_times, rr_intervals_ms, sampling_rate):
        """
        Export debug data for comparison with other implementations
        """
        output_file = "debug_neurokit2.txt"
        with open(output_file, 'w') as f:
            f.write("=== NEUROKIT2 DEBUG DATA ===\n")
            f.write(f"Total R-peaks detected: {len(r_peaks_indices)}\n")
            f.write(f"RR intervals count: {len(rr_intervals_ms)}\n")
            f.write(f"Sampling rate: {sampling_rate} Hz\n\n")
            
            f.write("First 20 R-peak times (seconds):\n")
            for i, time_sec in enumerate(rr_times[:20]):
                f.write(f"  Peak {i+1}: {time_sec:.6f}s\n")
            
            f.write("\nFirst 20 RR intervals (ms):\n")
            for i, rr_ms in enumerate(rr_intervals_ms[:20]):
                f.write(f"  RR {i+1}: {rr_ms:.3f}ms\n")
            
            f.write("\nAll RR intervals (ms):\n")
            f.write(str(rr_intervals_ms.tolist()))
        
        print(f"NeuroKit2 debug data exported to {output_file}")
        
        # Also print first few for immediate comparison
        print("\n=== NEUROKIT2 - FIRST 10 PEAKS ===")
        for i in range(min(10, len(rr_times))):
            print(f"Peak {i+1}: {rr_times[i]:.6f}s")
        
        print("\n=== NEUROKIT2 - FIRST 10 RR INTERVALS ===")
        for i in range(min(10, len(rr_intervals_ms))):
            print(f"RR {i+1}: {rr_intervals_ms[i]:.3f}ms")
        
        # Add this line after the first 10 RR intervals
        print(f"\n=== COMPLETE RR INTERVALS ===")
        print("ANALYSIS DEBUG: ALL RR INTERVALS:", [float(rr) for rr in rr_intervals_ms])
    
    def analyze_single_file(self, filepath):
        """
        Analyze a single EDF file and return results
        
        Parameters:
        filepath: Path to EDF file
        
        Returns:
        results: Dictionary with HRV metrics
        """
        print(f"\n{'='*50}")
        print(f"NEUROKIT2 ANALYSIS: {os.path.basename(filepath)}")
        print(f"{'='*50}")
        
        # Load EDF file
        ecg_signal, sampling_rate = self.load_edf_file(filepath)
        
        if ecg_signal is None:
            return None
        
        # Analyze with NeuroKit2
        results = self.analyze_hrv_with_neurokit(ecg_signal, sampling_rate)
        
        if results is None:
            return None
        
        # Print results in organized format
        self._print_results(results, filepath)
        
        return results
    
    def _print_results(self, results, filepath):
        """
        Print results in organized format for easy manual recording
        """
        print(f"\nRESULTS FOR: {os.path.basename(filepath)}")
        print("-" * 50)
        
        print("BASIC METRICS:")
        print(f"  Number of beats: {results['num_beats']}")
        print(f"  Average HR (BPM): {results['avg_hr_bpm']:.2f}")
        print(f"  Mean RR (ms): {results['mean_rr_ms']:.3f}")
        print(f"  Total time (sec): {results['total_time_sec']:.3f}")
        
        print("\nTIME DOMAIN:")
        print(f"  RMSSD (ms): {results['rmssd_ms']:.3f}")
        print(f"  SDNN (ms): {results['sdnn_ms']:.3f}")
        print(f"  pNN50 (%): {results['pnn50_percent']:.3f}")
        
        print("\nNONLINEAR (POINCARÉ):")
        print(f"  SD1 (ms): {results['sd1_ms']:.3f}")
        print(f"  SD2 (ms): {results['sd2_ms']:.3f}")
        print(f"  SD1/SD2: {results['sd1_sd2_ratio']:.3f}")
        
        print("\nFREQUENCY DOMAIN:")
        print(f"  LF Power (ms²): {results['lf_power_ms2']:.3f}")
        print(f"  HF Power (ms²): {results['hf_power_ms2']:.3f}")
        print(f"  LF/HF Ratio: {results['lf_hf_ratio']:.3f}")
        print(f"  LF (n.u.): {results['lf_nu']:.3f}")      # ADD THIS
        print(f"  HF (n.u.): {results['hf_nu']:.3f}")      # ADD THIS
        print("-" * 50)
    
    def analyze_validation_dataset(self, dataset_dir, output_file=None):
        """
        Analyze entire validation dataset
        
        Parameters:
        dataset_dir: Directory containing synthetic EDF files
        output_file: Optional CSV file to save results
        
        Returns:
        all_results: List of results dictionaries
        """
        dataset_path = Path(dataset_dir)
        edf_files = list(dataset_path.glob("*.edf"))
        
        print(f"Found {len(edf_files)} EDF files in {dataset_dir}")
        
        all_results = []
        
        for i, edf_file in enumerate(edf_files, 1):
            print(f"\nProcessing file {i}/{len(edf_files)}: {edf_file.name}")
            
            results = self.analyze_single_file(str(edf_file))
            
            if results is not None:
                results['filename'] = edf_file.name
                results['file_path'] = str(edf_file)
                all_results.append(results)
            else:
                print(f"Failed to analyze {edf_file.name}")
        
        # Save to CSV if requested
        if output_file and all_results:
            df = pd.DataFrame(all_results)
            df.to_csv(output_file, index=False)
            print(f"\nResults saved to {output_file}")
        
        print(f"\nCompleted analysis of {len(all_results)}/{len(edf_files)} files")
        
        return all_results

def main():
    """
    Main function for validation analysis
    """
    validator = NeuroKitValidator()
    
    # Test single file first
    test_file = input("Enter path to test EDF file (or press Enter to skip): ")
    
    if test_file.strip():
        print("Testing single file analysis...")
        results = validator.analyze_single_file(test_file.strip())
        
        if results:
            print("\n✅ Single file analysis successful!")
        else:
            print("\n❌ Single file analysis failed!")
            return
    
    # Analyze full dataset
    dataset_dir = input("Enter directory containing validation EDF files: ")
    
    if os.path.exists(dataset_dir):
        output_file = input("Enter output CSV filename (optional, press Enter to skip): ")
        output_file = output_file.strip() if output_file.strip() else None
        
        print(f"\nAnalyzing validation dataset in {dataset_dir}...")
        all_results = validator.analyze_validation_dataset(dataset_dir, output_file)
        
        print(f"\n✅ Dataset analysis complete! Processed {len(all_results)} files.")
        
    else:
        print(f"Directory {dataset_dir} not found!")

if __name__ == "__main__":
    main()