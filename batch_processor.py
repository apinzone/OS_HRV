# batch_processor.py
# Combined HRV validation processor for ChronOS and NeuroKit2
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import pyedflib
import neurokit2 as nk

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from analyzer import CardiovascularAnalyzer

def calculate_adaptive_ecg_params(ecg_signal, sample_rate):
    """
    Calculate adaptive ECG parameters for each signal mirroring ChronOS dataflow
    """
    #Match Exact ChronOS ECG Peak detection
    ecg_baseline = np.median(ecg_signal)
    ecg_max = np.max(ecg_signal)
    signal_range = ecg_max - ecg_baseline
    
    ecg_height_default = 0.55 * signal_range
    ecg_prominence_default = 0.6 * ecg_height_default
    ecg_distance_default = int(0.25 * sample_rate)
    
    # Blood pressure defaults (if needed)
    bp_height = 110
    bp_distance = 100
    bp_prominence = 5
    
    return {
        'ecg_height': ecg_height_default,
        'ecg_prominence': ecg_prominence_default,
        'ecg_distance': ecg_distance_default,
        'bp_height': bp_height,
        'bp_distance': bp_distance,
        'bp_prominence': bp_prominence
    }

def load_edf_file(filepath):
    """Load EDF file and extract ECG signal"""
    try:
        with pyedflib.EdfReader(filepath) as edf_file:
            sampling_rate = int(edf_file.getSampleFrequency(0))
            ecg_signal = edf_file.readSignal(0)
            
            print(f"  Loaded: {len(ecg_signal)} samples at {sampling_rate} Hz ({len(ecg_signal)/sampling_rate:.1f}s)")
            return ecg_signal, sampling_rate
            
    except Exception as e:
        print(f"  ERROR loading: {e}")
        return None, None

def analyze_with_chronos(file_path, edf_file):
    """Analyze using ChronOS pipeline"""
    try:
        # Initialize analyzer
        analyzer = CardiovascularAnalyzer()
        
        # Load file and detect channels - just like original script
        channels_info = analyzer.load_file_and_detect_channels(file_path)
        
        if not channels_info:
            return None
        
        # Configure ECG channel (assume channel 0 for single-channel files)
        ecg_channel_idx = 0
        success_msgs = analyzer.configure_channels(ecg_channel_idx, None)
        
        if not success_msgs or not any('ECG' in str(msg) for msg in success_msgs):
            return None
        
        # Get ECG signal for adaptive parameter calculation
        ecg_signal = analyzer.ecg_data['raw']
        sample_rate = analyzer.ecg_data['fs']
        
        # Adaptive parameters from gui 
        adaptive_params = calculate_adaptive_ecg_params(ecg_signal, sample_rate)

        # Run peak detection 
        analyzer.find_peaks_with_params(
            ecg_height=adaptive_params['ecg_height'],
            ecg_prominence=adaptive_params['ecg_prominence'],
            ecg_distance=adaptive_params['ecg_distance'],
            bp_height=adaptive_params['bp_height'],
            bp_distance=adaptive_params['bp_distance'],
            bp_prominence=adaptive_params['bp_prominence'],
        )
        
        # Check if peaks were detected
        if 'td_peaks' not in analyzer.ecg_data or len(analyzer.ecg_data['td_peaks']) == 0:
            return None
        
        peak_count = len(analyzer.ecg_data['td_peaks'])

        # Extract R-peak timestamps and R-R intervals from ChronOS
        r_peak_indices = analyzer.ecg_data['td_peaks']
        r_peak_timestamps = r_peak_indices / sample_rate  # Convert to seconds
        rr_intervals_ms = np.diff(r_peak_timestamps) * 1000  # Convert to milliseconds

        # Calculate HRV metrics 
        analyzer.calculate_time_domain()
        analyzer.calculate_frequency_domain()
        
        # Extract results from analyzer.results structure
        if hasattr(analyzer, 'results') and analyzer.results:
            results_data = analyzer.results
            hrv_data = {}
            
            # Time domain results
            if 'time_domain' in results_data and 'error' not in results_data['time_domain']:
                td = results_data['time_domain']
                hrv_data.update({
                    'mean_rr': td.get('mean_rr', np.nan),
                    'rmssd': td.get('rmssd', np.nan),
                    'sdnn': td.get('sdnn', np.nan),
                    'sdsd': td.get('sdsd', np.nan),
                    'pnn50': td.get('pnn50', np.nan),
                    'sd1': td.get('sd1', np.nan),
                    'sd2': td.get('sd2', np.nan),
                    'sd1_sd2_ratio': td.get('sd1_sd2_ratio', np.nan),
                    'sample_entropy': td.get('sample_entropy', np.nan)
                })

            # Frequency domain results  
            if 'frequency_domain' in results_data and 'error' not in results_data['frequency_domain']:
                fd = results_data['frequency_domain']
                hrv_data.update({
                    'vlf_power': fd.get('vlf_power', np.nan),
                    'lf_power': fd.get('lf_power', np.nan),
                    'hf_power': fd.get('hf_power', np.nan),
                    'total_power': fd.get('total_power', np.nan),
                    'lf_hf_ratio': fd.get('lf_hf_ratio', np.nan)
                })
            
            # Create result with EXACT validation metrics only
            if hrv_data:
                result = {
                    'filename': edf_file,
                    'num_peaks': peak_count,
                    'mean_rr_ms': hrv_data.get('mean_rr', np.nan),
                    'rmssd_ms': hrv_data.get('rmssd', np.nan),
                    'sdnn_ms': hrv_data.get('sdnn', np.nan),
                    'sd1_ms': hrv_data.get('sd1', np.nan),
                    'sd2_ms': hrv_data.get('sd2', np.nan),
                    'sd1_sd2_ratio': hrv_data.get('sd1_sd2_ratio', np.nan),
                    'sample_entropy': hrv_data.get('sample_entropy', np.nan),
                    'vlf_power_ms2': hrv_data.get('vlf_power', np.nan),
                    'lf_power_ms2': hrv_data.get('lf_power', np.nan),
                    'hf_power_ms2': hrv_data.get('hf_power', np.nan),
                    'total_power_ms2': hrv_data.get('total_power', np.nan),
                    'lf_hf_ratio': hrv_data.get('lf_hf_ratio', np.nan),
                    'r_peak_timestamps_sec': r_peak_timestamps.tolist(),
                    'rr_intervals_ms': rr_intervals_ms.tolist(),
                    'status': 'success'
                }
                return result
            
        return None
        
    except Exception as e:
        print(f"  ChronOS analysis error: {e}")
        return None

def analyze_with_neurokit(ecg_signal, sampling_rate, edf_file):
    """Analyze using NeuroKit2 with 4Hz interpolation"""
    try:
        # Find R-peaks using NeuroKit2
        _, r_peaks = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)

        if len(r_peaks['ECG_R_Peaks']) < 5:
            print(f"  WARNING: Only {len(r_peaks['ECG_R_Peaks'])} R-peaks detected")
            return None
        
        # Extract R-peak timestamps and R-R intervals from NeuroKit2
        r_peak_indices = r_peaks['ECG_R_Peaks']
        r_peak_timestamps = r_peak_indices / sampling_rate  # Convert to seconds
        rr_intervals_ms = np.diff(r_peak_timestamps) * 1000  # Convert to milliseconds

        time_domain = nk.hrv_time(r_peaks, sampling_rate=sampling_rate, show=False)
        nonlinear = nk.hrv_nonlinear(r_peaks, sampling_rate=sampling_rate, show=False)
        frequency_domain = nk.hrv_frequency(
            r_peaks, 
            sampling_rate=sampling_rate,
            interpolation_rate=4,  #4 hz instead of 100hz default
            normalize=False,       #ms² not normalized
            show=False
        )
        
        # Calculate basic metrics
        rr_times = r_peaks['ECG_R_Peaks'] / sampling_rate
        rr_intervals_ms = np.diff(rr_times) * 1000
        num_beats = len(r_peaks['ECG_R_Peaks'])
        total_time = rr_times[-1] - rr_times[0] if len(rr_times) > 1 else 0
        avg_hr = num_beats / (total_time / 60)
        mean_rr_ms = np.mean(rr_intervals_ms) if len(rr_intervals_ms) > 0 else 0
        
        # Time domain metrics from NeuroKit2
        rmssd = time_domain['HRV_RMSSD'].iloc[0] if 'HRV_RMSSD' in time_domain.columns else np.nan
        sdnn = time_domain['HRV_SDNN'].iloc[0] if 'HRV_SDNN' in time_domain.columns else np.nan
        pnn50 = time_domain['HRV_pNN50'].iloc[0] if 'HRV_pNN50' in time_domain.columns else np.nan
        
        # Nonlinear metrics from NeuroKit2 
        sd1 = nonlinear['HRV_SD1'].iloc[0] if 'HRV_SD1' in nonlinear.columns else np.nan
        sd2 = nonlinear['HRV_SD2'].iloc[0] if 'HRV_SD2' in nonlinear.columns else np.nan
        sd1_sd2_ratio = sd1 / sd2 if (sd2 > 0 and not np.isnan(sd1) and not np.isnan(sd2)) else np.nan
        sample_entropy = nonlinear['HRV_SampEn'].iloc[0] if 'HRV_SampEn' in nonlinear.columns else np.nan
        
        # Frequency domain metrics from NeuroKit2 
        vlf_power = frequency_domain['HRV_VLF'].iloc[0] if 'HRV_VLF' in frequency_domain.columns else np.nan
        lf_power = frequency_domain['HRV_LF'].iloc[0] if 'HRV_LF' in frequency_domain.columns else np.nan
        hf_power = frequency_domain['HRV_HF'].iloc[0] if 'HRV_HF' in frequency_domain.columns else np.nan
        lf_hf_ratio = frequency_domain['HRV_LFHF'].iloc[0] if 'HRV_LFHF' in frequency_domain.columns else np.nan
        lf_nu = frequency_domain['HRV_LFn'].iloc[0] if 'HRV_LFn' in frequency_domain.columns else np.nan
        hf_nu = frequency_domain['HRV_HFn'].iloc[0] if 'HRV_HFn' in frequency_domain.columns else np.nan
        total_power = frequency_domain['HRV_TP'].iloc[0] if 'HRV_TP' in frequency_domain.columns else np.nan
        
        # Compile results with EXACT validation metrics only
        result = {
            'filename': edf_file,
            'num_peaks': num_beats,
            'mean_rr_ms': mean_rr_ms,
            'rmssd_ms': rmssd,
            'sdnn_ms': sdnn,
            'sd1_ms': sd1,
            'sd2_ms': sd2,
            'sd1_sd2_ratio': sd1_sd2_ratio,
            'sample_entropy': sample_entropy,
            'vlf_power_ms2': vlf_power,
            'lf_power_ms2': lf_power,
            'hf_power_ms2': hf_power,
            'total_power_ms2': total_power,
            'lf_hf_ratio': lf_hf_ratio,
            'r_peak_timestamps_sec': r_peak_timestamps.tolist(),
            'rr_intervals_ms': rr_intervals_ms.tolist(),
            'status': 'success'
        }
        
        return result
        
    except Exception as e:
        print(f"  NeuroKit2 analysis error: {e}")
        return None

def batch_process_validation():
    """
    Batch process all EDF files using both ChronOS and NeuroKit2
    """
    
    # Validation directory
    edf_directory = r"C:\Users\Anthony\Desktop\peak_detector\data\validation_synthetic_ecg"
    output_file = "validation_results.xlsx"
    
    print(f"Processing directory: {edf_directory}")
    print(f"Output file: {output_file}")
    print()
    
    # Check if directory exists
    if not os.path.exists(edf_directory):
        return None
    
    # Find all EDF files
    edf_files = [f for f in os.listdir(edf_directory) if f.endswith('.edf')]
    edf_files.sort()
    
    print(f"Found {len(edf_files)} EDF files to process")
    
    if len(edf_files) == 0:
        print("No EDF files found in directory!")
        return None
    
    print(f"First few files: {edf_files[:5]}")
    print()
    
    chronos_results = []
    neurokit_results = []
    
    for i, edf_file in enumerate(edf_files, 1):
        print(f"[{i:3d}/{len(edf_files)}] Processing: {edf_file}")
        
        file_path = os.path.join(edf_directory, edf_file)
        
        # Load EDF file
        ecg_signal, sampling_rate = load_edf_file(file_path)
        
        if ecg_signal is None:
            print(f"  FAILED: Could not load file")
            # Add failed entries to maintain row alignment
            chronos_results.append({'filename': edf_file, 'status': 'load_failed'})
            neurokit_results.append({'filename': edf_file, 'status': 'load_failed'})
            continue
        
        # Analyze with ChronOS
        print(f"  Running ChronOS analysis...")
        chronos_result = analyze_with_chronos(file_path, edf_file)
        if chronos_result is None:
            chronos_result = {'filename': edf_file, 'status': 'analysis_failed'}
        chronos_results.append(chronos_result)
        
        # Analyze with NeuroKit2
        print(f"  Running NeuroKit2 analysis...")
        neurokit_result = analyze_with_neurokit(ecg_signal, sampling_rate, edf_file)
        if neurokit_result is None:
            neurokit_result = {'filename': edf_file, 'status': 'analysis_failed'}
        neurokit_results.append(neurokit_result)
        
        print()
    
    # Save results to Excel with separate tabs
    if chronos_results and neurokit_results:
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # ChronOS results
                chronos_df = pd.DataFrame(chronos_results)
                chronos_df.to_excel(writer, sheet_name='ChronOS_Results', index=False)
                
                # NeuroKit2 results  
                neurokit_df = pd.DataFrame(neurokit_results)
                neurokit_df.to_excel(writer, sheet_name='NeuroKit2_Results', index=False)
        except ImportError:
            print("Warning: openpyxl not installed. Saving as separate CSV files instead.")
            # Fallback to CSV files
            chronos_df = pd.DataFrame(chronos_results)
            neurokit_df = pd.DataFrame(neurokit_results)
            chronos_df.to_csv('ChronOS_validation_results.csv', index=False)
            neurokit_df.to_csv('NeuroKit2_validation_results.csv', index=False)
            print(f"Results saved to: ChronOS_validation_results.csv and NeuroKit2_validation_results.csv")
        
        print(f"\nResults saved to: {output_file}")
        print(f"ChronOS results: {len(chronos_results)} files")
        print(f"NeuroKit2 results: {len(neurokit_results)} files")
        
        # Summary statistics
        chronos_success = chronos_df[chronos_df['status'] == 'success']
        neurokit_success = neurokit_df[neurokit_df['status'] == 'success']
        
        print(f"\nSUCCESSFUL ANALYSES:")
        print(f"ChronOS: {len(chronos_success)}/{len(chronos_results)} files")
        print(f"NeuroKit2: {len(neurokit_success)}/{len(neurokit_results)} files")
        
        return {'chronos': chronos_df, 'neurokit': neurokit_df}
    
    return None

def main():
    """
    Main function - processes validation synthetic ECG directory with both methods
    """
    print("Starting Combined Validation Processing...")
    print("Running both ChronOS and NeuroKit2 analysis on identical datasets")
    print()
    
    results = batch_process_validation()
    
    if results is not None:
        print(f"\nCombined batch processing complete!")
        print(f"Results saved to: validation_results.xlsx")
        print(f"Ready for statistical validation analysis in R!")
    else:
        print(f"\nBatch processing failed!")

if __name__ == "__main__":
    main()