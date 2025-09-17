# batch_hrv_your_pipeline.py
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from analyzer import CardiovascularAnalyzer

def calculate_adaptive_ecg_params(ecg_signal, sample_rate):
    """
    Calculate adaptive ECG parameters exactly like your GUI
    This replicates the logic from your simple_gui.py calculate_adaptive_ecg_params function
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

def batch_process_hrv_metrics():
    """
    Batch process all EDF files to extract HRV metrics using ChronOS
    """
    
    #Validation directory
    edf_directory = r"C:\Users\Anthony\Desktop\peak_detector\data\validation_synthetic_ecg"
    output_file = "ChronOS_results.csv"
    
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
    
    results = []
    successful = 0
    failed = 0
    
    for i, edf_file in enumerate(edf_files, 1):
        print(f"[{i:3d}/{len(edf_files)}] Processing: {edf_file}")
        
        file_path = os.path.join(edf_directory, edf_file)
        
        try:
            # Initialize analyzer
            analyzer = CardiovascularAnalyzer()
            
            # Load file and detect channels
            channels_info = analyzer.load_file_and_detect_channels(file_path)
            
            if not channels_info:
                failed += 1
                results.append({
                    'filename': edf_file,
                    'status': 'load_failed'
                })
                continue
            
            # Configure ECG channel (assume channel 0 for single-channel files)
            ecg_channel_idx = 0
            success_msgs = analyzer.configure_channels(ecg_channel_idx, None)
            
            if not success_msgs or not any('ECG' in str(msg) for msg in success_msgs):
                failed += 1
                results.append({
                    'filename': edf_file,
                    'status': 'ecg_config_failed'
                })
                continue
            
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
                failed += 1
                results.append({
                    'filename': edf_file,
                    'status': 'no_peaks_detected'
                })
                continue
            
            peak_count = len(analyzer.ecg_data['td_peaks'])
            
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
                        'sd1_sd2_ratio': td.get('sd1_sd2_ratio', np.nan)
                        
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
                
                # Create result 
                if hrv_data:
                    result = {
                        'filename': edf_file,
                        'duration_sec': len(ecg_signal) / sample_rate,
                        'sample_rate': sample_rate,
                        'total_peaks': peak_count,
                        
                        'adaptive_height': adaptive_params['ecg_height'],
                        'adaptive_prominence': adaptive_params['ecg_prominence'], 
                        'adaptive_distance': adaptive_params['ecg_distance'],
                        
                        #HRV METRICS - Time Domain
                        'mean_rr_ms': hrv_data.get('mean_rr', np.nan),
                        'rmssd_ms': hrv_data.get('rmssd', np.nan),
                        'pnn50_percent': hrv_data.get('pnn50', np.nan),
                        'sdnn_ms': hrv_data.get('sdnn', np.nan),
                        'sdsd_ms': hrv_data.get('sdsd', np.nan),
                         
                        # Nonlinear metrics (Poincare analysis)
                        'sd1_ms': hrv_data.get('sd1', np.nan),
                        'sd2_ms': hrv_data.get('sd2', np.nan),
                        'sd1_sd2_ratio': hrv_data.get('sd1_sd2_ratio', np.nan),
                        
                        # Frequency domain metrics (VLF, LF, HF, TP, LF/HF ratio only)
                        'vlf_power_ms2': hrv_data.get('vlf_power', np.nan),
                        'lf_power_ms2': hrv_data.get('lf_power', np.nan),
                        'hf_power_ms2': hrv_data.get('hf_power', np.nan),
                        'total_power_ms2': hrv_data.get('total_power', np.nan),
                        'lf_hf_ratio': hrv_data.get('lf_hf_ratio', np.nan),
                        
                        'status': 'success'
                    }

                    successful += 1
                    
                else:
                    result = {
                        'filename': edf_file,
                        'duration_sec': len(ecg_signal) / sample_rate,
                        'sample_rate': sample_rate,
                        'total_peaks': peak_count,
                        'status': 'no_hrv_calculated'
                    }
                    failed += 1
            else:
                result = {
                    'filename': edf_file,
                    'duration_sec': len(ecg_signal) / sample_rate,
                    'sample_rate': sample_rate,
                    'total_peaks': peak_count,
                    'status': 'no_results_structure'
                }
                failed += 1
                
        except Exception as e:
            result = {
                'filename': edf_file,
                'status': f'error: {str(e)}'
            }
            failed += 1
        
        results.append(result)
        print()
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
    
    # Summary
    
    
    return df if results else None

def main():
    """
    Main function - processes your validation synthetic ECG directory
    """
    print("Starting Your Pipeline batch processing...")
    
    results_df = batch_process_hrv_metrics()
    
    if results_df is not None:
        print(f"\nBatch processing complete!")
        print(f"Results saved to: ChronOS_results.csv")
    else:
        print(f"\nBatch processing failed!")

if __name__ == "__main__":
    main()