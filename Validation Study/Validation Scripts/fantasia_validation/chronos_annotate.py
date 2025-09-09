# simple_peak_extractor.py
# Extract detected R-peaks from your analyzer

import os
import sys
import numpy as np
import pandas as pd

def extract_peaks_simple():
    """Simple extraction that works with your analyzer structure"""
    
    # Add current directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from analyzer import CardiovascularAnalyzer
    
    # Initialize analyzer
    analyzer = CardiovascularAnalyzer()
    
    # Load the EDF file
    edf_path = r"C:\Users\Anthony\Desktop\peak_detector\data\f1y01_from_csv\f1y01_from_csv.edf"
    
    print("Loading EDF file...")
    channels_info = analyzer.load_file_and_detect_channels(edf_path)
    
    if not channels_info:
        print("ERROR: Could not load EDF file")
        return
    
    print("Configuring ECG channel...")
    analyzer.configure_channels(0, None)  # ECG on channel 0
    
    # Get signal info
    ecg_signal = analyzer.ecg_data['raw']
    sample_rate = analyzer.ecg_data['fs']
    
    print(f"ECG signal: {len(ecg_signal)} samples at {sample_rate} Hz")
    
    # Calculate adaptive parameters (your method)
    ecg_baseline = np.median(ecg_signal)
    ecg_max = np.max(ecg_signal)
    ecg_min = np.min(ecg_signal)
    signal_range = ecg_max - ecg_baseline
    
    adaptive_params = {
        'ecg_height': 0.55 * signal_range,
        'ecg_prominence': 0.6 * (0.5 * signal_range),
        'ecg_distance': int(0.25 * sample_rate),
        'bp_height': 110,
        'bp_distance': 100,
        'bp_prominence': 5
    }
    
    print(f"Using adaptive parameters:")
    print(f"  Height: {adaptive_params['ecg_height']:.3f}")
    print(f"  Prominence: {adaptive_params['ecg_prominence']:.3f}")
    print(f"  Distance: {adaptive_params['ecg_distance']}")
    
    # Run peak detection
    print("Running peak detection...")
    analyzer.find_peaks_with_params(**adaptive_params, use_adaptive=True)
    
    # Extract peaks - try different access methods
    detected_peaks = None
    
    # Method 1: Direct access
    if 'peaks' in analyzer.ecg_data:
        detected_peaks = analyzer.ecg_data['peaks']
        print(f"Found peaks via ecg_data['peaks']: {len(detected_peaks)}")
    else:
        print("No peaks found in ecg_data['peaks']")
        print(f"Available keys in ecg_data: {list(analyzer.ecg_data.keys())}")
        return
    
    if detected_peaks is None or len(detected_peaks) == 0:
        print("ERROR: No peaks detected")
        return
    
    # Convert to timestamps
    peak_times_seconds = detected_peaks / sample_rate
    peak_times_ms = peak_times_seconds * 1000
    
    # Create DataFrame
    peaks_df = pd.DataFrame({
        'peak_number': range(1, len(detected_peaks) + 1),
        'sample_index': detected_peaks,
        'time_seconds': peak_times_seconds,
        'time_ms': peak_times_ms
    })
    
    # Save to CSV
    output_path = "detected_r_peaks_simple.csv"
    peaks_df.to_csv(output_path, index=False)
    
    print(f"\nSUCCESS!")
    print(f"Detected {len(detected_peaks)} R-peaks")
    print(f"Saved to: {output_path}")
    print(f"Time range: {peak_times_seconds[0]:.1f}s to {peak_times_seconds[-1]:.1f}s")
    
    # Show first few peaks
    print(f"\nFirst 10 peaks:")
    print(peaks_df.head(10).to_string(index=False))
    
    return peaks_df

if __name__ == "__main__":
    extract_peaks_simple()