import numpy as np
import pyedflib
from scipy.signal import savgol_filter
import os

def generate_ecg_with_noise(duration=300, fs=256, hr=60):
    """Generate noisy ECG signal that will benefit from bandpass filtering"""
    
    # Time vector
    t = np.arange(0, duration, 1/fs)
    n_samples = len(t)
    
    # Generate baseline clean ECG
    rr_mean = 60.0 / hr  # seconds
    
    # Create RR intervals with some variability
    n_beats = int(duration / rr_mean)
    rr_intervals = np.random.normal(rr_mean, rr_mean * 0.05, n_beats)  # 5% variability
    rr_intervals = np.clip(rr_intervals, 0.5, 1.5)  # Physiological limits
    
    # Generate beat times
    beat_times = np.cumsum(rr_intervals)
    beat_times = beat_times[beat_times < duration]
    
    # Create ECG signal
    ecg = np.zeros(n_samples)
    
    # Add QRS complexes
    for beat_time in beat_times:
        beat_idx = int(beat_time * fs)
        if beat_idx < n_samples - 20:
            # Simple QRS complex shape
            qrs_width = int(0.08 * fs)  # 80ms QRS
            qrs_indices = np.arange(beat_idx - qrs_width//2, beat_idx + qrs_width//2)
            qrs_indices = qrs_indices[(qrs_indices >= 0) & (qrs_indices < n_samples)]
            
            # Triangular QRS shape with some randomness
            qrs_shape = np.exp(-((qrs_indices - beat_idx) / (qrs_width/6))**2)
            amplitude = np.random.normal(1.0, 0.1)  # mV with variability
            ecg[qrs_indices] += amplitude * qrs_shape
    
    # ADD NOISE THAT REQUIRES FILTERING:
    
    # 1. Baseline wander (< 0.5 Hz) - should be removed by highpass
    baseline_wander = 0.3 * np.sin(2 * np.pi * 0.1 * t) + 0.2 * np.sin(2 * np.pi * 0.05 * t)
    
    # 2. Power line interference (50/60 Hz) - should be removed by lowpass
    power_line_50hz = 0.1 * np.sin(2 * np.pi * 50 * t)
    power_line_60hz = 0.08 * np.sin(2 * np.pi * 60 * t)
    
    # 3. High frequency muscle noise (> 40 Hz)
    muscle_noise = 0.15 * np.random.normal(0, 1, n_samples)
    muscle_noise = savgol_filter(muscle_noise, 5, 2)  # Smooth it slightly
    
    # 4. High frequency electronic noise
    electronic_noise = 0.05 * np.sin(2 * np.pi * 100 * t + np.random.uniform(0, 2*np.pi))
    
    # Combine clean ECG with all noise sources
    noisy_ecg = ecg + baseline_wander + power_line_50hz + power_line_60hz + muscle_noise + electronic_noise
    
    return noisy_ecg, t

def generate_ecg_with_ectopics(duration=300, fs=256, hr=60):
    """Generate ECG with isolated ectopic beats that should be detected"""
    
    # Time vector
    t = np.arange(0, duration, 1/fs)
    n_samples = len(t)
    
    # Generate baseline clean ECG
    rr_mean = 60.0 / hr
    n_beats = int(duration / rr_mean)
    
    # Start with regular RR intervals
    rr_intervals = np.random.normal(rr_mean, rr_mean * 0.03, n_beats)  # Less variability
    rr_intervals = np.clip(rr_intervals, 0.6, 1.4)
    
    # INJECT SPECIFIC ISOLATED ECTOPIC PATTERNS that bypass peak detection minimum distance:
    
    # 1. Short interval that passes peak detection but fails physiological limits (around 90 seconds)
    ectopic_idx_1 = int(90 / rr_mean)
    if ectopic_idx_1 < len(rr_intervals):
        rr_intervals[ectopic_idx_1] = 0.28  # 280ms - passes 250ms minimum distance but < 300ms physiological limit
        print(f"Injected short interval (280ms) at beat {ectopic_idx_1} (~90s)")
    
    # 2. Long interval - measurement error (around 150 seconds)
    ectopic_idx_2 = int(150 / rr_mean)
    if ectopic_idx_2 < len(rr_intervals):
        rr_intervals[ectopic_idx_2] = 2.1  # 2100ms - clearly > 2000ms threshold
        print(f"Injected long interval (2100ms) at beat {ectopic_idx_2} (~150s)")
    
    # 3. Very extreme statistical outlier (around 210 seconds)
    ectopic_idx_3 = int(210 / rr_mean)
    if ectopic_idx_3 < len(rr_intervals):
        # Create very extreme outlier that will definitely trigger z-score > 3
        mean_normal = np.mean(rr_intervals[:ectopic_idx_3])  # Use data before outlier
        std_normal = np.std(rr_intervals[:ectopic_idx_3])
        
        # Make it 5 standard deviations out to ensure detection
        outlier_value = mean_normal + 5.0 * std_normal
        # But keep it under 2000ms so it's caught by z-score, not physiological limits
        outlier_value = min(outlier_value, 1.8)  # Cap at 1800ms
        rr_intervals[ectopic_idx_3] = outlier_value
        print(f"Injected statistical outlier ({outlier_value*1000:.0f}ms, z-score=~5.0) at beat {ectopic_idx_3} (~210s)")
    
    # 4. Add one more - borderline long interval that's just at physiological limit
    ectopic_idx_4 = int(270 / rr_mean)
    if ectopic_idx_4 < len(rr_intervals):
        rr_intervals[ectopic_idx_4] = 2.05  # 2050ms - just over 2000ms limit
        print(f"Injected borderline long interval (2050ms) at beat {ectopic_idx_4} (~270s)")
    
    # Generate beat times
    beat_times = np.cumsum(rr_intervals)
    beat_times = beat_times[beat_times < duration]
    
    # Create clean ECG signal
    ecg = np.zeros(n_samples)
    
    # Add QRS complexes
    for beat_time in beat_times:
        beat_idx = int(beat_time * fs)
        if beat_idx < n_samples - 20:
            qrs_width = int(0.08 * fs)
            qrs_indices = np.arange(beat_idx - qrs_width//2, beat_idx + qrs_width//2)
            qrs_indices = qrs_indices[(qrs_indices >= 0) & (qrs_indices < n_samples)]
            
            qrs_shape = np.exp(-((qrs_indices - beat_idx) / (qrs_width/6))**2)
            amplitude = np.random.normal(0.8, 0.05)  # Smaller variability for cleaner signal
            ecg[qrs_indices] += amplitude * qrs_shape
    
    # Add minimal noise (just enough to be realistic)
    clean_noise = 0.02 * np.random.normal(0, 1, n_samples)
    ecg += clean_noise
    
    return ecg, t

def create_edf_file(ecg_signal, filename, fs=256, duration=None):
    """Create EDF file with ECG signal"""
    
    if duration is None:
        duration = len(ecg_signal) / fs
    
    # Create EDF file
    f = pyedflib.EdfWriter(filename, 1, file_type=pyedflib.FILETYPE_EDFPLUS)
    
    # Set signal header using correct method
    signal_header = {
        'label': 'ECG',
        'dimension': 'mV',
        'sample_frequency': fs,  # Updated parameter name
        'physical_min': -5.0,
        'physical_max': 5.0,
        'digital_min': -32768,
        'digital_max': 32767,
        'transducer': '',
        'prefilter': ''
    }
    
    # Set file header
    f.setPatientName('Test Patient')
    f.setPatientCode('TEST001')
    f.setPatientAdditional('Test Data')
    f.setRecordingAdditional('Generated ECG')
    f.setEquipment('Python Generator')
    
    f.setSignalHeader(0, signal_header)
    
    # Write the signal
    f.writeSamples([ecg_signal])
    f.close()
    
    print(f"Created {filename} - Duration: {duration:.1f}s, Samples: {len(ecg_signal)}")

def main():
    """Generate both test files"""
    
    # Check if pyedflib is available
    try:
        import pyedflib
    except ImportError:
        print("Error: pyedflib is required. Install with: pip install pyedflib")
        return
    
    # Parameters
    duration = 300  # 5 minutes
    fs = 256  # Sample rate
    hr = 65  # Heart rate
    
    print("Generating test EDF files...")
    
    # 1. Generate noisy ECG for filter testing
    print("\n1. Creating noisy ECG for bandpass filter demonstration...")
    noisy_ecg, time_vec = generate_ecg_with_noise(duration, fs, hr)
    create_edf_file(noisy_ecg, "test_noisy_ecg_for_filter.edf", fs, duration)
    
    print("   This file contains:")
    print("   - Baseline wander (< 0.5 Hz)")
    print("   - Power line interference (50/60 Hz)")
    print("   - High-frequency muscle noise (> 40 Hz)")
    print("   - Electronic noise (~100 Hz)")
    print("   Expected: 0.5-40 Hz bandpass filter will significantly clean the signal")
    
    # 2. Generate ECG with isolated ectopic beats
    print("\n2. Creating ECG with isolated ectopic beats for detection testing...")
    ectopic_ecg, time_vec = generate_ecg_with_ectopics(duration, fs, hr)
    create_edf_file(ectopic_ecg, "test_ecg_with_isolated_ectopics.edf", fs, duration)
    
    print("   This file contains 4 isolated ectopic patterns:")
    print("   - Short interval (280ms) - passes peak detection but fails physiological limits")  
    print("   - Long interval (2100ms) - physiological limit violation")
    print("   - Statistical outlier (capped at 1800ms, 5 std deviations)")
    print("   - Borderline long interval (2050ms) - just over physiological limit")
    print("   Expected: Your ectopic detection should flag exactly these 4 intervals")
    
    print("\nTest files created successfully!")
    print("\nTo test:")
    print("1. Load test_noisy_ecg_for_filter.edf")
    print("2. Configure ECG channel, preview peaks without filter")
    print("3. Enable bandpass filter, preview again - should see cleaner signal")
    print("4. Load test_ecg_with_isolated_ectopics.edf") 
    print("5. Configure channel, preview peaks, run ectopic detection")
    print("6. Should detect exactly 4 problematic intervals at the expected times")

if __name__ == "__main__":
    main()