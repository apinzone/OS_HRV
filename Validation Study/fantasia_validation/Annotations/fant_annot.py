# Extract R-peak annotations from f1y01.ecg file

import os
import wfdb
import pandas as pd
import numpy as np
from pathlib import Path

def extract_f1y01_annotations():
    """Extract R-peak annotations from f1y01 PhysioNet files"""
    
    # Paths
    physionet_dir = r"C:\Users\Anthony\Desktop\peak_detector\data\fantasia_validation"
    output_dir = r"C:\Users\Anthony\Desktop\peak_detector\data\fantasia_annotations"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    record_name = "f1y01"
    
    print("="*50)
    print(f"EXTRACTING R-PEAK ANNOTATIONS FOR {record_name}")
    print("="*50)
    
    # Check if required files exist
    required_files = [f"{record_name}.dat", f"{record_name}.hea", f"{record_name}.ecg"]
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(physionet_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print(f"ERROR: Missing required files in {physionet_dir}:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    # Change to physionet directory for wfdb
    original_dir = os.getcwd()
    os.chdir(physionet_dir)
    
    try:
        print(f"Reading annotations from {record_name}.ecg...")
        
        # Read the annotation file
        annotation = wfdb.rdann(record_name, extension='ecg')
        
        print(f"Total annotations found: {len(annotation.sample)}")
        print(f"Annotation symbols: {set(annotation.symbol)}")
        
        # Filter for actual heartbeat annotations
        # Common heartbeat symbols in PhysioNet databases
        heartbeat_symbols = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']
        
        valid_beat_indices = []
        valid_symbols = []
        
        for i, symbol in enumerate(annotation.symbol):
            if symbol in heartbeat_symbols:
                valid_beat_indices.append(i)
                valid_symbols.append(symbol)
        
        # Extract R-peak sample indices and times
        r_peak_samples = annotation.sample[valid_beat_indices]
        r_peak_times = r_peak_samples / 250.0  # 250 Hz sampling rate
        beat_types = valid_symbols
        
        print(f"Valid R-peak annotations: {len(r_peak_samples)}")
        print(f"Beat types found: {set(beat_types)}")
        print(f"Time range: {r_peak_times[0]:.1f}s to {r_peak_times[-1]:.1f}s")
        print(f"Duration: {(r_peak_times[-1] - r_peak_times[0])/60:.1f} minutes")
        
        # Calculate RR intervals for basic validation
        rr_intervals = np.diff(r_peak_times)
        heart_rates = 60.0 / rr_intervals
        
        print(f"\nBasic HRV statistics:")
        print(f"  Mean RR interval: {np.mean(rr_intervals)*1000:.1f} ms")
        print(f"  RR interval range: {np.min(rr_intervals)*1000:.1f} to {np.max(rr_intervals)*1000:.1f} ms")
        print(f"  Mean heart rate: {np.mean(heart_rates):.1f} BPM")
        print(f"  Heart rate range: {np.min(heart_rates):.1f} to {np.max(heart_rates):.1f} BPM")
        
        # Create DataFrame with annotations
        annotations_df = pd.DataFrame({
            'sample_index': r_peak_samples,
            'time_seconds': r_peak_times,
            'beat_type': beat_types,
            'rr_interval_ms': [np.nan] + (np.diff(r_peak_times) * 1000).tolist()  # RR interval in ms
        })
        
        # Save annotations
        csv_path = os.path.join(output_dir, f"{record_name}_ground_truth_annotations.csv")
        annotations_df.to_csv(csv_path, index=False)
        
        print(f"\nAnnotations saved to: {csv_path}")
        
        # Also create a summary file
        summary = {
            'record': record_name,
            'total_annotations': len(annotation.sample),
            'valid_r_peaks': len(r_peak_samples),
            'duration_minutes': (r_peak_times[-1] - r_peak_times[0]) / 60,
            'sampling_rate_hz': 250,
            'mean_rr_interval_ms': np.mean(rr_intervals) * 1000,
            'mean_heart_rate_bpm': np.mean(heart_rates),
            'beat_types': list(set(beat_types))
        }
        
        summary_df = pd.DataFrame([summary])
        summary_path = os.path.join(output_dir, f"{record_name}_annotation_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        print(f"Summary saved to: {summary_path}")
        
        # Display first few annotations for verification
        print(f"\nFirst 10 R-peak annotations:")
        print(annotations_df.head(10).to_string(index=False))
        
        return {
            'annotations_path': csv_path,
            'summary_path': summary_path,
            'r_peak_samples': r_peak_samples,
            'r_peak_times': r_peak_times,
            'total_peaks': len(r_peak_samples),
            'duration_minutes': (r_peak_times[-1] - r_peak_times[0]) / 60
        }
        
    except Exception as e:
        print(f"ERROR extracting annotations: {e}")
        return False
        
    finally:
        os.chdir(original_dir)

def validate_annotations(result):
    """Basic validation of extracted annotations"""
    
    if not result:
        return False
    
    print(f"\n{'='*50}")
    print("ANNOTATION VALIDATION")
    print(f"{'='*50}")
    
    r_peak_samples = result['r_peak_samples']
    r_peak_times = result['r_peak_times']
    
    # Check for reasonable values
    issues = []
    
    # Check RR intervals
    rr_intervals = np.diff(r_peak_times)
    min_rr = np.min(rr_intervals) * 1000  # ms
    max_rr = np.max(rr_intervals) * 1000  # ms
    
    if min_rr < 300:  # Less than 300ms (>200 BPM)
        issues.append(f"Very short RR interval detected: {min_rr:.1f}ms")
    
    if max_rr > 2000:  # More than 2000ms (<30 BPM)
        issues.append(f"Very long RR interval detected: {max_rr:.1f}ms")
    
    # Check for temporal order
    if not np.all(np.diff(r_peak_samples) > 0):
        issues.append("R-peaks not in chronological order")
    
    # Check sample indices are reasonable
    if np.min(r_peak_samples) < 0:
        issues.append("Negative sample indices found")
    
    if issues:
        print("POTENTIAL ISSUES DETECTED:")
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print("✅ All validation checks passed")
    
    print(f"\nFINAL STATISTICS:")
    print(f"  Total R-peaks: {len(r_peak_samples):,}")
    print(f"  Duration: {result['duration_minutes']:.1f} minutes")
    print(f"  Average beats per minute: {len(r_peak_samples)/result['duration_minutes']:.1f}")
    print(f"  Sample range: {np.min(r_peak_samples):,} to {np.max(r_peak_samples):,}")
    
    return len(issues) == 0

def main():
    """Main extraction function"""
    
    print("Extracting f1y01 R-peak annotations from PhysioNet data...")
    
    result = extract_f1y01_annotations()
    
    if result:
        validation_passed = validate_annotations(result)
        
        print(f"\n{'='*50}")
        print("EXTRACTION COMPLETE")
        print(f"{'='*50}")
        
        if validation_passed:
            print("✅ SUCCESS: Ground truth annotations extracted and validated")
            print(f"✅ Ready for validation with {result['total_peaks']:,} expert-annotated R-peaks")
            print(f"\nFiles created:")
            print(f"  📄 {result['annotations_path']}")
            print(f"  📄 {result['summary_path']}")
        else:
            print("⚠️  Annotations extracted but validation issues detected")
            print("   Review the warnings above before proceeding")
    else:
        print("❌ FAILED: Could not extract annotations")
        print("   Check that f1y01.dat, f1y01.hea, and f1y01.ecg files exist")

if __name__ == "__main__":
    main()