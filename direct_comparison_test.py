#!/usr/bin/env python3
"""
Direct comparison between your working validation script and batch script
Process the same file with both methods and compare results
"""

import neurokit2 as nk
import numpy as np
import pandas as pd
import pyedflib

def method_your_validation_script(filepath):
    """Your exact working validation script method"""
    print("🔬 METHOD 1: Your Validation Script (EXACT COPY)")
    
    try:
        with pyedflib.EdfReader(filepath) as edf_file:
            sampling_rate = int(edf_file.getSampleFrequency(0))
            ecg_signal = edf_file.readSignal(0)
            
            print(f"  Signal length: {len(ecg_signal)}, Rate: {sampling_rate} Hz")
            
            # EXACT COPY of your working script
            _, r_peaks = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)
            
            print(f"  R_peaks shape: {r_peaks['ECG_R_Peaks'].shape}")
            print(f"  R_peaks sum: {np.sum(r_peaks['ECG_R_Peaks'])}")
            print(f"  R_peaks unique values: {np.unique(r_peaks['ECG_R_Peaks'])}")
            
            # Calculate time domain metrics
            time_domain = nk.hrv_time(r_peaks, sampling_rate=sampling_rate, show=False)
            frequency_domain = nk.hrv_frequency(r_peaks, sampling_rate=sampling_rate, show=False)
            nonlinear = nk.hrv_nonlinear(r_peaks, sampling_rate=sampling_rate, show=False)
            
            print(f"  DataFrames - Time: {time_domain.shape}, Freq: {frequency_domain.shape}, NL: {nonlinear.shape}")
            
            # EXACT COPY of your metrics extraction
            rr_times = r_peaks['ECG_R_Peaks'] / sampling_rate
            rr_intervals_sec = np.diff(rr_times)
            rr_intervals_ms = rr_intervals_sec * 1000
            
            print(f"  RR calculation - Times shape: {rr_times.shape}, Intervals: {len(rr_intervals_ms)}")
            print(f"  First 5 RR times: {rr_times[:5]}")
            print(f"  First 5 RR intervals: {rr_intervals_ms[:5]}")
            
            # Basic counts and timing
            num_beats = len(r_peaks['ECG_R_Peaks'])
            total_time = rr_times[-1] - rr_times[0] if len(rr_times) > 1 else 0
            avg_hr = (num_beats - 1) / (total_time / 60) if total_time > 0 else 0
            mean_rr_ms = np.mean(rr_intervals_ms) if len(rr_intervals_ms) > 0 else 0
            
            # Time domain metrics
            rmssd = time_domain['HRV_RMSSD'].iloc[0] if 'HRV_RMSSD' in time_domain.columns else np.nan
            sdnn = time_domain['HRV_SDNN'].iloc[0] if 'HRV_SDNN' in time_domain.columns else np.nan
            pnn50 = time_domain['HRV_pNN50'].iloc[0] if 'HRV_pNN50' in time_domain.columns else np.nan
            
            # Nonlinear metrics
            sd1 = nonlinear['HRV_SD1'].iloc[0] if 'HRV_SD1' in nonlinear.columns else np.nan
            sd2 = nonlinear['HRV_SD2'].iloc[0] if 'HRV_SD2' in nonlinear.columns else np.nan
            sd1_sd2_ratio = sd1 / sd2 if (sd2 > 0 and not np.isnan(sd1) and not np.isnan(sd2)) else np.nan
            
            # Frequency domain metrics
            lf_power = frequency_domain['HRV_LF'].iloc[0] if 'HRV_LF' in frequency_domain.columns else np.nan
            hf_power = frequency_domain['HRV_HF'].iloc[0] if 'HRV_HF' in frequency_domain.columns else np.nan
            lf_hf_ratio = lf_power / hf_power if (hf_power > 0 and not np.isnan(lf_power) and not np.isnan(hf_power)) else np.nan
            
            # Normalized units
            total_power = lf_power + hf_power if (not np.isnan(lf_power) and not np.isnan(hf_power)) else np.nan
            lf_nu = (lf_power / total_power) if (total_power > 0 and not np.isnan(total_power)) else np.nan
            hf_nu = (hf_power / total_power) if (total_power > 0 and not np.isnan(total_power)) else np.nan
            
            return {
                'method': 'your_validation',
                'num_beats': num_beats,
                'avg_hr_bpm': avg_hr,
                'mean_rr_ms': mean_rr_ms,
                'total_time_sec': total_time,
                'rmssd_ms': rmssd,
                'sdnn_ms': sdnn,
                'pnn50_percent': pnn50,
                'sd1_ms': sd1,
                'sd2_ms': sd2,
                'sd1_sd2_ratio': sd1_sd2_ratio,
                'lf_power_ms2': lf_power,
                'hf_power_ms2': hf_power,
                'lf_hf_ratio': lf_hf_ratio,
                'lf_nu': lf_nu,
                'hf_nu': hf_nu,
                'debug_rr_times_first5': rr_times[:5].tolist(),
                'debug_rr_intervals_first5': rr_intervals_ms[:5].tolist(),
                'debug_r_peaks_sum': int(np.sum(r_peaks['ECG_R_Peaks'])),
                'debug_r_peaks_shape': r_peaks['ECG_R_Peaks'].shape[0]
            }
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def method_corrected_peaks(filepath):
    """Method using proper peak indices extraction"""
    print("\n🔬 METHOD 2: Corrected Peak Extraction")
    
    try:
        with pyedflib.EdfReader(filepath) as edf_file:
            sampling_rate = int(edf_file.getSampleFrequency(0))
            ecg_signal = edf_file.readSignal(0)
            
            # Same peak detection
            _, r_peaks = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)
            
            # CORRECTED: Extract actual peak indices
            peak_indices = np.where(r_peaks['ECG_R_Peaks'] == 1)[0]
            
            print(f"  Peak indices found: {len(peak_indices)}")
            print(f"  First 5 peak indices: {peak_indices[:5]}")
            
            if len(peak_indices) < 2:
                print("  Not enough peaks")
                return None
            
            # Calculate RR intervals correctly
            rr_times = peak_indices / sampling_rate
            rr_intervals_sec = np.diff(rr_times)
            rr_intervals_ms = rr_intervals_sec * 1000
            
            print(f"  RR calculation - Peak indices: {len(peak_indices)}, Intervals: {len(rr_intervals_ms)}")
            print(f"  First 5 RR times: {rr_times[:5]}")
            print(f"  First 5 RR intervals: {rr_intervals_ms[:5]}")
            
            # Same HRV calculations
            time_domain = nk.hrv_time(r_peaks, sampling_rate=sampling_rate, show=False)
            frequency_domain = nk.hrv_frequency(r_peaks, sampling_rate=sampling_rate, show=False)
            nonlinear = nk.hrv_nonlinear(r_peaks, sampling_rate=sampling_rate, show=False)
            
            # Basic metrics using corrected approach
            num_beats = len(peak_indices)
            total_time = rr_times[-1] - rr_times[0] if len(rr_times) > 1 else 0
            avg_hr = (num_beats - 1) / (total_time / 60) if total_time > 0 else 0
            mean_rr_ms = np.mean(rr_intervals_ms) if len(rr_intervals_ms) > 0 else 0
            
            # Same metric extractions
            rmssd = time_domain['HRV_RMSSD'].iloc[0] if 'HRV_RMSSD' in time_domain.columns else np.nan
            sdnn = time_domain['HRV_SDNN'].iloc[0] if 'HRV_SDNN' in time_domain.columns else np.nan
            pnn50 = time_domain['HRV_pNN50'].iloc[0] if 'HRV_pNN50' in time_domain.columns else np.nan
            
            sd1 = nonlinear['HRV_SD1'].iloc[0] if 'HRV_SD1' in nonlinear.columns else np.nan
            sd2 = nonlinear['HRV_SD2'].iloc[0] if 'HRV_SD2' in nonlinear.columns else np.nan
            sd1_sd2_ratio = sd1 / sd2 if (sd2 > 0 and not np.isnan(sd1) and not np.isnan(sd2)) else np.nan
            
            lf_power = frequency_domain['HRV_LF'].iloc[0] if 'HRV_LF' in frequency_domain.columns else np.nan
            hf_power = frequency_domain['HRV_HF'].iloc[0] if 'HRV_HF' in frequency_domain.columns else np.nan
            lf_hf_ratio = lf_power / hf_power if (hf_power > 0 and not np.isnan(lf_power) and not np.isnan(hf_power)) else np.nan
            
            total_power = lf_power + hf_power if (not np.isnan(lf_power) and not np.isnan(hf_power)) else np.nan
            lf_nu = (lf_power / total_power) if (total_power > 0 and not np.isnan(total_power)) else np.nan
            hf_nu = (hf_power / total_power) if (total_power > 0 and not np.isnan(total_power)) else np.nan
            
            return {
                'method': 'corrected_peaks',
                'num_beats': num_beats,
                'avg_hr_bpm': avg_hr,
                'mean_rr_ms': mean_rr_ms,
                'total_time_sec': total_time,
                'rmssd_ms': rmssd,
                'sdnn_ms': sdnn,
                'pnn50_percent': pnn50,
                'sd1_ms': sd1,
                'sd2_ms': sd2,
                'sd1_sd2_ratio': sd1_sd2_ratio,
                'lf_power_ms2': lf_power,
                'hf_power_ms2': hf_power,
                'lf_hf_ratio': lf_hf_ratio,
                'lf_nu': lf_nu,
                'hf_nu': hf_nu,
                'debug_rr_times_first5': rr_times[:5].tolist(),
                'debug_rr_intervals_first5': rr_intervals_ms[:5].tolist(),
                'debug_peak_indices_first5': peak_indices[:5].tolist(),
                'debug_num_peak_indices': len(peak_indices)
            }
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def compare_methods(filepath):
    """Compare both methods side by side"""
    print(f"🔍 COMPARING METHODS FOR: {filepath}")
    print("=" * 70)
    
    # Run both methods
    result1 = method_your_validation_script(filepath)
    result2 = method_corrected_peaks(filepath)
    
    if result1 is None or result2 is None:
        print("❌ One or both methods failed")
        return
    
    print(f"\n📊 COMPARISON RESULTS:")
    print("=" * 70)
    
    metrics_to_compare = [
        'num_beats', 'avg_hr_bpm', 'mean_rr_ms', 'rmssd_ms', 'sdnn_ms', 'pnn50_percent',
        'sd1_ms', 'sd2_ms', 'lf_power_ms2', 'hf_power_ms2', 'lf_hf_ratio', 'lf_nu', 'hf_nu'
    ]
    
    for metric in metrics_to_compare:
        val1 = result1.get(metric, 'N/A')
        val2 = result2.get(metric, 'N/A')
        
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            diff = val2 - val1
            pct_diff = (diff / val1 * 100) if val1 != 0 else 0
            print(f"{metric:15s}: {val1:8.3f} vs {val2:8.3f} | Diff: {diff:+8.3f} ({pct_diff:+6.1f}%)")
        else:
            print(f"{metric:15s}: {val1:>8s} vs {val2:>8s}")
    
    print("\n🔍 DEBUG INFO:")
    print("-" * 40)
    print("Your method:")
    print(f"  RR times first 5: {result1['debug_rr_times_first5']}")
    print(f"  RR intervals first 5: {result1['debug_rr_intervals_first5']}")
    print(f"  R_peaks sum: {result1['debug_r_peaks_sum']}")
    print(f"  R_peaks shape: {result1['debug_r_peaks_shape']}")
    
    print("\nCorrected method:")
    print(f"  RR times first 5: {result2['debug_rr_times_first5']}")
    print(f"  RR intervals first 5: {result2['debug_rr_intervals_first5']}")
    print(f"  Peak indices first 5: {result2['debug_peak_indices_first5']}")
    print(f"  Num peak indices: {result2['debug_num_peak_indices']}")

if __name__ == "__main__":
    # Test with the problematic file
    test_file = "validation_synthetic_ecg/synthetic_ecg_068.edf"
    compare_methods(test_file)