import os
import glob
import pandas as pd
import numpy as np
import neurokit2 as nk
import pyedflib
from datetime import datetime
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration
DATA_DIRECTORY = r"validation_synthetic_ecg"  # EDF files folder
OUTPUT_CSV = "neurokit_harmonized_validation_results.csv"

class NeuroKitHarmonizedBatchProcessor:
    """
    Batch processor using harmonized parameters to match PhysioKit exactly
    """
    
    def __init__(self):
        self.sampling_rate = 256  # Will be updated when loading files
        
    def load_edf_file(self, filepath):
        """
        Load EDF file and extract ECG signal
        """
        try:
            with pyedflib.EdfReader(filepath) as edf_file:
                # Get sampling rate
                self.sampling_rate = int(edf_file.getSampleFrequency(0))
                
                # Read ECG signal (assuming first channel is ECG)
                ecg_signal = edf_file.readSignal(0)
                
                return ecg_signal, self.sampling_rate
                
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None, None
    
    def analyze_hrv_harmonized(self, ecg_signal, sampling_rate):
        try:
            # Find peaks without cleaning 
            _, r_peaks = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)
            
            # Ensure min number of peaks for HRV analysis
            if len(r_peaks['ECG_R_Peaks']) < 5:
                return None
                
            # Calculate time domain metrics
            time_domain = nk.hrv_time(r_peaks, sampling_rate=sampling_rate, show=False)
            
            # Calculate frequency domain metrics 
            frequency_domain = nk.hrv_frequency(
                r_peaks, 
                sampling_rate=sampling_rate, 
                interpolation_rate=4,      # MATCH PhysioKit: 4 Hz instead of 100 Hz
                psd_method="welch",        # MATCH PhysioKit: Welch method
                normalize=False,           # MATCH PhysioKit: Absolute ms² units
                show=False
            )
            
            # Calculate nonlinear metrics
            nonlinear = nk.hrv_nonlinear(r_peaks, sampling_rate=sampling_rate, show=False)
            
            # Extract specific metrics 
            results = self._extract_harmonized_metrics(
                r_peaks, time_domain, frequency_domain, nonlinear, sampling_rate
            )
            
            return results
            
        except Exception as e:
            print(f"    ❌ NeuroKit2 harmonized analysis failed: {e}")
            return None
    
    def _extract_harmonized_metrics(self, r_peaks, time_domain, frequency_domain, nonlinear, sampling_rate):
        # Calculate RR intervals manually
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

        # Nonlinear metrics 
        sd1 = nonlinear['HRV_SD1'].iloc[0] if 'HRV_SD1' in nonlinear.columns else np.nan
        sd2 = nonlinear['HRV_SD2'].iloc[0] if 'HRV_SD2' in nonlinear.columns else np.nan
        sd1_sd2_ratio = sd1 / sd2 if (sd2 > 0 and not np.isnan(sd1) and not np.isnan(sd2)) else np.nan

        # Frequency domain metrics 
        vlf_power = frequency_domain['HRV_VLF'].iloc[0] if 'HRV_VLF' in frequency_domain.columns else np.nan
        lf_power = frequency_domain['HRV_LF'].iloc[0] if 'HRV_LF' in frequency_domain.columns else np.nan
        hf_power = frequency_domain['HRV_HF'].iloc[0] if 'HRV_HF' in frequency_domain.columns else np.nan
        total_power = frequency_domain['HRV_TP'].iloc[0] if 'HRV_TP' in frequency_domain.columns else (vlf_power + lf_power + hf_power)
        
        # LF/HF ratio
        lf_hf_ratio = lf_power / hf_power if (hf_power > 0 and not np.isnan(lf_power) and not np.isnan(hf_power)) else np.nan

        total_power_no_vlf = lf_power + hf_power if (not np.isnan(lf_power) and not np.isnan(hf_power)) else np.nan
        lf_nu = (lf_power / total_power_no_vlf) * 100 if (total_power_no_vlf > 0 and not np.isnan(total_power_no_vlf)) else np.nan
        hf_nu = (hf_power / total_power_no_vlf) * 100 if (total_power_no_vlf > 0 and not np.isnan(total_power_no_vlf)) else np.nan

        # Compile results 
        results = {
            # Basic metrics
            'nk_num_beats': num_beats,
            'nk_avg_hr_bpm': avg_hr,
            'nk_mean_rr_ms': mean_rr_ms,
            'nk_total_time_sec': total_time,
            
            # Time domain
            'nk_rmssd_ms': rmssd,
            'nk_sdnn_ms': sdnn,
            'nk_pnn50_percent': pnn50,
            
            # Nonlinear (Poincaré) 
            'nk_sd1_ms': sd1,
            'nk_sd2_ms': sd2,
            'nk_sd1_sd2_ratio': sd1_sd2_ratio,
            
            # Frequency domain - HARMONIZED (4Hz, Welch, absolute ms², VLF-excluded n.u.)
            'nk_vlf_power_ms2': vlf_power,
            'nk_lf_power_ms2': lf_power,
            'nk_hf_power_ms2': hf_power,
            'nk_total_power_ms2': total_power,
            'nk_lf_hf_ratio': lf_hf_ratio,
            'nk_lf_nu_percent': lf_nu,      # VLF-excluded normalization
            'nk_hf_nu_percent': hf_nu,      # VLF-excluded normalization
            
            # Analysis parameters for verification
            'nk_interpolation_rate': 4,
            'nk_psd_method': 'welch',
            'nk_normalize': False,
            'nk_vlf_excluded_nu': True
        }
        
        return results
    
    def process_single_file(self, filepath):
        """
        Process a single EDF file with harmonized parameters
        """
        filename = os.path.basename(filepath)
        
        # Initialize result row
        result = {
            'filename': filename,
            'status': 'failed',
            'error': None,
            'analysis_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            # Load EDF file
            ecg_signal, sampling_rate = self.load_edf_file(filepath)
            
            if ecg_signal is None:
                result['error'] = "Failed to load EDF file"
                return result
            
            result['sampling_rate'] = sampling_rate
            result['duration_sec'] = len(ecg_signal) / sampling_rate
            
            # Analyze with NeuroKit2 using HARMONIZED parameters
            nk_results = self.analyze_hrv_harmonized(ecg_signal, sampling_rate)
            
            if nk_results is not None:
                result.update(nk_results)
                result['status'] = 'success'
            else:
                result['error'] = "NeuroKit2 harmonized analysis failed - insufficient peaks"
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def batch_process_directory(self):
        """
        Process all EDF files in the directory 
        """
        print("NeuroKit2 Batch Processor")
        
        # Check dependencies
        try:
            import neurokit2 as nk
            print(f"✅ NeuroKit2 version: {nk.__version__}")
        except ImportError:
            print("❌ NeuroKit2 not installed. Install with: pip install neurokit2")
            return
        
        try:
            import pyedflib
            print("✅ pyedflib available")
        except ImportError:
            print("❌ pyedflib not installed. Install with: pip install pyedflib")
            return
        
        # Find EDF files
        if not os.path.exists(DATA_DIRECTORY):
            print(f"❌ Directory not found: {DATA_DIRECTORY}")
            return
        
        edf_files = glob.glob(os.path.join(DATA_DIRECTORY, "*.edf"))
        edf_files.sort()  # Sort for consistent processing order
        
        if not edf_files:
            print(f"❌ No EDF files found in {DATA_DIRECTORY}")
            return
        
        print(f"🔍 Found {len(edf_files)} EDF files")
        print(f"📊 Output: {OUTPUT_CSV}")
        print("🔑 Harmonized parameters: interpolation_rate=4, psd_method='welch', normalize=False")
        print()
        
        # Process all files
        results = []
        successful = 0
        failed = 0
        
        for file_path in tqdm(edf_files, desc="Processing with harmonized parameters"):
            result = self.process_single_file(file_path)
            results.append(result)
            
            if result['status'] == 'success':
                successful += 1
                beats = result.get('nk_num_beats', 0)
                hr = result.get('nk_avg_hr_bpm', 0)
                rmssd = result.get('nk_rmssd_ms', 0)
                lf_power = result.get('nk_lf_power_ms2', 0)
                hf_power = result.get('nk_hf_power_ms2', 0)
                print(f"✅ {result['filename']}: {beats} beats, HR: {hr:.1f} BPM, RMSSD: {rmssd:.1f}ms, LF: {lf_power:.3f}ms²")
            else:
                failed += 1
                error = result.get('error', 'Unknown error')
                print(f"❌ {result['filename']}: {error}")
        
        # Create DataFrame and save
        df = pd.DataFrame(results)
        
        # Reorder columns for better readability
        first_cols = ['filename', 'status', 'sampling_rate', 'duration_sec']
        hrv_cols = [col for col in df.columns if col.startswith('nk_')]
        param_cols = ['nk_interpolation_rate', 'nk_psd_method', 'nk_normalize', 'nk_vlf_excluded_nu']
        last_cols = ['analysis_timestamp', 'error']
        
        # Put parameter verification columns at the end of HRV columns
        hrv_data_cols = [col for col in hrv_cols if col not in param_cols]
        column_order = first_cols + hrv_data_cols + param_cols + last_cols
        available_cols = [col for col in column_order if col in df.columns]
        df = df[available_cols]
        
        # Save to CSV
        df.to_csv(OUTPUT_CSV, index=False)
        
        # Summary
        print()
        print("📊 BATCH PROCESSING COMPLETE!")
        print("=" * 80)
        print(f"✅ Successful: {successful}/{len(edf_files)} files")
        print(f"❌ Failed: {failed}/{len(edf_files)} files")
        print(f"💾 Results saved to: {OUTPUT_CSV}")
        
        # Show harmonization verification
        if successful > 0:
            success_df = df[df['status'] == 'success']
            print()
            print("🔍 HARMONIZATION VERIFICATION:")
            print(f"   Interpolation rate: {success_df['nk_interpolation_rate'].iloc[0]} Hz")
            print(f"   PSD method: {success_df['nk_psd_method'].iloc[0]}")
            print(f"   Normalize setting: {success_df['nk_normalize'].iloc[0]}")
            print(f"   VLF-excluded n.u.: {success_df['nk_vlf_excluded_nu'].iloc[0]}")
        
        # Show sample successful results
        if successful > 0:
            print()
            print("📋 Sample Harmonized Results:")
            display_cols = ['filename', 'nk_num_beats', 'nk_mean_rr_ms', 'nk_rmssd_ms', 'nk_lf_power_ms2', 'nk_hf_power_ms2', 'nk_lf_hf_ratio']
            available_display = [col for col in display_cols if col in success_df.columns]
            if len(success_df) > 0:
                print(success_df[available_display].head(5).to_string(index=False))
        
        print(f"\n🎉 Open {OUTPUT_CSV} to view all harmonized NeuroKit2 results!")
        print("🎯 These results should show MUCH better agreement with PhysioKit!")
        print("Ready for improved validation analysis!")
        return df

def main():
    """Main function"""
    processor = NeuroKitHarmonizedBatchProcessor()
    processor.batch_process_directory()

if __name__ == "__main__":
    main()