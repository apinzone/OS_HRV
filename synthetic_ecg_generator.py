# synthetic_ecg_generator.py
# Generate synthetic ECG signals using NeuroKit for HRV validation studies

import numpy as np
import neurokit2 as nk
import pyedflib
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

class SyntheticECGGenerator:
    """
    Generate synthetic ECG signals for HRV validation using NeuroKit2
    Based on ECGSYN methodology with controlled HRV parameters
    """
    
    def __init__(self):
        self.sampling_rate = 256  # Hz, matching validation study
        self.duration = 300  # seconds (5 minutes)
        self.mean_hr = 60  # bpm
        self.hr_std = 1  # bpm
        self.lf_freq = 0.1  # Hz, LF oscillation center
        self.hf_freq = 0.25  # Hz, HF oscillation center
        self.lf_hf_ratio = 1.0  # Target LF/HF power ratio
        
    def generate_synthetic_rr_intervals(self, n_samples=100):
        """
        Generate synthetic RR intervals with controlled HRV parameters
        
        Parameters:
        n_samples: Number of synthetic datasets to generate
        
        Returns:
        List of RR interval arrays
        """
        rr_datasets = []
        
        for i in range(n_samples):
            # Calculate number of beats for 5 minutes at target HR
            total_beats = int(self.duration * (self.mean_hr / 60))
            
            # Generate base RR intervals (mean HR with small variation)
            base_rr = 60 / self.mean_hr  # seconds per beat
            rr_noise = np.random.normal(0, self.hr_std/self.mean_hr, total_beats)
            base_rr_series = base_rr * (1 + rr_noise * 0.01)  # Small random variation
            
            # Add controlled LF and HF oscillations
            time_beats = np.cumsum(base_rr_series)
            
            # LF component (0.1 Hz oscillation)
            lf_amplitude = 0.02  # 2% of base RR
            lf_component = lf_amplitude * np.sin(2 * np.pi * self.lf_freq * time_beats)
            
            # HF component (0.25 Hz oscillation) 
            hf_amplitude = 0.02  # 2% of base RR (equal power for LF/HF ratio = 1.0)
            hf_component = hf_amplitude * np.sin(2 * np.pi * self.hf_freq * time_beats)
            
            # Combine components
            rr_intervals = base_rr_series * (1 + lf_component + hf_component)
            
            # Convert to milliseconds
            rr_intervals_ms = rr_intervals * 1000
            
            rr_datasets.append(rr_intervals_ms)
            
        return rr_datasets
    
    def rr_to_ecg_signal(self, rr_intervals_ms):
        """
        Convert RR intervals to synthetic ECG signal using NeuroKit
        
        Parameters:
        rr_intervals_ms: RR intervals in milliseconds
        
        Returns:
        ecg_signal: Synthetic ECG signal
        r_peaks: R-peak locations
        """
        # Convert RR intervals to R-peak times
        rr_seconds = np.array(rr_intervals_ms) / 1000
        r_peak_times = np.cumsum(np.concatenate([[0], rr_seconds[:-1]]))
        
        # Create time vector for full duration
        time_vector = np.arange(0, self.duration, 1/self.sampling_rate)
        
        # Convert R-peak times to sample indices
        r_peak_indices = np.round(r_peak_times * self.sampling_rate).astype(int)
        r_peak_indices = r_peak_indices[r_peak_indices < len(time_vector)]
        
        # Generate synthetic ECG using NeuroKit
        # Create clean ECG based on R-peak locations
        ecg_signal = np.zeros(len(time_vector))
        
        # Add R-peaks and surrounding ECG morphology
        for r_idx in r_peak_indices:
            if r_idx > 50 and r_idx < len(time_vector) - 50:
                # Simple ECG morphology: R-peak with P, Q, S, T waves
                # R-peak (main spike)
                ecg_signal[r_idx] = 1.0
                
                # Q-wave (before R)
                if r_idx > 10:
                    ecg_signal[r_idx-8:r_idx-3] = -0.1
                
                # S-wave (after R)
                if r_idx < len(time_vector) - 10:
                    ecg_signal[r_idx+3:r_idx+8] = -0.2
                
                # T-wave (later positive deflection)
                if r_idx < len(time_vector) - 30:
                    t_wave = 0.3 * np.exp(-((np.arange(20) - 10)**2) / 50)
                    ecg_signal[r_idx+15:r_idx+35] += t_wave
                
                # P-wave (before QRS complex)
                if r_idx > 40:
                    p_wave = 0.15 * np.exp(-((np.arange(15) - 7)**2) / 25)
                    ecg_signal[r_idx-35:r_idx-20] += p_wave
        
        # Smooth the signal and add small amount of noise
        from scipy import signal
        ecg_signal = signal.savgol_filter(ecg_signal, 5, 2)
        noise = np.random.normal(0, 0.02, len(ecg_signal))
        ecg_signal += noise
        
        return ecg_signal, r_peak_indices
    
    def save_to_edf(self, ecg_signal, filename, subject_id="SYNTH"):
        """
        Save synthetic ECG signal to EDF file
        
        Parameters:
        ecg_signal: ECG signal array
        filename: Output filename
        subject_id: Subject identifier
        """
        # Ensure signal is the right length and type
        ecg_signal = np.array(ecg_signal, dtype=np.float64)
        
        # Calculate file parameters
        file_duration = len(ecg_signal) / self.sampling_rate
        n_data_records = int(file_duration)  # 1-second data records
        samples_per_record = self.sampling_rate
        
        # Adjust signal length to match exact data records
        target_length = n_data_records * samples_per_record
        if len(ecg_signal) > target_length:
            ecg_signal = ecg_signal[:target_length]
        elif len(ecg_signal) < target_length:
            # Pad with zeros if needed
            padding = target_length - len(ecg_signal)
            ecg_signal = np.concatenate([ecg_signal, np.zeros(padding)])
        
        try:
            with pyedflib.EdfWriter(filename, 1, file_type=pyedflib.FILETYPE_EDFPLUS) as writer:
                # Set file info
                writer.setPatientCode(subject_id)
                writer.setPatientName(f"Synthetic_{subject_id}")
                writer.setTechnician("SyntheticECGGenerator")
                writer.setEquipment("Python_NeuroKit2")
                
                # Set header info
                writer.setDatarecordDuration(1000000)  # 1 second in microseconds
                
                # Set signal info
                signal_headers = {
                    'label': 'ECG',
                    'dimension': 'mV',
                    'sample_frequency': self.sampling_rate,  # Changed from sample_rate
                    'physical_min': float(np.min(ecg_signal)) - 0.1,
                    'physical_max': float(np.max(ecg_signal)) + 0.1,
                    'digital_min': -32768,
                    'digital_max': 32767,
                    'transducer': 'Synthetic',
                    'prefilter': 'None'
                }
                
                writer.setSignalHeader(0, signal_headers)
                
                # Write data in 1-second chunks
                for i in range(n_data_records):
                    start_idx = i * samples_per_record
                    end_idx = (i + 1) * samples_per_record
                    data_chunk = ecg_signal[start_idx:end_idx]
                    writer.writeSamples([data_chunk])
                
            print(f"Saved synthetic ECG to {filename}")
            
        except Exception as e:
            print(f"Error saving EDF file {filename}: {e}")
            # Try alternative approach with simpler EDF format
            self._save_simple_edf(ecg_signal, filename, subject_id)
    
    def _save_simple_edf(self, ecg_signal, filename, subject_id="SYNTH"):
        """
        Fallback method to save EDF with simpler format
        """
        try:
            with pyedflib.EdfWriter(filename, 1) as writer:  # Use basic EDF format
                
                # Simple signal header
                signal_headers = [{
                    'label': 'ECG',
                    'dimension': 'mV',
                    'sample_frequency': self.sampling_rate,
                    'physical_min': -5.0,
                    'physical_max': 5.0,
                    'digital_min': -32768,
                    'digital_max': 32767
                }]
                
                writer.setSignalHeaders(signal_headers)
                writer.writeSamples([ecg_signal])
                
            print(f"Saved synthetic ECG to {filename} (simple format)")
            
        except Exception as e:
            print(f"Failed to save EDF file {filename}: {e}")
            print("Consider using a different file format or checking pyedflib installation")
    
    def generate_validation_dataset(self, n_datasets=100, output_dir="synthetic_ecg"):
        """
        Generate complete validation dataset
        
        Parameters:
        n_datasets: Number of synthetic ECG files to generate
        output_dir: Directory to save EDF files
        
        Returns:
        validation_info: Dictionary with ground truth parameters
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate RR intervals for all datasets
        print(f"Generating {n_datasets} synthetic ECG datasets...")
        rr_datasets = self.generate_synthetic_rr_intervals(n_datasets)
        
        validation_info = {
            'files': [],
            'ground_truth': {
                'mean_hr_bpm': self.mean_hr,
                'hr_std_bpm': self.hr_std,
                'mean_rr_ms': 60000 / self.mean_hr,  # Should be ~1000ms
                'lf_freq_hz': self.lf_freq,
                'hf_freq_hz': self.hf_freq,
                'lf_hf_ratio': self.lf_hf_ratio,
                'duration_sec': self.duration,
                'sampling_rate_hz': self.sampling_rate
            },
            'actual_parameters': []
        }
        
        for i, rr_intervals in enumerate(rr_datasets):
            # Generate ECG signal
            ecg_signal, r_peaks = self.rr_to_ecg_signal(rr_intervals)
            
            # Save to EDF
            filename = os.path.join(output_dir, f"synthetic_ecg_{i+1:03d}.edf")
            self.save_to_edf(ecg_signal, filename, f"SYNTH_{i+1:03d}")
            
            # Calculate actual parameters for this dataset
            actual_rr_mean = np.mean(rr_intervals)
            actual_rr_std = np.std(rr_intervals)
            actual_hr = 60000 / actual_rr_mean
            
            validation_info['files'].append(filename)
            validation_info['actual_parameters'].append({
                'file': filename,
                'mean_rr_ms': actual_rr_mean,
                'rr_std_ms': actual_rr_std,
                'mean_hr_bpm': actual_hr,
                'n_beats': len(rr_intervals)
            })
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{n_datasets} datasets")
        
        # Save validation info
        info_file = os.path.join(output_dir, "validation_info.txt")
        with open(info_file, 'w') as f:
            f.write("SYNTHETIC ECG VALIDATION DATASET\n")
            f.write("=" * 40 + "\n\n")
            f.write("Ground Truth Parameters:\n")
            for key, value in validation_info['ground_truth'].items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\nGenerated {len(validation_info['files'])} synthetic ECG files\n")
            f.write(f"Files saved to: {output_dir}\n")
            
        print(f"\nValidation dataset complete!")
        print(f"Files saved to: {output_dir}")
        print(f"Ground truth parameters saved to: {info_file}")
        
        return validation_info
    
    def plot_sample_ecg(self, ecg_signal, r_peaks=None, title="Synthetic ECG Sample"):
        """
        Plot a sample of the generated ECG for quality check
        """
        time_vector = np.arange(len(ecg_signal)) / self.sampling_rate
        
        plt.figure(figsize=(12, 6))
        plt.plot(time_vector[:2560], ecg_signal[:2560])  # First 10 seconds
        
        if r_peaks is not None:
            r_peaks_time = r_peaks[r_peaks < 2560] / self.sampling_rate
            plt.plot(r_peaks_time, ecg_signal[r_peaks[r_peaks < 2560]], 'ro', markersize=8)
        
        plt.xlabel('Time (s)')
        plt.ylabel('ECG (mV)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def main():
    """
    Main function to generate synthetic ECG validation dataset
    """
    # Initialize generator
    generator = SyntheticECGGenerator()
    
    # Generate small test dataset first
    print("Generating test dataset (5 files)...")
    test_info = generator.generate_validation_dataset(n_datasets=5, output_dir="test_synthetic_ecg")
    
    # Generate sample ECG for visualization
    test_rr = generator.generate_synthetic_rr_intervals(1)[0]
    test_ecg, test_r_peaks = generator.rr_to_ecg_signal(test_rr)
    generator.plot_sample_ecg(test_ecg, test_r_peaks, "Sample Synthetic ECG (10 seconds)")
    
    # Ask user if they want to generate full dataset
    response = input("\nGenerate full dataset (100 files)? [y/n]: ")
    if response.lower() == 'y':
        print("Generating full validation dataset...")
        full_info = generator.generate_validation_dataset(n_datasets=100, output_dir="validation_synthetic_ecg")
        print("Full dataset generation complete!")
    
    return test_info

if __name__ == "__main__":
    # Example usage
    validation_info = main()