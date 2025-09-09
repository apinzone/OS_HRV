# single_fantasia_converter.py
# Test conversion with just f1y01 - FIXED VERSION

import os
import wfdb
import pandas as pd
import numpy as np
import pyedflib
from datetime import datetime

def test_f1y01_conversion():
    """Test conversion of just f1y01 to debug the EDF issue"""
    
    fantasia_dir = r"C:\Users\Anthony\Desktop\peak_detector\data\fantasia_validation"
    output_dir = r"C:\Users\Anthony\Desktop\peak_detector\data\fantasia_converted"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    record_name = "f1y01"
    
    print("=" * 60)
    print(f"TESTING SINGLE FILE CONVERSION: {record_name}")
    print("=" * 60)
    
    original_dir = os.getcwd()
    os.chdir(fantasia_dir)
    
    try:
        # Step 1: Extract annotations
        print("Step 1: Extracting annotations...")
        annotation = wfdb.rdann(record_name, extension='ecg')
        
        # Filter for normal beats
        normal_beat_symbols = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'F', 'e']
        normal_indices = []
        for i, symbol in enumerate(annotation.symbol):
            if symbol in normal_beat_symbols:
                normal_indices.append(i)
        
        r_peak_samples = annotation.sample[normal_indices]
        r_peak_times = r_peak_samples / 250.0
        r_peak_symbols = np.array(annotation.symbol)[normal_indices]
        
        annotations_df = pd.DataFrame({
            'sample_index': r_peak_samples,
            'time_seconds': r_peak_times,
            'beat_type': r_peak_symbols
        })
        
        annotations_path = os.path.join(output_dir, f"{record_name}_annotations.csv")
        annotations_df.to_csv(annotations_path, index=False)
        
        print(f"  Extracted {len(r_peak_samples)} R-peak annotations")
        print(f"  Saved to: {annotations_path}")
        
        # Step 2: Load ECG signal
        print("\nStep 2: Loading ECG signal...")
        record = wfdb.rdrecord(record_name)
        ecg_signal = record.p_signal[:, 0]
        sampling_rate = record.fs
        
        print(f"  Signal shape: {ecg_signal.shape}")
        print(f"  Sampling rate: {sampling_rate} Hz")
        print(f"  Duration: {len(ecg_signal)/sampling_rate/60:.1f} minutes")
        print(f"  Signal range: {np.min(ecg_signal):.3f} to {np.max(ecg_signal):.3f}")
        print(f"  Data type: {ecg_signal.dtype}")
        
        # Step 3: Try different EDF conversion approaches
        print("\nStep 3: Testing EDF conversion approaches...")
        
        edf_path = os.path.join(output_dir, f"{record_name}.edf")
        
        # First, check pyedflib version and available methods
        print(f"  pyedflib version: {pyedflib.__version__}")
        
        # Check what methods are available
        print("  Checking available EdfWriter methods...")
        writer_test = pyedflib.EdfWriter("temp_test.edf", 1)
        available_methods = [method for method in dir(writer_test) if not method.startswith('_')]
        writer_test.close()
        os.remove("temp_test.edf")
        print(f"  Available methods: {available_methods}")
        
        # Try the most basic approach first
        print("  Trying most basic EDF creation...")
        try:
            with pyedflib.EdfWriter(edf_path, 1) as writer:
                # Try different parameter names based on pyedflib version
                try:
                    # Newer API
                    writer.setSignalHeader(0, 
                                         label='ECG', 
                                         dimension='mV',
                                         sample_rate=int(sampling_rate))
                except TypeError as e1:
                    print(f"    Newer API failed: {e1}")
                    try:
                        # Older API - different parameter names
                        writer.setLabel(0, 'ECG')
                        writer.setDimension(0, 'mV') 
                        writer.setSamplefrequency(0, int(sampling_rate))
                        writer.setPhysicalMinimum(0, float(np.min(ecg_signal)))
                        writer.setPhysicalMaximum(0, float(np.max(ecg_signal)))
                    except Exception as e2:
                        print(f"    Older API also failed: {e2}")
                        raise e1
                
                writer.writeSamples([ecg_signal.astype(np.float64)])
            
            print(f"  SUCCESS: Basic approach worked!")
            print(f"  EDF file saved: {edf_path}")
            
            # Verify the file
            file_size = os.path.getsize(edf_path) / (1024*1024)
            print(f"  File size: {file_size:.1f} MB")
            
            return True
            
        except Exception as e:
            print(f"  Basic approach failed: {e}")
        
        # FIXED: Try compliant EDF+ creation using proper data records
        print("  Trying EDF+ compliant creation with proper data records...")
        try:
            edf_path_compliant = os.path.join(output_dir, f"{record_name}_compliant.edf")
            
            # Ensure signal is float64
            ecg_signal = ecg_signal.astype(np.float64)
            
            # Calculate proper data record parameters
            data_record_duration = 1.0  # 1 second per data record (standard)
            samples_per_data_record = int(sampling_rate * data_record_duration)
            
            # Calculate total number of data records needed
            total_samples = len(ecg_signal)
            num_data_records = int(np.ceil(total_samples / samples_per_data_record))
            
            # Pad signal if necessary to fit exact number of data records
            padded_length = num_data_records * samples_per_data_record
            if padded_length > total_samples:
                padding_needed = padded_length - total_samples
                ecg_signal = np.concatenate([ecg_signal, np.zeros(padding_needed)])
                print(f"    Padded signal with {padding_needed} zeros to {len(ecg_signal)} samples")
            
            # Create EDF+ file with proper compliance
            with pyedflib.EdfWriter(edf_path_compliant, 1, file_type=pyedflib.FILETYPE_EDFPLUS) as writer:
                
                # Set file header information (required for EDF+)
                writer.setPatientCode(f"Fantasia_{record_name}")
                writer.setPatientName(f"Subject_{record_name}")
                writer.setTechnician("PhysioNet")
                writer.setEquipment("Fantasia Database")
                
                # Set recording start time (required for EDF+)
                start_time = datetime(2000, 1, 1, 12, 0, 0)
                writer.setStartdatetime(start_time)
                
                # Set data record duration (in units of 100 microseconds)
                writer.setDatarecordDuration(int(data_record_duration * 10000))
                
                # Configure signal parameters using your API
                signal_min = float(np.min(ecg_signal))
                signal_max = float(np.max(ecg_signal))
                
                # Ensure min and max are different (EDF requirement)
                if abs(signal_max - signal_min) < 1e-6:
                    signal_max = signal_min + 1.0
                
                # Set signal parameters using the correct method names for your version
                writer.setLabel(0, 'ECG')
                writer.setPhysicalDimension(0, 'mV')
                writer.setSamplefrequency(0, samples_per_data_record)  # Samples per data record
                writer.setPhysicalMinimum(0, signal_min)
                writer.setPhysicalMaximum(0, signal_max)
                writer.setDigitalMinimum(0, -32768)
                writer.setDigitalMaximum(0, 32767)
                writer.setTransducer(0, 'ECG electrode')
                writer.setPrefilter(0, 'None')
                
                print(f"    Writing {num_data_records} data records of {samples_per_data_record} samples each")
                
                # Write data in chunks corresponding to data records
                for i in range(num_data_records):
                    start_idx = i * samples_per_data_record
                    end_idx = start_idx + samples_per_data_record
                    chunk = ecg_signal[start_idx:end_idx]
                    
                    # Write this data record
                    writer.writePhysicalSamples(chunk)
            
            print(f"  SUCCESS: EDF+ compliant creation worked!")
            print(f"  EDF file saved: {edf_path_compliant}")
            
            file_size = os.path.getsize(edf_path_compliant) / (1024*1024)
            print(f"  File size: {file_size:.1f} MB")
            
            # Verify with pyedflib
            try:
                with pyedflib.EdfReader(edf_path_compliant) as reader:
                    n_signals = reader.signals_in_file
                    n_data_records = reader.datarecords_in_file
                    samples_in_file = reader.getNSamples()[0]
                    duration = reader.file_duration
                    
                    print(f"    Verification: {n_signals} signals, {n_data_records} data records")
                    print(f"    Total samples: {samples_in_file}, Duration: {duration:.1f}s")
                    
            except Exception as e:
                print(f"    Verification warning: {e}")
            
            return True
            
        except Exception as e:
            print(f"  EDF+ compliant approach failed: {e}")
        
        print("  All EDF conversion approaches failed")
        return False
        
    finally:
        os.chdir(original_dir)

def test_edf_loading():
    """Test if the created EDF can be loaded by your analyzer"""
    
    output_dir = r"C:\Users\Anthony\Desktop\peak_detector\data\fantasia_converted"
    
    # Find the EDF file that was created - prioritize compliant version
    edf_files = [f for f in os.listdir(output_dir) if f.endswith('.edf') and 'f1y01' in f]
    
    if not edf_files:
        print("No EDF file found to test")
        return
    
    # Prioritize compliant version if it exists
    edf_file = None
    for f in edf_files:
        if 'compliant' in f:
            edf_file = f
            break
    if not edf_file:
        edf_file = edf_files[0]
    
    edf_path = os.path.join(output_dir, edf_file)
    
    print(f"\nStep 4: Testing EDF loading with your analyzer...")
    print(f"Testing file: {edf_file}")
    
    try:
        # Add current directory to path for imports
        current_dir = os.path.dirname(os.path.abspath(__file__))
        import sys
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        from analyzer import CardiovascularAnalyzer
        
        analyzer = CardiovascularAnalyzer()
        channels_info = analyzer.load_file_and_detect_channels(edf_path)
        
        if channels_info:
            print(f"  SUCCESS: Your analyzer loaded the EDF file!")
            print(f"  Channels detected: {len(channels_info)}")
            for ch in channels_info:
                print(f"    Channel {ch['index']}: {ch['name']} ({ch['units']}) - {ch['likely_type']}")
            
            # Try configuring ECG channel
            analyzer.configure_channels(0, None)
            
            if hasattr(analyzer, 'ecg_data') and analyzer.ecg_data:
                ecg_signal = analyzer.ecg_data['raw']
                sample_rate = analyzer.ecg_data['fs']
                
                print(f"  ECG signal loaded: {len(ecg_signal)} samples at {sample_rate} Hz")
                print(f"  Signal range: {np.min(ecg_signal):.3f} to {np.max(ecg_signal):.3f}")
                
                return True
            else:
                print(f"  WARNING: File loaded but ECG data not accessible")
                return False
        else:
            print("  FAILED: Your analyzer could not load the EDF file")
            return False
            
    except Exception as e:
        print(f"  ERROR testing with analyzer: {e}")
        return False

if __name__ == "__main__":
    # Test conversion
    conversion_success = test_f1y01_conversion()
    
    if conversion_success:
        # Test loading
        loading_success = test_edf_loading()
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        if loading_success:
            print("✅ Annotations extracted successfully")
            print("✅ EDF conversion succeeded")
            print("✅ Analyzer can load the EDF file")
            print("✅ Ready to run validation on f1y01")
        else:
            print("✅ Annotations extracted successfully")
            print("✅ EDF conversion succeeded")
            print("❌ Analyzer cannot load the EDF file")
            print("💡 EDF created but not compatible with your analyzer")
        
    else:
        print("\n" + "=" * 60) 
        print("SUMMARY")
        print("=" * 60)
        print("✅ Annotations extracted successfully")
        print("❌ EDF conversion failed")
        print("💡 Try manual debugging or different pyedflib version")