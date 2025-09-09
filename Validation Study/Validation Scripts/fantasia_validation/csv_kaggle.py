# f1y01_csv_to_edf_converter.py
# Extract f1y01 ECG data from Kaggle CSV and convert to EDF

import os
import pandas as pd
import numpy as np
import pyedflib
from datetime import datetime
from pathlib import Path

def extract_f1y01_from_csv():
    """Extract f1y01 ECG data from the Kaggle CSV file"""
    
    csv_path = r"C:\Users\Anthony\Desktop\peak_detector\data\kaggle_fantasia\fantasia_ecg_respiration_signals.csv"
    output_dir = r"C:\Users\Anthony\Desktop\peak_detector\data\f1y01_from_csv"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("EXTRACTING F1Y01 ECG DATA FROM KAGGLE CSV")
    print("="*60)
    
    # Check if CSV file exists
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}")
        return False
    
    print(f"Reading CSV file: {os.path.basename(csv_path)}")
    
    try:
        # Read the CSV file - start with a small sample to check structure
        print("Examining CSV structure...")
        df_sample = pd.read_csv(csv_path, nrows=20)
        
        print(f"CSV columns: {list(df_sample.columns)}")
        print(f"Sample data:")
        print(df_sample.head().to_string(index=False))
        
        # Based on your earlier description, the structure should be:
        # ECG, RESP, Participant, Sample, Sampling_Rate, Database
        expected_columns = ['ECG', 'RESP', 'Participant', 'Sample', 'Sampling_Rate', 'Database']
        
        missing_columns = [col for col in expected_columns if col not in df_sample.columns]
        if missing_columns:
            print(f"WARNING: Expected columns missing: {missing_columns}")
            print("Available columns:", list(df_sample.columns))
        
        # Instead of reading the entire 3.6GB file, read in chunks and filter
        target_participant = "Fantasia_f1y01"
        print(f"Reading CSV in chunks and filtering for {target_participant}...")
        
        chunk_size = 100000  # Process 100K rows at a time
        f1y01_data_chunks = []
        total_rows_processed = 0
        
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            total_rows_processed += len(chunk)
            
            # Filter this chunk for f1y01 data
            f1y01_chunk = chunk[chunk['Participant'] == target_participant]
            
            if len(f1y01_chunk) > 0:
                f1y01_data_chunks.append(f1y01_chunk)
                print(f"  Chunk {total_rows_processed//chunk_size}: Found {len(f1y01_chunk):,} f1y01 samples")
            
            # Progress indicator
            if total_rows_processed % 1000000 == 0:
                print(f"  Processed {total_rows_processed:,} rows...")
        
        if not f1y01_data_chunks:
            print(f"ERROR: No data found for {target_participant}")
            print("Available participants in first chunk:")
            first_chunk = pd.read_csv(csv_path, nrows=chunk_size)
            for participant in sorted(first_chunk['Participant'].unique()):
                print(f"  - {participant}")
            return False
        
        # Combine all f1y01 chunks
        print("Combining f1y01 data chunks...")
        f1y01_data = pd.concat(f1y01_data_chunks, ignore_index=True)
        
        if len(f1y01_data) == 0:
            print(f"ERROR: No data found for {target_participant}")
            print("Available participants:")
            for participant in sorted(df['Participant'].unique()):
                print(f"  - {participant}")
            return False
        
        print(f"Found {len(f1y01_data):,} samples for {target_participant}")
        
        # Sort by sample number to ensure correct chronological order
        f1y01_data = f1y01_data.sort_values('Sample').reset_index(drop=True)
        
        # Extract ECG signal
        ecg_signal_raw = f1y01_data['ECG'].values.astype(np.float64)
        ecg_signal = ecg_signal_raw - np.mean(ecg_signal_raw)
        
        # Get sampling rate
        sampling_rate = f1y01_data['Sampling_Rate'].iloc[0]
        
        # Calculate duration
        duration_seconds = len(ecg_signal) / sampling_rate
        duration_minutes = duration_seconds / 60
        
        # Signal statistics
        signal_min = np.min(ecg_signal)
        signal_max = np.max(ecg_signal)
        signal_mean = np.mean(ecg_signal)
        signal_std = np.std(ecg_signal)
        
        print(f"\nECG Signal Information:")
        print(f"  Samples: {len(ecg_signal):,}")
        print(f"  Sampling rate: {sampling_rate} Hz")
        print(f"  Duration: {duration_minutes:.1f} minutes ({duration_seconds:.1f} seconds)")
        print(f"  Signal range: {signal_min:.3f} to {signal_max:.3f}")
        print(f"  Mean: {signal_mean:.3f}, Std: {signal_std:.3f}")
        
        # Save extracted data as CSV for reference
        extracted_csv_path = os.path.join(output_dir, "f1y01_ecg_signal.csv")
        ecg_df = pd.DataFrame({
            'sample_number': f1y01_data['Sample'].values,
            'ecg_signal': ecg_signal,
            'time_seconds': np.arange(len(ecg_signal)) / sampling_rate
        })
        ecg_df.to_csv(extracted_csv_path, index=False)
        print(f"  Extracted ECG saved to: {extracted_csv_path}")
        
        return {
            'ecg_signal': ecg_signal,
            'sampling_rate': int(sampling_rate),
            'duration_minutes': duration_minutes,
            'output_dir': output_dir
        }
        
    except Exception as e:
        print(f"ERROR reading CSV file: {e}")
        return False

def convert_ecg_to_edf(ecg_data):
    """Convert the extracted ECG signal to EDF format"""
    
    print(f"\n{'='*60}")
    print("CONVERTING ECG SIGNAL TO EDF")
    print(f"{'='*60}")
    
    ecg_signal = ecg_data['ecg_signal']
    sampling_rate = ecg_data['sampling_rate']
    output_dir = ecg_data['output_dir']
    
    edf_path = os.path.join(output_dir, "f1y01_from_csv.edf")
    
    try:
        print(f"Creating EDF file: {os.path.basename(edf_path)}")
        
        # Try a simpler approach without forcing data record duration
        print(f"  Creating EDF file without custom data record duration...")
        
        # Create EDF+ file with default settings
        with pyedflib.EdfWriter(edf_path, 1, file_type=pyedflib.FILETYPE_EDFPLUS) as writer:
            
            # Set file header information
            writer.setPatientCode("F1Y01_CSV")
            writer.setPatientName("Fantasia_f1y01")
            writer.setTechnician("Kaggle_Dataset")
            writer.setEquipment("CSV_Conversion")
            
            # Set recording start time
            start_time = datetime(2000, 1, 1, 12, 0, 0)
            writer.setStartdatetime(start_time)
            
            # Configure signal parameters using the API that works with pyedflib 0.1.40
            signal_min = float(np.min(ecg_signal))
            signal_max = float(np.max(ecg_signal))
            
            # Ensure min and max are different (EDF requirement)
            if abs(signal_max - signal_min) < 1e-6:
                signal_max = signal_min + 1.0
            
            # Set signal parameters - let pyedflib handle data record duration automatically
            writer.setLabel(0, 'ECG')
            writer.setPhysicalDimension(0, 'mV')
            writer.setSamplefrequency(0, sampling_rate)  # Use actual sampling rate
            writer.setPhysicalMinimum(0, signal_min)
            writer.setPhysicalMaximum(0, signal_max)
            writer.setDigitalMinimum(0, -32768)
            writer.setDigitalMaximum(0, 32767)
            writer.setTransducer(0, 'ECG electrode')
            writer.setPrefilter(0, 'None')
            
            print(f"  Signal parameters:")
            print(f"    Sampling rate: {sampling_rate} Hz")
            print(f"    Physical range: {signal_min:.3f} to {signal_max:.3f}")
            print(f"    Digital range: -32768 to 32767")
            
            # Write all data at once - let pyedflib handle data record structure
            print(f"  Writing {len(ecg_signal):,} samples...")
            writer.writeSamples([ecg_signal])
            print(f"  Data written successfully")
        
        print(f"  EDF file created successfully!")
        
        # Verify the file
        file_size = os.path.getsize(edf_path) / (1024*1024)
        print(f"  File size: {file_size:.1f} MB")
        
        # Test reading the file back
        try:
            with pyedflib.EdfReader(edf_path) as reader:
                n_signals = reader.signals_in_file
                n_data_records = reader.datarecords_in_file
                samples_in_file = reader.getNSamples()[0]
                duration = reader.file_duration
                
                print(f"  Verification:")
                print(f"    Signals: {n_signals}")
                print(f"    Data records: {n_data_records}")
                print(f"    Total samples: {samples_in_file:,}")
                print(f"    Duration: {duration:.1f} seconds")
                
                # Read a small sample to verify data integrity
                test_data = reader.readSignal(0, start=0, n=min(1000, samples_in_file))
                print(f"    Sample data range: {np.min(test_data):.3f} to {np.max(test_data):.3f}")
                
        except Exception as e:
            print(f"  WARNING: Verification failed: {e}")
        
        return edf_path
        
    except Exception as e:
        print(f"ERROR creating EDF file: {e}")
        return False

def test_with_analyzer(edf_path):
    """Test the created EDF file with your analyzer"""
    
    print(f"\n{'='*60}")
    print("TESTING EDF WITH YOUR ANALYZER")
    print(f"{'='*60}")
    
    try:
        # Import your analyzer
        current_dir = os.path.dirname(os.path.abspath(__file__))
        import sys
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        from analyzer import CardiovascularAnalyzer
        
        print(f"Testing file: {os.path.basename(edf_path)}")
        
        # Initialize analyzer
        analyzer = CardiovascularAnalyzer()
        
        # Try to load the file
        channels_info = analyzer.load_file_and_detect_channels(edf_path)
        
        if not channels_info:
            print("ERROR: Analyzer could not load the EDF file")
            return False
        
        print(f"SUCCESS: Analyzer loaded the EDF file")
        print(f"Channels detected: {len(channels_info)}")
        
        for ch in channels_info:
            print(f"  Channel {ch['index']}: {ch['name']} ({ch['units']}) - {ch['likely_type']}")
        
        # Configure ECG channel
        analyzer.configure_channels(0, None)
        
        if not (hasattr(analyzer, 'ecg_data') and analyzer.ecg_data):
            print("ERROR: ECG data not accessible after configuration")
            return False
        
        # Get ECG data
        ecg_signal = analyzer.ecg_data['raw']
        sample_rate = analyzer.ecg_data['fs']
        
        print(f"ECG data loaded successfully:")
        print(f"  Samples: {len(ecg_signal):,}")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Duration: {len(ecg_signal)/sample_rate/60:.1f} minutes")
        print(f"  Signal range: {np.min(ecg_signal):.3f} to {np.max(ecg_signal):.3f}")
        
        return True
        
    except Exception as e:
        print(f"ERROR testing with analyzer: {e}")
        return False

def main():
    """Main conversion workflow"""
    
    print("Converting f1y01 ECG data from Kaggle CSV to EDF format...")
    
    # Step 1: Extract ECG data from CSV
    ecg_data = extract_f1y01_from_csv()
    if not ecg_data:
        print("Failed to extract ECG data from CSV")
        return
    
    # Step 2: Convert to EDF
    edf_path = convert_ecg_to_edf(ecg_data)
    if not edf_path:
        print("Failed to convert ECG to EDF")
        return
    
    # Step 3: Test with analyzer
    analyzer_success = test_with_analyzer(edf_path)
    
    # Summary
    print(f"\n{'='*60}")
    print("CONVERSION SUMMARY")
    print(f"{'='*60}")
    
    if analyzer_success:
        print("SUCCESS: f1y01 ECG data converted and validated")
        print(f"EDF file ready: {edf_path}")
        print(f"Duration: {ecg_data['duration_minutes']:.1f} minutes")
        print(f"Ready for validation against ground truth annotations")
    else:
        print("PARTIAL SUCCESS: EDF created but analyzer test failed")
        print(f"EDF file: {edf_path}")
        print("Manual inspection may be needed")

if __name__ == "__main__":
    main()