# analyzer.py - Enhanced with ECG scale detection, flexible channel selection, and EDF support
import numpy as np
import math
from scipy.signal import find_peaks, welch, coherence, csd
from scipy.interpolate import interp1d
import bioread
import os
from functions import *  # All your existing functions

# EDF support with graceful fallback
try:
    import pyedflib
    EDF_AVAILABLE = True
except ImportError:
    EDF_AVAILABLE = False

class CardiovascularAnalyzer:
    def __init__(self):
        self.ecg_data = {}
        self.bp_data = {}
        self.results = {}
        self.time_window = None
        # Scale detection attributes
        self.ecg_scale = 'unknown'
        self.ecg_scale_factor = 1.0
        # Channel selection attributes
        self.file_channels = []
        self.file_object = None
        self.filepath = None
        self.file_type = None  # NEW: Track file type ('acq' or 'edf')
        self.ecg_channel = None
        self.bp_channel = None
        self.channels_configured = False

    def detect_ecg_scale(self, ecg_signal):
        """
        Detect if ECG is in mV or μV based on signal characteristics
        Returns: ('mV', factor) or ('μV', factor) or ('V', factor)
        """
        # Calculate signal statistics
        signal_std = np.std(ecg_signal)
        signal_max = np.max(np.abs(ecg_signal))
        signal_range = np.ptp(ecg_signal)  # peak-to-peak
        
        # Try basic peak detection with different thresholds
        signal_abs_max = np.max(np.abs(ecg_signal))
        
        # Test with mV-scale thresholds (typical: 0.5-5 mV range)
        if 0.1 <= signal_abs_max <= 10:
            # Signal is likely in mV range
            test_height = signal_abs_max * 0.3  # 30% of max as threshold
            test_prominence = signal_std * 0.5
            
            peaks_mv, _ = find_peaks(ecg_signal, 
                                   height=test_height, 
                                   distance=50,  # Lower distance for testing
                                   prominence=test_prominence)
            
            if len(peaks_mv) > 10:  # Found reasonable number of peaks
                return 'mV', 1.0
        
        # Test with μV-scale thresholds (typical: 500-5000 μV range)
        elif 100 <= signal_abs_max <= 10000:
            # Signal is likely in μV range
            test_height = signal_abs_max * 0.3
            test_prominence = signal_std * 0.5
            
            peaks_uv, _ = find_peaks(ecg_signal, 
                                   height=test_height, 
                                   distance=50,
                                   prominence=test_prominence)
            
            if len(peaks_uv) > 10:  # Found reasonable number of peaks
                return 'μV', 0.001  # Convert μV to mV
        
        # Test with V-scale thresholds (if signal is very small)
        elif 0.0001 <= signal_abs_max <= 0.01:
            # Signal might be in V range (very small values)
            test_height = signal_abs_max * 0.3
            test_prominence = signal_std * 0.5
            
            peaks_v, _ = find_peaks(ecg_signal, 
                                  height=test_height, 
                                  distance=50,
                                  prominence=test_prominence)
            
            if len(peaks_v) > 10:
                return 'V', 1000.0  # Convert V to mV
        
        # If auto-detection fails, make educated guess based on amplitude
        if signal_abs_max > 50:
            print(f"⚠️  ECG amplitude ({signal_abs_max:.1f}) suggests μV scale")
            return 'μV', 0.001
        elif signal_abs_max < 0.1:
            print(f"⚠️  ECG amplitude ({signal_abs_max:.4f}) suggests V scale")
            return 'V', 1000.0
        else:
            print(f"⚠️  ECG amplitude ({signal_abs_max:.3f}) suggests mV scale")
            return 'mV', 1.0

    def _guess_channel_type(self, channel_info):
        """Guess if channel is ECG, BP, or Other based on characteristics"""
        # For EDF files, channel_info is a dict with keys: name, units, data, sample_rate
        # For ACQ files, channel_info is the actual channel object
        
        if isinstance(channel_info, dict):
            # EDF format
            name = channel_info['name'].lower()
            units = channel_info['units'].lower()
            data = channel_info['data']
        else:
            # ACQ format (original code)
            name = getattr(channel_info, 'name', '').lower()
            units = getattr(channel_info, 'units', '').lower()
            data = channel_info.raw_data
        
        # Check name patterns
        if any(keyword in name for keyword in ['ecg', 'ekg', 'heart', 'cardiac', 'lead', 'ii', 'v1', 'v2']):
            return 'ECG (likely)'
        elif any(keyword in name for keyword in ['bp', 'blood', 'pressure', 'arterial', 'abp', 'systolic']):
            return 'BP (likely)'
        elif any(keyword in name for keyword in ['resp', 'breathing', 'airflow']):
            return 'Respiratory (maybe)'
        elif any(keyword in name for keyword in ['emg', 'muscle']):
            return 'EMG (maybe)'
        
        # Check units
        if any(unit in units for unit in ['mv', 'millivolt', 'volt', 'µv', 'uv', 'microvolt']):
            return 'ECG (likely)'
        elif any(unit in units for unit in ['mmhg', 'pressure', 'torr', 'kpa']):
            return 'BP (likely)'
        
        # Check data characteristics
        data_range = np.ptp(data)  # peak-to-peak
        data_mean = np.mean(np.abs(data))
        
        # ECG typically has smaller amplitude variations
        if data_range < 10 and data_mean < 5:
            return 'ECG (maybe)'
        # BP typically has higher values
        elif data_mean > 50 and data_range > 20:
            return 'BP (maybe)'
        
        return 'Unknown'

    def _load_edf_file(self, filepath):
            """Load EDF file and extract channel information - FIXED for current pyedflib API"""
            if not EDF_AVAILABLE:
                raise Exception("EDF support requires pyedflib. Install with: pip install pyedflib")
            
            try:
                # Open EDF file
                edf_file = pyedflib.EdfReader(filepath)
                
                # Get file info
                n_channels = edf_file.signals_in_file
                file_duration = edf_file.file_duration
                
                print(f"📁 EDF File Info: {n_channels} channels, {file_duration:.1f}s duration")
                
                # Extract channel information using robust method detection
                channels_info = []
                
                # Try to get all signal headers at once (newer API)
                try:
                    signal_headers = edf_file.getSignalHeaders()
                    use_headers = True
                    print("✅ Using getSignalHeaders() method")
                except AttributeError:
                    use_headers = False
                    print("⚠️  getSignalHeaders() not available, using individual methods")
                
                for i in range(n_channels):
                    if use_headers:
                        # Use signal headers (newer API)
                        header = signal_headers[i]
                        signal_label = header.get('label', f'Channel_{i}').strip()
                        sample_rate = header.get('sample_rate', 1000.0)
                        physical_dimension = header.get('dimension', 'Unknown').strip()
                        physical_min = header.get('physical_min', -1000.0)
                        physical_max = header.get('physical_max', 1000.0)
                    else:
                        # Use individual methods (try multiple naming conventions)
                        try:
                            # Try different method names that exist in various pyedflib versions
                            if hasattr(edf_file, 'signal_label'):
                                signal_label = edf_file.signal_label(i)
                            elif hasattr(edf_file, 'getSignalLabel'):
                                signal_label = edf_file.getSignalLabel(i)
                            else:
                                signal_label = f'Channel_{i}'
                            
                            if hasattr(edf_file, 'samplefrequency'):
                                sample_rate = edf_file.samplefrequency(i)
                            elif hasattr(edf_file, 'getSampleFreency'):  # Note: typo in some versions
                                sample_rate = edf_file.getSampleFreency(i)
                            elif hasattr(edf_file, 'getSampleFrequency'):
                                sample_rate = edf_file.getSampleFrequency(i)
                            else:
                                sample_rate = 1000.0
                            
                            if hasattr(edf_file, 'physical_dimension'):
                                physical_dimension = edf_file.physical_dimension(i)
                            elif hasattr(edf_file, 'getPhysicalDimension'):
                                physical_dimension = edf_file.getPhysicalDimension(i)
                            else:
                                physical_dimension = 'Unknown'
                            
                            if hasattr(edf_file, 'physical_min'):
                                physical_min = edf_file.physical_min(i)
                            elif hasattr(edf_file, 'getPhysicalMinimum'):
                                physical_min = edf_file.getPhysicalMinimum(i)
                            else:
                                physical_min = -1000.0
                            
                            if hasattr(edf_file, 'physical_max'):
                                physical_max = edf_file.physical_max(i)
                            elif hasattr(edf_file, 'getPhysicalMaximum'):
                                physical_max = edf_file.getPhysicalMaximum(i)
                            else:
                                physical_max = 1000.0
                                
                        except Exception as e:
                            print(f"⚠️  Using fallback values for channel {i}: {e}")
                            signal_label = f'Channel_{i}'
                            sample_rate = 1000.0
                            physical_dimension = 'Unknown'
                            physical_min = -1000.0
                            physical_max = 1000.0
                    
                    # Read signal data
                    signal_data = edf_file.readSignal(i)
                    
                    # Create time index
                    time_index = np.arange(len(signal_data)) / sample_rate
                    
                    # Clean up strings
                    signal_label = str(signal_label).strip()
                    physical_dimension = str(physical_dimension).strip()
                    
                    # Create channel info dictionary
                    channel_info = {
                        'index': i,
                        'name': signal_label,
                        'units': physical_dimension,
                        'samples': len(signal_data),
                        'duration': len(signal_data) / sample_rate,
                        'sample_rate': sample_rate,
                        'data_range': f"{np.min(signal_data):.3f} to {np.max(signal_data):.3f}",
                        'physical_min': physical_min,
                        'physical_max': physical_max,
                        'data': signal_data,
                        'time_index': time_index,
                        'likely_type': self._guess_channel_type({
                            'name': signal_label,
                            'units': physical_dimension,
                            'data': signal_data,
                            'sample_rate': sample_rate
                        })
                    }
                    
                    channels_info.append(channel_info)
                
                # Close the file
                edf_file.close()
                
                return channels_info
                
            except Exception as e:
                raise Exception(f"Failed to load EDF file: {str(e)}")

    def _load_acq_file(self, filepath):
        """Load ACQ file and extract channel information (original method)"""
        try:
            file = bioread.read_file(filepath)
            self.file_object = file  # Store for later use
            
            # Extract channel information
            channels_info = []
            for i, channel in enumerate(file.channels):
                channel_info = {
                    'index': i,
                    'name': getattr(channel, 'name', f'Channel {i}'),
                    'units': getattr(channel, 'units', 'Unknown'),
                    'samples': len(channel.raw_data),
                    'duration': max(channel.time_index) if len(channel.time_index) > 0 else 0,
                    'sample_rate': len(channel.raw_data) / max(channel.time_index) if max(channel.time_index) > 0 else 0,
                    'data_range': f"{np.min(channel.raw_data):.3f} to {np.max(channel.raw_data):.3f}",
                    'likely_type': self._guess_channel_type(channel),
                    'channel_object': channel  # Store original channel object for ACQ files
                }
                channels_info.append(channel_info)
            
            return channels_info
            
        except Exception as e:
            raise Exception(f"Failed to load ACQ file: {str(e)}")

    def load_file_and_detect_channels(self, filepath):
        """Load file and return available channel information for user selection"""
        try:
            # Determine file type
            file_ext = os.path.splitext(filepath)[1].lower()
            
            if file_ext == '.acq':
                self.file_type = 'acq'
                channels_info = self._load_acq_file(filepath)
            elif file_ext in ['.edf', '.bdf']:
                self.file_type = 'edf'
                channels_info = self._load_edf_file(filepath)
                # For EDF files, we store the channel info directly since we don't have a file object
                self.file_object = None
            else:
                raise Exception(f"Unsupported file format: {file_ext}")
            
            self.filepath = filepath
            self.file_channels = channels_info
            self.channels_configured = False
            self.analyzed = False
            self.preview_mode = False
            
            print(f"✅ {self.file_type.upper()} file loaded: {len(channels_info)} channels detected")
            for ch in channels_info:
                print(f"   Channel {ch['index']}: {ch['name']} ({ch['units']}) - {ch['likely_type']}")
            
            return channels_info
            
        except Exception as e:
            raise Exception(f"Failed to load file: {str(e)}")

    def configure_channels(self, ecg_channel_idx=None, bp_channel_idx=None):
        """Configure which channels to use for ECG and BP analysis"""
        if not self.file_channels:
            raise Exception("No file loaded. Load file first.")
        
        # Reset previous data
        self.ecg_data = {}
        self.bp_data = {}
        self.results = {}
        
        success_messages = []
        
        # Configure ECG channel
        if ecg_channel_idx is not None:
            try:
                if self.file_type == 'acq':
                    # ACQ file handling (original method)
                    ecg_channel = self.file_object.channels[ecg_channel_idx]
                    ECG_Data = ecg_channel.raw_data
                    Time = ecg_channel.time_index
                    ECG_fs = len(ECG_Data)/max(Time) if max(Time) > 0 else 1000
                    channel_name = getattr(ecg_channel, 'name', f'Channel {ecg_channel_idx}')
                    
                elif self.file_type == 'edf':
                    # EDF file handling
                    ecg_channel_info = self.file_channels[ecg_channel_idx]
                    ECG_Data = ecg_channel_info['data']
                    Time = ecg_channel_info['time_index']
                    ECG_fs = ecg_channel_info['sample_rate']
                    channel_name = ecg_channel_info['name']
                
                # Scale detection (same for both file types)
                self.ecg_scale, self.ecg_scale_factor = self.detect_ecg_scale(ECG_Data)
                ECG_Data_mV = ECG_Data * self.ecg_scale_factor
                
                self.ecg_data = {
                    'raw': ECG_Data_mV,
                    'raw_original': ECG_Data,
                    'time': Time,
                    'fs': ECG_fs,
                    'detected_scale': self.ecg_scale,
                    'scale_factor': self.ecg_scale_factor,
                    'channel_index': ecg_channel_idx,
                    'channel_name': channel_name
                }
                
                self.ecg_channel = ecg_channel_idx
                success_messages.append(f"✅ ECG: Channel {ecg_channel_idx} configured ({self.ecg_scale} detected)")
                
            except Exception as e:
                raise Exception(f"Failed to configure ECG channel {ecg_channel_idx}: {str(e)}")
        
        # Configure BP channel  
        if bp_channel_idx is not None:
            try:
                if self.file_type == 'acq':
                    # ACQ file handling (original method)
                    bp_channel = self.file_object.channels[bp_channel_idx]
                    BP_Data = bp_channel.raw_data
                    BP_Time = bp_channel.time_index
                    BP_fs = len(BP_Data)/max(BP_Time) if max(BP_Time) > 0 else 1000
                    channel_name = getattr(bp_channel, 'name', f'Channel {bp_channel_idx}')
                    
                elif self.file_type == 'edf':
                    # EDF file handling
                    bp_channel_info = self.file_channels[bp_channel_idx]
                    BP_Data = bp_channel_info['data']
                    BP_Time = bp_channel_info['time_index']
                    BP_fs = bp_channel_info['sample_rate']
                    channel_name = bp_channel_info['name']
                
                self.bp_data = {
                    'raw': BP_Data,
                    'time': BP_Time,
                    'fs': BP_fs,
                    'channel_index': bp_channel_idx,
                    'channel_name': channel_name
                }
                
                self.bp_channel = bp_channel_idx
                success_messages.append(f"✅ BP: Channel {bp_channel_idx} configured")
                
            except Exception as e:
                raise Exception(f"Failed to configure BP channel {bp_channel_idx}: {str(e)}")
        
        # Mark as configured if at least one channel is set
        if ecg_channel_idx is not None or bp_channel_idx is not None:
            self.channels_configured = True
        
        return success_messages

    def get_analysis_capabilities(self):
        """Return what types of analysis can be performed based on configured channels"""
        capabilities = {
            'time_domain_hrv': bool(self.ecg_data),
            'frequency_domain_hrv': bool(self.ecg_data),
            'brs_sequence': bool(self.ecg_data and self.bp_data),
            'brs_spectral': bool(self.ecg_data and self.bp_data),
            'bp_analysis': bool(self.bp_data)
        }
        
        return capabilities

    def get_scale_info(self):
        """Get information about detected ECG scale"""
        return {
            'detected_scale': self.ecg_scale,
            'scale_factor': self.ecg_scale_factor,
            'conversion_applied': self.ecg_scale != 'mV'
        }

    def set_time_window(self, start_time, end_time):
        """Set time window for analysis"""
        self.time_window = {
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time
        }
        
    def filter_data_by_time_window(self, data_array, time_array, start_time, end_time):
        """Filter data to only include points within the specified time window"""
        mask = (time_array >= start_time) & (time_array <= end_time)
        return data_array[mask], time_array[mask], mask

    def filter_peaks_by_time_window(self, peaks, time_array, start_time, end_time):
        """Filter peaks to only include those within the specified time window"""
        # Convert peak indices to times
        peak_times = np.array([time_array[p] for p in peaks if p < len(time_array)])
        
        # Find peaks within time window
        time_mask = (peak_times >= start_time) & (peak_times <= end_time)
        
        # Get the original peak indices that fall within the time window
        valid_peaks = [peaks[i] for i in range(len(peaks)) if i < len(time_mask) and time_mask[i]]
        
        return np.array(valid_peaks), time_mask

    def find_peaks_adaptive(self):
        """
        Adaptive peak detection that works with the detected scale
        """
        if not self.ecg_data:
            raise Exception("ECG channel not configured. Cannot perform peak detection.")
            
        # ECG peaks - use adaptive thresholds based on signal characteristics
        x = self.ecg_data['raw']  # Already converted to mV
        
        # Calculate adaptive thresholds
        signal_std = np.std(x)
        signal_max = np.max(np.abs(x))
        
        # Scale-aware thresholds (now that signal is in mV)
        height_threshold = max(0.3, signal_max * 0.4)  # At least 0.3 mV or 40% of max
        prominence_threshold = max(0.2, signal_std * 0.8)  # At least 0.2 mV or 80% of std
        
        print(f"🔧 ECG Peak Detection:")
        print(f"   Height threshold: {height_threshold:.3f} mV")
        print(f"   Prominence threshold: {prominence_threshold:.3f} mV")
        
        peaks, properties = find_peaks(x, 
                                     height=height_threshold, 
                                     threshold=None, 
                                     distance=100, 
                                     prominence=(prominence_threshold, None), 
                                     width=None, 
                                     wlen=None, 
                                     rel_height=None, 
                                     plateau_size=None)
        
        print(f"   Found {len(peaks)} ECG peaks")
        
        td_peaks = (peaks / self.ecg_data['fs'])
        RRDistance = distancefinder(td_peaks)  # Your function
        RRDistance_ms = [element * 1000 for element in RRDistance]
        
        self.ecg_data.update({
            'peaks': peaks,
            'td_peaks': td_peaks,
            'rr_intervals': RRDistance_ms,
            'peak_detection_method': 'adaptive'
        })
        
        # BP peaks - only if BP channel is configured
        if self.bp_data:
            BP = self.bp_data['raw']
            BP_peaks, _ = find_peaks(BP, height=110, threshold=None, distance=100, 
                                    prominence=(5,None), width=None, wlen=None, 
                                    rel_height=None, plateau_size=None)
            
            td_BP_peaks = (BP_peaks/self.bp_data['fs'])
            Systolic_Array = BP[BP_peaks]
            
            self.bp_data.update({
                'peaks': BP_peaks,
                'td_peaks': td_BP_peaks,
                'systolic': Systolic_Array
            })

    def find_peaks(self):
        """Peak detection with fallback to adaptive"""
        if not self.ecg_data:
            raise Exception("ECG channel not configured. Cannot perform peak detection.")
            
        try:
            # Try original method first (works if signal is in mV)
            x = self.ecg_data['raw']
            peaks, _ = find_peaks(x, height=0.8, threshold=None, distance=100, 
                                 prominence=(0.7,None), width=None, wlen=None, 
                                 rel_height=None, plateau_size=None)
            
            # Check if we found reasonable number of peaks
            if len(peaks) < 10:
                print("⚠️  Few peaks found with standard thresholds, trying adaptive detection...")
                self.find_peaks_adaptive()
                return
            
            # Original method worked
            td_peaks = (peaks / self.ecg_data['fs'])
            RRDistance = distancefinder(td_peaks)  # Your function
            RRDistance_ms = [element * 1000 for element in RRDistance]
            
            self.ecg_data.update({
                'peaks': peaks,
                'td_peaks': td_peaks,
                'rr_intervals': RRDistance_ms,
                'peak_detection_method': 'original'
            })
            
            print(f"✅ Original peak detection worked: {len(peaks)} peaks found")
            
        except Exception as e:
            print(f"⚠️  Original peak detection failed: {e}")
            print("Falling back to adaptive detection...")
            self.find_peaks_adaptive()
        
        # BP peaks - only if BP channel is configured
        if self.bp_data:
            BP = self.bp_data['raw']
            BP_peaks, _ = find_peaks(BP, height=110, threshold=None, distance=100, 
                                    prominence=(5,None), width=None, wlen=None, 
                                    rel_height=None, plateau_size=None)
            
            td_BP_peaks = (BP_peaks/self.bp_data['fs'])
            Systolic_Array = BP[BP_peaks]
            
            self.bp_data.update({
                'peaks': BP_peaks,
                'td_peaks': td_BP_peaks,
                'systolic': Systolic_Array
            })

    def find_peaks_with_params(self, ecg_height=0.8, ecg_distance=100, ecg_prominence=0.7,
                            bp_height=110, bp_distance=100, bp_prominence=5, use_adaptive=False):
        """Find peaks with custom parameters OR adaptive detection"""
        
        if not self.ecg_data:
            raise Exception("ECG channel not configured. Cannot perform peak detection.")
        
        if use_adaptive:
            # Use adaptive detection instead of manual parameters
            self.find_peaks_adaptive()
            return
        
        # ECG peaks with custom parameters
        x = self.ecg_data['raw']
        peaks, _ = find_peaks(x, 
                            height=ecg_height, 
                            threshold=None, 
                            distance=ecg_distance, 
                            prominence=(ecg_prominence, None), 
                            width=None, 
                            wlen=None, 
                            rel_height=None, 
                            plateau_size=None)
        
        td_peaks = (peaks / self.ecg_data['fs'])
        RRDistance = distancefinder(td_peaks)
        RRDistance_ms = [element * 1000 for element in RRDistance]
        
        self.ecg_data.update({
            'peaks': peaks,
            'td_peaks': td_peaks,
            'rr_intervals': RRDistance_ms,
            'peak_detection_method': 'manual_params'
        })
        
        # BP peaks with custom parameters - only if BP channel is configured
        if self.bp_data:
            BP = self.bp_data['raw']
            BP_peaks, _ = find_peaks(BP, 
                                    height=bp_height, 
                                    threshold=None, 
                                    distance=bp_distance, 
                                    prominence=(bp_prominence, None), 
                                    width=None, 
                                    wlen=None, 
                                    rel_height=None, 
                                    plateau_size=None)
            
            td_BP_peaks = (BP_peaks/self.bp_data['fs'])
            Systolic_Array = BP[BP_peaks]
            
            self.bp_data.update({
                'peaks': BP_peaks,
                'td_peaks': td_BP_peaks,
                'systolic': Systolic_Array
            })
        
        # Store the parameters used
        self.peak_detection_params = {
            'ecg_height': ecg_height,
            'ecg_distance': ecg_distance,
            'ecg_prominence': ecg_prominence,
            'bp_height': bp_height,
            'bp_distance': bp_distance,
            'bp_prominence': bp_prominence,
            'use_adaptive': use_adaptive
        }

    def analyze_with_current_peaks(self, time_window=None):
        """Run analysis with current peak detection results and optional time window"""
        if time_window:
            self.set_time_window(time_window['start_time'], time_window['end_time'])
        
        capabilities = self.get_analysis_capabilities()
        
        if capabilities['time_domain_hrv']:
            self.calculate_time_domain()
        if capabilities['frequency_domain_hrv']:
            self.calculate_frequency_domain()
        if capabilities['brs_sequence']:
            self.calculate_brs_sequence()
        if capabilities['brs_spectral']:
            self.calculate_brs_spectral()
        
    def get_windowed_data(self):
        """Get ECG and BP data filtered by time window if set"""
        if self.time_window is None:
            # No time window set, return all data
            return {
                'ecg_peaks': self.ecg_data.get('peaks', []),
                'ecg_td_peaks': self.ecg_data.get('td_peaks', []),
                'ecg_rr_intervals': self.ecg_data.get('rr_intervals', []),
                'bp_peaks': self.bp_data.get('peaks', []),
                'bp_td_peaks': self.bp_data.get('td_peaks', []),
                'bp_systolic': self.bp_data.get('systolic', [])
            }
        
        # Filter ECG data by time window
        start_time = self.time_window['start_time']
        end_time = self.time_window['end_time']
        
        result = {
            'ecg_peaks': [],
            'ecg_td_peaks': [],
            'ecg_rr_intervals': [],
            'bp_peaks': [],
            'bp_td_peaks': [],
            'bp_systolic': []
        }
        
        # Filter ECG peaks if available
        if self.ecg_data and 'peaks' in self.ecg_data:
            ecg_peaks_windowed, _ = self.filter_peaks_by_time_window(
                self.ecg_data['peaks'], self.ecg_data['time'], start_time, end_time
            )
            
            # Recalculate td_peaks and RR intervals for windowed data
            if len(ecg_peaks_windowed) > 1:
                ecg_td_peaks_windowed = ecg_peaks_windowed / self.ecg_data['fs']
                ecg_rr_windowed = distancefinder(ecg_td_peaks_windowed)
                ecg_rr_windowed_ms = [element * 1000 for element in ecg_rr_windowed]
            else:
                ecg_td_peaks_windowed = np.array([])
                ecg_rr_windowed_ms = []
            
            result.update({
                'ecg_peaks': ecg_peaks_windowed,
                'ecg_td_peaks': ecg_td_peaks_windowed,
                'ecg_rr_intervals': ecg_rr_windowed_ms
            })
        
        # Filter BP peaks if available
        if self.bp_data and 'peaks' in self.bp_data:
            bp_peaks_windowed, _ = self.filter_peaks_by_time_window(
                self.bp_data['peaks'], self.bp_data['time'], start_time, end_time
            )
            
            # Recalculate BP data for windowed peaks
            if len(bp_peaks_windowed) > 1:
                bp_td_peaks_windowed = bp_peaks_windowed / self.bp_data['fs']
                bp_systolic_windowed = self.bp_data['raw'][bp_peaks_windowed]
            else:
                bp_td_peaks_windowed = np.array([])
                bp_systolic_windowed = np.array([])
            
            result.update({
                'bp_peaks': bp_peaks_windowed,
                'bp_td_peaks': bp_td_peaks_windowed,
                'bp_systolic': bp_systolic_windowed
            })
        
        return result
        
    def calculate_time_domain(self):
        """Updated to use windowed data"""
        windowed_data = self.get_windowed_data()
        RRDistance_ms = windowed_data['ecg_rr_intervals']
        td_peaks = windowed_data['ecg_td_peaks']
        
        if len(RRDistance_ms) < 2:
            # Not enough data for analysis
            self.results['time_domain'] = {
                'error': 'Insufficient RR intervals in selected time window',
                'num_beats': len(RRDistance_ms)
            }
            return
        
        # Your exact calculations
        Successive_time_diff = SuccessiveDiff(RRDistance_ms)  # Your function
        AvgDiff = np.average(Successive_time_diff)
        SDNN = np.std(RRDistance_ms)
        SDSD = np.std(Successive_time_diff)
        NN50 = NNCounter(Successive_time_diff, 50)  # Your function
        pNN50 = (NN50/len(td_peaks))*100 if len(td_peaks) > 0 else 0
        RMSSD = np.sqrt(np.average(rms(Successive_time_diff)))  # Your function
        SD1 = np.sqrt(0.5*math.pow(SDSD,2))
        SD2 = np.sqrt((2*math.pow(SDNN,2) - (0.5*math.pow(SDSD,2))))
        S = math.pi * SD1 * SD2
        
        if len(td_peaks) > 0:
            Sampling_Time = max(td_peaks) - min(td_peaks) if self.time_window else max(td_peaks)
        else:
            Sampling_Time = 0
            
        Num_Beats = len(RRDistance_ms)
        HR = np.round(Num_Beats/(Sampling_Time/60),2) if Sampling_Time > 0 else 0
        
        # Sample entropy with your exact parameters
        m = 2
        r = 0.2 * np.std(RRDistance_ms)
        sampen = SampEn(RRDistance_ms, m, r) if len(RRDistance_ms) > m else 0  # Your function
        
        self.results['time_domain'] = {
            'num_beats': Num_Beats,
            'sampling_time': Sampling_Time,
            'hr': HR,
            'avg_diff': AvgDiff,
            'mean_rr': np.average(RRDistance_ms),
            'rmssd': RMSSD,
            'sdnn': SDNN,
            'sdsd': SDSD,
            'pnn50': pNN50,
            'sd1': SD1,
            'sd2': SD2,
            'sd1_sd2_ratio': SD1/SD2 if SD2 > 0 else 0,
            'ellipse_area': S,
            'sample_entropy': sampen
        }
        
    def calculate_frequency_domain(self):
        """Updated to use windowed data with proper time alignment"""
        windowed_data = self.get_windowed_data()
        td_peaks = windowed_data['ecg_td_peaks']
        RRDistance_ms = windowed_data['ecg_rr_intervals']
        
        if len(RRDistance_ms) < 10:  # Need minimum data for frequency analysis
            self.results['frequency_domain'] = {
                'error': 'Insufficient data for frequency domain analysis',
                'num_rr': len(RRDistance_ms)
            }
            return
        
        # IMPORTANT: Reset time to start from 0 for windowed data
        if len(td_peaks) > 1:
            # Shift time to start from 0
            time_offset = td_peaks[0]
            td_peaks_normalized = td_peaks - time_offset
            max_time = td_peaks_normalized[-1]
        else:
            self.results['frequency_domain'] = {
                'error': 'Insufficient peak data for frequency analysis',
                'num_peaks': len(td_peaks)
            }
            return
        
        # Your exact interpolation with normalized time
        interp_fs = 4
        uniform_time = np.arange(0, max_time, 1/interp_fs)
        
        if len(uniform_time) < 10:
            self.results['frequency_domain'] = {
                'error': 'Time window too short for frequency analysis',
                'window_duration': max_time
            }
            return
        
        # Use the normalized time for interpolation (excluding the last peak for RR intervals)
        if len(td_peaks_normalized) > len(RRDistance_ms):
            time_for_interp = td_peaks_normalized[:-1]  # Remove last peak to match RR intervals
        else:
            time_for_interp = td_peaks_normalized[:len(RRDistance_ms)]
            
        if len(time_for_interp) != len(RRDistance_ms):
            # Ensure arrays match
            min_len = min(len(time_for_interp), len(RRDistance_ms))
            time_for_interp = time_for_interp[:min_len]
            RRDistance_ms = RRDistance_ms[:min_len]
        
        if len(time_for_interp) < 3:
            self.results['frequency_domain'] = {
                'error': 'Insufficient data points for interpolation',
                'available_points': len(time_for_interp)
            }
            return
            
        rr_interp_func = interp1d(time_for_interp, RRDistance_ms, kind='cubic', fill_value="extrapolate")
        rr_fft = rr_interp_func(uniform_time)
        
        # Use nperseg that's appropriate for the data length
        nperseg = min(256, len(rr_fft)//4)
        if nperseg < 8:  # Minimum for meaningful frequency analysis
            self.results['frequency_domain'] = {
                'error': 'Window too short for reliable frequency analysis',
                'data_points': len(rr_fft),
                'min_required': 32
            }
            return
        
        frequencies, psd = welch(rr_fft, fs=interp_fs, nperseg=nperseg)
        
        # Your exact band definitions
        lf_band = (frequencies >= 0.04) & (frequencies < 0.15)
        hf_band = (frequencies >= 0.15) & (frequencies < 0.4)
        vlf_band = (frequencies >= 0.003) & (frequencies < 0.04)
        
        # Your exact power calculations (no unit conversion here to match original)
        vlf_power = np.trapezoid(psd[vlf_band], frequencies[vlf_band]) if np.any(vlf_band) else 0
        lf_power = np.trapezoid(psd[lf_band], frequencies[lf_band]) if np.any(lf_band) else 0
        hf_power = np.trapezoid(psd[hf_band], frequencies[hf_band]) if np.any(hf_band) else 0
        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
        total_power = lf_power + hf_power
        
        # Normalized units
        lf_nu = (lf_power/total_power) if total_power > 0 else 0
        hf_nu = (hf_power/total_power) if total_power > 0 else 0
        
        self.results['frequency_domain'] = {
            'vlf_power': vlf_power,
            'lf_power': lf_power,
            'hf_power': hf_power,
            'total_power': total_power,
            'lf_hf_ratio': lf_hf_ratio,
            'lf_nu': lf_nu,
            'hf_nu': hf_nu,
            'frequencies': frequencies,
            'psd': psd,
            'rr_fft': rr_fft,  # Store for BRS analysis
            'uniform_time': uniform_time,
            'time_offset': time_offset,  # Store offset for reference
            'window_duration': max_time
        }
        
    def calculate_brs_sequence(self):
        """Updated to use windowed data and store sequence details for plotting"""
        windowed_data = self.get_windowed_data()
        Systolic_Array = windowed_data['bp_systolic']
        RRDistance_ms = windowed_data['ecg_rr_intervals']
        td_BP_peaks = windowed_data['bp_td_peaks']
        td_peaks = windowed_data['ecg_td_peaks']
        
        if len(Systolic_Array) < 3 or len(RRDistance_ms) < 3:
            self.results['brs_sequence'] = {
                'error': 'Insufficient data for BRS sequence analysis',
                'bp_beats': len(Systolic_Array),
                'ecg_beats': len(RRDistance_ms)
            }
            return
        
        # Your exact function call with exact parameters
        results = full_sequence_brs(
            sbp=Systolic_Array,
            pi=RRDistance_ms,
            min_len=3,
            delay_range=(0, 4),
            r_threshold=0.8,
            thresh_sbp=1,
            thresh_pi=4
        )
        
        # Store additional data needed for plotting
        results['plotting_data'] = {
            'sbp': Systolic_Array,
            'rri': RRDistance_ms,
            'sap_times': td_BP_peaks,
            'rri_times': td_peaks[:-1] if len(td_peaks) > len(RRDistance_ms) else td_peaks[:len(RRDistance_ms)],
            'ramps': find_sap_ramps(Systolic_Array, min_len=3, thresh=1),  # Your function
            'best_delay': results.get('best_delay', 1),
            'r_threshold': 0.8,  # Use the same threshold as analysis
            'thresh_pi': 4
        }
        
        self.results['brs_sequence'] = results

    def calculate_brs_spectral(self):
        """Updated to exactly match main.py spectral calculations"""
        windowed_data = self.get_windowed_data()
        td_BP_peaks = windowed_data['bp_td_peaks']
        Systolic_Array = windowed_data['bp_systolic']
        
        if len(Systolic_Array) < 10:
            self.results['brs_spectral'] = {
                'error': 'Insufficient BP data for spectral BRS analysis',
                'bp_beats': len(Systolic_Array)
            }
            return
        
        # Check if we have frequency domain results
        if 'frequency_domain' not in self.results or 'error' in self.results['frequency_domain']:
            self.results['brs_spectral'] = {
                'error': 'Frequency domain analysis required for spectral BRS',
                'dependency': 'frequency_domain'
            }
            return
        
        # IMPORTANT: Normalize BP time to start from 0 (same as ECG)
        if len(td_BP_peaks) > 1:
            # Use the same time offset as ECG if available
            if 'time_offset' in self.results['frequency_domain']:
                time_offset = self.results['frequency_domain']['time_offset']
            else:
                time_offset = td_BP_peaks[0]
            
            td_BP_peaks_normalized = td_BP_peaks - time_offset
            max_bp_time = td_BP_peaks_normalized[-1]
        else:
            self.results['brs_spectral'] = {
                'error': 'Insufficient BP peaks for spectral analysis',
                'bp_peaks': len(td_BP_peaks)
            }
            return
        
        # Interpolation for BP - exactly matching main.py
        interp_fs = 4  # Fixed 4 Hz like main.py
        uniform_time_bp = np.arange(0, max_bp_time, 1/interp_fs)
        
        if len(uniform_time_bp) < 10:
            self.results['brs_spectral'] = {
                'error': 'BP time window too short for spectral analysis',
                'bp_duration': max_bp_time
            }
            return
            
        bp_interp_func = interp1d(td_BP_peaks_normalized, Systolic_Array, kind='cubic', fill_value="extrapolate")
        bp_fft = bp_interp_func(uniform_time_bp)
        
        # Get RR data from frequency domain results
        rr_fft = self.results['frequency_domain']['rr_fft']
        
        # Ensure RR and BP data have compatible lengths
        min_length = min(len(rr_fft), len(bp_fft))
        rr_fft_trimmed = rr_fft[:min_length]
        bp_fft_trimmed = bp_fft[:min_length]
        
        # Use FIXED nperseg=256 to match main.py exactly
        nperseg = 256
        
        # Only use adaptive sizing if data is genuinely too short
        if min_length < nperseg * 2:  # Need at least 2 windows for reliable estimates
            if min_length < 64:  # Absolute minimum for any spectral analysis
                self.results['brs_spectral'] = {
                    'error': 'Insufficient data length for reliable spectral BRS analysis',
                    'min_length': min_length,
                    'required_minimum': 64
                }
                return
            nperseg = min_length // 4  # Fallback only if absolutely necessary
        
        # Band definitions - exactly matching main.py
        lf_band = lambda freqs: (freqs >= 0.04) & (freqs < 0.15)
        hf_band = lambda freqs: (freqs >= 0.15) & (freqs < 0.4)
        
        # Calculate BP PSD - exactly matching main.py approach
        frequencies_bp, psd_bp = welch(bp_fft_trimmed, fs=interp_fs, nperseg=nperseg)
        lf_power_BP = np.trapezoid(psd_bp[lf_band(frequencies_bp)], frequencies_bp[lf_band(frequencies_bp)]) if np.any(lf_band(frequencies_bp)) else 0
        hf_power_BP = np.trapezoid(psd_bp[hf_band(frequencies_bp)], frequencies_bp[hf_band(frequencies_bp)]) if np.any(hf_band(frequencies_bp)) else 0
        
        # Coherence calculation - exactly matching main.py
        try:
            frequencies_coh, coherence_values = coherence(rr_fft_trimmed, bp_fft_trimmed, fs=interp_fs, nperseg=nperseg)
            lf_coherence = np.mean(coherence_values[lf_band(frequencies_coh)]) if np.any(lf_band(frequencies_coh)) else 0
            hf_coherence = np.mean(coherence_values[hf_band(frequencies_coh)]) if np.any(hf_band(frequencies_coh)) else 0
            
            # CSD calculation - EXACTLY matching main.py (note: bp_fft, rr_fft order like main.py)
            frequencies_csd, csd_bp_rr = csd(bp_fft_trimmed, rr_fft_trimmed, fs=interp_fs, nperseg=nperseg)
            _, psd_bp_auto = welch(bp_fft_trimmed, fs=interp_fs, nperseg=nperseg)
            
            # Transfer function - exactly matching main.py
            transfer_gain = np.abs(csd_bp_rr) / psd_bp_auto
            
            # BRS calculations - exactly matching main.py
            lf_band_tf = lf_band(frequencies_csd)
            hf_band_tf = hf_band(frequencies_csd)
            brs_lf_tf = np.mean(transfer_gain[lf_band_tf]) if np.any(lf_band_tf) else 0
            brs_hf_tf = np.mean(transfer_gain[hf_band_tf]) if np.any(hf_band_tf) else 0
            
            # Store results exactly matching your main.py variables
            self.results['brs_spectral'] = {
                'brs_lf_tf': brs_lf_tf,
                'brs_hf_tf': brs_hf_tf,
                'lf_coherence': lf_coherence,
                'hf_coherence': hf_coherence,
                'lf_power_bp': lf_power_BP,
                'hf_power_bp': hf_power_BP,
                'frequencies_coh': frequencies_coh,
                'coherence_values': coherence_values,
                'frequencies_csd': frequencies_csd,
                'csd_bp_rr': csd_bp_rr,  # Store the actual CSD for plotting
                'psd_bp_auto': psd_bp_auto,  # Store BP PSD for plotting
                'transfer_gain': transfer_gain,
                'frequencies_bp': frequencies_bp,  # Store BP frequencies for plotting
                'psd_bp': psd_bp,  # Store BP PSD for plotting
                'valid_lf': lf_coherence > 0.5,
                'valid_hf': hf_coherence > 0.5,
                'data_length_used': min_length,
                'nperseg_used': nperseg,  # Track what nperseg was actually used
                'bp_fft': bp_fft_trimmed,  # Store for plotting if needed
                'analysis_method': 'matches_main_py'  # Flag for verification
            }
            
        except Exception as e:
            self.results['brs_spectral'] = {
                'error': f'Spectral BRS calculation failed: {str(e)}',
                'rr_length': len(rr_fft),
                'bp_length': len(bp_fft),
                'min_length': min_length,
                'nperseg_attempted': nperseg
            }
        
    def analyze_all(self, time_window=None):
        """Run complete analysis based on available channels"""
        if not self.channels_configured:
            raise Exception("Channels not configured. Please configure channels first.")
        
        if time_window:
            self.set_time_window(time_window['start_time'], time_window['end_time'])
        
        capabilities = self.get_analysis_capabilities()
        
        # Only run analyses for which we have data
        if capabilities['time_domain_hrv'] or capabilities['brs_sequence']:
            self.find_peaks()
        
        if capabilities['time_domain_hrv']:
            self.calculate_time_domain()
            
        if capabilities['frequency_domain_hrv']:
            self.calculate_frequency_domain()
            
        if capabilities['brs_sequence']:
            self.calculate_brs_sequence()
            
        if capabilities['brs_spectral']:
            self.calculate_brs_spectral()
        
    def get_summary(self):
        """Enhanced summary including all your original metrics, with channel configuration info"""
        if not self.results:
            return "No analysis completed yet"
            
        summary = []
        
        # Channel configuration info
        summary.append("=== CHANNEL CONFIGURATION ===")
        summary.append(f"File Type: {self.file_type.upper()}")
        if self.ecg_data:
            summary.append(f"ECG Channel: {self.ecg_channel} ({self.ecg_data.get('channel_name', 'Unknown')})")
            summary.append(f"ECG Scale: {self.ecg_data.get('detected_scale', 'Unknown')}")
        else:
            summary.append("ECG Channel: Not configured")
            
        if self.bp_data:
            summary.append(f"BP Channel: {self.bp_channel} ({self.bp_data.get('channel_name', 'Unknown')})")
        else:
            summary.append("BP Channel: Not configured")
        summary.append("")
        
        # Time window information
        if self.time_window:
            tw = self.time_window
            summary.append("=== ANALYSIS WINDOW ===")
            summary.append(f"Time window: {tw['start_time']:.1f}s to {tw['end_time']:.1f}s")
            summary.append(f"Window duration: {tw['duration']:.1f} seconds ({tw['duration']/60:.1f} minutes)")
            summary.append("")
        
        # Time domain (exactly matching your original print statements)
        if 'time_domain' in self.results:
            td = self.results['time_domain']
            if 'error' in td:
                summary.append("=== TIME DOMAIN RESULTS ===")
                summary.append(f"ERROR: {td['error']}")
                summary.append("")
            else:
                summary.append("=== TIME DOMAIN RESULTS ===")
                summary.append(f"n = {td['num_beats']} beats are included for analysis")
                summary.append(f"The total sampling time is {td['sampling_time']:.3f} seconds")
                summary.append(f"The average heart rate during the sampling time is = {td['hr']} BPM")
                summary.append(f"the mean difference between successive R-R intervals is = {td['avg_diff']:.3f} ms")
                summary.append(f"The mean R-R Interval duration is {td['mean_rr']:.3f} ms")
                summary.append(f"pNN50 = {td['pnn50']:.3f} %")
                summary.append(f"RMSSD = {td['rmssd']:.3f} ms")
                summary.append(f"SDNN = {td['sdnn']:.3f} ms")
                summary.append(f"SDSD = {td['sdsd']:.3f} ms")
                summary.append(f"Sample Entropy = {td['sample_entropy']}")
                summary.append(f"SD1 = {td['sd1']:.3f} ms")
                summary.append(f"SD2 = {td['sd2']:.3f} ms")
                summary.append(f"SD1/SD2 = {td['sd1_sd2_ratio']:.3f}")
                summary.append(f"The area of the ellipse fitted over the Poincaré Plot (S) is {td['ellipse_area']:.3f} ms^2")
                summary.append("")
        
        # Blood pressure info
        windowed_data = self.get_windowed_data()
        if len(windowed_data['bp_systolic']) > 0:
            Avg_BP = np.round((np.average(windowed_data['bp_systolic'])),3)
            SD_BP = np.round((np.std(windowed_data['bp_systolic'])),3)
            summary.append("=== BLOOD PRESSURE ===")
            summary.append(f"The average systolic blood pressure during the sampling time is {Avg_BP} + - {SD_BP} mmHg")
            summary.append(f"{len(windowed_data['bp_systolic'])} pressure waves are included in the analysis")
            summary.append("")
        
        # Frequency domain (matching your original print format)
        if 'frequency_domain' in self.results:
            fd = self.results['frequency_domain']
            if 'error' in fd:
                summary.append("=== FREQUENCY DOMAIN ===")
                summary.append(f"ERROR: {fd['error']}")
                summary.append("")
            else:
                summary.append("=== FREQUENCY DOMAIN ===")
                summary.append(f"LF Power: {fd['lf_power']:.2f} ms²")
                summary.append(f"HF Power: {fd['hf_power']:.2f} ms²")
                summary.append(f"Total Power: {fd['total_power']:.2f} ms²")
                summary.append(f"LF/HF Ratio: {fd['lf_hf_ratio']:.2f}")
                summary.append(f"LF Power: {fd['lf_nu']:.2f} n.u.")
                summary.append(f"HF Power: {fd['hf_nu']:.2f} n.u.")
                summary.append("")
        
        # BRS sequence (matching your original print format)
        if 'brs_sequence' in self.results:
            brs = self.results['brs_sequence']
            if 'error' in brs:
                summary.append("=== BRS SEQUENCE METHOD ===")
                summary.append(f"ERROR: {brs['error']}")
                summary.append("")
            else:
                summary.append("=== BRS SEQUENCE METHOD ===")
                summary.append(f"BRS (mean): {brs['BRS_mean']:.2f} ms/mmHg")
                summary.append(f"BEI: {brs['BEI']:.2f}")
                summary.append(f"Valid BRS sequences: {brs['num_sequences']}")
                summary.append(f"Total SAP ramps: {brs['num_sbp_ramps']}")
                summary.append(f"Up BRS sequences: {brs['n_up']}")
                summary.append(f"Down BRS sequences: {brs['n_down']}")
                summary.append(f"Best delay: {brs['best_delay']} beats")
                summary.append("")
        
        # BRS spectral (matching your original print format)
        if 'brs_spectral' in self.results:
            brs_spec = self.results['brs_spectral']
            if 'error' in brs_spec:
                summary.append("=== BRS CROSS-SPECTRAL METHOD ===")
                summary.append(f"ERROR: {brs_spec['error']}")
                summary.append("")
            else:
                summary.append("=== BRS CROSS-SPECTRAL METHOD ===")
                if brs_spec['valid_lf']:
                    summary.append(f"Spectral BRS (Transfer Function, LF): {brs_spec['brs_lf_tf']:.3f} ms/mmHg (LF coherence OK)")
                else:
                    summary.append(f"Low coherence ({brs_spec['lf_coherence']:.3f}) – Spectral BRS not reliable")
                
                if brs_spec['valid_hf']:
                    summary.append(f"Spectral BRS (Transfer Function, HF): {brs_spec['brs_hf_tf']:.3f} ms/mmHg (HF coherence OK)")
                else:
                    summary.append(f"Low HF coherence ({brs_spec['hf_coherence']:.3f}) – HF BRS not reliable")
        
        return "\n".join(summary)