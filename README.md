# ChronOS - Open Source HRV Analysis Platform

A validated, web-accessible, peak detection and heart rate variability toolkit for cardiovascular research.

ChronOS was designed as a free, open-source research tool for physiological signals. This project began during my Ph.D. at Kent State University in an effort to overcome lack of user control in WinCPRS, a previously discontinued HRV and BRS analysis pipeline. Today, ChronOS has grown to include all commonly-implement time domain, frequency domain, and non-linear HRV metrics and includes scaffolding for BRS analysis. 

## Validation Status - Manuscript in Review

ChronOS has been rigorously validated through multiple approaches (complete validation methodology detailed in `/Validation Study/VALIDATION STUDY README`):

**Ground Truth Validation:** 99.99% sensitivity and positive predictive value on expert-annotated Fantasia database ECG data with F1 score of 1.000

**Criterion Validation:** Strong agreement with NeuroKit2 across 100 synthetic ECG recordings:
- Excellent agreement (ICC ≥ 0.90): Peak detection, mean RR, RMSSD, pNN50, SD1, HF power
- Good agreement (ICC 0.75-0.89): SDNN, SD2, LF power, LF/HF ratio, total power

**Methodological Compliance:** Follows 1996 Task Force guidelines for time domain, frequency domain, and nonlinear HRV metrics HRV

*Complete validation methodology and statistical results available in `/Validation Study/`*

## Features

### Core Analysis Capabilities
- **Time Domain:** RMSSD, SDNN, pNN50, mean RR, heart rate metrics
- **Frequency Domain:** VLF/LF/HF power analysis (4Hz interpolation, Welch method)
- **Nonlinear Metrics:** Poincaré plot analysis (SD1, SD2), sample entropy calculation
- **Baroreflex Sensitivity:** Sequence method and spectral transfer function analysis

### Technical Architecture
- **File Format Support:** Native ACQ (BIOPAC) and EDF (European Data Format) compatibility
- **Adaptive Peak Detection:** Signal-specific threshold calculation with physiological constraints
- **Scale Detection:** Automatic ECG amplitude detection (mV/μV/V) with conversion
- **Quality Assurance:** Pan-Tompkins validation and ectopic beat detection with manual review
- **Preprocessing:** Optional ECG bandpass filtering (0.5-40 Hz) for noise reduction

### Interface Options
- **Web GUI:** Streamlit-based interface with real-time parameter adjustment and interactive visualization
- **Programmatic API:** Direct Python access for batch processing and custom workflows
- **Interactive Analysis:** Time window selection, peak detection preview, and manual parameter optimization

## Installation

### Required Dependencies
```bash
pip install streamlit matplotlib plotly numpy pandas scipy bioread
```

### Optional Dependencies
```bash
# For EDF file support
pip install pyedflib
```

### Quick Setup
```bash
git clone https://github.com/apinzone/OS_HRV.git
cd OS_HRV
pip install -r requirements.txt
streamlit run simple_gui.py
```

### Cloud Access
ChronOS is available online through the Streamlit Community Cloud at chronoshrv.streamlit.app - no installation required. Access the full analysis platform directly in your browser.

## Usage

### Web Interface
```bash
streamlit run simple_gui.py
```

### Programmatic Analysis
```python
from analyzer import CardiovascularAnalyzer

# Initialize analyzer
analyzer = CardiovascularAnalyzer()

# Load file and detect available channels
channels = analyzer.load_file_and_detect_channels("your_file.edf")
print(f"Detected {len(channels)} channels")

# Configure analysis channels
analyzer.configure_channels(ecg_channel_idx=0, bp_channel_idx=1)

# Optional: Set analysis window for focused analysis
analyzer.set_time_window(start_time=60, end_time=300)  # Analyze seconds 60-300

# Optional: Configure preprocessing
analyzer.configure_preprocessing(enable_bandpass=True, lowcut=0.5, highcut=40.0)

# Run comprehensive analysis
analyzer.analyze_all()

# Extract results
results = analyzer.get_validation_metrics()
print(f"RMSSD: {results['rmssd_ms']:.1f} ms")
print(f"SDNN: {results['sdnn_ms']:.1f} ms")
print(f"LF/HF Ratio: {results['lf_hf_ratio']:.2f}")
```
    
### Data Requirements
- **Minimum:** One ECG channel for HRV analysis
- **Recommended:** ECG + Blood Pressure channels for comprehensive BRS analysis
- **Sampling Rate:** 256 Hz or higher recommended for optimal peak detection
- **Duration:** Minimum 2 minutes for reliable frequency domain analysis; 5+ minutes preferred

## Advanced Features

### Adaptive Peak Detection Algorithm
ChronOS employs signal-specific parameter calculation rather than fixed thresholds:

- **Height Threshold:** 0.55 × (signal maximum - baseline) 
- **Prominence Threshold:** 0.6 × height threshold 
- **Distance Constraint:** 250ms minimum 

### Quality Assurance Framework
- **Pan-Tompkins Validation:** Parallel QRS detection algorithm identifies potential missed peaks
- **Ectopic Beat Detection:** Physiological limits (300-2000ms) and statistical outlier detection
- **Manual Review Interface:** User oversight with correction capabilities for flagged intervals
- **Visual Inspection Tools:** Real-time ECG visualization with adjustable parameters

### Analysis Window Selection
- **Time Domain Windowing:** Focus analysis on specific recording segments
- **Interactive Selection:** Visual time window adjustment with immediate feedback
- **Artifact Avoidance:** Optional 5.0-40.0 Hz bandpass filter with real-time visualization of signal correction

## Repository Structure

```
├── analyzer.py              # Core cardiovascular analysis engine
├── simple_gui.py            # Streamlit web interface with real-time visualization
├── functions.py             # Signal processing utilities and HRV calculations
├── Validation Study/        # Complete validation analysis and results
│   ├── README.md            # Validation methodology and reproduction instructions
│   ├── batch_processor.py   # ChronOS vs NeuroKit2 validation script
│   ├── peak_diff.py         # Peak detection difference analyzer
│   ├── HRV_validity.R       # Statistical analysis and plot generation
│   └── fantasia_validation/ # Ground truth validation scripts
├── test_data/               # Sample physiological data files (including test files with noise and ectopics)
└── requirements.txt         # Python dependencies
```

## Validation Study

The `/Validation Study/` directory contains comprehensive validation materials:

### Reproducing Validation Results
```bash
cd "Validation Study"
python batch_processor.py    # Generate validation data
python peak_diff.py         # Analyze peak detection differences  
Rscript HRV_validity.R      # Statistical analysis and plots
```

### Key Validation Findings
- **Ground Truth Performance:** 8,708/8,709 R-peaks correctly detected on clinical ECG data
- **Peak Detection Superiority:** ChronOS avoided spurious detections in baseline noise where NeuroKit2 detected false positives
- **HRV Metric Agreement:** Strong correlation across time, frequency, and nonlinear domains
- **Method Transparency:** All validation code and data publicly available for independent verification


## Troubleshooting

### Installation Issues

**"EDF files require pyedflib" error:**
```bash
pip install pyedflib
```

**Import errors for bioread:**
```bash
pip install bioread
```

### Peak Detection Issues

**Too few peaks detected:**
- Use Preview function to visualize current detection
- Lower height threshold (try 0.3-0.4 × signal range)
- Try enabling sensitive detection settings (parameters will utilize a more permissive height and higher prominence; 250ms distance minimum remains)
- Check ECG scale detection (should show mV conversion)
- Consider enabling bandpass filter for noisy signals

**Too many peaks detected:**
- Increase height threshold (try 0.7-0.8 × signal range)
- Increase prominence requirement
- Check for baseline drift or movement artifacts
- Use ectopic beat detection to identify false positives

### File Format Issues

**Channel detection errors:**
- Verify file format is supported (ACQ or EDF)
- Check that file contains physiological data channels
- Use channel preview to identify ECG vs other signal types
- Ensure file is not corrupted or truncated

## Contributing

**Future Goals:**
- Including support for .csv files 
- Batch processing mode to analyze all files in a given directory, or multiple files simultaneously
- Intake of an R-R interval array, bypassing default peak detection and immmediately initiating HRV analysis 
- Validation of BRS functionality using laboratory-collected PPG and ECG data

We welcome contributions from the cardiovascular research community:

### Development Guidelines
1. Fork the repository and create a feature branch
2. Follow existing code style and documentation patterns
3. Add comprehensive docstrings to new functions
4. Test with both ACQ and EDF file formats
5. Validate results against NeuroKit2 when implementing new features
6. Update validation tests for significant algorithm changes

### Reporting Issues
- Provide example data files when possible
- Include complete error messages and system information
- Describe expected vs actual behavior
- Test with multiple file formats if applicable

### Feature Requests
- Describe the physiological or analytical motivation
- Provide literature references for new metrics or methods
- Consider computational feasibility for real-time analysis
- Ensure compatibility with existing validation framework

## Contact

**Anthony G. Pinzone, Ph.D., CSCS*D**  
Department of Kinesiology  
California State University of San Marcos  
Email: apinzone@csusm.edu or apinzone10@gmail.com  
GitHub: https://github.com/apinzone  
Website: https://apinzone.github.io/  
LinkedIn: https://www.linkedin.com/in/apinzone/  

## License
ChronOS and all source code are open-soruce and freely accessible under the MIT License - Free for academic, research, educational, and commercial use.
The full license can be viewed in our LICENSE.txt file.