# ChronOS - Open Source HRV Analysis Platform

A validated heart rate variability analysis toolkit for cardiovascular research and education.

ChronOS provides accessible HRV analysis for researchers, clinicians, and educators. Developed as a free alternative to commercial software, it delivers research-grade cardiovascular analysis with transparent, literature-based methodology.

## Validation

ChronOS has been validated against NeuroKit2 using 100 synthetic ECG recordings, demonstrating excellent agreement for core HRV metrics (ICC ≥ 0.90). The validation follows established guidelines with publication-quality statistical analysis.

*Validation study manuscript in preparation - see `/validation_study/` for complete analysis.*

## Features

**HRV Analysis**
- Time Domain: RMSSD, SDNN, pNN50, heart rate metrics
- Frequency Domain: LF/HF power analysis with Task Force compliance
- Nonlinear Metrics: Poincaré plot analysis (SD1, SD2)
- Sample Entropy: Signal complexity quantification

**Interface Options**
- GUI application using Streamlit for easy analysis
- Command-line interface for batch processing
- Flexible file support: `.acq` (BIOPAC) and `.edf` formats
- Manual parameter adjustment for optimal peak detection

**Methodology**
- Physics-based peak detection with physiological constraints
- Task Force compliant frequency domain analysis (4 Hz interpolation, VLF-excluded normalization)
- Literature-based calculations following established guidelines

## Quick Start

### Installation
```bash
pip install -r requirements.txt
streamlit run simple_gui.py
```

### Programmatic Usage
```python
from analyzer import CardiovascularAnalyzer

analyzer = CardiovascularAnalyzer()
analyzer.load_file_and_detect_channels("your_file.edf")
analyzer.configure_channels(ecg_channel_idx=0)
analyzer.analyze_all()
results = analyzer.get_validation_metrics()
```

## Repository Structure

```
├── analyzer.py           # Core HRV analysis engine
├── simple_gui.py         # GUI interface  
├── main.py              # Command-line interface
├── functions.py         # Supporting functions
└── validation_study/    # Validation analysis & results
```

## Requirements

- ECG signal (minimum one channel)
- File formats: `.acq` or `.edf`
- Recommended duration: ≥2 minutes for frequency domain analysis
- Optional: Blood pressure channel for baroreflex analysis

## Output Metrics

**Time Domain**
- RMSSD: Root mean square of successive differences
- SDNN: Standard deviation of normal-to-normal intervals  
- pNN50: Percentage of successive intervals >50ms

**Frequency Domain**  
- VLF Power: Very Low frequency power (0.003-0.04 Hz)
- LF Power: Low frequency power (0.04-0.15 Hz)
- HF Power: High frequency power (0.15-0.40 Hz)
- LF/HF Ratio: Autonomic balance measure

**Nonlinear Analysis**
- SD1/SD2: Poincaré plot short and long-term variability
- Sample Entropy: Signal regularity quantification

## Scientific Approach

ChronOS implements established cardiovascular analysis standards:
- Task Force 1996 guidelines for frequency domain analysis
- Validated peak detection algorithms
- Literature-based parameter selection
- Open methodology for reproducible research

The platform addresses cost barriers in cardiovascular research by providing free, validated analysis tools that match commercial software performance while maintaining full methodological transparency.

## Usage Applications

- Research laboratories requiring validated HRV analysis
- Educational settings for cardiovascular physiology instruction  
- Clinical applications needing reliable HRV assessment
- Student research projects involving physiological signal analysis

## Validation Study

See `/validation_study/` for:
- Complete validation methodology and statistical results
- Batch processing scripts for large datasets
- Comparison analysis with established software

## Contact

**Anthony G. Pinzone, Ph.D., CSCS*D**  
Email: apinzone10@gmail.com  
GitHub: https://github.com/apinzone  
Website: https://apinzone.github.io/

## License

MIT License - Free for academic, research, and educational use.
