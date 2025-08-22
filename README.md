# PhysioKit - Open Source Heart Rate Variability Analysis Platform

**A comprehensive, validated HRV analysis toolkit designed to democratize cardiovascular research**

PhysioKit provides accessible, cost-effective heart rate variability (HRV) analysis for researchers, clinicians, and educators. Developed as a free alternative to expensive commercial software, PhysioKit delivers research-grade cardiovascular analysis with user-friendly interfaces and transparent, literature-based methodology.

## 🏆 Validation & Scientific Rigor

PhysioKit has been rigorously validated against NeuroKit2 using 100 synthetic ECG recordings, demonstrating:
- **Excellent agreement** for core HRV metrics (ICC ≥ 0.90)
- **Publication-quality validation** following established guidelines
- **Literature-based methodology** adhering to Task Force standards
- **Open methodology** for reproducible research

*Validation study manuscript in preparation - see `/validation_study/` for complete analysis.*

## ✨ Key Features

### 📊 **Comprehensive HRV Analysis**
- **Time Domain**: RMSSD, SDNN, pNN50, and more
- **Frequency Domain**: LF/HF power 
- **Nonlinear Metrics**: Poincaré plot analysis (SD1, SD2)
- **Sample Entropy**: Signal complexity analysis

### 🔧 **User-Friendly Interface**
- **GUI Application**: Streamlit-based interface for easy analysis
- **Flexible File Support**: `.acq` (BIOPAC) and `.edf` files
- **Manual Parameter Control**: Adjustable peak detection for optimal results
- **Real-time Visualization**: Interactive plots and immediate feedback

### 🎯 **Research-Grade Accuracy**
- **Physics-based peak detection** with physiological constraints
- **Task Force compliant** frequency domain analysis
- **Validated algorithms** with excellent agreement to established software
- **Transparent methodology** following published guidelines

## 🚀 Quick Start

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Launch GUI interface
streamlit run simple_gui.py
```

### Programmatic Usage
```python
from analyzer import CardiovascularAnalyzer

# Initialize analyzer
analyzer = CardiovascularAnalyzer()

# Load and configure your data
analyzer.load_file_and_detect_channels("your_file.edf")
analyzer.configure_channels(ecg_channel_idx=0)

# Run complete analysis
analyzer.analyze_all()

# Get results
results = analyzer.get_validation_metrics()
```

## 📁 Repository Structure

```
├── analyzer.py           # Core HRV analysis engine
├── simple_gui.py         # Streamlit GUI interface  
├── main.py              # Command-line interface
├── functions.py         # Supporting analysis functions
├── README.md            # This file
├── requirements.txt     # Dependencies
└── validation_study/    # Validation analysis & results
```

## 📋 Input Requirements

- **ECG signal**: Minimum one ECG channel for HRV analysis
- **File formats**: `.acq` (BIOPAC) or `.edf` (European Data Format)
- **Optional**: Blood pressure channel for baroreflex sensitivity analysis
- **Duration**: Recommend ≥2 minutes for reliable frequency domain analysis

## 📈 Key Outputs

### Time Domain Metrics
- **RMSSD**: Root mean square of successive differences
- **SDNN**: Standard deviation of normal-to-normal intervals  
- **pNN50**: Percentage of successive NN intervals >50ms
- **Sample Entropy**: Signal regularity measure

### Frequency Domain Metrics  
- **LF Power**: Low frequency power (0.04-0.15 Hz)
- **HF Power**: High frequency power (0.15-0.40 Hz)
- **LF/HF Ratio**: Autonomic balance indicator
- **Normalized Units**: Task Force compliant calculations

### Nonlinear Analysis
- **SD1**: Short-term variability (Poincaré plot)
- **SD2**: Long-term variability (Poincaré plot)
- **Poincaré Visualization**: Interactive ellipse fitting

## 🎓 Educational Use

PhysioKit is designed for:
- **Research laboratories** needing cost-effective HRV analysis
- **Educational institutions** teaching cardiovascular physiology
- **Clinical applications** requiring validated HRV assessment
- **Student projects** with hands-on physiological signal analysis

## 🔬 Scientific Methodology

PhysioKit implements established HRV analysis guidelines:
- **Task Force 1996** standards for frequency domain analysis
- **4 Hz interpolation** for spectral analysis
- **VLF-excluded normalization** following literature recommendations
- **Sample standard deviation** for time domain calculations

## 🎯 Mission Statement

**Democratizing cardiovascular research through accessible, validated analysis tools.**

Many researchers face barriers accessing expensive HRV software. PhysioKit removes these obstacles by providing free, open-source analysis that matches commercial software performance while maintaining full methodological transparency.

## 🤝 Contributing

We welcome contributions! Whether you're:
- 🐛 **Reporting bugs**
- 💡 **Suggesting features** 
- 📖 **Improving documentation**
- 🔬 **Adding analysis methods**

See our contribution guidelines and open an issue or pull request.

## For Researchers & Developers
See `/validation_study/` for:
- Validation methodology and results
- Batch processing tools for large datasets
- Comparison scripts and statistical analysis

## 📧 Contact & Support

**Anthony G. Pinzone, Ph.D., CSCS*D**  
📧 [apinzone10@gmail.com](mailto:apinzone10@gmail.com)  
🔗 [GitHub](https://github.com/apinzone) | [Website](https://apinzone.github.io/) | [Google Scholar](https://scholar.google.com/citations?user=GMi1gHsAAAAJ&hl=en)

## 📄 License

Open-source under the MIT License. Free for academic, research, and educational use.

---
