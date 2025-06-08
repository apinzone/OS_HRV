# Heart Rate Variability (HRV) Analysis Toolkit

This project provides a comprehensive analysis pipeline for assessing heart rate variability (HRV) and related cardiovascular metrics from ECG and blood pressure (BP) signals using Python. It was developed for educational and research purposes in exercise physiology and physiological signal processing.

## Overview

The script extracts time- and frequency-domain HRV metrics, generates common HRV visualizations (e.g., tachograms, Poincaré plots, power spectral density), and outputs relevant summary statistics. This project includes:

* ECG and BP signal parsing from `.acq` files (BIOPAC format)
* Peak detection for R-R intervals (ECG) and systolic BP
* Time-domain HRV metrics (e.g., RMSSD, SDNN, SDSD, pNN50)
* Frequency-domain HRV metrics (LF, HF, LF/HF ratio, total power)
* Poincaré plot with fitted ellipse and axis analysis
* RRI tachograms and annotated ECG segments
* High-resolution figure exports 

## Input Requirements

* `.acq` file recorded with a BIOPAC system
* At least one ECG channel (ideally 3-lead ECG)
* Optional: blood pressure waveform channel for systolic peak analysis

## Dependencies

```bash
pip install numpy scipy matplotlib seaborn bioread
```

## Core Files

* `main.py`: The core script to run ECG/BP signal analysis and generate figures
* `functions.py`: Utility functions for peak detection, RR interval analysis, entropy, and plotting

## Key Outputs

* **Time-Domain HRV Metrics**:

  * RMSSD: Root Mean Square of Successive Differences
  * SDNN: Standard deviation of all RR intervals
  * SDSD: Standard deviation of successive differences
  * pNN50: % of RR intervals that differ >50 ms
  * Sample Entropy (SampEn)

* **Frequency-Domain HRV Metrics**:

  * LF Power (ms² and n.u.): 0.04 to 0.15 Hz
  * HF Power (ms² and n.u.): 0.15 to 0.4 Hz
  * LF/HF Ratio: Balance between sympathetic and parasympathetic input
  * Total Power (ms²): LF + HF

* **Visualizations**:

  * ECG segment plots (labeled and unlabeled)
  * RRI Tachogram
  * Poincaré Plot with SD1/SD2 ellipse
  * Power Spectral Density plot for frequency domain HRV

All figures are exported as high-resolution `.tif` images under the `/figures` directory.

## Example Workflow

```python
# Load .acq file and extract ECG
file = bioread.read_file("SAMPLE.acq")
ECG_Data = file.channels[0].raw_data
Time = file.channels[0].time_index

# Detect R-peaks and compute RR intervals
peaks, _ = find_peaks(ECG_Data, ...)
RR = distancefinder(peaks / fs)

# Compute HRV metrics
RMSSD = ...
SDNN = ...

# Generate figures
plt.plot(...)
```

## Notes
* To run the appropriate sample script, please make sure that the file path specified for ECG SOURCE in line 7 is /data/SAMPLE.acq
* Frequency domain HRV metrics use Welch's method and cubic interpolation of RR intervals.
* All frequency bands follow standard guidelines (LF = 0.04–0.15 Hz, HF = 0.15–0.4 Hz).
* Sample entropy is computed with configurable parameters `m` and `r`.
* A future goal of the project is to incorporate sequencing methods calculate measurements of baroreflex sensitvity. We have extended the peak detector to SBP waves. 
  Feel free to comment back in this code if you'd like to run it 

## Author

Developed by Anthony G. Pinzone, Ph.D., CSCS\*D
[GitHub](https://github.com/apinzone) | [Website](https://apinzone.github.io/) | [Google Scholar](https://scholar.google.com/citations?user=GMi1gHsAAAAJ&hl=en)

## License

This project is open-source and available under the MIT License.

---

For inquiries, collaborations, or feedback, feel free to reach out via email at **[apinzone10@gmail.com](mailto:apinzone10@gmail.com)**.
