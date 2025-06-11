# streamlit_hrv_app_with_peak_controls.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tempfile
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from analyzer import CardiovascularAnalyzer  # Use your existing analyzer
from functions import *

# Configure the page
st.set_page_config(
    page_title="HRV Analysis Tool with Peak Controls", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🫀 Heart Rate Variability Analysis Tool")
st.markdown("Upload an ACQ file to analyze HRV and BRS metrics with adjustable peak detection and time window selection")

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = CardiovascularAnalyzer()
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'file_loaded' not in st.session_state:
    st.session_state.file_loaded = False
if 'preview_mode' not in st.session_state:
    st.session_state.preview_mode = False

# Sidebar for controls
with st.sidebar:
    st.header("📁 File Upload")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an ACQ file", 
        type="acq",
        help="Upload your ACQ file containing ECG and blood pressure data"
    )
    
    if uploaded_file is not None:
        if st.button("🔄 Load File", type="primary"):
            with st.spinner("Loading file..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".acq") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Load file only (don't analyze yet)
                    st.session_state.analyzer.load_file(tmp_file_path)
                    
                    # Clean up temp file
                    os.unlink(tmp_file_path)
                    
                    st.session_state.file_loaded = True
                    st.session_state.analyzed = False
                    st.session_state.preview_mode = False
                    st.success("✅ File loaded! Now adjust parameters below and preview results.")
                    
                except Exception as e:
                    st.error(f"❌ File loading failed: {str(e)}")
                    st.session_state.file_loaded = False

    # Show parameters only if file is loaded but not analyzed
    if st.session_state.file_loaded and not st.session_state.analyzed:
        
        # NEW: Time Window Selection
        st.header("⏱️ Analysis Time Window")
        st.markdown("**Select the time range for HRV analysis:**")
        
        # Get total recording duration
        if hasattr(st.session_state.analyzer, 'ecg_data') and 'time' in st.session_state.analyzer.ecg_data:
            max_time = max(st.session_state.analyzer.ecg_data['time'])
            max_time_min = max_time / 60
            
            st.write(f"📊 **Total recording duration:** {max_time:.1f} seconds ({max_time_min:.1f} minutes)")
            
            # Time window sliders
            col1, col2 = st.columns(2)
            
            with col1:
                start_time = st.slider(
                    "Start Time (seconds)", 
                    min_value=0.0, 
                    max_value=max_time-1, 
                    value=0.0, 
                    step=1.0,
                    help="Start of analysis window"
                )
            
            with col2:
                end_time = st.slider(
                    "End Time (seconds)", 
                    min_value=start_time+10, 
                    max_value=max_time, 
                    value=max_time, 
                    step=1.0,
                    help="End of analysis window"
                )
            
            # Calculate and display window duration
            window_duration = end_time - start_time
            window_duration_min = window_duration / 60
            
            # Show selected window info
            st.info(f"🎯 **Selected window:** {start_time:.0f}s to {end_time:.0f}s ({window_duration:.0f}s = {window_duration_min:.1f} min)")
            
            # Store time window in session state
            st.session_state.time_window = {
                'start_time': start_time,
                'end_time': end_time,
                'duration': window_duration
            }
        
        st.markdown("---")
        
        # Peak Detection Parameters
        st.header("🎛️ Peak Detection Parameters")
        
        # ECG Peak Detection Parameters
        st.subheader("⚡ ECG R-peak Detection")
        
        ecg_height = st.slider(
            "ECG Height Threshold", 
            min_value=0.1, max_value=2.0, value=0.8, step=0.1,
            help="Minimum height for R-peak detection"
        )
        
        ecg_distance = st.slider(
            "ECG Minimum Distance", 
            min_value=50, max_value=200, value=100, step=10,
            help="Minimum distance between R-peaks (samples)"
        )
        
        ecg_prominence = st.slider(
            "ECG Prominence", 
            min_value=0.1, max_value=1.5, value=0.7, step=0.1,
            help="Minimum prominence for R-peak detection"
        )
        
        # BP Peak Detection Parameters
        st.subheader("🩸 Blood Pressure Peak Detection")
        
        bp_height = st.slider(
            "BP Height Threshold", 
            min_value=80, max_value=150, value=110, step=5,
            help="Minimum height for systolic peak detection (mmHg)"
        )
        
        bp_distance = st.slider(
            "BP Minimum Distance", 
            min_value=50, max_value=200, value=100, step=10,
            help="Minimum distance between systolic peaks (samples)"
        )
        
        bp_prominence = st.slider(
            "BP Prominence", 
            min_value=1, max_value=10, value=5, step=1,
            help="Minimum prominence for systolic peak detection"
        )
        
        # Store parameters in session state
        st.session_state.peak_params = {
            'ecg_height': ecg_height,
            'ecg_distance': ecg_distance,
            'ecg_prominence': ecg_prominence,
            'bp_height': bp_height,
            'bp_distance': bp_distance,
            'bp_prominence': bp_prominence
        }
        
        # Preview button
        if st.button("🔍 Preview Peak Detection & Time Window", use_container_width=True):
            with st.spinner("Detecting peaks with new parameters and time window..."):
                try:
                    # Update peak detection with new parameters
                    st.session_state.analyzer.find_peaks_with_params(
                        ecg_height=ecg_height,
                        ecg_distance=ecg_distance,
                        ecg_prominence=ecg_prominence,
                        bp_height=bp_height,
                        bp_distance=bp_distance,
                        bp_prominence=bp_prominence
                    )
                    st.session_state.preview_mode = True
                    st.success("✅ Peak detection updated! Check the preview plots with your selected time window.")
                except Exception as e:
                    st.error(f"❌ Peak detection failed: {str(e)}")

    # Plot selection (only show if analyzed)
    if st.session_state.analyzed:
        st.header("📊 Plot Selection")
        
        # Interactive plot options
        st.subheader("🎯 Interactive Plots")
        interactive_plots = st.multiselect(
            "Interactive visualizations:",
            [
                "🔍 Interactive ECG (R-peaks)",
                "🔍 Interactive BP (Systolic peaks)", 
                "🔍 Interactive Tachogram"
            ]
        )
        
        # Static plot options
        st.subheader("📈 Static Analysis Plots")
        static_plots = st.multiselect(
            "Analysis results:",
            [
                "📊 Frequency Domain",
                "🔄 Poincaré Plot", 
                "🩺 BRS Sequence Analysis",
                "🩺 BRS Time Domain Visualization"
            ]
        )
        
        # Combine all selected plots
        all_selected_plots = interactive_plots + static_plots
        
        if st.button("🎨 Generate Selected Plots"):
            st.session_state.selected_plots = all_selected_plots

# HELPER FUNCTIONS FOR TIME WINDOW FILTERING
def filter_data_by_time_window(data_array, time_array, start_time, end_time):
    """Filter data to only include points within the specified time window"""
    mask = (time_array >= start_time) & (time_array <= end_time)
    return data_array[mask], time_array[mask], mask

def filter_peaks_by_time_window(peaks, time_array, start_time, end_time):
    """Filter peaks to only include those within the specified time window"""
    # Convert peak indices to times
    peak_times = time_array[peaks]
    # Find peaks within time window
    mask = (peak_times >= start_time) & (peak_times <= end_time)
    filtered_peaks = peaks[mask]
    
    # Adjust peak indices relative to the filtered data
    start_idx = np.where(time_array >= start_time)[0][0] if len(np.where(time_array >= start_time)[0]) > 0 else 0
    adjusted_peaks = filtered_peaks - start_idx
    
    return adjusted_peaks, mask

# MAIN CONTENT AREA
# Case 1: Analysis is complete - show results and plots
if st.session_state.analyzed:
    # Results summary
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📋 Analysis Results")
        
        # Show the time window used for analysis
        if 'time_window' in st.session_state:
            tw = st.session_state.time_window
            st.info(f"🎯 **Analysis window:** {tw['start_time']:.0f}s - {tw['end_time']:.0f}s ({tw['duration']:.0f}s)")
        
        # Get comprehensive results
        results = st.session_state.analyzer.get_summary()
        st.text_area("Results", results, height=600, disabled=True)
        
        # Download results button
        if st.download_button(
            label="⬇️ Download Results",
            data=results,
            file_name="hrv_analysis_results.txt",
            mime="text/plain"
        ):
            st.success("Results downloaded!")
    
    with col2:
        st.subheader("📊 Analysis Plots")
        
        # Display selected plots
        if 'selected_plots' in st.session_state:
            for plot_type in st.session_state.selected_plots:
                
                # Interactive ECG with R-peaks
                if "Interactive ECG" in plot_type:
                    st.subheader("🔍 Interactive ECG Signal with R-peaks")
                    
                    fig = go.Figure()
                    
                    # Add ECG trace
                    fig.add_trace(go.Scatter(
                        x=st.session_state.analyzer.ecg_data['time'],
                        y=st.session_state.analyzer.ecg_data['raw'],
                        mode='lines',
                        name='ECG',
                        line=dict(color='blue', width=1)
                    ))
                    
                    # Add R-peaks
                    peaks = st.session_state.analyzer.ecg_data['peaks']
                    time_data = st.session_state.analyzer.ecg_data['time']
                    ecg_data = st.session_state.analyzer.ecg_data['raw']
                    
                    if len(peaks) > 0:
                        valid_peaks = [p for p in peaks if p < len(time_data) and p < len(ecg_data)]
                        peak_times = [time_data[p] for p in valid_peaks]
                        peak_values = [ecg_data[p] for p in valid_peaks]
                        
                        fig.add_trace(go.Scatter(
                            x=peak_times,
                            y=peak_values,
                            mode='markers',
                            name=f'R-peaks (n={len(valid_peaks)})',
                            marker=dict(color='red', size=8)
                        ))
                    
                    # Highlight analysis window if it exists
                    if 'time_window' in st.session_state:
                        tw = st.session_state.time_window
                        fig.add_vrect(
                            x0=tw['start_time'], x1=tw['end_time'],
                            fillcolor="yellow", opacity=0.2,
                            annotation_text="Analysis Window", annotation_position="top left"
                        )
                    
                    fig.update_layout(
                        title='Interactive ECG Signal with R-peak Detection',
                        xaxis_title='Time (s)',
                        yaxis_title='ECG (mV)',
                        hovermode='x unified',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Interactive BP with systolic peaks
                elif "Interactive BP" in plot_type:
                    st.subheader("🔍 Interactive Blood Pressure with Systolic Peaks")
                    
                    fig = go.Figure()
                    
                    # Add BP trace
                    fig.add_trace(go.Scatter(
                        x=st.session_state.analyzer.bp_data['time'],
                        y=st.session_state.analyzer.bp_data['raw'],
                        mode='lines',
                        name='Blood Pressure',
                        line=dict(color='red', width=1)
                    ))
                    
                    # Add systolic peaks
                    bp_peaks = st.session_state.analyzer.bp_data['peaks']
                    time_data = st.session_state.analyzer.bp_data['time']
                    bp_data = st.session_state.analyzer.bp_data['raw']
                    
                    if len(bp_peaks) > 0:
                        valid_peaks = [p for p in bp_peaks if p < len(time_data) and p < len(bp_data)]
                        peak_times = [time_data[p] for p in valid_peaks]
                        peak_values = [bp_data[p] for p in valid_peaks]
                        
                        fig.add_trace(go.Scatter(
                            x=peak_times,
                            y=peak_values,
                            mode='markers',
                            name=f'Systolic Peaks (n={len(valid_peaks)})',
                            marker=dict(color='green', size=8)
                        ))
                    
                    # Highlight analysis window if it exists
                    if 'time_window' in st.session_state:
                        tw = st.session_state.time_window
                        fig.add_vrect(
                            x0=tw['start_time'], x1=tw['end_time'],
                            fillcolor="yellow", opacity=0.2,
                            annotation_text="Analysis Window", annotation_position="top left"
                        )
                    
                    # Add statistics
                    avg_systolic = np.mean(st.session_state.analyzer.bp_data['systolic'])
                    std_systolic = np.std(st.session_state.analyzer.bp_data['systolic'])
                    
                    fig.update_layout(
                        title=f'Interactive Blood Pressure Signal (Mean: {avg_systolic:.1f}±{std_systolic:.1f} mmHg)',
                        xaxis_title='Time (s)',
                        yaxis_title='Blood Pressure (mmHg)',
                        hovermode='x unified',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Interactive tachogram
                elif "Interactive Tachogram" in plot_type:
                    st.subheader("🔍 Interactive Heart Rate Variability Tachogram")
                    
                    fig = go.Figure()
                    
                    rr_intervals = st.session_state.analyzer.ecg_data['rr_intervals']
                    time_points = st.session_state.analyzer.ecg_data['td_peaks'][:-1]
                    
                    fig.add_trace(go.Scatter(
                        x=time_points,
                        y=rr_intervals,
                        mode='lines+markers',
                        name='RR Intervals',
                        line=dict(color='blue', width=1),
                        marker=dict(size=4)
                    ))
                    
                    # Highlight analysis window if it exists
                    if 'time_window' in st.session_state:
                        tw = st.session_state.time_window
                        fig.add_vrect(
                            x0=tw['start_time'], x1=tw['end_time'],
                            fillcolor="yellow", opacity=0.2,
                            annotation_text="Analysis Window", annotation_position="top left"
                        )
                    
                    # Add statistics
                    mean_rr = np.mean(rr_intervals)
                    std_rr = np.std(rr_intervals)
                    
                    fig.update_layout(
                        title=f'Interactive RR Interval Tachogram (Mean: {mean_rr:.1f}±{std_rr:.1f} ms)',
                        xaxis_title='Time (s)',
                        yaxis_title='RR Interval (ms)',
                        hovermode='x unified',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Frequency Domain
                elif "Frequency Domain" in plot_type:
                    st.subheader("📊 Frequency Domain Analysis")
                    
                    freq_data = st.session_state.analyzer.results['frequency_domain']
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    frequencies = freq_data['frequencies']
                    psd = freq_data['psd']
                    
                    ax.plot(frequencies, psd * 1e6, 'b-', linewidth=1.5, label='PSD')
                    
                    # Highlight frequency bands
                    lf_band = (frequencies >= 0.04) & (frequencies < 0.15)
                    hf_band = (frequencies >= 0.15) & (frequencies < 0.4)
                    vlf_band = (frequencies >= 0.003) & (frequencies < 0.04)
                    
                    ax.fill_between(frequencies[vlf_band], psd[vlf_band] * 1e6, 
                                   color='gray', alpha=0.3, label='VLF (0.003-0.04 Hz)')
                    ax.fill_between(frequencies[lf_band], psd[lf_band] * 1e6, 
                                   color='lightblue', alpha=0.5, label='LF (0.04-0.15 Hz)')
                    ax.fill_between(frequencies[hf_band], psd[hf_band] * 1e6, 
                                   color='lightcoral', alpha=0.5, label='HF (0.15-0.4 Hz)')
                    
                    ax.set_xlabel('Frequency (Hz)')
                    ax.set_ylabel('Power Spectral Density (ms²/Hz)')
                    ax.set_title('HRV Frequency Domain Analysis')
                    ax.set_xlim(0, 0.5)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Add power values as text
                    power_text = (f"VLF Power: {freq_data['vlf_power']:.2f} ms²\n"
                                f"LF Power: {freq_data['lf_power']:.2f} ms²\n"
                                f"HF Power: {freq_data['hf_power']:.2f} ms²\n"
                                f"Total Power: {freq_data['total_power']:.2f} ms²\n"
                                f"LF/HF Ratio: {freq_data['lf_hf_ratio']:.2f}\n"
                                f"LF n.u.: {freq_data['lf_nu']:.2f}\n"
                                f"HF n.u.: {freq_data['hf_nu']:.2f}")
                    
                    ax.text(0.98, 0.98, power_text, transform=ax.transAxes, 
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Poincaré Plot
                elif "Poincaré Plot" in plot_type:
                    st.subheader("🔄 Poincaré Plot")
                    
                    fig, ax = plt.subplots(figsize=(8, 8))
                    
                    # Use your exact implementation
                    RRDistance_ms = st.session_state.analyzer.ecg_data['rr_intervals']
                    RRIplusOne = Poincare(RRDistance_ms)  # Your function from functions.py
                    
                    # Your exact center calculation
                    EllipseCenterX = np.average(np.delete(RRDistance_ms, -1))
                    EllipseCenterY = np.average(RRIplusOne)
                    Center_coords = EllipseCenterX, EllipseCenterY
                    
                    # Your exact regression calculation
                    z = np.polyfit(np.delete(RRDistance_ms, -1), RRIplusOne, 1)
                    p = np.poly1d(z)
                    slope = z[0]
                    theta = np.degrees(np.arctan(slope))
                    theta_rad = np.radians(theta)
                    
                    # Scatter plot
                    ax.scatter(np.delete(RRDistance_ms, -1), RRIplusOne, alpha=0.6, s=20, color='blue')
                    
                    # Regression line
                    ax.plot(np.delete(RRDistance_ms, -1), p(np.delete(RRDistance_ms, -1)), color="red", linewidth=2, label='Regression Line')
                    
                    # Get SD1, SD2 from results
                    sd1 = st.session_state.analyzer.results['time_domain']['sd1']
                    sd2 = st.session_state.analyzer.results['time_domain']['sd2']
                    
                    # Ellipse
                    from matplotlib.patches import Ellipse
                    e = Ellipse(xy=Center_coords, width=sd2*2, height=sd1*2, angle=theta,
                                edgecolor='black', facecolor='none', linewidth=2)
                    ax.add_patch(e)
                    
                    # SD2 (major axis)
                    x_sd2 = [EllipseCenterX, EllipseCenterX + sd2 * np.cos(theta_rad)]
                    y_sd2 = [EllipseCenterY, EllipseCenterY + sd2 * np.sin(theta_rad)]
                    ax.plot(x_sd2, y_sd2, 'b-', linewidth=2, label='SD2 (Major Axis)')
                    
                    # SD1 (minor axis)
                    x_sd1 = [EllipseCenterX, EllipseCenterX - sd1 * np.sin(theta_rad)]
                    y_sd1 = [EllipseCenterY, EllipseCenterY + sd1 * np.cos(theta_rad)]
                    ax.plot(x_sd1, y_sd1, 'g-', linewidth=2, label='SD1 (Minor Axis)')
                    
                    ax.set_xlabel("RRI (ms)")
                    ax.set_ylabel("RRI + 1 (ms)")
                    ax.set_title('Poincaré Plot')
                    ax.grid(True, alpha=0.3)
                    
                    # Add SD1/SD2 text box
                    textstr = f'SD1 = {sd1:.1f} ms\nSD2 = {sd2:.1f} ms\nSD1/SD2 = {sd1/sd2:.3f}'
                    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # BRS Time Domain Visualization
                elif "BRS Time Domain Visualization" in plot_type:
                    st.subheader("🩺 Interactive BRS Time Domain Analysis")
                    
                    if 'brs_sequence' in st.session_state.analyzer.results:
                        brs_data = st.session_state.analyzer.results['brs_sequence']
                        
                        if 'error' in brs_data:
                            st.error(f"BRS Analysis Error: {brs_data['error']}")
                        elif 'plotting_data' in brs_data:
                            plot_data = brs_data['plotting_data']
                            
                            try:
                                # Create interactive subplot using Plotly
                                fig = make_subplots(
                                    rows=2, cols=1,
                                    subplot_titles=('Systolic Blood Pressure', 'RR Intervals'),
                                    vertical_spacing=0.1,
                                    shared_xaxes=True
                                )
                                
                                # Plot full SBP signal
                                fig.add_trace(go.Scatter(
                                    x=plot_data['sap_times'],
                                    y=plot_data['sbp'],
                                    mode='lines+markers',
                                    name='Systolic BP',
                                    line=dict(color='red', width=1),
                                    marker=dict(size=4),
                                    opacity=0.7
                                ), row=1, col=1)
                                
                                # Plot full RRI signal
                                fig.add_trace(go.Scatter(
                                    x=plot_data['rri_times'],
                                    y=plot_data['rri'],
                                    mode='lines+markers',
                                    name='RR Intervals',
                                    line=dict(color='blue', width=1),
                                    marker=dict(size=4),
                                    opacity=0.7
                                ), row=2, col=1)
                                
                                # Highlight valid BRS sequences
                                from scipy.stats import linregress
                                
                                sbp = plot_data['sbp']
                                rri = plot_data['rri']
                                ramps = plot_data['ramps']
                                delay = plot_data['best_delay']
                                r_threshold = plot_data['r_threshold']
                                thresh_pi = plot_data['thresh_pi']
                                sap_times = plot_data['sap_times']
                                rri_times = plot_data['rri_times']
                                
                                sequence_count = 0
                                valid_sequences = []
                                
                                for i, (start, end, direction) in enumerate(ramps):
                                    if end + delay >= len(rri):
                                        continue
                                    
                                    sbp_ramp = sbp[start:end + 1]
                                    rri_ramp = rri[start + delay:end + 1 + delay]
                                    
                                    if len(sbp_ramp) != len(rri_ramp):
                                        continue
                                    if np.any(np.abs(np.diff(rri_ramp)) < thresh_pi):
                                        continue
                                    
                                    slope, intercept, r_value, _, _ = linregress(sbp_ramp, rri_ramp)
                                    if abs(r_value) < r_threshold or slope <= 0:
                                        continue
                                    
                                    # This is a valid BRS sequence - highlight it
                                    sequence_count += 1
                                    color = 'green' if direction == 'up' else 'orange'
                                    
                                    # Highlight SBP portion
                                    if start < len(sap_times) and end < len(sap_times):
                                        fig.add_trace(go.Scatter(
                                            x=sap_times[start:end + 1],
                                            y=sbp[start:end + 1],
                                            mode='lines+markers',
                                            name=f'{direction.upper()} sequence' if sequence_count == 1 else None,
                                            line=dict(color=color, width=3),
                                            marker=dict(size=6, color=color),
                                            showlegend=(sequence_count == 1),  # Only show legend for first sequence of each type
                                            legendgroup=direction
                                        ), row=1, col=1)
                                    
                                    # Highlight RRI portion
                                    if start + delay < len(rri_times) and end + 1 + delay <= len(rri_times):
                                        fig.add_trace(go.Scatter(
                                            x=rri_times[start + delay:end + 1 + delay],
                                            y=rri[start + delay:end + 1 + delay],
                                            mode='lines+markers',
                                            name=None,
                                            line=dict(color=color, width=3),
                                            marker=dict(size=6, color=color),
                                            showlegend=False,
                                            legendgroup=direction
                                        ), row=2, col=1)
                                    
                                    valid_sequences.append({
                                        'sequence': sequence_count,
                                        'direction': direction,
                                        'slope': slope,
                                        'r_value': r_value,
                                        'start': start,
                                        'end': end
                                    })
                                
                                # Highlight analysis window if it exists
                                if 'time_window' in st.session_state:
                                    tw = st.session_state.time_window
                                    fig.add_vrect(
                                        x0=tw['start_time'], x1=tw['end_time'],
                                        fillcolor="yellow", opacity=0.2,
                                        annotation_text="Analysis Window", annotation_position="top left",
                                        row=1, col=1
                                    )
                                    fig.add_vrect(
                                        x0=tw['start_time'], x1=tw['end_time'],
                                        fillcolor="yellow", opacity=0.2,
                                        row=2, col=1
                                    )
                                
                                # Update layout
                                fig.update_xaxes(title_text="Time (s)", row=2, col=1)
                                fig.update_yaxes(title_text="Systolic BP (mmHg)", row=1, col=1)
                                fig.update_yaxes(title_text="RR Interval (ms)", row=2, col=1)
                                
                                fig.update_layout(
                                    title=f'Interactive BRS Time Domain Analysis - {sequence_count} Valid Sequences Found',
                                    height=700,
                                    hovermode='x unified',
                                    showlegend=True,
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                    )
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Add BRS summary info
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("BRS Mean", f"{brs_data.get('BRS_mean', 0):.2f} ms/mmHg")
                                with col2:
                                    st.metric("BEI", f"{brs_data.get('BEI', 0):.2f}")
                                with col3:
                                    st.metric("Valid Sequences", sequence_count)
                                with col4:
                                    st.metric("Best Delay", f"{plot_data['best_delay']} beats")
                                
                                # Show sequence details
                                if valid_sequences:
                                    st.subheader("📋 Valid BRS Sequence Details")
                                    
                                    sequence_df = pd.DataFrame(valid_sequences)
                                    sequence_df['slope'] = sequence_df['slope'].round(3)
                                    sequence_df['r_value'] = sequence_df['r_value'].round(3)
                                    
                                    st.dataframe(
                                        sequence_df,
                                        column_config={
                                            "sequence": "Seq #",
                                            "direction": "Direction", 
                                            "slope": "Slope (ms/mmHg)",
                                            "r_value": "Correlation (r)",
                                            "start": "Start Index",
                                            "end": "End Index"
                                        },
                                        hide_index=True,
                                        use_container_width=True
                                    )
                                
                                st.info(f"📊 **Analysis Parameters:** delay={plot_data['best_delay']} beats, r_threshold={plot_data['r_threshold']}, thresh_pi={plot_data['thresh_pi']} ms")
                                
                            except Exception as e:
                                st.error(f"Plotting error: {str(e)}")
                                st.write("Debug info:")
                                st.write(f"SBP points: {len(plot_data['sbp'])}")
                                st.write(f"RRI points: {len(plot_data['rri'])}")
                                st.write(f"SAP times: {len(plot_data['sap_times'])}")
                                st.write(f"RRI times: {len(plot_data['rri_times'])}")
                                
                        else:
                            st.warning("No plotting data available in BRS results")
                    else:
                        st.warning("No BRS sequence analysis results available")
                
                # BRS Sequence Analysis (existing)
                elif "BRS Sequence Analysis" in plot_type:
                    st.subheader("🩺 BRS Sequence Analysis Summary")
                    
                    if 'brs_sequence' in st.session_state.analyzer.results:
                        brs_data = st.session_state.analyzer.results['brs_sequence']
                        
                        if 'error' in brs_data:
                            st.error(f"BRS Analysis Error: {brs_data['error']}")
                        else:
                            # Display BRS metrics in a nice format
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("BRS Mean", f"{brs_data.get('BRS_mean', 0):.2f} ms/mmHg")
                                st.metric("BEI (Baroreflex Effectiveness Index)", f"{brs_data.get('BEI', 0):.2f}")
                                st.metric("Best Delay", f"{brs_data.get('best_delay', 0)} beats")
                            
                            with col2:
                                st.metric("Valid BRS Sequences", brs_data.get('num_sequences', 0))
                                st.metric("Total SAP Ramps", brs_data.get('num_sbp_ramps', 0))
                                st.metric("Up Sequences", brs_data.get('n_up', 0))
                                st.metric("Down Sequences", brs_data.get('n_down', 0))
                            
                            # Show analysis parameters
                            st.info("**Analysis Parameters:** min_len=3, delay_range=(0,4), r_threshold=0.8, thresh_sbp=1, thresh_pi=4")
                    else:
                        st.warning("No BRS sequence analysis results available")

# Case 2: File loaded and in preview mode - show peak detection preview with time window
elif st.session_state.file_loaded and st.session_state.preview_mode:
    st.subheader("🔍 Peak Detection Preview with Time Window Selection")
    st.markdown("**Review the detected peaks and time window below. Adjust parameters in the sidebar if needed.**")
    
    # Get time window info
    time_window_info = ""
    if 'time_window' in st.session_state:
        tw = st.session_state.time_window
        time_window_info = f" | **Selected Window:** {tw['start_time']:.0f}s - {tw['end_time']:.0f}s ({tw['duration']:.0f}s)"
    
    st.info(f"🎯 **Preview Mode** {time_window_info}")
    
    # Create two columns for ECG and BP previews
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**ECG Peak Detection Preview:**")
        
        # Show peak detection stats
        peaks = st.session_state.analyzer.ecg_data['peaks']
        time_data = st.session_state.analyzer.ecg_data['time']
        
        if len(peaks) > 1:
            peak_intervals = np.diff([time_data[p] for p in peaks if p < len(time_data)])
            avg_interval = np.mean(peak_intervals)
            hr_from_peaks = 60 / avg_interval
            
            # Metrics in a row
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Detected R-peaks", len(peaks))
            with metric_col2:
                st.metric("Estimated HR", f"{hr_from_peaks:.1f} BPM")
        
        # Interactive ECG preview - ENTIRE TIME SERIES with highlighted window
        fig = go.Figure()
        
        # Show entire ECG signal
        fig.add_trace(go.Scatter(
            x=time_data,
            y=st.session_state.analyzer.ecg_data['raw'],
            mode='lines',
            name='ECG (Full Recording)',
            line=dict(color='blue', width=0.8)
        ))
        
        # Add ALL detected peaks
        if len(peaks) > 0:
            valid_peaks = [p for p in peaks if p < len(time_data)]
            if len(valid_peaks) > 0:
                fig.add_trace(go.Scatter(
                    x=[time_data[p] for p in valid_peaks],
                    y=[st.session_state.analyzer.ecg_data['raw'][p] for p in valid_peaks],
                    mode='markers',
                    name=f'All R-peaks (n={len(valid_peaks)})',
                    marker=dict(color='red', size=4, symbol='circle')
                ))
        
        # Highlight the selected analysis window
        if 'time_window' in st.session_state:
            tw = st.session_state.time_window
            fig.add_vrect(
                x0=tw['start_time'], x1=tw['end_time'],
                fillcolor="yellow", opacity=0.3,
                annotation_text=f"Analysis Window ({tw['duration']:.0f}s)", 
                annotation_position="top left"
            )
        
        # Calculate recording duration
        duration_min = time_data[-1] / 60 if len(time_data) > 0 else 0
        
        fig.update_layout(
            title=f'ECG Peak Detection - Full Recording ({duration_min:.1f} minutes)',
            xaxis_title='Time (s)',
            yaxis_title='ECG (mV)',
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**BP Peak Detection Preview:**")
        
        # Show peak detection stats
        bp_peaks = st.session_state.analyzer.bp_data['peaks']
        bp_time_data = st.session_state.analyzer.bp_data['time']
        
        if len(bp_peaks) > 1:
            systolic_values = st.session_state.analyzer.bp_data['systolic']
            
            # Metrics in a row
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Systolic Peaks", len(bp_peaks))
            with metric_col2:
                st.metric("Mean Systolic", f"{np.mean(systolic_values):.1f} mmHg")
        
        # Interactive BP preview - ENTIRE TIME SERIES with highlighted window
        fig = go.Figure()
        
        # Show entire BP signal
        fig.add_trace(go.Scatter(
            x=bp_time_data,
            y=st.session_state.analyzer.bp_data['raw'],
            mode='lines',
            name='Blood Pressure (Full Recording)',
            line=dict(color='red', width=0.8)
        ))
        
        # Add ALL detected systolic peaks
        if len(bp_peaks) > 0:
            valid_peaks = [p for p in bp_peaks if p < len(bp_time_data)]
            if len(valid_peaks) > 0:
                fig.add_trace(go.Scatter(
                    x=[bp_time_data[p] for p in valid_peaks],
                    y=[st.session_state.analyzer.bp_data['raw'][p] for p in valid_peaks],
                    mode='markers',
                    name=f'All Systolic Peaks (n={len(valid_peaks)})',
                    marker=dict(color='green', size=4, symbol='circle')
                ))
        
        # Highlight the selected analysis window
        if 'time_window' in st.session_state:
            tw = st.session_state.time_window
            fig.add_vrect(
                x0=tw['start_time'], x1=tw['end_time'],
                fillcolor="yellow", opacity=0.3,
                annotation_text=f"Analysis Window ({tw['duration']:.0f}s)", 
                annotation_position="top left"
            )
        
        # Calculate recording duration
        duration_min = bp_time_data[-1] / 60 if len(bp_time_data) > 0 else 0
        
        fig.update_layout(
            title=f'BP Peak Detection - Full Recording ({duration_min:.1f} minutes)',
            xaxis_title='Time (s)',
            yaxis_title='Blood Pressure (mmHg)',
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Add RR Interval Tachogram Preview with time window highlighting
    st.markdown("---")
    st.subheader("📈 RR Interval Tachogram Preview")
    st.write("**Heart Rate Variability across the entire recording:**")
    
    # Calculate RR intervals and time points
    if len(peaks) > 1:
        rr_intervals = st.session_state.analyzer.ecg_data['rr_intervals']
        rr_time_points = st.session_state.analyzer.ecg_data['td_peaks'][:-1]  # Remove last point
        
        # Create tachogram
        fig_tacho = go.Figure()
        
        fig_tacho.add_trace(go.Scatter(
            x=rr_time_points,
            y=rr_intervals,
            mode='lines+markers',
            name='RR Intervals',
            line=dict(color='purple', width=1),
            marker=dict(size=3, color='purple')
        ))
        
        # Highlight the selected analysis window
        if 'time_window' in st.session_state:
            tw = st.session_state.time_window
            fig_tacho.add_vrect(
                x0=tw['start_time'], x1=tw['end_time'],
                fillcolor="yellow", opacity=0.3,
                annotation_text=f"Analysis Window ({tw['duration']:.0f}s)", 
                annotation_position="top left"
            )
        
        # Add statistics
        mean_rr = np.mean(rr_intervals)
        std_rr = np.std(rr_intervals)
        min_rr = np.min(rr_intervals)
        max_rr = np.max(rr_intervals)
        
        # Add horizontal lines for mean ± std
        fig_tacho.add_hline(y=mean_rr, line_dash="dash", line_color="green", 
                           annotation_text=f"Mean: {mean_rr:.1f} ms")
        fig_tacho.add_hline(y=mean_rr + std_rr, line_dash="dot", line_color="orange", 
                           annotation_text=f"+1 SD: {mean_rr + std_rr:.1f} ms")
        fig_tacho.add_hline(y=mean_rr - std_rr, line_dash="dot", line_color="orange", 
                           annotation_text=f"-1 SD: {mean_rr - std_rr:.1f} ms")
        
        fig_tacho.update_layout(
            title=f'RR Interval Tachogram - Full Recording (Range: {min_rr:.0f}-{max_rr:.0f} ms)',
            xaxis_title='Time (s)',
            yaxis_title='RR Interval (ms)',
            height=400,
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig_tacho, use_container_width=True)
        
        # Add summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean RR", f"{mean_rr:.1f} ms")
        with col2:
            st.metric("RR Std Dev", f"{std_rr:.1f} ms")
        with col3:
            st.metric("RR Range", f"{max_rr - min_rr:.0f} ms")
        with col4:
            cv_rr = (std_rr / mean_rr) * 100
            st.metric("RR CV%", f"{cv_rr:.1f}%")
    
    else:
        st.warning("⚠️ Not enough R-peaks detected for RR interval analysis. Try adjusting ECG parameters.")
    
    # Show window-specific statistics if time window is selected
    if 'time_window' in st.session_state and len(peaks) > 1:
        st.markdown("---")
        st.subheader("🎯 Analysis Window Statistics")
        
        tw = st.session_state.time_window
        
        # Convert to numpy arrays to ensure proper indexing
        rr_intervals_np = np.array(rr_intervals)
        rr_time_points_np = np.array(rr_time_points)
        
        # Filter RR intervals to analysis window
        window_mask = (rr_time_points_np >= tw['start_time']) & (rr_time_points_np <= tw['end_time'])
        window_rr = rr_intervals_np[window_mask]
        
        if len(window_rr) > 0:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Window RR Count", len(window_rr))
            with col2:
                st.metric("Window Mean RR", f"{np.mean(window_rr):.1f} ms")
            with col3:
                st.metric("Window RR Std", f"{np.std(window_rr):.1f} ms")
            with col4:
                window_cv = (np.std(window_rr) / np.mean(window_rr)) * 100
                st.metric("Window RR CV%", f"{window_cv:.1f}%")
            
            st.info(f"ℹ️ Analysis will use {len(window_rr)} RR intervals from the selected {tw['duration']:.0f}-second window.")
        else:
            st.warning("⚠️ No RR intervals found in the selected time window. Try adjusting the window or peak detection parameters.")
    
    # Action buttons
    st.markdown("---")
    st.markdown("**What would you like to do next?**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Accept & Run Full Analysis", type="primary", use_container_width=True):
            with st.spinner("Running complete cardiovascular analysis on selected time window..."):
                try:
                    # Pass the time window to the analyzer
                    time_window = st.session_state.get('time_window', None)
                    st.session_state.analyzer.analyze_with_current_peaks(time_window)
                    st.session_state.analyzed = True
                    st.session_state.preview_mode = False
                    st.success("✅ Complete analysis finished!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
    
    with col2:
        if st.button("🔄 Adjust Parameters", use_container_width=True):
            st.session_state.preview_mode = False
            st.info("👈 Adjust parameters or time window in the sidebar and click 'Preview' again")
            st.rerun()
    
    with col3:
        if st.button("📊 Use Default Settings", use_container_width=True):
            with st.spinner("Analyzing with default parameters..."):
                try:
                    st.session_state.analyzer.find_peaks()  # Original method
                    # Pass the time window to the analyzer even with default settings
                    time_window = st.session_state.get('time_window', None)
                    st.session_state.analyzer.analyze_with_current_peaks(time_window)
                    st.session_state.analyzed = True
                    st.session_state.preview_mode = False
                    st.success("✅ Analysis with defaults completed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")

# Case 3: No file loaded or other states - show instructions
else:
    # Instructions when no file is loaded
    st.info("👆 Please upload an ACQ file using the sidebar to begin analysis")
    
    # Show information about the tool
    st.markdown("""
    ## About This HRV Analysis Tool
    
    This tool provides comprehensive cardiovascular analysis with **adjustable peak detection parameters** and **time window selection**.
    
    ### 🎛️ New Features:
    
    **Time Window Selection:**
    - Select specific time segments for analysis (e.g., first 5 minutes)
    - Visual preview of selected window on full time series
    - Window-specific statistics before analysis
    - Zoom into stable recording periods
    
    **Adjustable Peak Detection:**
    - Customize ECG R-peak detection parameters
    - Adjust blood pressure systolic peak detection
    - Real-time preview of detected peaks
    - Fine-tune before running full analysis
    
    **Interactive Preview:**
    - See entire time series with highlighted analysis window
    - Validate peak detection quality across full recording
    - Preview RR intervals within selected window
    - Adjust parameters and preview again
    
    ### 📊 Analysis Features:
    
    **Time Domain HRV:**
    - RMSSD, SDNN, SDSD, pNN50
    - Poincaré plot metrics (SD1, SD2)
    - Sample entropy
    
    **Frequency Domain HRV:**
    - VLF, LF, HF power analysis
    - LF/HF ratio and normalized units
    - Power spectral density visualization
    
    **Baroreflex Sensitivity:**
    - Sequence method (time domain)
    - Cross-spectral analysis (frequency domain)
    - Transfer function and coherence
    
    ### 🔧 How to Use:
    
    1. **Upload** your ACQ file
    2. **Select time window** for analysis (start/end times)
    3. **Adjust** peak detection parameters using sliders
    4. **Preview** peak detection results and time window
    5. **Accept** parameters and run full analysis
    6. **Select** and generate plots
    7. **Download** your complete results
    
    ### ⏱️ Time Window Selection:
    
    **Why use time windows?**
    - Focus on stable recording periods
    - Exclude artifacts or movement at beginning/end
    - Analyze specific conditions or time periods
    - Compare different segments of same recording
    
    **Window Selection Tips:**
    - Start after initial settling period (first 30-60s)
    - Choose stable periods without artifacts
    - Minimum 2-3 minutes for reliable HRV analysis
    - 5+ minutes recommended for frequency domain
    
    ### 🎯 Peak Detection Parameters:
    
    **ECG Parameters:**
    - **Height**: Minimum amplitude for R-peak detection
    - **Distance**: Minimum samples between consecutive peaks
    - **Prominence**: How much peaks stand out from baseline
    
    **Blood Pressure Parameters:**
    - **Height**: Minimum systolic pressure threshold
    - **Distance**: Minimum samples between systolic peaks
    - **Prominence**: Peak prominence above surrounding signal
    
    ### 📈 Enhanced Visualizations:
    
    - **Full time series ECG and BP signals** with analysis window highlighted
    - **Complete RR interval tachogram** with window overlay
    - **Zoomable plots** for detailed inspection
    - **Window-specific statistics** before analysis
    - **Peak detection validation** across entire recording
    
    ### 🩺 Advanced Analysis:
    
    - **Frequency domain analysis** with band highlighting
    - **Poincaré plot** with regression line and ellipse
    - **Baroreflex sensitivity** using multiple methods
    - **Transfer function and coherence** analysis
    
    **Quality Control:**
    - Real-time peak count and heart rate estimation
    - Visual validation of detection accuracy across entire recording
    - Window-specific RR interval statistics
    - Parameter optimization before analysis
    - Comprehensive statistical summaries
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit • Interactive Peak Detection • Time Window Selection • Full Time Series Preview")