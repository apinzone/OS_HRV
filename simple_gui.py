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
st.markdown("Upload an ACQ file to analyze HRV and BRS metrics with adjustable peak detection")

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
                    st.success("✅ File loaded! Now adjust peak detection parameters below and preview results.")
                    
                except Exception as e:
                    st.error(f"❌ File loading failed: {str(e)}")
                    st.session_state.file_loaded = False

    # Peak Detection Parameters (only show if file is loaded but not analyzed)
    if st.session_state.file_loaded and not st.session_state.analyzed:
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
        if st.button("🔍 Preview Peak Detection", use_container_width=True):
            with st.spinner("Detecting peaks with new parameters..."):
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
                    st.success("✅ Peak detection updated! Check the preview plots.")
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
                "🔍 Interactive Combined Signals",
                "🔍 Interactive Tachogram",
                "🔍 Peak Detection Validation"
            ]
        )
        
        # Static plot options
        st.subheader("📈 Static Analysis Plots")
        static_plots = st.multiselect(
            "Analysis results:",
            [
                "📊 Frequency Domain",
                "🔄 Poincaré Plot", 
                "🩺 BRS Analysis",
                "🌊 Cross-Spectral BRS",
                "📈 Transfer Function & Coherence"
            ]
        )
        
        # Combine all selected plots
        all_selected_plots = interactive_plots + static_plots
        
        if st.button("🎨 Generate Selected Plots"):
            st.session_state.selected_plots = all_selected_plots

# MAIN CONTENT AREA
# Case 1: Analysis is complete - show results and plots
if st.session_state.analyzed:
    # Results summary
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📋 Analysis Results")
        
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
                
                # Interactive combined signals
                elif "Interactive Combined" in plot_type:
                    st.subheader("🔍 Interactive Combined ECG and BP Signals")
                    
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('ECG Signal', 'Blood Pressure Signal'),
                        vertical_spacing=0.1
                    )
                    
                    # ECG subplot
                    fig.add_trace(go.Scatter(
                        x=st.session_state.analyzer.ecg_data['time'],
                        y=st.session_state.analyzer.ecg_data['raw'],
                        mode='lines',
                        name='ECG',
                        line=dict(color='blue', width=1)
                    ), row=1, col=1)
                    
                    # BP subplot
                    fig.add_trace(go.Scatter(
                        x=st.session_state.analyzer.bp_data['time'],
                        y=st.session_state.analyzer.bp_data['raw'],
                        mode='lines',
                        name='Blood Pressure',
                        line=dict(color='red', width=1)
                    ), row=2, col=1)
                    
                    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
                    fig.update_yaxes(title_text="ECG (mV)", row=1, col=1)
                    fig.update_yaxes(title_text="BP (mmHg)", row=2, col=1)
                    
                    fig.update_layout(
                        title='Interactive Combined ECG and Blood Pressure Signals',
                        height=600,
                        hovermode='x unified'
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

# Case 2: File loaded and in preview mode - show peak detection preview
elif st.session_state.file_loaded and st.session_state.preview_mode:
    st.subheader("🔍 Peak Detection Preview")
    st.markdown("**Review the detected peaks below. Adjust parameters in the sidebar if needed.**")
    
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
        
        # Interactive ECG preview - ENTIRE TIME SERIES
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
        
        # Interactive BP preview - ENTIRE TIME SERIES
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
    
    # Add RR Interval Tachogram Preview
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
    
    # Action buttons
    st.markdown("---")
    st.markdown("**What would you like to do next?**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Accept & Run Full Analysis", type="primary", use_container_width=True):
            with st.spinner("Running complete cardiovascular analysis..."):
                try:
                    st.session_state.analyzer.analyze_with_current_peaks()
                    st.session_state.analyzed = True
                    st.session_state.preview_mode = False
                    st.success("✅ Complete analysis finished!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
    
    with col2:
        if st.button("🔄 Adjust Parameters", use_container_width=True):
            st.session_state.preview_mode = False
            st.info("👈 Adjust parameters in the sidebar and click 'Preview Peak Detection' again")
            st.rerun()
    
    with col3:
        if st.button("📊 Use Default Settings", use_container_width=True):
            with st.spinner("Analyzing with default parameters..."):
                try:
                    st.session_state.analyzer.find_peaks()  # Original method
                    st.session_state.analyzer.analyze_with_current_peaks()
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
    
    This tool provides comprehensive cardiovascular analysis with **adjustable peak detection parameters**.
    
    ### 🎛️ New Features:
    
    **Adjustable Peak Detection:**
    - Customize ECG R-peak detection parameters
    - Adjust blood pressure systolic peak detection
    - Real-time preview of detected peaks
    - Fine-tune before running full analysis
    
    **Interactive Preview:**
    - See entire time series of your data
    - Validate peak detection quality across full recording
    - Adjust parameters and preview again
    - Accept settings or try different ones
    
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
    2. **Adjust** peak detection parameters using sliders
    3. **Preview** peak detection results on full time series
    4. **Accept** parameters and run full analysis
    5. **Select** and generate plots
    6. **Download** your complete results
    
    ### 🎯 Peak Detection Parameters:
    
    **ECG Parameters:**
    - **Height**: Minimum amplitude for R-peak detection
    - **Distance**: Minimum samples between consecutive peaks
    - **Prominence**: How much peaks stand out from baseline
    
    **Blood Pressure Parameters:**
    - **Height**: Minimum systolic pressure threshold
    - **Distance**: Minimum samples between systolic peaks
    - **Prominence**: Peak prominence above surrounding signal
    
    ### 📈 Interactive Visualizations:
    
    - **Full time series ECG and BP signals** with detected peaks
    - **Complete RR interval tachogram** with statistical reference lines
    - **Zoomable plots** for detailed inspection
    - **Peak detection validation tools**
    
    ### 🩺 Advanced Analysis:
    
    - **Frequency domain analysis** with band highlighting
    - **Poincaré plot** with regression line and ellipse
    - **Baroreflex sensitivity** using multiple methods
    - **Transfer function and coherence** analysis
    
    **Quality Control:**
    - Real-time peak count and heart rate estimation
    - Visual validation of detection accuracy across entire recording
    - Parameter optimization before analysis
    - Comprehensive statistical summaries
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit • Interactive Peak Detection • Full Time Series Preview")