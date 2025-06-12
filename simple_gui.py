import sys
import os

# Handle paths when running as executable
if getattr(sys, 'frozen', False):
    # Running as executable
    current_dir = os.path.dirname(sys.executable)
else:
    # Running as script
    current_dir = os.path.dirname(os.path.abspath(__file__))

# Add current directory to path for imports
sys.path.insert(0, current_dir)

# enhanced_professional_gui.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tempfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
from analyzer import CardiovascularAnalyzer
from scipy.interpolate import interp1d  
from scipy.signal import welch
from functions import *

# ============================================================================
# PAGE CONFIGURATION & STYLING
# ============================================================================

st.set_page_config(
    page_title="PhysioKit - HRV & BRS Analysis",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: #f0f0f0;
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .version-info {
        color: #d0d0d0;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        opacity: 0.8;
    }
    
    /* Status indicator styling */
    .status-container {
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    
    .status-item {
        text-align: center;
        padding: 0.5rem;
    }
    
    .status-success {
        color: #28a745;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    .status-warning {
        color: #fd7e14;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    .status-info {
        color: #17a2b8;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    .status-pending {
        color: #6c757d;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    /* Enhanced metric cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #007bff;
        margin: 0.5rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    
    /* Progress styling */
    .progress-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    
    /* Analysis window highlight */
    .window-info {
        background: linear-gradient(90deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Results area styling */
    .results-header {
        background: linear-gradient(90deg, #e8f5e8 0%, #d4edda 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
    }
    
    /* Plot section styling */
    .plot-section {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    /* Error/warning styling */
    .error-box {
        background: linear-gradient(90deg, #f8d7da 0%, #f5c6cb 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(90deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'analyzer' not in st.session_state:
    st.session_state.analyzer = CardiovascularAnalyzer()
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'file_loaded' not in st.session_state:
    st.session_state.file_loaded = False
if 'preview_mode' not in st.session_state:
    st.session_state.preview_mode = False
if 'analysis_started' not in st.session_state:
    st.session_state.analysis_started = False

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def show_professional_header():
    """Display the professional header with branding"""
    st.markdown("""
    <div class="main-header">
        <h1>🫀 PhysioKit</h1>
        <p>Professional HRV & Baroreflex Sensitivity Analysis Platform</p>
        <div class="version-info">Version 1.0 | Advanced Peak Detection & Time Window Analysis</div>
    </div>
    """, unsafe_allow_html=True)

def show_analysis_status():
    """Display current analysis status with professional indicators"""
    st.markdown('<div class="status-container">', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.session_state.file_loaded:
            st.markdown('<div class="status-item"><div class="status-success">✅ File Loaded</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-item"><div class="status-pending">📁 No File</div></div>', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.preview_mode:
            st.markdown('<div class="status-item"><div class="status-warning">🔍 Preview Mode</div></div>', unsafe_allow_html=True)
        elif st.session_state.file_loaded:
            st.markdown('<div class="status-item"><div class="status-info">⚙️ Configure</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-item"><div class="status-pending">⚙️ Waiting</div></div>', unsafe_allow_html=True)
    
    with col3:
        if st.session_state.analyzed:
            st.markdown('<div class="status-item"><div class="status-success">✅ Analyzed</div></div>', unsafe_allow_html=True)
        elif st.session_state.analysis_started:
            st.markdown('<div class="status-item"><div class="status-warning">⏳ Processing</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-item"><div class="status-pending">⏳ Pending</div></div>', unsafe_allow_html=True)
    
    with col4:
        if 'selected_plots' in st.session_state and st.session_state.selected_plots:
            st.markdown('<div class="status-item"><div class="status-success">📊 Plots Ready</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-item"><div class="status-pending">📈 No Plots</div></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def run_analysis_with_progress(time_window=None):
    """Run analysis with detailed progress feedback and professional styling"""
    
    # Create progress container
    st.markdown('<div class="progress-container">', unsafe_allow_html=True)
    st.markdown("### 🔄 Analysis in Progress")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Peak Detection
        status_text.markdown("**🔍 Step 1/5:** Detecting ECG R-peaks and BP systolic peaks...")
        progress_bar.progress(10)
        
        if hasattr(st.session_state, 'peak_params'):
            st.session_state.analyzer.find_peaks_with_params(**st.session_state.peak_params)
        else:
            st.session_state.analyzer.find_peaks()
        
        progress_bar.progress(20)
        status_text.markdown("**✅ Step 1 Complete:** Peak detection finished")
        
        # Step 2: Time Domain
        status_text.markdown("**📊 Step 2/5:** Calculating time domain HRV metrics...")
        progress_bar.progress(30)
        
        if time_window:
            st.session_state.analyzer.set_time_window(time_window['start_time'], time_window['end_time'])
        st.session_state.analyzer.calculate_time_domain()
        
        progress_bar.progress(45)
        status_text.markdown("**✅ Step 2 Complete:** Time domain analysis finished")
        
        # Step 3: Frequency Domain
        status_text.markdown("**🌊 Step 3/5:** Analyzing frequency domain characteristics...")
        progress_bar.progress(60)
        
        st.session_state.analyzer.calculate_frequency_domain()
        
        progress_bar.progress(70)
        status_text.markdown("**✅ Step 3 Complete:** Frequency domain analysis finished")
        
        # Step 4: BRS Sequence
        status_text.markdown("**🩺 Step 4/5:** Computing baroreflex sensitivity (sequence method)...")
        progress_bar.progress(80)
        
        st.session_state.analyzer.calculate_brs_sequence()
        
        progress_bar.progress(90)
        status_text.markdown("**✅ Step 4 Complete:** BRS sequence analysis finished")
        
        # Step 5: BRS Spectral
        status_text.markdown("**📈 Step 5/5:** Computing spectral baroreflex sensitivity...")
        progress_bar.progress(95)
        
        st.session_state.analyzer.calculate_brs_spectral()
        
        progress_bar.progress(100)
        status_text.markdown("**🎉 Analysis Complete!** All cardiovascular metrics calculated successfully.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        return True
        
    except Exception as e:
        progress_bar.progress(0)
        status_text.markdown(f"**❌ Analysis Failed:** {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)
        return False

def show_enhanced_metrics(results_dict, title, icon="📊"):
    """Display metrics in professional cards"""
    st.markdown(f"### {icon} {title}")
    
    if 'error' in results_dict:
        st.markdown(f"""
        <div class="error-box">
            <strong>❌ Analysis Error:</strong><br>
            {results_dict['error']}
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Create metric cards based on available data
    if title == "Time Domain Analysis":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Basic Measurements</h4>
                <p><strong>Beats:</strong> {results_dict.get('num_beats', 'N/A')}</p>
                <p><strong>Heart Rate:</strong> {results_dict.get('hr', 'N/A')} BPM</p>
                <p><strong>Mean RR:</strong> {results_dict.get('mean_rr', 'N/A'):.1f} ms</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Variability Metrics</h4>
                <p><strong>RMSSD:</strong> {results_dict.get('rmssd', 'N/A'):.1f} ms</p>
                <p><strong>SDNN:</strong> {results_dict.get('sdnn', 'N/A'):.1f} ms</p>
                <p><strong>pNN50:</strong> {results_dict.get('pnn50', 'N/A'):.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Poincaré Analysis</h4>
                <p><strong>SD1:</strong> {results_dict.get('sd1', 'N/A'):.1f} ms</p>
                <p><strong>SD2:</strong> {results_dict.get('sd2', 'N/A'):.1f} ms</p>
                <p><strong>SD1/SD2:</strong> {results_dict.get('sd1_sd2_ratio', 'N/A'):.3f}</p>
            </div>
            """, unsafe_allow_html=True)

def create_professional_plot_header(title, description=""):
    """Create a professional header for plot sections"""
    st.markdown(f"""
    <div class="plot-section">
        <h3 style="margin-top: 0; color: #2c3e50;">{title}</h3>
        {f'<p style="color: #6c757d; margin-bottom: 1rem;">{description}</p>' if description else ''}
    """, unsafe_allow_html=True)

def close_plot_section():
    """Close the plot section div"""
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Professional Header
show_professional_header()

# Analysis Status Dashboard
show_analysis_status()

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.markdown("## 📁 File Upload")
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    
    # File upload with enhanced styling
    uploaded_file = st.file_uploader(
        "Choose an ACQ file", 
        type="acq",
        help="Upload your ACQ file containing ECG and blood pressure data"
    )
    
    if uploaded_file is not None:
        file_info = f"**File:** {uploaded_file.name}\n\n**Size:** {uploaded_file.size / 1024:.1f} KB"
        st.info(file_info)
        
        if st.button("🔄 Load File", type="primary", use_container_width=True):
            with st.spinner("Loading and validating file..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".acq") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Load file
                    st.session_state.analyzer.load_file(tmp_file_path)
                    
                    # Clean up temp file
                    os.unlink(tmp_file_path)
                    
                    st.session_state.file_loaded = True
                    st.session_state.analyzed = False
                    st.session_state.preview_mode = False
                    
                    st.success("✅ File loaded successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ File loading failed: {str(e)}")
                    st.session_state.file_loaded = False
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Time Window Selection (only if file is loaded)
    if st.session_state.file_loaded and not st.session_state.analyzed:
        st.markdown("## ⏱️ Analysis Window")
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        
        if hasattr(st.session_state.analyzer, 'ecg_data') and 'time' in st.session_state.analyzer.ecg_data:
            max_time = max(st.session_state.analyzer.ecg_data['time'])
            max_time_min = max_time / 60
            
            st.info(f"📊 **Recording:** {max_time:.1f}s ({max_time_min:.1f} min)")
            
            # Time window sliders with better styling
            start_time = st.slider(
                "🎯 Start Time (seconds)", 
                min_value=0.0, 
                max_value=max_time-10, 
                value=0.0, 
                step=1.0,
                help="Start of analysis window"
            )
            
            end_time = st.slider(
                "🏁 End Time (seconds)", 
                min_value=start_time+10, 
                max_value=max_time, 
                value=max_time, 
                step=1.0,
                help="End of analysis window"
            )
            
            # Window duration display
            window_duration = end_time - start_time
            window_duration_min = window_duration / 60
            
            st.success(f"📐 **Window:** {window_duration:.0f}s ({window_duration_min:.1f} min)")
            
            st.session_state.time_window = {
                'start_time': start_time,
                'end_time': end_time,
                'duration': window_duration
            }
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Peak Detection Parameters
        st.markdown("## 🎛️ Peak Detection")
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        
        # ECG Parameters
        with st.expander("⚡ ECG R-peak Detection", expanded=True):
            ecg_height = st.slider("Height Threshold", 0.1, 2.0, 0.8, 0.1, help="Minimum R-peak amplitude")
            ecg_distance = st.slider("Min Distance", 50, 200, 100, 10, help="Min samples between peaks")
            ecg_prominence = st.slider("Prominence", 0.1, 1.5, 0.7, 0.1, help="Peak prominence")
        
        # BP Parameters
        with st.expander("🩸 BP Systolic Detection", expanded=True):
            bp_height = st.slider("BP Height (mmHg)", 80, 150, 110, 5, help="Min systolic pressure")
            bp_distance = st.slider("BP Min Distance", 50, 200, 100, 10, help="Min samples between peaks")
            bp_prominence = st.slider("BP Prominence", 1, 10, 5, 1, help="Peak prominence")
        
        st.session_state.peak_params = {
            'ecg_height': ecg_height,
            'ecg_distance': ecg_distance,
            'ecg_prominence': ecg_prominence,
            'bp_height': bp_height,
            'bp_distance': bp_distance,
            'bp_prominence': bp_prominence
        }
        
        # Preview button with enhanced styling
        if st.button("🔍 Preview Detection", use_container_width=True, type="secondary"):
            with st.spinner("Updating peak detection..."):
                try:
                    st.session_state.analyzer.find_peaks_with_params(**st.session_state.peak_params)
                    st.session_state.preview_mode = True
                    st.success("✅ Preview updated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Preview failed: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Plot Selection (only if analyzed)
    if st.session_state.analyzed:
        st.markdown("## 📊 Visualizations")
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        
        plot_options = [
            "🔍 Interactive Tachogram",
            "📊 Frequency Domain",
            "🔄 Poincaré Plot",
            "🌊 Spectral BRS Analysis",  # ← ADD THIS NEW OPTION
            "🩺 BRS Sequence Analysis",
            "🩺 BRS Time Domain Visualization"
        ]

        selected_plots = st.multiselect(
            "Select visualizations:",
            plot_options,
            default=plot_options[:3],
            help="Choose which plots to generate"
        )

        if st.button("🎨 Generate Plots", use_container_width=True, type="primary"):
            st.session_state.selected_plots = selected_plots
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
        

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

# Case 1: Analysis Complete - Show Results
if st.session_state.analyzed:
    # Results Header
    st.markdown("""
    <div class="results-header">
        <h2 style="margin: 0; color: #155724;">🎉 Analysis Complete</h2>
        <p style="margin: 0.5rem 0 0 0; color: #155724;">Comprehensive cardiovascular analysis finished successfully</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Time window info if applicable
    if 'time_window' in st.session_state:
        tw = st.session_state.time_window
        st.markdown(f"""
        <div class="window-info">
            <strong>🎯 Analysis Window:</strong> {tw['start_time']:.1f}s to {tw['end_time']:.1f}s 
            ({tw['duration']:.1f} seconds = {tw['duration']/60:.1f} minutes)
        </div>
        """, unsafe_allow_html=True)
    
    # Results in two columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📋 Analysis Summary")
        
        # Enhanced metrics display
        if 'time_domain' in st.session_state.analyzer.results:
            show_enhanced_metrics(
                st.session_state.analyzer.results['time_domain'], 
                "Time Domain Analysis", 
                "⏱️"
            )
        
        # BRS Results
        if 'brs_sequence' in st.session_state.analyzer.results:
            brs_data = st.session_state.analyzer.results['brs_sequence']
            if 'error' not in brs_data:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🩺 Baroreflex Sensitivity</h4>
                    <p><strong>BRS Mean:</strong> {brs_data.get('BRS_mean', 'N/A'):.2f} ms/mmHg</p>
                    <p><strong>BEI:</strong> {brs_data.get('BEI', 'N/A'):.2f}</p>
                    <p><strong>Valid Sequences:</strong> {brs_data.get('num_sequences', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Download results
        results_text = st.session_state.analyzer.get_summary()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        st.download_button(
            label="⬇️ Download Complete Report",
            data=results_text,
            file_name=f"cardio_analysis_{timestamp}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col2:
        st.markdown("### 📊 Interactive Visualizations")
        
        # Display selected plots with professional styling
        if 'selected_plots' in st.session_state:
            for plot_type in st.session_state.selected_plots:
                
                if "Interactive Tachogram" in plot_type:
                    create_professional_plot_header(
                        "🔍 Heart Rate Variability Tachogram",
                        "Interactive visualization of RR interval variations over time"
                    )
                    
                    fig = go.Figure()
                    
                    rr_intervals = st.session_state.analyzer.ecg_data['rr_intervals']
                    time_points = st.session_state.analyzer.ecg_data['td_peaks'][:-1]
                    
                    fig.add_trace(go.Scatter(
                        x=time_points,
                        y=rr_intervals,
                        mode='lines+markers',
                        name='RR Intervals',
                        line=dict(color='#667eea', width=2),
                        marker=dict(size=4, color='#764ba2')
                    ))
                    
                    # Highlight analysis window
                    if 'time_window' in st.session_state:
                        tw = st.session_state.time_window
                        fig.add_vrect(
                            x0=tw['start_time'], x1=tw['end_time'],
                            fillcolor="rgba(255, 193, 7, 0.3)", opacity=0.8,
                            annotation_text="Analysis Window", annotation_position="top left"
                        )
                    
                    # Enhanced metrics panel
                    mean_rr = np.mean(rr_intervals)
                    std_rr = np.std(rr_intervals)
                    
                    metrics_text = "<b>📊 HRV Metrics</b><br><br>"
                    
                    if 'time_domain' in st.session_state.analyzer.results:
                        td = st.session_state.analyzer.results['time_domain']
                        if 'error' not in td:
                            metrics_text += f"<b>📈 Basic Measures</b><br>"
                            metrics_text += f"Beats: {td['num_beats']}<br>"
                            metrics_text += f"Mean RR: {td['mean_rr']:.1f} ms<br>"
                            metrics_text += f"HR: {td['hr']:.1f} BPM<br><br>"
                            
                            metrics_text += f"<b>🔢 Time Domain</b><br>"
                            metrics_text += f"RMSSD: {td['rmssd']:.1f} ms<br>"
                            metrics_text += f"SDNN: {td['sdnn']:.1f} ms<br>"
                            metrics_text += f"pNN50: {td['pnn50']:.1f}%<br>"
                            metrics_text += f"SampEn: {td['sample_entropy']:.3f}"
                    
                    fig.add_annotation(
                        x=1.02, y=1.0, xref="paper", yref="paper",
                        text=metrics_text, showarrow=False,
                        font=dict(family="Arial", size=11, color="black"),
                        align="left", bgcolor="rgba(248, 249, 250, 0.95)",
                        bordercolor="rgba(108, 117, 125, 0.5)", borderwidth=1, borderpad=10,
                        xanchor="left", yanchor="top"
                    )
                    
                    fig.update_layout(
                        title=f'Heart Rate Variability Analysis (Mean: {mean_rr:.1f}±{std_rr:.1f} ms)',
                        xaxis_title='Time (s)', yaxis_title='RR Interval (ms)',
                        hovermode='x unified', height=600, margin=dict(r=250),
                        plot_bgcolor='rgba(248,249,250,0.8)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    close_plot_section()
                
                elif "Frequency Domain" in plot_type:
                    create_professional_plot_header(
                        "📊 Frequency Domain Analysis",
                        "Power spectral density analysis of heart rate variability"
                    )
                    
                    freq_data = st.session_state.analyzer.results['frequency_domain']
                    
                    if 'error' not in freq_data:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        frequencies = freq_data['frequencies']
                        psd = freq_data['psd']
                        
                        ax.plot(frequencies, psd * 1e6, 'b-', linewidth=2, label='PSD', color='#667eea')
                        
                        # Highlight frequency bands with better colors
                        lf_band = (frequencies >= 0.04) & (frequencies < 0.15)
                        hf_band = (frequencies >= 0.15) & (frequencies < 0.4)
                        vlf_band = (frequencies >= 0.003) & (frequencies < 0.04)
                        
                        ax.fill_between(frequencies[vlf_band], psd[vlf_band] * 1e6, 
                                       color='#95a5a6', alpha=0.4, label='VLF (0.003-0.04 Hz)')
                        ax.fill_between(frequencies[lf_band], psd[lf_band] * 1e6, 
                                       color='#3498db', alpha=0.5, label='LF (0.04-0.15 Hz)')
                        ax.fill_between(frequencies[hf_band], psd[hf_band] * 1e6, 
                                       color='#e74c3c', alpha=0.5, label='HF (0.15-0.4 Hz)')
                        
                        ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
                        ax.set_ylabel('Power Spectral Density (ms²/Hz)', fontsize=12, fontweight='bold')
                        ax.set_title('HRV Frequency Domain Analysis', fontsize=14, fontweight='bold')
                        ax.set_xlim(0, 0.5)
                        ax.legend(loc='center right')
                        ax.grid(True, alpha=0.3)
                        ax.set_facecolor('#f8f9fa')
                        
                        # Enhanced power values text box
                        power_text = (f"VLF: {freq_data['vlf_power']:.2f} ms²\n"
                                    f"LF: {freq_data['lf_power']:.2f} ms²\n"
                                    f"HF: {freq_data['hf_power']:.2f} ms²\n"
                                    f"Total: {freq_data['total_power']:.2f} ms²\n"
                                    f"LF/HF: {freq_data['lf_hf_ratio']:.2f}\n"
                                    f"LF n.u.: {freq_data['lf_nu']:.2f}\n"
                                    f"HF n.u.: {freq_data['hf_nu']:.2f}")
                        
                        ax.text(0.98, 0.98, power_text, transform=ax.transAxes, 
                               verticalalignment='top', horizontalalignment='right',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#dee2e6'))
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.error(f"Frequency domain analysis error: {freq_data['error']}")
                    
                    close_plot_section()
                
                elif "Poincaré Plot" in plot_type:
                    create_professional_plot_header(
                        "🔄 Poincaré Plot Analysis",
                        "Nonlinear analysis of heart rate variability patterns"
                    )
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    RRDistance_ms = st.session_state.analyzer.ecg_data['rr_intervals']
                    RRIplusOne = Poincare(RRDistance_ms)
                    
                    EllipseCenterX = np.average(np.delete(RRDistance_ms, -1))
                    EllipseCenterY = np.average(RRIplusOne)
                    Center_coords = EllipseCenterX, EllipseCenterY
                    
                    z = np.polyfit(np.delete(RRDistance_ms, -1), RRIplusOne, 1)
                    p = np.poly1d(z)
                    slope = z[0]
                    theta = np.degrees(np.arctan(slope))
                    theta_rad = np.radians(theta)
                    
                    # Enhanced scatter plot
                    ax.scatter(np.delete(RRDistance_ms, -1), RRIplusOne, 
                              alpha=0.7, s=25, color='#667eea', edgecolors='white', linewidth=0.5)
                    
                    # Enhanced regression line
                    ax.plot(np.delete(RRDistance_ms, -1), p(np.delete(RRDistance_ms, -1)), 
                           color="#e74c3c", linewidth=3, label='Identity Line', alpha=0.8)
                    
                    # Get SD values
                    if 'time_domain' in st.session_state.analyzer.results:
                        td_results = st.session_state.analyzer.results['time_domain']
                        sd1 = td_results['sd1']
                        sd2 = td_results['sd2']
                        
                        # Enhanced ellipse
                        from matplotlib.patches import Ellipse
                        e = Ellipse(xy=Center_coords, width=sd2*2, height=sd1*2, angle=theta,
                                    edgecolor='#2c3e50', facecolor='none', linewidth=2, alpha=0.8)
                        ax.add_patch(e)
                        
                        # Enhanced axis lines
                        x_sd2 = [EllipseCenterX, EllipseCenterX + sd2 * np.cos(theta_rad)]
                        y_sd2 = [EllipseCenterY, EllipseCenterY + sd2 * np.sin(theta_rad)]
                        ax.plot(x_sd2, y_sd2, color='#3498db', linewidth=3, label='SD2 (Long-term)', alpha=0.9)
                        
                        x_sd1 = [EllipseCenterX, EllipseCenterX - sd1 * np.sin(theta_rad)]
                        y_sd1 = [EllipseCenterY, EllipseCenterY + sd1 * np.cos(theta_rad)]
                        ax.plot(x_sd1, y_sd1, color='#27ae60', linewidth=3, label='SD1 (Short-term)', alpha=0.9)
                        
                        # Enhanced text box
                        textstr = f'SD1 = {sd1:.1f} ms\nSD2 = {sd2:.1f} ms\nSD1/SD2 = {sd1/sd2:.3f}\nEllipse Area = {td_results["ellipse_area"]:.1f} ms²'
                        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, verticalalignment='top',
                                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#dee2e6'),
                                fontsize=11, fontweight='bold')
                    
                    ax.set_xlabel("RR Interval (ms)", fontsize=12, fontweight='bold')
                    ax.set_ylabel("RR Interval + 1 (ms)", fontsize=12, fontweight='bold')
                    ax.set_title('Poincaré Plot - Nonlinear HRV Analysis', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='upper right')
                    ax.set_facecolor('#f8f9fa')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    close_plot_section()
                
                elif "BRS Time Domain Visualization" in plot_type:
                    create_professional_plot_header(
                        "🩺 Baroreflex Sensitivity - Time Domain Analysis",
                        "Interactive visualization of blood pressure and RR interval sequences"
                    )
                    
                    if 'brs_sequence' in st.session_state.analyzer.results:
                        brs_data = st.session_state.analyzer.results['brs_sequence']
                        
                        if 'error' in brs_data:
                            st.markdown(f"""
                            <div class="error-box">
                                <strong>BRS Analysis Error:</strong> {brs_data['error']}
                            </div>
                            """, unsafe_allow_html=True)
                        elif 'plotting_data' in brs_data:
                            plot_data = brs_data['plotting_data']
                            
                            # Enhanced interactive BRS plot
                            fig = make_subplots(
                                rows=2, cols=1,
                                subplot_titles=('Systolic Blood Pressure', 'RR Intervals'),
                                vertical_spacing=0.1,
                                shared_xaxes=True
                            )
                            
                            # Enhanced SBP trace
                            fig.add_trace(go.Scatter(
                                x=plot_data['sap_times'],
                                y=plot_data['sbp'],
                                mode='lines+markers',
                                name='Systolic BP',
                                line=dict(color='#e74c3c', width=2),
                                marker=dict(size=5, color='#c0392b'),
                                opacity=0.8
                            ), row=1, col=1)
                            
                            # Enhanced RRI trace
                            fig.add_trace(go.Scatter(
                                x=plot_data['rri_times'],
                                y=plot_data['rri'],
                                mode='lines+markers',
                                name='RR Intervals',
                                line=dict(color='#3498db', width=2),
                                marker=dict(size=5, color='#2980b9'),
                                opacity=0.8
                            ), row=2, col=1)
                            
                            # Highlight valid sequences with enhanced colors
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
                                
                                if len(sbp_ramp) != len(rri_ramp) or np.any(np.abs(np.diff(rri_ramp)) < thresh_pi):
                                    continue
                                
                                slope, intercept, r_value, _, _ = linregress(sbp_ramp, rri_ramp)
                                if abs(r_value) < r_threshold or slope <= 0:
                                    continue
                                
                                sequence_count += 1
                                color = '#27ae60' if direction == 'up' else '#f39c12'
                                
                                # Highlight sequences with enhanced styling
                                if start < len(sap_times) and end < len(sap_times):
                                    fig.add_trace(go.Scatter(
                                        x=sap_times[start:end + 1],
                                        y=sbp[start:end + 1],
                                        mode='lines+markers',
                                        name=f'{direction.upper()} sequence' if sequence_count == 1 else None,
                                        line=dict(color=color, width=4),
                                        marker=dict(size=8, color=color),
                                        showlegend=(sequence_count == 1),
                                        legendgroup=direction
                                    ), row=1, col=1)
                                
                                if start + delay < len(rri_times) and end + 1 + delay <= len(rri_times):
                                    fig.add_trace(go.Scatter(
                                        x=rri_times[start + delay:end + 1 + delay],
                                        y=rri[start + delay:end + 1 + delay],
                                        mode='lines+markers',
                                        name=None,
                                        line=dict(color=color, width=4),
                                        marker=dict(size=8, color=color),
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
                            
                            # Highlight analysis window
                            if 'time_window' in st.session_state:
                                tw = st.session_state.time_window
                                fig.add_vrect(
                                    x0=tw['start_time'], x1=tw['end_time'],
                                    fillcolor="rgba(255, 193, 7, 0.2)", opacity=0.8,
                                    annotation_text="Analysis Window", annotation_position="top left",
                                    row=1, col=1
                                )
                                fig.add_vrect(
                                    x0=tw['start_time'], x1=tw['end_time'],
                                    fillcolor="rgba(255, 193, 7, 0.2)", opacity=0.8,
                                    row=2, col=1
                                )
                            
                            # Enhanced layout
                            fig.update_xaxes(title_text="Time (s)", row=2, col=1)
                            fig.update_yaxes(title_text="Systolic BP (mmHg)", row=1, col=1)
                            fig.update_yaxes(title_text="RR Interval (ms)", row=2, col=1)
                            
                            fig.update_layout(
                                title=f'Baroreflex Sensitivity Analysis - {sequence_count} Valid Sequences Found',
                                height=700,
                                hovermode='x unified',
                                showlegend=True,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                plot_bgcolor='rgba(248,249,250,0.8)'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Enhanced BRS metrics display
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("🎯 BRS Mean", f"{brs_data.get('BRS_mean', 0):.2f} ms/mmHg", 
                                         help="Average baroreflex sensitivity")
                            with col2:
                                st.metric("📊 BEI", f"{brs_data.get('BEI', 0):.2f}", 
                                         help="Baroreflex Effectiveness Index")
                            with col3:
                                st.metric("✅ Valid Sequences", sequence_count,
                                         help="Number of valid BRS sequences detected")
                            with col4:
                                st.metric("⏱️ Best Delay", f"{plot_data['best_delay']} beats",
                                         help="Optimal delay between BP and RR changes")
                            
                            # Enhanced sequence details table
                            if valid_sequences:
                                st.markdown("#### 📋 Valid BRS Sequence Details")
                                
                                sequence_df = pd.DataFrame(valid_sequences)
                                sequence_df['slope'] = sequence_df['slope'].round(3)
                                sequence_df['r_value'] = sequence_df['r_value'].round(3)
                                
                                st.dataframe(
                                    sequence_df,
                                    column_config={
                                        "sequence": st.column_config.NumberColumn("Seq #", width="small"),
                                        "direction": st.column_config.TextColumn("Direction", width="small"), 
                                        "slope": st.column_config.NumberColumn("Slope (ms/mmHg)", format="%.3f"),
                                        "r_value": st.column_config.NumberColumn("Correlation (r)", format="%.3f"),
                                        "start": st.column_config.NumberColumn("Start Index", width="small"),
                                        "end": st.column_config.NumberColumn("End Index", width="small")
                                    },
                                    hide_index=True,
                                    use_container_width=True
                                )
                                
                                # Analysis parameters info
                                st.info(f"📋 **Analysis Parameters:** delay={plot_data['best_delay']} beats, "
                                       f"r_threshold={plot_data['r_threshold']}, thresh_pi={plot_data['thresh_pi']} ms")
                    
                    close_plot_section()
                        # STEP 2A: Update plot options list in simple_gui.py (around line 820)

                elif "Spectral BRS Analysis" in plot_type:
                    create_professional_plot_header(
                        "🌊 Spectral Baroreflex Sensitivity Analysis",
                        "Cross-spectral analysis showing RRI and SBP power spectra, coherence, and transfer function"
                    )
                    
                    # Check if we have the required spectral data
                    if 'brs_spectral' not in st.session_state.analyzer.results or 'frequency_domain' not in st.session_state.analyzer.results:
                        st.markdown("""
                        <div class="error-box">
                            <strong>❌ Spectral Analysis Error:</strong> Required spectral data not available. 
                            Please ensure frequency domain and BRS spectral analysis completed successfully.
                        </div>
                        """, unsafe_allow_html=True)
                    
                    else:
                        brs_spec_data = st.session_state.analyzer.results['brs_spectral']
                        freq_data = st.session_state.analyzer.results['frequency_domain']
                        
                        if 'error' in brs_spec_data:
                            st.markdown(f"""
                            <div class="error-box">
                                <strong>❌ Spectral BRS Error:</strong> {brs_spec_data['error']}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        elif 'error' in freq_data:
                            st.markdown(f"""
                            <div class="error-box">
                                <strong>❌ Frequency Domain Error:</strong> {freq_data['error']}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        else:
                            # Create comprehensive spectral analysis plot
                            fig = make_subplots(
                                rows=2, cols=2,
                                subplot_titles=(
                                    'RRI Power Spectral Density', 
                                    'SBP Power Spectral Density',
                                    'RRI-SBP Coherence', 
                                    'Transfer Function |H(f)| = |CSD|/PSD_BP'
                                ),
                                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                                    [{"secondary_y": False}, {"secondary_y": False}]],
                                vertical_spacing=0.12,
                                horizontal_spacing=0.1
                            )
                            
                            # Get the RRI spectral data
                            frequencies_rr = freq_data['frequencies']
                            psd_rr = freq_data['psd']
                            
                            # RRI PSD (Top Left)
                            fig.add_trace(go.Scatter(
                                x=frequencies_rr,
                                y=psd_rr * 1e6,  # Convert to ms²/Hz for display
                                mode='lines',
                                name='RRI PSD',
                                line=dict(color='#3498db', width=2),
                                showlegend=False
                            ), row=1, col=1)
                            
                            # Highlight frequency bands on RRI PSD
                            lf_band = (frequencies_rr >= 0.04) & (frequencies_rr < 0.15)
                            hf_band = (frequencies_rr >= 0.15) & (frequencies_rr < 0.4)
                            
                            if np.any(lf_band):
                                fig.add_trace(go.Scatter(
                                    x=frequencies_rr[lf_band],
                                    y=psd_rr[lf_band] * 1e6,
                                    mode='lines',
                                    fill='tonexty',
                                    name='LF Band',
                                    line=dict(color='#e74c3c', width=0),
                                    fillcolor='rgba(231, 76, 60, 0.3)',
                                    showlegend=True
                                ), row=1, col=1)
                            
                            if np.any(hf_band):
                                fig.add_trace(go.Scatter(
                                    x=frequencies_rr[hf_band],
                                    y=psd_rr[hf_band] * 1e6,
                                    mode='lines',
                                    fill='tonexty',
                                    name='HF Band',
                                    line=dict(color='#27ae60', width=0),
                                    fillcolor='rgba(39, 174, 96, 0.3)',
                                    showlegend=True
                                ), row=1, col=1)
                            
                            # SBP PSD (Top Right) - use the calculated data from analyzer
                            if 'frequencies_bp' in brs_spec_data and 'psd_bp' in brs_spec_data:
                                frequencies_bp = brs_spec_data['frequencies_bp']
                                psd_bp = brs_spec_data['psd_bp']
                                
                                fig.add_trace(go.Scatter(
                                    x=frequencies_bp,
                                    y=psd_bp,
                                    mode='lines',
                                    name='SBP PSD',
                                    line=dict(color='#e74c3c', width=2),
                                    showlegend=False
                                ), row=1, col=2)
                                
                                # Highlight bands on SBP
                                lf_band_bp = (frequencies_bp >= 0.04) & (frequencies_bp < 0.15)
                                hf_band_bp = (frequencies_bp >= 0.15) & (frequencies_bp < 0.4)
                                
                                if np.any(lf_band_bp):
                                    fig.add_trace(go.Scatter(
                                        x=frequencies_bp[lf_band_bp],
                                        y=psd_bp[lf_band_bp],
                                        mode='lines',
                                        fill='tonexty',
                                        name='LF Band (BP)',
                                        line=dict(color='#e74c3c', width=0),
                                        fillcolor='rgba(231, 76, 60, 0.2)',
                                        showlegend=False
                                    ), row=1, col=2)
                                
                                if np.any(hf_band_bp):
                                    fig.add_trace(go.Scatter(
                                        x=frequencies_bp[hf_band_bp],
                                        y=psd_bp[hf_band_bp],
                                        mode='lines',
                                        fill='tonexty',
                                        name='HF Band (BP)',
                                        line=dict(color='#27ae60', width=0),
                                        fillcolor='rgba(39, 174, 96, 0.2)',
                                        showlegend=False
                                    ), row=1, col=2)
                            
                            # Coherence plot (Bottom Left)
                            if 'frequencies_coh' in brs_spec_data and 'coherence_values' in brs_spec_data:
                                frequencies_coh = brs_spec_data['frequencies_coh']
                                coherence_values = brs_spec_data['coherence_values']
                                
                                fig.add_trace(go.Scatter(
                                    x=frequencies_coh,
                                    y=coherence_values,
                                    mode='lines',
                                    name='Coherence',
                                    line=dict(color='#9b59b6', width=2),
                                    showlegend=False
                                ), row=2, col=1)
                                
                                # Add coherence threshold line
                                fig.add_hline(y=0.5, line_dash="dash", line_color="#f39c12", 
                                            annotation_text="Threshold (0.5)", row=2, col=1)
                                
                                # Highlight significant coherence regions
                                significant_mask = coherence_values >= 0.5
                                if np.any(significant_mask):
                                    fig.add_trace(go.Scatter(
                                        x=frequencies_coh[significant_mask],
                                        y=coherence_values[significant_mask],
                                        mode='markers',
                                        name='Valid Coherence (≥0.5)',
                                        marker=dict(color='#e74c3c', size=4),
                                        showlegend=False
                                    ), row=2, col=1)
                            
                            # Transfer Function (Bottom Right) - exactly matching main.py calculations
                            if 'transfer_gain' in brs_spec_data and 'frequencies_csd' in brs_spec_data:
                                frequencies_csd = brs_spec_data['frequencies_csd']
                                transfer_gain = brs_spec_data['transfer_gain']
                                
                                fig.add_trace(go.Scatter(
                                    x=frequencies_csd,
                                    y=transfer_gain,
                                    mode='lines',
                                    name='Transfer Function |H(f)|',
                                    line=dict(color='#2c3e50', width=2),
                                    showlegend=False
                                ), row=2, col=2)
                                
                                # Add comprehensive BRS results annotation
                                lf_coherence = brs_spec_data.get('lf_coherence', 0)
                                hf_coherence = brs_spec_data.get('hf_coherence', 0)
                                brs_lf_tf = brs_spec_data.get('brs_lf_tf', 0)
                                brs_hf_tf = brs_spec_data.get('brs_hf_tf', 0)
                                nperseg_used = brs_spec_data.get('nperseg_used', 'N/A')
                                
                                fig.add_annotation(
                                    x=0.98, y=0.95, xref="paper", yref="paper",
                                    text=f"<b>🔍 Spectral BRS Results</b><br><br>"
                                        f"<b>LF Band (0.04-0.15 Hz)</b><br>"
                                        f"BRS: {brs_lf_tf:.3f} ms/mmHg<br>"
                                        f"Coherence: {lf_coherence:.3f} {'✅' if lf_coherence > 0.5 else '❌'}<br><br>"
                                        f"<b>HF Band (0.15-0.4 Hz)</b><br>"
                                        f"BRS: {brs_hf_tf:.3f} ms/mmHg<br>"
                                        f"Coherence: {hf_coherence:.3f} {'✅' if hf_coherence > 0.5 else '❌'}<br><br>"
                                        f"<b>Method</b><br>"
                                        f"CSD: |csd(bp,rr)| / psd_bp<br>"
                                        f"nperseg: {nperseg_used}<br>"
                                        f"Interp: 4 Hz",
                                    showarrow=False,
                                    font=dict(family="Arial", size=10, color="black"),
                                    align="left", bgcolor="rgba(248, 249, 250, 0.95)",
                                    bordercolor="rgba(108, 117, 125, 0.5)", borderwidth=1, borderpad=10,
                                    xanchor="right", yanchor="top"
                                )
                            
                            # Update axes labels and formatting
                            fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=1)
                            fig.update_yaxes(title_text="PSD (ms²/Hz)", row=1, col=1)
                            
                            fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
                            fig.update_yaxes(title_text="PSD (mmHg²/Hz)", row=1, col=2)
                            
                            fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
                            fig.update_yaxes(title_text="Coherence", row=2, col=1)
                            
                            fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=2)
                            fig.update_yaxes(title_text="Transfer Gain (ms/mmHg)", row=2, col=2)
                            
                            # Set consistent frequency range for all subplots (focus on relevant HRV bands)
                            for row in [1, 2]:
                                for col in [1, 2]:
                                    fig.update_xaxes(range=[0, 0.5], row=row, col=col)
                            
                            # Overall layout
                            fig.update_layout(
                                title=f'Spectral BRS Analysis - Matches main.py Implementation',
                                height=700,
                                showlegend=True,
                                legend=dict(
                                    orientation="h", 
                                    yanchor="bottom", 
                                    y=1.02, 
                                    xanchor="center", 
                                    x=0.5
                                ),
                                plot_bgcolor='rgba(248,249,250,0.8)',
                                margin=dict(r=250)  # Extra space for annotations
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Enhanced summary metrics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                lf_valid = "✅ Valid" if brs_spec_data.get('valid_lf', False) else "❌ Invalid"
                                st.metric(
                                    "🔵 LF BRS", 
                                    f"{brs_spec_data.get('brs_lf_tf', 0):.3f} ms/mmHg",
                                    help=f"Low frequency BRS - {lf_valid}"
                                )
                            
                            with col2:
                                hf_valid = "✅ Valid" if brs_spec_data.get('valid_hf', False) else "❌ Invalid"
                                st.metric(
                                    "🟢 HF BRS", 
                                    f"{brs_spec_data.get('brs_hf_tf', 0):.3f} ms/mmHg",
                                    help=f"High frequency BRS - {hf_valid}"
                                )
                            
                            with col3:
                                st.metric(
                                    "🔵 LF Coherence", 
                                    f"{brs_spec_data.get('lf_coherence', 0):.3f}",
                                    help="Coherence in LF band (>0.5 required for validity)"
                                )
                            
                            with col4:
                                st.metric(
                                    "🟢 HF Coherence", 
                                    f"{brs_spec_data.get('hf_coherence', 0):.3f}",
                                    help="Coherence in HF band (>0.5 required for validity)"
                                )
                            
                            # Technical details in expandable section
                            with st.expander("📋 Technical Analysis Details (matches main.py)", expanded=False):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**🔧 Analysis Parameters (matches main.py exactly):**")
                                    st.markdown(f"- Interpolation frequency: 4 Hz")
                                    st.markdown(f"- Spectral method: Welch's method") 
                                    st.markdown(f"- CSD calculation: csd(bp_fft, rr_fft)")
                                    st.markdown(f"- Transfer function: |CSD| / PSD_bp")
                                    st.markdown(f"- nperseg used: {brs_spec_data.get('nperseg_used', 'N/A')}")
                                    st.markdown(f"- Data points used: {brs_spec_data.get('data_length_used', 'N/A')}")
                                    
                                    if 'time_offset' in freq_data:
                                        st.markdown(f"- Time normalization: {freq_data['time_offset']:.1f}s offset")
                                    
                                    if 'window_duration' in freq_data:
                                        st.markdown(f"- Analysis duration: {freq_data['window_duration']:.1f}s")
                                
                                with col2:
                                    st.markdown("**📊 Frequency Bands:**")
                                    st.markdown("- VLF: 0.003 - 0.04 Hz")
                                    st.markdown("- LF: 0.04 - 0.15 Hz")
                                    st.markdown("- HF: 0.15 - 0.4 Hz")
                                    st.markdown("")
                                    st.markdown("**✅ Validity Criteria:**")
                                    st.markdown("- Coherence > 0.5 required")
                                    st.markdown("- Sufficient data length")
                                    st.markdown("- Stable spectral estimates")
                                    
                                    if 'analysis_method' in brs_spec_data:
                                        st.markdown(f"- Method: {brs_spec_data['analysis_method']}")
                            
                            # Interpretation guide
                            st.markdown("### 📚 Interpretation Guide")
                            
                            interpretation_col1, interpretation_col2 = st.columns(2)
                            
                            with interpretation_col1:
                                st.markdown("""
                                **🔍 Understanding the Plots:**
                                - **Top Left:** RRI power spectral density with frequency bands
                                - **Top Right:** SBP power spectral density with frequency bands
                                - **Bottom Left:** Coherence shows strength of linear relationship
                                - **Bottom Right:** Transfer function |H(f)| = |CSD|/PSD_BP quantifies BRS
                                """)
                                
                                st.markdown("""
                                **📊 Coherence Interpretation:**
                                - **> 0.5:** Strong linear relationship (BRS values reliable)
                                - **< 0.5:** Weak relationship (BRS values unreliable)
                                - **Peak coherence:** Indicates dominant coupling frequencies
                                """)
                            
                            with interpretation_col2:
                                st.markdown("""
                                **🩺 Clinical Significance:**
                                - **Higher BRS:** Better cardiovascular regulation
                                - **LF BRS:** Reflects sympathetic and parasympathetic modulation
                                - **HF BRS:** Primarily reflects parasympathetic activity
                                """)
                                
                                st.markdown("""
                                **⚠️ Quality Assessment:**
                                - Check coherence before interpreting BRS values
                                - Look for consistent patterns across frequency bands
                                - Consider data length and artifact presence
                                - This implementation exactly matches your main.py calculations
                                """)
                    
                    close_plot_section()
                elif "BRS Sequence Analysis" in plot_type:
                    create_professional_plot_header(
                        "🩺 BRS Sequence Analysis Summary",
                        "Comprehensive baroreflex sensitivity metrics and statistics"
                    )
                    
                    if 'brs_sequence' in st.session_state.analyzer.results:
                        brs_data = st.session_state.analyzer.results['brs_sequence']
                        
                        if 'error' in brs_data:
                            st.markdown(f"""
                            <div class="error-box">
                                <strong>BRS Analysis Error:</strong> {brs_data['error']}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # Enhanced BRS metrics display
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4>🎯 Primary Metrics</h4>
                                    <p><strong>BRS Mean:</strong> {brs_data.get('BRS_mean', 0):.2f} ms/mmHg</p>
                                    <p><strong>BEI:</strong> {brs_data.get('BEI', 0):.2f}</p>
                                    <p><strong>Best Delay:</strong> {brs_data.get('best_delay', 0)} beats</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4>📊 Sequence Counts</h4>
                                    <p><strong>Valid Sequences:</strong> {brs_data.get('num_sequences', 0)}</p>
                                    <p><strong>Total SAP Ramps:</strong> {brs_data.get('num_sbp_ramps', 0)}</p>
                                    <p><strong>Success Rate:</strong> {(brs_data.get('num_sequences', 0) / max(brs_data.get('num_sbp_ramps', 1), 1) * 100):.1f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4>🔄 Direction Analysis</h4>
                                    <p><strong>Up Sequences:</strong> {brs_data.get('n_up', 0)}</p>
                                    <p><strong>Down Sequences:</strong> {brs_data.get('n_down', 0)}</p>
                                    <p><strong>Up/Down Ratio:</strong> {(brs_data.get('n_up', 0) / max(brs_data.get('n_down', 1), 1)):.2f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Analysis parameters
                            st.markdown("""
                            <div class="window-info">
                                <strong>📋 Analysis Parameters:</strong> min_len=3, delay_range=(0,4), r_threshold=0.8, thresh_sbp=1, thresh_pi=4
                            </div>
                            """, unsafe_allow_html=True)
                    
                    close_plot_section()

# Case 2: Preview Mode - Show Enhanced Peak Detection Preview
elif st.session_state.file_loaded and st.session_state.preview_mode:
    st.markdown("""
    <div class="window-info">
        <h3 style="margin: 0;">🔍 Peak Detection Preview & Time Window Selection</h3>
        <p style="margin: 0.5rem 0 0 0;">Review detected peaks and selected analysis window. Adjust parameters in sidebar if needed.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Time window info
    if 'time_window' in st.session_state:
        tw = st.session_state.time_window
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎯 Selected Analysis Window</h4>
            <p><strong>Start:</strong> {tw['start_time']:.0f}s | <strong>End:</strong> {tw['end_time']:.0f}s | <strong>Duration:</strong> {tw['duration']:.0f}s ({tw['duration']/60:.1f} min)</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Get peaks data safely
    peaks = st.session_state.analyzer.ecg_data.get('peaks', [])
    time_data = st.session_state.analyzer.ecg_data.get('time', [])
    
    # Enhanced preview plots
    col1, col2 = st.columns(2)
    
    with col1:
        create_professional_plot_header("⚡ ECG Peak Detection Preview")
        
        # Peak detection stats
        if len(peaks) > 1 and len(time_data) > 0:
            peak_intervals = np.diff([time_data[p] for p in peaks if p < len(time_data)])
            if len(peak_intervals) > 0:
                avg_interval = np.mean(peak_intervals)
                hr_from_peaks = 60 / avg_interval
                
                # Enhanced metrics
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("🫀 R-peaks Detected", len(peaks))
                with metric_col2:
                    st.metric("💓 Estimated HR", f"{hr_from_peaks:.1f} BPM")
        
        # Enhanced ECG preview plot
        if len(time_data) > 0:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=time_data,
                y=st.session_state.analyzer.ecg_data['raw'],
                mode='lines',
                name='ECG Signal',
                line=dict(color='#3498db', width=1),
                opacity=0.8
            ))
            
            # Add detected peaks
            if len(peaks) > 0:
                valid_peaks = [p for p in peaks if p < len(time_data)]
                if len(valid_peaks) > 0:
                    fig.add_trace(go.Scatter(
                        x=[time_data[p] for p in valid_peaks],
                        y=[st.session_state.analyzer.ecg_data['raw'][p] for p in valid_peaks],
                        mode='markers',
                        name=f'R-peaks (n={len(valid_peaks)})',
                        marker=dict(color='#e74c3c', size=6, symbol='circle')
                    ))
            
            # Highlight analysis window
            if 'time_window' in st.session_state:
                tw = st.session_state.time_window
                fig.add_vrect(
                    x0=tw['start_time'], x1=tw['end_time'],
                    fillcolor="rgba(255, 193, 7, 0.3)", opacity=0.8,
                    annotation_text=f"Analysis Window ({tw['duration']:.0f}s)", 
                    annotation_position="top left"
                )
            
            duration_min = time_data[-1] / 60 if len(time_data) > 0 else 0
            
            fig.update_layout(
                title=f'ECG Peak Detection - Full Recording ({duration_min:.1f} min)',
                xaxis_title='Time (s)',
                yaxis_title='ECG (mV)',
                height=400,
                showlegend=True,
                hovermode='x unified',
                plot_bgcolor='rgba(248,249,250,0.8)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        close_plot_section()
    
    with col2:
        create_professional_plot_header("🩸 BP Peak Detection Preview")
        
        # BP peak detection stats
        bp_peaks = st.session_state.analyzer.bp_data.get('peaks', [])
        bp_time_data = st.session_state.analyzer.bp_data.get('time', [])
        
        if len(bp_peaks) > 1:
            systolic_values = st.session_state.analyzer.bp_data.get('systolic', [])
            
            if len(systolic_values) > 0:
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("📈 Systolic Peaks", len(bp_peaks))
                with metric_col2:
                    st.metric("🩸 Mean Systolic", f"{np.mean(systolic_values):.1f} mmHg")
        
        # Enhanced BP preview plot
        if len(bp_time_data) > 0:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=bp_time_data,
                y=st.session_state.analyzer.bp_data['raw'],
                mode='lines',
                name='Blood Pressure',
                line=dict(color='#e74c3c', width=1),
                opacity=0.8
            ))
            
            # Add detected peaks
            if len(bp_peaks) > 0:
                valid_peaks = [p for p in bp_peaks if p < len(bp_time_data)]
                if len(valid_peaks) > 0:
                    fig.add_trace(go.Scatter(
                        x=[bp_time_data[p] for p in valid_peaks],
                        y=[st.session_state.analyzer.bp_data['raw'][p] for p in valid_peaks],
                        mode='markers',
                        name=f'Systolic Peaks (n={len(valid_peaks)})',
                        marker=dict(color='#27ae60', size=6, symbol='circle')
                    ))
            
            # Highlight analysis window
            if 'time_window' in st.session_state:
                tw = st.session_state.time_window
                fig.add_vrect(
                    x0=tw['start_time'], x1=tw['end_time'],
                    fillcolor="rgba(255, 193, 7, 0.3)", opacity=0.8,
                    annotation_text=f"Analysis Window ({tw['duration']:.0f}s)", 
                    annotation_position="top left"
                )
            
            duration_min = bp_time_data[-1] / 60 if len(bp_time_data) > 0 else 0
            
            fig.update_layout(
                title=f'BP Peak Detection - Full Recording ({duration_min:.1f} min)',
                xaxis_title='Time (s)',
                yaxis_title='Blood Pressure (mmHg)',
                height=400,
                showlegend=True,
                hovermode='x unified',
                plot_bgcolor='rgba(248,249,250,0.8)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        close_plot_section()
    
    # Enhanced RR Interval Tachogram Preview
    create_professional_plot_header(
        "📈 RR Interval Tachogram Preview",
        "Heart rate variability across the complete recording with analysis window highlighted"
    )
    
    if len(peaks) > 1 and 'rr_intervals' in st.session_state.analyzer.ecg_data:
        rr_intervals = st.session_state.analyzer.ecg_data['rr_intervals']
        rr_time_points = st.session_state.analyzer.ecg_data['td_peaks'][:-1]
        
        # Enhanced tachogram
        fig_tacho = go.Figure()
        
        fig_tacho.add_trace(go.Scatter(
            x=rr_time_points,
            y=rr_intervals,
            mode='lines+markers',
            name='RR Intervals',
            line=dict(color='#9b59b6', width=2),
            marker=dict(size=4, color='#8e44ad')
        ))
        
        # Highlight analysis window
        if 'time_window' in st.session_state:
            tw = st.session_state.time_window
            fig_tacho.add_vrect(
                x0=tw['start_time'], x1=tw['end_time'],
                fillcolor="rgba(255, 193, 7, 0.3)", opacity=0.8,
                annotation_text=f"Analysis Window ({tw['duration']:.0f}s)", 
                annotation_position="top left"
            )
        
        # Enhanced statistics
        mean_rr = np.mean(rr_intervals)
        std_rr = np.std(rr_intervals)
        min_rr = np.min(rr_intervals)
        max_rr = np.max(rr_intervals)
        
        # Add reference lines with better styling
        fig_tacho.add_hline(y=mean_rr, line_dash="dash", line_color="#27ae60", line_width=2,
                           annotation_text=f"Mean: {mean_rr:.1f} ms")
        fig_tacho.add_hline(y=mean_rr + std_rr, line_dash="dot", line_color="#f39c12", line_width=2,
                           annotation_text=f"+1 SD: {mean_rr + std_rr:.1f} ms")
        fig_tacho.add_hline(y=mean_rr - std_rr, line_dash="dot", line_color="#f39c12", line_width=2,
                           annotation_text=f"-1 SD: {mean_rr - std_rr:.1f} ms")
        
        fig_tacho.update_layout(
            title=f'RR Interval Tachogram - Full Recording (Range: {min_rr:.0f}-{max_rr:.0f} ms)',
            xaxis_title='Time (s)',
            yaxis_title='RR Interval (ms)',
            height=450,
            hovermode='x unified',
            showlegend=True,
            plot_bgcolor='rgba(248,249,250,0.8)'
        )
        
        st.plotly_chart(fig_tacho, use_container_width=True)
        
        # Enhanced summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Mean RR", f"{mean_rr:.1f} ms", help="Average RR interval")
        with col2:
            st.metric("📈 RR Std Dev", f"{std_rr:.1f} ms", help="Standard deviation")
        with col3:
            st.metric("📏 RR Range", f"{max_rr - min_rr:.0f} ms", help="Max - Min RR interval")
        with col4:
            cv_rr = (std_rr / mean_rr) * 100
            st.metric("📊 RR CV%", f"{cv_rr:.1f}%", help="Coefficient of variation")
        
        # Window-specific statistics
        if 'time_window' in st.session_state:
            st.markdown("### 🎯 Analysis Window Statistics")
            
            tw = st.session_state.time_window
            
            rr_intervals_np = np.array(rr_intervals)
            rr_time_points_np = np.array(rr_time_points)
            
            # Filter RR intervals to analysis window
            window_mask = (rr_time_points_np >= tw['start_time']) & (rr_time_points_np <= tw['end_time'])
            window_rr = rr_intervals_np[window_mask]
            
            if len(window_rr) > 0:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🔢 Window RR Count", len(window_rr), help="RR intervals in selected window")
                with col2:
                    st.metric("💓 Window Mean RR", f"{np.mean(window_rr):.1f} ms", help="Average RR in window")
                with col3:
                    st.metric("📊 Window RR Std", f"{np.std(window_rr):.1f} ms", help="Std deviation in window")
                with col4:
                    window_cv = (np.std(window_rr) / np.mean(window_rr)) * 100
                    st.metric("📈 Window RR CV%", f"{window_cv:.1f}%", help="CV in selected window")
                
                st.markdown(f"""
                <div class="window-info">
                    <strong>ℹ️ Analysis Preview:</strong> The selected {tw['duration']:.0f}-second window contains 
                    {len(window_rr)} RR intervals ready for comprehensive HRV analysis.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-box">
                    <strong>⚠️ Warning:</strong> No RR intervals found in the selected time window. 
                    Please adjust the window or peak detection parameters.
                </div>
                """, unsafe_allow_html=True)
        
    else:
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Warning:</strong> Not enough R-peaks detected for RR interval analysis. 
            Try adjusting ECG parameters in the sidebar.
        </div>
        """, unsafe_allow_html=True)
    
    close_plot_section()
    
    # Enhanced action buttons
    st.markdown("---")
    st.markdown("### 🚀 Next Steps")
    st.markdown("Choose your next action based on the preview results above:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Accept & Run Full Analysis", type="primary", use_container_width=True,
                    help="Proceed with comprehensive HRV and BRS analysis using current settings"):
            st.session_state.analysis_started = True
            
            with st.spinner("🔄 Running comprehensive cardiovascular analysis..."):
                try:
                    time_window = st.session_state.get('time_window', None)
                    success = run_analysis_with_progress(time_window)
                    
                    if success:
                        st.session_state.analyzed = True
                        st.session_state.preview_mode = False
                        st.session_state.analysis_started = False
                        st.success("🎉 Complete analysis finished successfully!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.session_state.analysis_started = False
                        
                except Exception as e:
                    st.session_state.analysis_started = False
                    st.error(f"❌ Analysis failed: {str(e)}")
    
    with col2:
        if st.button("🔄 Adjust Parameters", use_container_width=True,
                    help="Return to parameter adjustment mode"):
            st.session_state.preview_mode = False
            st.info("👈 Adjust parameters or time window in the sidebar and click 'Preview' again")
            st.rerun()
    
    with col3:
        if st.button("📊 Use Default Settings", use_container_width=True,
                    help="Run analysis with original default parameters"):
            st.session_state.analysis_started = True
            
            with st.spinner("🔄 Analyzing with default parameters..."):
                try:
                    st.session_state.analyzer.find_peaks()  # Original method
                    time_window = st.session_state.get('time_window', None)
                    success = run_analysis_with_progress(time_window)
                    
                    if success:
                        st.session_state.analyzed = True
                        st.session_state.preview_mode = False
                        st.session_state.analysis_started = False
                        st.success("🎉 Analysis with defaults completed!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.session_state.analysis_started = False
                        
                except Exception as e:
                    st.session_state.analysis_started = False
                    st.error(f"❌ Analysis failed: {str(e)}")

# Case 3: No file loaded - Enhanced welcome screen
else:
    # Professional welcome message
    st.markdown("""
    <div class="window-info">
        <h3 style="margin: 0;">👋 Welcome to PhysioKit</h3>
        <p style="margin: 0.5rem 0 0 0;">Upload an ACQ file using the sidebar to begin your cardiovascular analysis journey</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced feature showcase
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎛️ Advanced Features</h3>
            <ul style="margin: 0; padding-left: 1.2rem;">
                <li><strong>Time Window Selection:</strong> Focus analysis on specific recording segments</li>
                <li><strong>Adjustable Peak Detection:</strong> Fine-tune ECG and BP peak identification</li>
                <li><strong>Real-time Preview:</strong> Validate settings before full analysis</li>
                <li><strong>Interactive Visualizations:</strong> Explore results with dynamic plots</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Analysis Capabilities</h3>
            <ul style="margin: 0; padding-left: 1.2rem;">
                <li><strong>Time Domain HRV:</strong> RMSSD, SDNN, pNN50, Poincaré analysis</li>
                <li><strong>Frequency Domain:</strong> VLF, LF, HF power analysis</li>
                <li><strong>Baroreflex Sensitivity:</strong> Sequence and spectral methods</li>
                <li><strong>Professional Reports:</strong> Comprehensive analysis summaries</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🚀 How to Get Started</h3>
            <ol style="margin: 0; padding-left: 1.2rem;">
                <li><strong>Upload:</strong> Select your ACQ file in the sidebar</li>
                <li><strong>Configure:</strong> Choose analysis time window</li>
                <li><strong>Adjust:</strong> Fine-tune peak detection parameters</li>
                <li><strong>Preview:</strong> Validate settings with real-time preview</li>
                <li><strong>Analyze:</strong> Run comprehensive cardiovascular analysis</li>
                <li><strong>Explore:</strong> Generate interactive visualizations</li>
                <li><strong>Download:</strong> Export professional analysis reports</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h3>⏱️ Time Window Benefits</h3>
            <ul style="margin: 0; padding-left: 1.2rem;">
                <li><strong>Quality Control:</strong> Exclude artifacts and movement periods</li>
                <li><strong>Focused Analysis:</strong> Analyze specific conditions or timeframes</li>
                <li><strong>Reproducibility:</strong> Standardize analysis windows</li>
                <li><strong>Efficiency:</strong> Process relevant data segments only</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Professional footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%); 
           border-radius: 10px; margin-top: 2rem;">
    <p style="margin: 0; color: #6c757d; font-size: 0.9rem;">
        <strong>PhysioKit v1.0</strong> | Professional HRV & BRS Analysis Platform<br>
        Built with Streamlit • Enhanced User Experience • Advanced Peak Detection • Time Window Selection
    </p>
</div>
""", unsafe_allow_html=True)