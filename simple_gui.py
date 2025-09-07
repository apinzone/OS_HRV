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
from matplotlib.patches import Polygon
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# EDF support with graceful fallback

# Professional color constants
COLORS = {
    'primary': '#2563eb',
    'secondary': '#64748b', 
    'success': '#059669',
    'warning': '#d97706',
    'danger': '#dc2626',
    'ecg': '#3498db',
    'bp': '#e74c3c',
    'rr': '#9b59b6',
    'background': 'rgba(248,249,250,0.8)',
    'grid': 'rgba(108,117,125,0.2)',
    'window': 'rgba(255, 193, 7, 0.3)'
}

def apply_professional_layout(fig, title, xaxis_title, yaxis_title, height=500):
    """Apply consistent professional styling to all plots"""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, weight=600, family="Inter"),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(size=12, weight=500, color="black")),
            gridcolor=COLORS['grid'],
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(size=12, weight=500, color="black")),
            gridcolor=COLORS['grid'],
            showgrid=True,
            zeroline=False
        ),
        plot_bgcolor=COLORS['background'],
        paper_bgcolor='white',
        font=dict(family="Inter, sans-serif", size=11, color="black"),
        height=height,
        margin=dict(l=60, r=60, t=60, b=60),
        hovermode='x unified'
    )
    return fig

try:
    import pyedflib
    EDF_AVAILABLE = True
except ImportError:
    EDF_AVAILABLE = False
    print("⚠️  pyedflib not available. Install with: pip install pyedflib")

# ============================================================================
# PAGE CONFIGURATION & STYLING
# ============================================================================

import base64
def get_page_icon():
    try:
        with open("logo.png", "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{data}"
    except:
        return "⚡"  # Fallback to emoji

st.set_page_config(
    page_title="ChronOS - HRV & BRS Analysis", 
    page_icon=get_page_icon(),
    layout="wide",
    initial_sidebar_state="expanded"
)
# Professional CSS styling
st.markdown("""
<style>
    /* Import professional font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables for consistent theming */
    /* These are just variables that get substituted throughout the CSS */
    :root {
        --primary: #2563eb;
        --primary-dark: #1d4ed8;
        --secondary: #64748b;
        --success: #059669;
        --warning: #d97706;
        --danger: #dc2626;
        --notification-color: 
        --surface: #ffffff;
        --surface-alt: #f8fafc;
        --border: rgba(0, 0, 0, 0.7);
        --highlight-border: rgb(255, 75, 25);
        --text: #1e293b;
        --text-muted: #64748b;
        --shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        --border-opacity: 0.5;
    }
    
    /* FORCE LIGHT MODE */
    *, *::before, *::after {
        color-scheme: light !important;
    }
    
    html[data-theme="dark"], 
    body[data-theme="dark"],
    .stApp[data-theme="dark"] {
        color-scheme: light !important;
        background-color: var(--surface) !important;
        color: var(--text) !important;
    }
    
    /* Force sidebar to light theme */
    .css-1d391kg, .css-17eq0hr, .css-1lcbmhc, .css-1cypcdb,
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div,
    .css-1544g2n, .sidebar .sidebar-content,
    section[data-testid="stSidebar"] {
        background-color: var(--surface-alt) !important;
        color: var(--text) !important;
    }
    
    /* Force ALL text elements */
    [data-testid="stSidebar"] *, 
    .css-1d391kg *,
    .sidebar-section *,
    .stSelectbox *,
    .stSlider *,
    .stExpander *,
    .stMarkdown * {
        color: var(--text) !important;
        background-color: transparent !important;
    }
    
    /* Force input elements */
    input, textarea, select, option {
        background-color: var(--surface) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
    }
    
    /* Override Streamlit app defaults */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
        color: var(--text) !important;
        color-scheme: light !important;
    }

    /* Professional glassmorphism header */
    .main-header {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.95) 0%, rgba(79, 70, 229, 0.95) 100%) !important;
        backdrop-filter: blur(20px);
        border: 1px solid var(--border);
        color: white !important;
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-lg);
        text-align: center;
    }
    
    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9) !important;
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
    }
    
    .version-info {
        color: rgba(255, 255, 255, 0.8) !important;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Clean status container */
    .status-container {
        background: var(--surface);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
    }
    
    .status-item {
        text-align: center;
        padding: 1rem;
        background: var(--surface-alt);
        border-radius: 8px;
        border: 1px solid var(--border);
        transition: all 0.2s ease;
    }
    
    .status-success { color: var(--success); font-weight: 600; font-size: 0.9rem; }
    .status-warning { color: var(--warning); font-weight: 600; font-size: 0.9rem; }
    .status-info { color: var(--primary); font-weight: 600; font-size: 0.9rem; }
    .status-pending { color: var(--secondary); font-weight: 600; font-size: 0.9rem; }
    
    /* Kubios-style metric cards */
    .metric-card {
        background: var(--surface);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid var(--primary);
        margin: 1rem 0;
        box-shadow: var(--shadow);
        border: 1px solid var(--border, var(--border-opacity));
    }
    
    .metric-card h4 {
        color: var(--text);
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .metric-card p {
        color: var(--text);
        margin: 0.5rem 0;
        font-weight: 500;
    }

    .invis-card {
        background: var(--surface);
        padding: 1.5rem;
        # border-radius: 12px;
        # border: 1px solid var(--border, var(--border-opacity));
        margin: 1rem 0;
    }
    
    /* Professional sidebar */
    .sidebar-section {
        background: var(--surface);
        padding: 1.25rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
    }
    
    /* Professional info boxes */
    .window-info {
        background: linear-gradient(90deg, rgba(234, 179, 8, 0.1) 0%, rgba(251, 191, 36, 0.1) 100%);
        padding: 1.25rem;
        border-radius: 10px;
        border-left: 4px solid var(--warning);
        margin: 1rem 0;
        border: 1px solid rgba(234, 179, 8, var(--border-opacity));
    }
    
    /* Clean button styling */
    .stButton > button {
        border-radius: 8px;
        border: 1px solid var(--border);
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.2s ease;
        background: var(--surface);
        color: var(--text);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow);
        background: var(--primary);
        color: white;
        border-color: var(--highlight-border);
    }
    
    /* Professional results header */
    .results-header {
        background: linear-gradient(90deg, rgba(5, 150, 105, 0.1) 0%, rgba(16, 185, 129, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid var(--success);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(5, 150, 105, 0.2);
        box-shadow: var(--shadow);
    }
    
    /* Clean plot sections */
    .plot-section {
        background: var(--surface);
        padding: 1.5rem;
        # border-radius: 12px;
        # border: 1px solid var(--border);
        margin: 1rem 0;
        # box-shadow: var(--shadow);
    }
    
    .plot-section h3 {
        color: var(--text);
        font-weight: 600;
        margin-top: 0;
    }
    
    /* Professional error/warning styling */
    .error-box {
        background: linear-gradient(90deg, rgba(220, 38, 38, 0.1) 0%, rgba(239, 68, 68, 0.1) 100%);
        padding: 1.25rem;
        border-radius: 10px;
        border-left: 4px solid var(--danger);
        margin: 1rem 0;
        border: 1px solid rgba(220, 38, 38, 0.2);
        color: var(--text);
    }
    
    .warning-box {
        background: linear-gradient(90deg, rgba(234, 179, 8, 0.1) 0%, rgba(251, 191, 36, 0.1) 100%);
        padding: 1.25rem;
        border-radius: 10px;
        border-left: 4px solid var(--warning);
        margin: 1rem 0;
        border: 1px solid rgba(234, 179, 8, var(--border-opacity));
        color: var(--text);
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Force all text to use our color scheme */
    h1, h2, h3, h4, h5, h6 { color: var(--text) !important; }
    p, span, div, label { color: var(--text) !important; }
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
if 'channels_info' not in st.session_state:
    st.session_state.channels_info = []
if 'channels_configured' not in st.session_state:
    st.session_state.channels_configured = False

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _contiguous_runs(mask):
    """Return (start_idx, end_idx) for True runs in a boolean mask (end exclusive)."""
    if mask.size == 0: 
        return []
    m = mask.astype(int)
    dm = np.diff(np.r_[0, m, 0])
    starts = np.where(dm == 1)[0]
    ends   = np.where(dm == -1)[0]
    return list(zip(starts, ends))

def _band_fill(ax, f, y, lo, hi, *, scale=1.0, facecolor="#cccccc", alpha=0.4, label=None, zorder=2):
    """
    Fill area under y between lo..hi with vertical edges and no diagonal bridges.
    Handles internal NaNs by splitting into contiguous segments.
    """
    f = np.asarray(f, float)
    y = np.asarray(y, float)

    # Clean & sort
    good = np.isfinite(f) & np.isfinite(y) & (f > 0)
    f, y = f[good], y[good]
    order = np.argsort(f)
    f, y = f[order], y[order]
    f, uniq_idx = np.unique(f, return_index=True)
    y = y[uniq_idx]

    if f.size == 0 or hi <= f[0] or lo >= f[-1]:
        return None

    lo_c = max(lo, f[0])
    hi_c = min(hi, f[-1])
    if lo_c >= hi_c:
        return None

    # In-band mask and split into contiguous runs (avoids NaN gaps)
    inside = (f > lo_c) & (f < hi_c)
    runs = _contiguous_runs(inside)
    legend_patch = None

    for s, e in runs:
        fx = f[s:e]
        yx = y[s:e]

        # Interpolate values at exact band edges for vertical sides
        y_lo = float(np.interp(lo_c, f, y))
        y_hi = float(np.interp(hi_c, f, y))

        # Build polygon: (lo,0) -> (lo,ylo) -> (fx,yx) -> (hi,yhi) -> (hi,0)
        x_poly = np.concatenate([[lo_c], [lo_c], fx, [hi_c], [hi_c]])
        y_poly = np.concatenate([[0.0], [y_lo], yx, [y_hi], [0.0]]) * scale

        patch = Polygon(np.column_stack([x_poly, y_poly]),
                        closed=True, facecolor=facecolor, edgecolor='none',
                        alpha=alpha, zorder=zorder)
        ax.add_patch(patch)
        if legend_patch is None and label:
            # one invisible patch for legend entry
            legend_patch = Polygon([[0,0],[1,0],[1,1]], closed=True,
                                   facecolor=facecolor, edgecolor='none', alpha=alpha, label=label)
            ax.add_patch(legend_patch)
            legend_patch.set_visible(False)

    return legend_patch

def show_professional_header():
    """Display header with PNG logo"""
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
            <div style="width: 115px; height: 115px; margin-right: 10px;">
                <img src="data:image/png;base64,{}" 
                     style="width: 100%; height: 100%; object-fit: contain;" 
                     alt="ChronOS Logo"/>
            </div>
            <h1 style="margin: 0; font-size: 48px;">ChronOS</h1>
        </div>
        <p>Professional HRV & Baroreflex Sensitivity Analysis Platform</p>
        <div class="version-info">Version 1.3 | Advanced Peak Detection | HRV and BRS Analysis</div>
    </div>
    """.format(get_base64_of_image("logo.png")), unsafe_allow_html=True)

def get_base64_of_image(path):
    """Convert image to base64 string for embedding in HTML"""
    import base64
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
    
def show_analysis_status():
    """Display current analysis status with professional indicators"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.session_state.file_loaded:
            st.markdown('<div class="status-item"><div class="status-success">✅ File Loaded</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-item"><div class="status-pending">📁 Upload File</div></div>', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.channels_configured:
            st.markdown('<div class="status-item"><div class="status-success">✅ Channels Set</div></div>', unsafe_allow_html=True)
        elif st.session_state.file_loaded:
            st.markdown('<div class="status-item"><div class="status-warning">🔧 Configure Channels</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-item"><div class="status-pending">🔧 Channels Not Set</div></div>', unsafe_allow_html=True)
    
    with col3:
        if st.session_state.analyzed:
            st.markdown('<div class="status-item"><div class="status-success">✅ Analyzed</div></div>', unsafe_allow_html=True)
        elif st.session_state.analysis_started:
            st.markdown('<div class="status-item"><div class="status-warning">⏳ Processing</div></div>', unsafe_allow_html=True)
        elif st.session_state.preview_mode:
            st.markdown('<div class="status-item"><div class="status-info">🔍 Analysis Preview</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-item"><div class="status-pending">⏳ Analysis Pending</div></div>', unsafe_allow_html=True)
    
    with col4:
        if 'selected_plots' in st.session_state and st.session_state.selected_plots:
            st.markdown('<div class="status-item"><div class="status-success">📊 Plots Ready</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-item"><div class="status-pending">📈 No Plots Generated</div></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def run_analysis_with_progress(time_window=None):
    """Run analysis with detailed progress feedback and professional styling"""
    
    # Create progress container
    st.markdown("### 🔄 Analysis in Progress")
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:

    # Step 1: Peak Detection
        status_text.markdown("**🔍 Step 1/5:** Detecting ECG R-peaks and BP systolic peaks...")
        progress_bar.progress(10)

        # Check analysis capabilities first
        if not st.session_state.analyzer.channels_configured:
            raise Exception("Channels not configured. Please configure channels first.")
        
        capabilities = st.session_state.analyzer.get_analysis_capabilities()

        try:
            if hasattr(st.session_state, 'peak_params'):
                st.session_state.analyzer.find_peaks_with_params(**st.session_state.peak_params)
            else:
                st.session_state.analyzer.find_peaks()
        except Exception as peak_error:
            status_text.markdown("**⚠️ Step 1:** Standard detection failed, trying adaptive method...")
            st.session_state.analyzer.find_peaks_adaptive()
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

def auto_scale_peak_parameters(analyzer):
    """Automatically calculate optimal peak detection parameters"""
    auto_params = {}
    
    try:
        # ECG Auto-scaling
        if hasattr(analyzer, 'ecg_data') and analyzer.ecg_data and 'raw' in analyzer.ecg_data:
            ecg_signal = analyzer.ecg_data['raw']
            ecg_max = np.max(ecg_signal)
            ecg_range = ecg_max - np.min(ecg_signal)
            
            auto_params['ecg_height'] = max(0.1, ecg_max * 0.7)
            auto_params['ecg_prominence'] = max(0.1, ecg_range * 0.25)
            auto_params['ecg_distance'] = 100
        else:
            auto_params['ecg_height'] = 0.8
            auto_params['ecg_prominence'] = 0.7
            auto_params['ecg_distance'] = 100
        
        # BP Auto-scaling 
        if hasattr(analyzer, 'bp_data') and analyzer.bp_data and 'raw' in analyzer.bp_data:
            bp_signal = analyzer.bp_data['raw']
            bp_max = np.max(bp_signal)
            bp_range = bp_max - np.min(bp_signal)
            bp_mean = np.mean(bp_signal)
            bp_std = np.std(bp_signal, ddof=1)
            
            auto_params['bp_height'] = int(max(80, bp_max * 0.6))  
            calculated_prominence = max(2, bp_std * 0.5)
            auto_params['bp_prominence'] = int(min(8, calculated_prominence))        
            auto_params['bp_distance'] = 100
        else:
            auto_params['bp_height'] = 110
            auto_params['bp_prominence'] = 5
            auto_params['bp_distance'] = 100
        
        return auto_params, True, "Auto-scale successful"
        
    except Exception as e:
        return {
            'ecg_height': 0.8, 'ecg_prominence': 0.7, 'ecg_distance': 100,
            'bp_height': 110, 'bp_prominence': 5, 'bp_distance': 100
        }, False, f"Auto-scale failed: {str(e)}"

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
    
    # Enhanced file upload with EDF support (CHANGE 2)
    if EDF_AVAILABLE:
        file_types = ["acq", "edf"]
        help_text = "Upload your ACQ file (AcqKnowledge) or EDF file (European Data Format) containing ECG and blood pressure data"
    else:
        file_types = ["acq"]
        help_text = "Upload your ACQ file containing ECG and blood pressure data. For EDF support, install pyedflib: pip install pyedflib"
    
    uploaded_file = st.file_uploader(
        "Choose a physiological data file", 
        type=file_types,
        help=help_text
    )
    
    if uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1].lower()  # CHANGE 3
        
        if file_ext in ['edf'] and not EDF_AVAILABLE:  # CHANGE 3
            st.error("❌ EDF files require pyedflib. Install with: pip install pyedflib")
        else:
            file_info = f"**File:** {uploaded_file.name}\n\n**Size:** {uploaded_file.size / 1024:.1f} KB\n\n**Type:** {file_ext.upper()}"  # CHANGE 3
            st.info(file_info)
            
            if st.button("🔄 Load File", type="primary", use_container_width=True):
                with st.spinner(f"Loading {file_ext.upper()} file and detecting channels..."):  # CHANGE 3
                    try:
                        # Save uploaded file temporarily (CHANGE 4)
                        file_suffix = f".{file_ext}"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        # Load file and detect channels
                        channels_info = st.session_state.analyzer.load_file_and_detect_channels(tmp_file_path)
                        
                        # Clean up temp file
                        os.unlink(tmp_file_path)
                        
                        # Store channel info for selection
                        st.session_state.channels_info = channels_info
                        st.session_state.file_loaded = True
                        st.session_state.channels_configured = False
                        st.session_state.analyzed = False
                        st.session_state.preview_mode = False
                        
                        st.success(f"✅ {file_ext.upper()} file loaded! Found {len(channels_info)} channels. Please configure channels below.")  # CHANGE 5
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ {file_ext.upper()} file loading failed: {str(e)}")  # CHANGE 6
                        st.session_state.file_loaded = False
                        # Clean up temp file on error (CHANGE 6)
                        try:
                            os.unlink(tmp_file_path)
                        except:
                            pass
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Channel Selection Interface
    if st.session_state.file_loaded and not st.session_state.channels_configured:
        st.markdown("## 🔧 Channel Configuration")
        
        # Show file type information (CHANGE 7)
        file_type = getattr(st.session_state.analyzer, 'file_type', 'unknown').upper()
        st.markdown(f"**📁 File Type:** {file_type}")
    
    # Display available channels
    st.markdown("### Available Channels:")
    for ch in st.session_state.channels_info:
        # Create color coding based on likely type (CHANGE 8)
        if 'ECG' in ch['likely_type']:
            type_color = "🟢"
        elif 'BP' in ch['likely_type']:
            type_color = "🔵"
        elif 'Respiratory' in ch['likely_type']:
            type_color = "🟡"
        elif 'EMG' in ch['likely_type']:
            type_color = "🟠"
        else:
            type_color = "⚪"
        
        st.markdown(f"""
        {type_color} **Channel {ch['index']}:** {ch['name']} ({ch['units']})  
        📊 Range: {ch['data_range']} | Type: {ch['likely_type']}  
        ⏱️ Duration: {ch['duration']:.1f}s | 📈 Rate: {ch['sample_rate']:.1f} Hz
        """)
    
    st.markdown("---")
    
    # Channel selection dropdowns
    ecg_options = ["None"] + [f"Channel {ch['index']}: {ch['name']}" for ch in st.session_state.channels_info]
    bp_options = ["None"] + [f"Channel {ch['index']}: {ch['name']}" for ch in st.session_state.channels_info]
    
    # Find suggested defaults
    ecg_default = 0  # Default to "None"
    bp_default = 0   # Default to "None"
    
    for i, ch in enumerate(st.session_state.channels_info):
        if 'ECG' in ch['likely_type'] and ecg_default == 0:
            ecg_default = i + 1  # +1 because "None" is first option
        elif 'BP' in ch['likely_type'] and bp_default == 0:
            bp_default = i + 1   # +1 because "None" is first option
    
    ecg_selection = st.selectbox(
        "⚡ Select ECG Channel:",
        ecg_options,
        index=ecg_default,
        help="Choose the channel containing ECG/EKG data"
    )
    
    bp_selection = st.selectbox(
        "🩸 Select BP Channel:",
        bp_options,
        index=bp_default,
        help="Choose the channel containing blood pressure data"
    )
    
    # Show what analysis will be available
    ecg_selected = ecg_selection != "None"
    bp_selected = bp_selection != "None"
    
    if ecg_selected or bp_selected:
        st.markdown("### 📊 Available Analyses:")
        available_analyses = []
        if ecg_selected:
            available_analyses.extend(["Time Domain HRV", "Frequency Domain HRV"])
        if ecg_selected and bp_selected:
            available_analyses.extend(["BRS Sequence Method", "BRS Spectral Method"])
        elif bp_selected and not ecg_selected:
            available_analyses.append("Blood Pressure Analysis")
        
        for analysis in available_analyses:
            st.markdown(f"- {analysis}")
    
    # Configure button
    if st.button("Configure Channels", type="primary", use_container_width=True):
        try:
            # Parse selections
            ecg_idx = None if ecg_selection == "None" else int(ecg_selection.split(":")[0].replace("Channel ", ""))
            bp_idx = None if bp_selection == "None" else int(bp_selection.split(":")[0].replace("Channel ", ""))
            
            if ecg_idx is None and bp_idx is None:
                st.error("❌ Please select at least one channel (ECG or BP)")
            else:
                # Configure channels
                success_msgs = st.session_state.analyzer.configure_channels(ecg_idx, bp_idx)
                
                # Show capabilities
                capabilities = st.session_state.analyzer.get_analysis_capabilities()
                
                st.session_state.channels_configured = True
                st.session_state.analyzed = False
                
                success_text = "\n".join(success_msgs)
                
                # Show what analysis can be performed
                available_analyses = []
                if capabilities['time_domain_hrv']:
                    available_analyses.append("Time Domain HRV")
                if capabilities['frequency_domain_hrv']:
                    available_analyses.append("Frequency Domain HRV")
                if capabilities['brs_sequence']:
                    available_analyses.append("BRS Sequence Method")
                if capabilities['brs_spectral']:
                    available_analyses.append("BRS Spectral Method")
                
                success_text += f"\n\n📊 Available analyses: {', '.join(available_analyses)}"
                
                st.success(success_text)
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Channel configuration failed: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Time Window Selection (only if file is loaded)
    # Time Window Selection and Peak Detection Parameters (only if channels are configured)
    if st.session_state.file_loaded and st.session_state.channels_configured:
        
        # Show current channel configuration
        analyzer = st.session_state.analyzer
        
        st.markdown("## Current Configuration")

        config_info = f"**📁 File Type:** {getattr(analyzer, 'file_type', 'Unknown').upper()}\n\n"
        if analyzer.ecg_data:
            scale_info = f" ({analyzer.ecg_data['detected_scale']} detected)" if 'detected_scale' in analyzer.ecg_data else ""
            config_info += f"⚡ **ECG:** Channel {analyzer.ecg_channel} - {analyzer.ecg_data['channel_name']}{scale_info}\n\n"
        if analyzer.bp_data:
            config_info += f"🩸 **BP:** Channel {analyzer.bp_channel} - {analyzer.bp_data['channel_name']}"

        st.markdown(config_info)
        st.markdown('</div>', unsafe_allow_html=True)
    
        # Time Window Selection (only if file is loaded)
        if not st.session_state.analyzed:
            st.markdown("## ⏱️ Analysis Window")
            
            if hasattr(st.session_state.analyzer, 'ecg_data') and 'time' in st.session_state.analyzer.ecg_data:
                max_time = max(st.session_state.analyzer.ecg_data['time'])
                max_time_min = max_time / 60
                
                st.info(f"**Recording:** {max_time:.1f}s ({max_time_min:.1f} min)")
                
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
                
                st.success(f"**Window:** {window_duration:.0f}s ({window_duration_min:.1f} min)")
                
                st.session_state.time_window = {
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': window_duration
                }
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Peak Detection Parameters
        st.markdown("## Peak Detection")

        # Auto-scale button
        if st.button("🎯 Auto-Scale", help="Automatically calculate optimal parameters based on your signals", use_container_width=True):
            with st.spinner("Calculating optimal parameters..."):
                auto_params, success, message = auto_scale_peak_parameters(st.session_state.analyzer)
                
                if success:
                    # Force update session state
                    st.session_state.ecg_height = auto_params['ecg_height']
                    st.session_state.ecg_prominence = auto_params['ecg_prominence'] 
                    st.session_state.ecg_distance = auto_params['ecg_distance']
                    st.session_state.bp_height = auto_params['bp_height']
                    st.session_state.bp_prominence = auto_params['bp_prominence']
                    st.session_state.bp_distance = auto_params['bp_distance']
                    
                    # Force slider reset by incrementing counter
                    st.session_state.reset_counter = st.session_state.get('reset_counter', 0) + 1
                    st.session_state.force_slider_reset = True
                
                    st.success(f"✅ {message}")
                    st.info(f"📊 **Auto-calculated parameters:**\n"
                        f"• ECG Height: {auto_params['ecg_height']:.2f}\n" 
                        f"• ECG Prominence: {auto_params['ecg_prominence']:.2f}\n"
                        f"• BP Height: {auto_params['bp_height']:.1f} mmHg\n"
                        f"• BP Prominence: {auto_params['bp_prominence']:.1f}")
                    st.rerun()
                else:
                    st.warning(f"⚠️ {message}")

        st.markdown("**Parameter Settings:**")

        # ECG Parameters
        with st.expander("⚡ ECG R-peak Detection", expanded=True):
            # Check for reset and clear flag
            if st.session_state.get('force_slider_reset', False):
                st.session_state.force_slider_reset = False  # Clear the flag immediately
            
            ecg_height_default = st.session_state.get('ecg_height', 0.8)
            ecg_distance_default = st.session_state.get('ecg_distance', 100)
            ecg_prominence_default = st.session_state.get('ecg_prominence', 0.7)

            # Force unique keys when resetting
            reset_key = st.session_state.get('reset_counter', 0)
            
            ecg_height = st.slider("Height Threshold", 0.1, 2.0, ecg_height_default, 0.1, 
                                key=f"ecg_height_{reset_key}", help="Minimum R-peak amplitude")
            ecg_distance = st.slider("Min Distance", 50, 200, ecg_distance_default, 10, 
                                    key=f"ecg_distance_{reset_key}", help="Min samples between peaks")
            ecg_prominence = st.slider("Prominence", 0.1, 1.5, ecg_prominence_default, 0.1, 
                                    key=f"ecg_prominence_{reset_key}", help="Peak prominence")

        # BP Parameters
        with st.expander("🩸 BP Systolic Detection", expanded=True):
            bp_height_default = st.session_state.get('bp_height', 110)
            bp_distance_default = st.session_state.get('bp_distance', 100)
            bp_prominence_default = st.session_state.get('bp_prominence', 5)
            
            reset_key = st.session_state.get('reset_counter', 0)
            
            bp_height = st.slider("BP Height (mmHg)", 80, 150, bp_height_default, 5, 
                                key=f"bp_height_{reset_key}", help="Min systolic pressure")
            bp_distance = st.slider("BP Min Distance", 50, 200, bp_distance_default, 10, 
                                key=f"bp_distance_{reset_key}", help="Min samples between peaks")
            bp_prominence = st.slider("BP Prominence", 1, 10, bp_prominence_default, 1, 
                                    key=f"bp_prominence_{reset_key}", help="Peak prominence")
        st.session_state.peak_params = {
            'ecg_height': ecg_height,
            'ecg_distance': ecg_distance,
            'ecg_prominence': ecg_prominence,
            'bp_height': bp_height,
            'bp_distance': bp_distance,
            'bp_prominence': bp_prominence
        }

        # Enhanced preview button
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
        
        plot_options = [
            "Interactive Tachogram",
            "RRI Histogram",
            "Frequency Domain",
            "Poincaré Plot",
            "Spectral BRS Analysis", 
            "BRS Sequence Analysis",
            "BRS Time Domain Visualization"
        ]

        selected_plots = st.multiselect(
            "Select visualizations:",
            plot_options,
            default=plot_options[:4],
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
if st.session_state.analyzed and st.session_state.channels_configured:
    # Results Header with scale info and file type
    scale_note = ""
    if hasattr(st.session_state.analyzer, 'get_scale_info'):
        scale_info = st.session_state.analyzer.get_scale_info()
        if scale_info['conversion_applied']:
            scale_note = f" | ECG converted from {scale_info['detected_scale']} to mV"

    file_type_note = f" | {getattr(st.session_state.analyzer, 'file_type', 'Unknown').upper()} file"

    st.markdown(f"""
    <div class="results-header">
        <h2 style="margin: 0; color: #155724;">🎉 Analysis Complete</h2>
        <p style="margin: 0.5rem 0 0 0; color: #155724;">Comprehensive cardiovascular analysis finished successfully{scale_note}{file_type_note}</p>
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
    # col1, col2 = st.columns([1, 2])
    
        
    #     # BRS Results 
    #     if 'brs_sequence' in st.session_state.analyzer.results:
    #         brs_data = st.session_state.analyzer.results['brs_sequence']
    #         if 'error' not in brs_data:
    #             st.markdown(f"""
    #             <div class="metric-card">
    #                 <h4>🩺 Baroreflex Sensitivity</h4>
    #                 <p><strong>BRS Mean:</strong> {brs_data.get('BRS_mean', 'N/A'):.2f} ms/mmHg</p>
    #                 <p><strong>BEI:</strong> {brs_data.get('BEI', 'N/A'):.2f}</p>
    #                 <p><strong>Valid Sequences:</strong> {brs_data.get('num_sequences', 'N/A')}</p>
    #             </div>
    #             """, unsafe_allow_html=True)



    
    st.markdown("### 📊 Interactive Visualizations")
    
    # Display selected plots with professional styling
    if 'selected_plots' in st.session_state:
        if "Interactive Tachogram" in st.session_state.selected_plots:
            with st.container(border=True, key="tachogram_container"):

                create_professional_plot_header(
                    "Heart Rate Variability Tachogram",
                    "Interactive visualization of RR interval variations over time"
                )

                col1, col2 = st.columns([0.3, 0.7])
                with col1:
                    td_results = st.session_state.analyzer.results['time_domain']
                    if 'error' not in td_results:
                        # Time Domain Metrics
                        st.markdown(f"""
                        <div class="invis-card">
                            <h4> HRV Metrics </h4>
                            <h5>Basic Measures</h5>
                            <p><strong>Beats:</strong> {td_results.get('num_beats', 'N/A')}</p>
                            <p><strong>Mean RR:</strong> {td_results.get('mean_rr', 'N/A'):.1f} ms</p>
                            <p><strong>HR:</strong> {td_results.get('hr', 'N/A'):.1f} BPM</p>
                            <br>
                            <h5>Time Domain Metrics</h5>
                            <p><strong>Mean RR:</strong> {td_results.get('mean_rr', 'N/A'):.1f} ms</p>
                            <p><strong>RMSSD:</strong> {td_results.get('rmssd', 'N/A'):.1f} ms</p>
                            <p><strong>SDNN:</strong> {td_results.get('sdnn', 'N/A'):.1f} ms</p>
                            <p><strong>SDSD:</strong> {td_results.get('sdsd', 'N/A'):.1f} ms</p>
                            <p><strong>pNN50:</strong> {td_results.get('pnn50', 'N/A'):.1f} %</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    fig = go.Figure()
                    
                    rr_intervals = st.session_state.analyzer.ecg_data['rr_intervals']
                    time_points = st.session_state.analyzer.ecg_data['td_peaks'][:-1]
                    
                    # Enhanced RR intervals trace
                    fig.add_trace(go.Scatter(
                        x=time_points,
                        y=rr_intervals,
                        mode='lines+markers',
                        name='RR Intervals',
                        line=dict(color=COLORS['rr'], width=2.5),
                        marker=dict(size=4, color=COLORS['rr'], opacity=0.8),
                        hovertemplate='<b>Time:</b> %{x:.1f}s<br><b>RR:</b> %{y:.1f}ms<extra></extra>'
                    ))
                    
                    # Highlight analysis window with professional styling
                    if 'time_window' in st.session_state:
                        tw = st.session_state.time_window
                        fig.add_vrect(
                            x0=tw['start_time'], x1=tw['end_time'],
                            fillcolor=COLORS['window'], opacity=0.6,
                            line=dict(color=COLORS['warning'], width=2),
                            annotation_text="Analysis Window", 
                            annotation_position="top left",
                            annotation=dict(font=dict(size=12, color=COLORS['warning']))
                        )
                    
                    # Enhanced statistics reference lines
                    mean_rr = np.mean(rr_intervals)
                    std_rr = np.std(rr_intervals,ddof=1)
                    
                    fig.add_hline(y=mean_rr, line_dash="dash", line_color=COLORS['success'], 
                                line_width=2, opacity=0.8,
                                annotation_text=f"Mean: {mean_rr:.1f}ms")
                    fig.add_hline(y=mean_rr + std_rr, line_dash="dot", line_color=COLORS['secondary'], 
                                line_width=1.5, opacity=0.6,
                                annotation_text=f"+1σ: {mean_rr + std_rr:.1f}ms")
                    fig.add_hline(y=mean_rr - std_rr, line_dash="dot", line_color=COLORS['secondary'], 
                                line_width=1.5, opacity=0.6,
                                annotation_text=f"-1σ: {mean_rr - std_rr:.1f}ms")
                    
                    # Apply professional layout
                    fig = apply_professional_layout(
                        fig, 
                        f'Heart Rate Variability Analysis (μ={mean_rr:.1f}±{std_rr:.1f}ms)',
                        'Time (seconds)', 
                        'RR Interval (ms)', 
                        height=600
                    )
                    fig.update_layout() 
                    
                    st.plotly_chart(fig, use_container_width=True)
                    close_plot_section()
            
            if "RRI Histogram" in st.session_state.selected_plots:
                with st.container(border=True):
                    create_professional_plot_header(
                        "RRI Histogram",
                        "Distribution of RR intervals"
                    )
                    
                    if hasattr(st.session_state.analyzer, 'ecg_data') and 'rr_intervals' in st.session_state.analyzer.ecg_data:
                        rr_intervals = st.session_state.analyzer.ecg_data['rr_intervals']
                        
                        # Calculate basic statistics
                        rr_mean = np.mean(rr_intervals)
                        rr_std = np.std(rr_intervals, ddof=1)
                        
                        # Create histogram 
                        plt.rcParams.update({
                            'font.family': 'sans-serif',
                            'font.size': 11,
                            'axes.titlesize': 16,
                            'axes.titleweight': 'bold',
                            'axes.labelsize': 12,
                            'axes.labelweight': '500',
                            'axes.facecolor': '#f8f9fa',
                            'figure.facecolor': 'white'
                        })
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Create histogram
                        counts, bins, patches = ax.hist(rr_intervals, bins=30, 
                                                    color='#3498db', alpha=0.7, 
                                                    edgecolor='#2980b9', linewidth=0.5)
                        
                        # Add mean line
                        ax.axvline(rr_mean, color='#e74c3c', linestyle='--', linewidth=2,
                                label=f'Mean: {rr_mean:.1f} ms')
                        
                        # Styling
                        ax.set_xlabel('RR Interval (ms)', fontsize=12, fontweight='500')
                        ax.set_ylabel('Frequency', fontsize=12, fontweight='500')
                        ax.set_title(f'RR Interval Distribution (n={len(rr_intervals)})', 
                                    fontsize=16, fontweight='bold', pad=20)
                        
                        # Add statistics text box
                        stats_text = f'Mean: {rr_mean:.1f} ms\nStd: {rr_std:.1f} ms\nCount: {len(rr_intervals)}'
                        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                                verticalalignment='top', horizontalalignment='left',
                                bbox=dict(boxstyle='round,pad=0.8', facecolor='white', 
                                        alpha=0.95, edgecolor='#dee2e6', linewidth=1),
                                fontsize=11, fontweight='500')
                        
                        # Grid and spines
                        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                        ax.set_axisbelow(True)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_linewidth(0.8)
                        ax.spines['bottom'].set_linewidth(0.8)
                        
                        # Legend
                        legend = ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
                        legend.get_frame().set_facecolor('white')
                        legend.get_frame().set_alpha(0.9)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                    else:
                        st.error("❌ No RR interval data available. Complete peak detection analysis first.")
                    
                    close_plot_section()

        
        if "Frequency Domain" in st.session_state.selected_plots:
            with st.container(border=True):
                create_professional_plot_header(
                    "Frequency Domain Analysis",
                    "Power spectral density analysis of heart rate variability"
                )

                col1, col2 = st.columns([0.3, 0.7])
                with col1:
                    results = st.session_state.analyzer.results
                    if 'frequency_domain' in results and 'error' not in results['frequency_domain']:
                        fd = results['frequency_domain']
                        # Time Domain Metrics
                        st.markdown(f"""
                        <div class="invis-card">
                            <h4> 📊 Frequency Domain HRV Metrics </h4>
                            <p><strong>VLF Power:</strong> {fd.get('vlf_power', 'N/A'):.2f} ms²</p>
                            <p><strong>LF Power:</strong> {fd.get('lf_power', 'N/A'):.2f} ms²</p>
                            <p><strong>HF Power:</strong> {fd.get('hf_power', 'N/A'):.2f} ms²</p>
                            <p><strong>Total Power:</strong> {fd.get('total_power', 'N/A'):.2f} ms²</p>
                            <p><strong>LF/HF Ratio:</strong> {fd.get('lf_hf_ratio', 'N/A'):.2f}</p>
                            <p><strong>LF n.u.:</strong> {fd.get('lf_nu', 'N/A'):.2f}</p>
                            <p><strong>HF n.u.:</strong> {fd.get('hf_nu', 'N/A'):.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)

            
            with col2:
                freq_data = st.session_state.analyzer.results['frequency_domain']
                if 'error' not in freq_data:
                    # Enhanced matplotlib styling
                    plt.style.use('default')
                    plt.rcParams.update({
                        'font.family': 'sans-serif',
                        'font.size': 11,
                        'axes.titlesize': 16,
                        'axes.titleweight': 'bold',
                        'axes.labelsize': 12,
                        'axes.labelweight': '500',
                        'axes.facecolor': '#f8f9fa',
                        'figure.facecolor': 'white'
                    })
                    
                    fig, ax = plt.subplots(figsize=(14, 7))
                    frequencies = np.asarray(freq_data['frequencies'], dtype=float)
                    psd = np.asarray(freq_data['psd'], dtype=float)

                    # Clean + strictly increasing + unique x
                    good = np.isfinite(frequencies) & np.isfinite(psd) & (frequencies > 0)
                    frequencies, psd = frequencies[good], psd[good]
                    order = np.argsort(frequencies)
                    frequencies, psd = frequencies[order], psd[order]
                    frequencies, uniq_idx = np.unique(frequencies, return_index=True)
                    psd = psd[uniq_idx]

                    # Band masks with epsilon ~ half a bin to "touch" edges
                    df = np.median(np.diff(frequencies))
                    eps = float(df) * 0.51 if np.isfinite(df) and df > 0 else 1e-12

                    vlf_mask = (frequencies >= (0.003 - eps)) & (frequencies <= (0.04 + eps))
                    lf_mask  = (frequencies >= (0.04  - eps)) & (frequencies <= (0.15 + eps))
                    hf_mask  = (frequencies >= (0.15  - eps)) & (frequencies <= (0.40 + eps))

                    scale = 1e6
                    baseline = np.zeros_like(psd)

                    p_vlf = _band_fill(ax, frequencies, psd, 0.003, 0.04, scale=scale,
                                    facecolor='#95a5a6', alpha=0.4, label='VLF (0.003–0.04 Hz)')
                    p_lf  = _band_fill(ax, frequencies, psd, 0.04,  0.15, scale=scale,
                                    facecolor='#346edb', alpha=0.5, label='LF (0.04–0.15 Hz)')
                    p_hf  = _band_fill(ax, frequencies, psd, 0.15,  0.40, scale=scale,
                                    facecolor='#e74c3c', alpha=0.5, label='HF (0.15–0.40 Hz)')

                    # PSD curve 
                    ax.plot(frequencies, psd*scale, color='#2c3e50', linewidth=2.5, label='PSD', zorder=5)
                    
                    #Graph styling
                    ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='500')
                    ax.set_ylabel('Power Spectral Density (ms²/Hz)', fontsize=12, fontweight='500')
                    ax.set_title('Heart Rate Variability - Frequency Domain Analysis', 
                                fontsize=16, fontweight='bold', pad=20)
                    ax.set_xlim(0, 0.5)
                    
                    # Legend
                    legend_handles = [
                        Patch(facecolor='#95a5a6', edgecolor='none', alpha=0.4, label='VLF (0.003–0.04 Hz)'),
                        Patch(facecolor='#346edb', edgecolor='none', alpha=0.5, label='LF (0.04–0.15 Hz)'),
                        Patch(facecolor='#e74c3c', edgecolor='none', alpha=0.5, label='HF (0.15–0.40 Hz)'),
                        Line2D([0], [0], color='#2c3e50', linewidth=2.5, label='PSD')
                    ]

                    legend = ax.legend(
                        handles=legend_handles,
                        loc='upper right',
                        frameon=True,
                        fancybox=True,
                        shadow=True,
                        fontsize=10,
                        handlelength=1.8,
                        borderaxespad=0.8
                    )
                    legend.get_frame().set_facecolor('white')
                    legend.get_frame().set_alpha(0.9)

                    
                    #Grid
                    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                    ax.set_axisbelow(True)
                    
                    # Remove top and right spines
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_linewidth(0.8)
                    ax.spines['bottom'].set_linewidth(0.8)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.error(f"Frequency domain analysis error: {freq_data['error']}")
                
                close_plot_section()


        if "Poincaré Plot" in st.session_state.selected_plots:
            with st.container(border=True):

                create_professional_plot_header(
                    "Poincaré Plot Analysis",
                    "Nonlinear analysis of heart rate variability patterns"
                )

                col1, col2 = st.columns([0.4, 0.6])
                with col1:
                    td_results = st.session_state.analyzer.results['time_domain']
                    if 'error' not in td_results:
                        # Nonlinear Metrics
                        st.markdown(f"""
                        <div class="invis-card">
                            <h4> 📊 Nonlinear Metrics </h4>
                            <p><strong>SD1:</strong> {td_results.get('sd1', 'N/A'):.1f} ms</p>
                            <p><strong>SD2:</strong> {td_results.get('sd2', 'N/A'):.1f} ms</p>
                            <p><strong>SD1/SD2:</strong> {td_results.get('sd1_sd2_ratio', 'N/A'):.3f}</p>
                            <p><strong>Ellipse Area:</strong> {td_results.get('ellipse_area', 'N/A'):.1f} ms²</p>
                            <p><strong>Sample Entropy:</strong> {td_results.get('sample_entropy', 'N/A'):.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
                with col2:
                        plt.rcParams.update({
                            'font.family': 'sans-serif',
                            'font.size': 11,
                            'axes.titlesize': 16,
                            'axes.titleweight': 'bold',
                            'axes.labelsize': 12,
                            'axes.labelweight': '500',
                            'axes.facecolor': '#f8f9fa',
                            'figure.facecolor': 'white'
                        })
                        
                        fig, ax = plt.subplots(figsize=(11, 9))
                        
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
                        
                        # Enhanced scatter plot with better styling
                        scatter = ax.scatter(np.delete(RRDistance_ms, -1), RRIplusOne, 
                                            alpha=0.7, s=30, c='#667eea', edgecolors='white', 
                                            linewidth=0.5, zorder=5)
                        
                        # Professional identity line
                        ax.plot(np.delete(RRDistance_ms, -1), p(np.delete(RRDistance_ms, -1)), 
                            color="#e74c3c", linewidth=3, label='Identity Line', 
                            alpha=0.5, zorder=8)
                        
                        # Get SD values and draw enhanced ellipse
                        if 'time_domain' in st.session_state.analyzer.results:
                            td_results = st.session_state.analyzer.results['time_domain']
                            sd1 = td_results['sd1']
                            sd2 = td_results['sd2']
                            
                            # Enhanced ellipse with professional styling
                            from matplotlib.patches import Ellipse
                            e = Ellipse(xy=Center_coords, width=sd2*2, height=sd1*2, angle=theta,
                                        edgecolor='#2c3e50', facecolor='none', linewidth=2.5, 
                                        alpha=0.8, linestyle='-', zorder=10)
                            ax.add_patch(e)
                            
                            # Enhanced axis lines with better colors
                            x_sd2 = [EllipseCenterX, EllipseCenterX + sd2 * np.cos(theta_rad)]
                            y_sd2 = [EllipseCenterY, EllipseCenterY + sd2 * np.sin(theta_rad)]
                            ax.plot(x_sd2, y_sd2, color='#3498db', linewidth=3.5, 
                                    label='SD2 (Long-term)', alpha=0.9, zorder=9)
                            
                            x_sd1 = [EllipseCenterX, EllipseCenterX - sd1 * np.sin(theta_rad)]
                            y_sd1 = [EllipseCenterY, EllipseCenterY + sd1 * np.cos(theta_rad)]
                            ax.plot(x_sd1, y_sd1, color='#27ae60', linewidth=3.5, 
                                    label='SD1 (Short-term)', alpha=0.9, zorder=9)
                        
                        # Professional styling
                        ax.set_xlabel("RR Interval (ms)", fontsize=12, fontweight='500')
                        ax.set_ylabel("RR Interval + 1 (ms)", fontsize=12, fontweight='500')
                        ax.set_title('Poincaré Plot - Nonlinear HRV Analysis', 
                                    fontsize=16, fontweight='bold', pad=20)
                        
                        # Enhanced grid and spines
                        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                        ax.set_axisbelow(True)
                        
                        # Remove top and right spines
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_linewidth(0.8)
                        ax.spines['bottom'].set_linewidth(0.8)
                        
                        # Enhanced legend
                        legend = ax.legend(loc='upper right', frameon=True, fancybox=True, 
                                        shadow=True, fontsize=11)
                        legend.get_frame().set_facecolor('white')
                        legend.get_frame().set_alpha(0.9)
                        
                        # Equal aspect ratio for proper ellipse display
                        ax.set_aspect('equal', adjustable='box')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        close_plot_section()
        
        if "BRS Time Domain Visualization" in st.session_state.selected_plots:
            with st.container(border=True):
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

        if "Spectral BRS Analysis" in st.session_state.selected_plots:
            with st.container(border=True):
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
                            """)
                
                close_plot_section()

        if "BRS Sequence Analysis" in st.session_state.selected_plots:
            with st.container(border=True):
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

        # Simple download button for complete report
        st.markdown("### 📄 Export Results")

        if hasattr(st.session_state, 'analyzer') and st.session_state.analyzer:
            try:
                # Create comprehensive summary manually
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                summary_lines = []
                summary_lines.append("=== PHYSIOKIT HRV ANALYSIS REPORT ===")
                summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                summary_lines.append("")
                
                # Channel info
                summary_lines.append("=== CHANNEL CONFIGURATION ===")
                summary_lines.append(f"File Type: {getattr(st.session_state.analyzer, 'file_type', 'Unknown').upper()}")
                if st.session_state.analyzer.ecg_data:
                    summary_lines.append(f"ECG Channel: {st.session_state.analyzer.ecg_channel} ({st.session_state.analyzer.ecg_data['channel_name']})")
                    summary_lines.append(f"ECG Scale: {st.session_state.analyzer.ecg_data.get('detected_scale', 'Unknown')}")
                if st.session_state.analyzer.bp_data:
                    summary_lines.append(f"BP Channel: {st.session_state.analyzer.bp_channel} ({st.session_state.analyzer.bp_data['channel_name']})")
                summary_lines.append("")
                
                # Time window info
                if 'time_window' in st.session_state:
                    tw = st.session_state.time_window
                    summary_lines.append("=== ANALYSIS WINDOW ===")
                    summary_lines.append(f"Start Time: {tw['start_time']:.1f} seconds")
                    summary_lines.append(f"End Time: {tw['end_time']:.1f} seconds")
                    summary_lines.append(f"Duration: {tw['duration']:.1f} seconds ({tw['duration']/60:.1f} minutes)")
                    summary_lines.append("")
                
                # HRV Results
                if hasattr(st.session_state.analyzer, 'results'):
                    results = st.session_state.analyzer.results
                    
                    # Time Domain
                    if 'time_domain' in results and 'error' not in results['time_domain']:
                        td = results['time_domain']
                        summary_lines.append("=== TIME DOMAIN HRV METRICS ===")
                        summary_lines.append(f"Number of Beats: {td.get('num_beats', 'N/A')}")
                        summary_lines.append(f"Heart Rate: {td.get('hr', 'N/A'):.1f} BPM")
                        summary_lines.append(f"Mean RR: {td.get('mean_rr', 'N/A'):.1f} ms")
                        summary_lines.append(f"RMSSD: {td.get('rmssd', 'N/A'):.1f} ms")
                        summary_lines.append(f"SDNN: {td.get('sdnn', 'N/A'):.1f} ms")
                        summary_lines.append(f"pNN50: {td.get('pnn50', 'N/A'):.1f} %")
                        summary_lines.append(f"Sample Entropy: {td.get('sample_entropy', 'N/A'):.3f}")
                        summary_lines.append("")
                        
                        summary_lines.append("=== NONLINEAR HRV METRICS ===")
                        summary_lines.append(f"SD1: {td.get('sd1', 'N/A'):.1f} ms")
                        summary_lines.append(f"SD2: {td.get('sd2', 'N/A'):.1f} ms")
                        summary_lines.append(f"SD1/SD2 Ratio: {td.get('sd1_sd2_ratio', 'N/A'):.3f}")
                        summary_lines.append(f"Ellipse Area: {td.get('ellipse_area', 'N/A'):.1f} ms²")
                        summary_lines.append("")
                    
                    # Frequency Domain
                    if 'frequency_domain' in results and 'error' not in results['frequency_domain']:
                        fd = results['frequency_domain']
                        summary_lines.append("=== FREQUENCY DOMAIN HRV METRICS ===")
                        summary_lines.append(f"VLF Power: {fd.get('vlf_power', 'N/A'):.2f} ms²")
                        summary_lines.append(f"LF Power: {fd.get('lf_power', 'N/A'):.2f} ms²")
                        summary_lines.append(f"HF Power: {fd.get('hf_power', 'N/A'):.2f} ms²")
                        summary_lines.append(f"Total Power: {fd.get('total_power', 'N/A'):.2f} ms²")
                        summary_lines.append(f"LF/HF Ratio: {fd.get('lf_hf_ratio', 'N/A'):.2f}")
                        summary_lines.append(f"LF n.u.: {fd.get('lf_nu', 'N/A'):.2f}")
                        summary_lines.append(f"HF n.u.: {fd.get('hf_nu', 'N/A'):.2f}")
                        summary_lines.append("")
                    
                    # BRS Results
                    if 'brs_sequence' in results and 'error' not in results['brs_sequence']:
                        brs = results['brs_sequence']
                        summary_lines.append("=== BAROREFLEX SENSITIVITY (SEQUENCE METHOD) ===")
                        summary_lines.append(f"BRS Mean: {brs.get('BRS_mean', 'N/A'):.2f} ms/mmHg")
                        summary_lines.append(f"BEI: {brs.get('BEI', 'N/A'):.2f}")
                        summary_lines.append(f"Valid Sequences: {brs.get('num_sequences', 'N/A')}")
                        summary_lines.append(f"Up Sequences: {brs.get('n_up', 'N/A')}")
                        summary_lines.append(f"Down Sequences: {brs.get('n_down', 'N/A')}")
                        summary_lines.append("")
                
                # Join all lines
                complete_summary = "\n".join(summary_lines)
                
                if len(complete_summary.strip()) > 100:  # Make sure we have real content
                    st.download_button(
                        label="⬇️ Download Complete Report",
                        data=complete_summary,
                        file_name=f"hrv_analysis_{timestamp}.txt",
                        mime="text/plain",
                        use_container_width=True,
                        help="Download comprehensive HRV analysis results"
                    )
                    st.success("✅ Complete analysis report ready for download!")
                else:
                    st.warning("⚠️ Analysis results not complete. Run analysis first.")
                    
            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")
        else:
            st.info("ℹ️ Complete analysis first to enable download")

# Case 2: Preview Mode - Show Enhanced Peak Detection Preview
elif st.session_state.file_loaded and st.session_state.channels_configured and st.session_state.preview_mode:
    st.markdown("""
    <div class="window-info">
        <h3 style="margin: 0;">🔍 Peak Detection Preview & Time Window Selection</h3>
        <p style="margin: 0.5rem 0 0 0;">Review detected peaks and selected analysis window. Adjust parameters in sidebar if needed.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show channel configuration in preview
    if hasattr(st.session_state.analyzer, 'ecg_data') or hasattr(st.session_state.analyzer, 'bp_data'):
        analyzer = st.session_state.analyzer
        
        config_text = "<strong>Configured Channels:</strong> "
        channel_parts = []
        
        if analyzer.ecg_data:
            scale_note = f" ({analyzer.ecg_data.get('detected_scale', 'Unknown')} scale)" 
            channel_parts.append(f"ECG Ch{analyzer.ecg_channel}{scale_note}")
        
        if analyzer.bp_data:
            channel_parts.append(f"BP Ch{analyzer.bp_channel}")
        
        config_text += " | ".join(channel_parts)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>🔧 Channel Configuration</h4>
            <p>{config_text}</p>
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
    
    if (len(peaks) > 1 and len(time_data) > 0) or (len(time_data) > 0):
        with st.container(border=True):
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
    
    # BP peak detection stats
    bp_peaks = st.session_state.analyzer.bp_data.get('peaks', [])
    bp_time_data = st.session_state.analyzer.bp_data.get('time', [])

    if (len(bp_peaks) > 1) or (len(bp_time_data) > 0):
        with st.container(border=True):
            create_professional_plot_header("🩸 BP Peak Detection Preview")
            
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
    with st.container(border=True):
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
            std_rr = np.std(rr_intervals, ddof=1)
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
            if st.button("✅ Accept & Run Full Analysis", use_container_width=True,
                        help="Proceed with comprehensive HRV and BRS analysis using current settings"):
                
                # Check analysis capabilities before starting
                capabilities = st.session_state.analyzer.get_analysis_capabilities()
                if not any(capabilities.values()):
                    st.error("❌ No analysis capabilities available. Please check channel configuration.")
                else:
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
                                plot_options = [
                                    "Interactive Tachogram",
                                    "RRI Histogram",
                                    "Frequency Domain",
                                    "Poincaré Plot"
                                ]
                                st.session_state.selected_plots = plot_options
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
        <h3 style="margin: 0;">⌚ Welcome to ChronOS</h3>
        <p style="margin: 0.5rem 0 0 0;">Upload an ACQ or EDF file using the sidebar to begin your cardiovascular analysis journey</p>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced feature showcase
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Platform Capabilities</h3>
            <ul style="margin: 0; padding-left: 1.2rem; line-height: 1.6;">
                <li><strong>File Format Support:</strong> ACQ (AcqKnowledge) and EDF files</li>
                <li><strong>Channel Configuration:</strong> Flexible ECG and BP channel selection</li>
                <li><strong>Analysis Window:</strong> Customizable time segments for focused analysis</li>
                <li><strong>Peak Detection:</strong> Adaptive parameter scaling with options for manual user adjustment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h3>Analysis Methods</h3>
            <ul style="margin: 0; padding-left: 1.2rem; line-height: 1.6;">
                <li><strong>Time Domain:</strong> RMSSD, SDNN, pNN50</li>
                <li><strong>Nonlinear:</strong> Poincaré analysis (SD1 and SD2), Sample Entropy</li>
                <li><strong>Frequency Domain:</strong> VLF, LF, HF power spectral analysis</li>
                <li><strong>Baroreflex Sensitivity:</strong> Sequence and spectral methods</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Quick Start</h3>
            <ol style="margin: 0; padding-left: 1.2rem; line-height: 1.6;">
                <li>Upload ACQ or EDF file</li>
                <li>Select ECG and BP channels</li>
                <li>Configure analysis parameters</li>
                <li>Preview peak detection</li>
                <li>Run comprehensive analysis</li>
                <li>Generate visualizations and export results</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

# Professional footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%); 
           border-radius: 10px; margin-top: 2rem; display: flex; align-items: center; justify-content: center;">
    <div style="display: flex; align-items: center; gap: 15px;">
        <img src="data:image/png;base64,{get_base64_of_image("logo.png")}" 
             style="width: 40px; height: 40px; object-fit: contain;" 
             alt="ChronOS Logo"/>
        <div>
            <p style="margin: 0; color: #6c757d; font-size: 0.9rem;">
                <strong>ChronOS v1.3</strong> | Professional HRV & BRS Analysis Platform<br>
                Built with Streamlit • Enhanced User Experience • Advanced Peak Detection • Time Window Selection
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)