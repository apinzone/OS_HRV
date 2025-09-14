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
    """Apply consistent professional styling to all plots - updated for better legibility"""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color='black', family='Inter'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(size=14, color='black', family='Inter')),
            tickfont=dict(size=12, color='black', family='Inter'),
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(size=14, color='black', family='Inter')),
            tickfont=dict(size=12, color='black', family='Inter'),
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            zeroline=False
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter', size=12, color='black'),
        height=height,
        margin=dict(l=60, r=60, t=60, b=60),
        hovermode='x unified',
        showlegend=False
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
    
    /* Force sidebar to always stay expanded */
    [data-testid="stSidebar"] {
        min-width: 300px !important;
        width: 300px !important;
    }

    /* Hide the sidebar collapse button */
    [data-testid="stSidebar"] button[kind="header"] {
        display: none !important;
    }

    /* Alternative: hide the entire sidebar header that contains collapse button */
    [data-testid="stSidebar"] .css-1lcbmhc {
        display: none !important;
    }

    /* Force sidebar content to always be visible */
    [data-testid="stSidebar"] .css-1d391kg {
        transform: none !important;
        transition: none !important;
    }

    /* Prevent sidebar from being hidden on mobile */
    @media (max-width: 768px) {
        [data-testid="stSidebar"] {
            transform: translateX(0) !important;
            visibility: visible !important;
        }
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
        background: white !important;
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
    
    /* Footer Rules */
    .footer-section {
        color: white !important;
    }

    .footer-section * {
        color: white !important;
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
        border: 1px solid var(--border, var(--border-opacity));
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
    
    /* Preview Boxes */       
    .preview-header {
        background: white;
        padding: 1.25rem;
        border-radius: 10px;
        border-left: 4px solid #2563eb;
        margin: 1rem 0;
        border: 1px solid rgba(0,0,0,0.1);
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
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
        <div class="version-info">Version 1.4 | Advanced Peak Detection | HRV and BRS Analysis</div>
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
        status_text.markdown("**Analysis Complete!** All cardiovascular metrics calculated successfully.")
        
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
# show_analysis_status()
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

                        analysis_keys_to_clear = [
                            'ectopic_results', 
                            'peak_params', 
                            'time_window', 
                            'selected_plots',
                            'ecg_reset_counter'
                        ]
                        for key in analysis_keys_to_clear:
                            if key in st.session_state:
                                del st.session_state[key]
                                
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
        "Select ECG Channel:",
        ecg_options,
        index=ecg_default,
        help="Choose the channel containing ECG/EKG data"
    )
    
    bp_selection = st.selectbox(
        "Select BP Channel:",
        bp_options,
        index=bp_default,
        help="Choose the channel containing blood pressure data"
    )
    
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
        # ECG Preprocessing Section
        st.markdown("## 🔧 ECG Preprocessing")

        # Bandpass filter checkbox
        enable_bandpass = st.checkbox(
            "Enable ECG Bandpass Filter (0.5-40 Hz)", 
            value=False,
            help="Apply standard clinical ECG filtering to reduce baseline wander and high-frequency noise"
        )

        if enable_bandpass:
            st.info("🔧 **Filter Applied:** 0.5-40 Hz bandpass filter will remove baseline drift (<0.5 Hz) and muscle noise (>40 Hz)")
            
            # Optional: Add advanced filter parameters in an expander
            with st.expander("🔧 Advanced Filter Settings", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    lowcut = st.number_input("Low cutoff (Hz)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
                with col2:
                    highcut = st.number_input("High cutoff (Hz)", min_value=10.0, max_value=150.0, value=40.0, step=5.0)
                
                filter_order = st.selectbox("Filter order", [2, 4, 6, 8], index=1, help="Higher order = sharper cutoff")
        else:
            # Use default values when filter is disabled
            lowcut = 0.5
            highcut = 40.0
            filter_order = 4

        # Configure preprocessing before channel configuration
        st.session_state.analyzer.configure_preprocessing(
            enable_bandpass=enable_bandpass,
            lowcut=lowcut,
            highcut=highcut,
            order=filter_order
        )

        st.markdown('</div>', unsafe_allow_html=True)

        # Peak Detection Parameters
        st.markdown("## Peak Detection")

        st.markdown("**Parameter Settings:**")

        # ECG Parameters
        with st.expander("⚡ ECG R-peak Detection", expanded=True):
            
            # Calculate dynamic ECG parameters based on signal characteristics
            if hasattr(st.session_state.analyzer, 'ecg_data') and 'raw' in st.session_state.analyzer.ecg_data:
                ecg_signal = st.session_state.analyzer.ecg_data['raw']
                ecg_baseline = np.median(ecg_signal)
                ecg_max = np.max(ecg_signal)
                signal_range = ecg_max - ecg_baseline
                
                # Get sample rate FIRST
                if 'fs' in st.session_state.analyzer.ecg_data:
                    sample_rate = st.session_state.analyzer.ecg_data['fs']
                else:
                    sample_rate = 256  # Default fallback
                
                # Adaptive defaults
                if (st.session_state.get('apply_sensitive', False) or 
                    st.session_state.get('sensitive_height') is not None):
                    # Use stored sensitive values
                    ecg_height_default = st.session_state.get('sensitive_height', 0.55 * signal_range)
                    ecg_prominence_default = st.session_state.get('sensitive_prominence', 0.6 * ecg_height_default)
                    ecg_distance_default = int((st.session_state.get('sensitive_distance_ms', 250) / 1000) * sample_rate)
                    # Only clear apply_sensitive flag, keep the values stored
                    if st.session_state.get('apply_sensitive', False):
                        st.session_state.apply_sensitive = False
                else:
                    # Your existing default calculations
                    ecg_height_default = 0.55 * signal_range
                    ecg_prominence_default = 0.6 * ecg_height_default
                    ecg_distance_default = int(0.25 * sample_rate)
                
                # Store defaults for reset functionality
                st.session_state.ecg_defaults = {
                    'height': ecg_height_default,
                    'prominence': ecg_prominence_default,
                    'distance': ecg_distance_default
                }
                
                # ADAPTIVE SLIDER RANGES based on signal characteristics
                height_min = 0.1 * signal_range
                height_max = 1.5 * signal_range
                height_step = signal_range / 1000
                
                prom_min = 0.1 * ecg_height_default
                prom_max = 1.0 * signal_range
                prom_step = signal_range / 1000
                
                # Distance ranges - now sample_rate is available
                dist_min = max(20, int(0.1 * sample_rate))  # 100ms minimum
                dist_max = min(800, int(0.5 * sample_rate))  # 500ms maximum
                
                
            else:
                # Fallback values when no signal loaded
                ecg_height_default = 0.8
                ecg_prominence_default = 0.4
                ecg_distance_default = 64
                sample_rate = 256  # Add fallback sample rate
                
                # Store fallback defaults
                st.session_state.ecg_defaults = {
                    'height': ecg_height_default,
                    'prominence': ecg_prominence_default,
                    'distance': ecg_distance_default
                }
                
                # Conservative fallback ranges
                height_min, height_max, height_step = 0.1, 2.0, 0.01
                prom_min, prom_max, prom_step = 0.1, 1.5, 0.01
                dist_min, dist_max = 50, 400
            
            if 'ecg_reset_counter' not in st.session_state:
                st.session_state.ecg_reset_counter = 0

            # Restore Defaults Button
            if st.button("🔄 Restore ECG Defaults", help="Reset sliders to calculated optimal values", key="restore_ecg"):
                # Clear sensitive parameters
                for key in ['apply_sensitive', 'sensitive_height', 'sensitive_distance_ms', 'sensitive_prominence']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                # Clear the slider session state keys
                for key in ['ecg_height', 'ecg_prominence', 'ecg_distance', 'ecg_distance_ms']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                # Increment reset counter to force new slider instances
                st.session_state.ecg_reset_counter += 1
                st.success("Parameters restored to conservative defaults!")
                st.rerun()

            # Use the reset counter in slider keys to force recreation
            reset_suffix = f"_{st.session_state.ecg_reset_counter}"

            # ADAPTIVE SLIDERS with reset counter in keys
            ecg_height = st.slider(
                "Height Threshold", 
                min_value=float(height_min), 
                max_value=float(height_max), 
                value=float(ecg_height_default), 
                step=float(height_step),
                key=f"ecg_height{reset_suffix}",  # Dynamic key
                help="Minimum R-peak amplitude (auto-scaled to signal range)",
                format="%.4f"
            )

            # Distance slider - convert to/from milliseconds for display
            if hasattr(st.session_state.analyzer, 'ecg_data') and 'fs' in st.session_state.analyzer.ecg_data:
                sample_rate = st.session_state.analyzer.ecg_data['fs']
                
                # Convert sample-based limits to milliseconds
                dist_min_ms = int((dist_min / sample_rate) * 1000)
                dist_max_ms = int((dist_max / sample_rate) * 1000)
                ecg_distance_default_ms = int((ecg_distance_default / sample_rate) * 1000)
                
                # Slider in milliseconds
                ecg_distance_ms = st.slider(
                    "Min Distance (ms)", 
                    min_value=dist_min_ms, 
                    max_value=dist_max_ms, 
                    value=ecg_distance_default_ms, 
                    step=10,
                    key=f"ecg_distance_ms{reset_suffix}",
                    help="Minimum time between R-peaks (physiological refractory period)"
                )
                
                # Convert back to samples for internal use
                ecg_distance = int((ecg_distance_ms / 1000) * sample_rate)
            else:
                # Fallback when no ECG data loaded
                ecg_distance_ms = st.slider(
                    "Min Distance (ms)", 
                    min_value=100, 
                    max_value=800, 
                    value=250, 
                    step=10,
                    help="Minimum time between R-peaks - will be converted to samples when ECG is loaded"
                )
                ecg_distance = 64  # Default fallback in samples

            ecg_prominence = st.slider(
                "Prominence", 
                min_value=float(prom_min), 
                max_value=float(prom_max), 
                value=float(ecg_prominence_default), 
                step=float(prom_step),
                key=f"ecg_prominence{reset_suffix}",  # Dynamic key
                help="Peak prominence (auto-scaled to signal characteristics)",
                format="%.4f"
            )

            # Sensitive Parameters Button 
            st.markdown("---")
            if st.button("Apply Sensitive Parameters", use_container_width=True, 
                         help="Set parameters for maximum sensitivity (45% signal range, 250ms distance, 150% prominence)"):
                
                # Calculate sensitive parameters
                if hasattr(st.session_state.analyzer, 'ecg_data') and 'raw' in st.session_state.analyzer.ecg_data:
                    ecg_signal = st.session_state.analyzer.ecg_data['raw']
                    signal_range = np.max(ecg_signal) - np.median(ecg_signal)
                    sample_rate = st.session_state.analyzer.ecg_data['fs']
                    
                    # Calculate your specified sensitive parameters
                    sensitive_height = 0.45 * signal_range
                    sensitive_distance_ms = 250
                    sensitive_prominence = 1.5 * sensitive_height
                    
                    # Clear existing slider states and set new values
                    for key in ['ecg_height', 'ecg_prominence', 'ecg_distance_ms']:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    # Store the sensitive values for next reset
                    st.session_state.sensitive_height = sensitive_height
                    st.session_state.sensitive_distance_ms = sensitive_distance_ms
                    st.session_state.sensitive_prominence = sensitive_prominence
                    st.session_state.apply_sensitive = True

                    # Force slider recreation by incrementing reset counter
                    st.session_state.ecg_reset_counter += 1
                    
                    st.success(f"Sensitive parameters applied: Height={sensitive_height:.3f}, Distance={sensitive_distance_ms}ms, Prominence={sensitive_prominence:.3f}")
                    st.rerun()
                else:
                    st.warning("Load ECG data first to calculate sensitive parameters")

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
            with st.spinner("Updating preprocessing and peak detection..."):
                try:
                    # First, reapply current preprocessing settings to configured channels
                    if st.session_state.analyzer.ecg_data:
                        # Get the original raw data and reapply current filter settings
                        original_raw = st.session_state.analyzer.ecg_data.get('raw_original')
                        if original_raw is not None:
                            # Apply current preprocessing
                            if st.session_state.analyzer.preprocessing_options['enable_bandpass']:
                                filtered_data = st.session_state.analyzer.bandpass_filter(
                                    original_raw * st.session_state.analyzer.ecg_data['scale_factor'],
                                    lowcut=st.session_state.analyzer.preprocessing_options['bandpass_lowcut'],
                                    highcut=st.session_state.analyzer.preprocessing_options['bandpass_highcut'],
                                    fs=st.session_state.analyzer.ecg_data['fs'],
                                    order=st.session_state.analyzer.preprocessing_options['bandpass_order']
                                )
                            else:
                                filtered_data = original_raw * st.session_state.analyzer.ecg_data['scale_factor']
                            
                            # Update the processed signal
                            st.session_state.analyzer.ecg_data['raw'] = filtered_data
                    
                    # Then run peak detection with current parameters
                    st.session_state.analyzer.find_peaks_with_params(**st.session_state.peak_params)
                    st.session_state.preview_mode = True
                    st.success("✅ Preview updated with current settings!")
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
        <h2 style="margin: 0; color: #155724;">Analysis Complete</h2>
        <p style="margin: 0.5rem 0 0 0; color: #155724;">Comprehensive cardiovascular analysis finished successfully{scale_note}{file_type_note}</p>
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



    
    st.markdown("### Analysis Results 📋")
    

    # Display selected plots with professional styling
    if "Interactive Tachogram" in st.session_state.selected_plots:
        with st.container(border=True, key="tachogram_container"):

            fig = go.Figure()
            
            rr_intervals = np.array(st.session_state.analyzer.ecg_data['rr_intervals'])
            time_points = np.array(st.session_state.analyzer.ecg_data['td_peaks'][:-1])
            
            # Filter data to analysis window if specified
            if 'time_window' in st.session_state:
                tw = st.session_state.time_window
                # Create mask for data within the analysis window
                window_mask = (time_points >= tw['start_time']) & (time_points <= tw['end_time'])
                # Filter both time points and RR intervals to the window
                time_points_filtered = time_points[window_mask]
                rr_intervals_filtered = rr_intervals[window_mask]
            else:
                # Use all data if no window specified
                time_points_filtered = time_points
                rr_intervals_filtered = rr_intervals
            
            # Enhanced RR intervals trace (using filtered data)
            fig.add_trace(go.Scatter(
                x=time_points_filtered,
                y=rr_intervals_filtered,
                mode='lines+markers',
                name='RR Intervals',
                line=dict(color='#3498db', width=2.5),
                marker=dict(size=5, color='#3498db', opacity=0.8),
                hovertemplate='<b>Time:</b> %{x:.1f}s<br><b>RR:</b> %{y:.1f}ms<extra></extra>'
            ))
            
            # Reference lines without annotations (using filtered data statistics)
            mean_rr = np.mean(rr_intervals_filtered)
            std_rr = np.std(rr_intervals_filtered, ddof=1)
            
            fig.add_hline(y=mean_rr, line_dash="dash", line_color=COLORS['secondary'], 
                        line_width=2, opacity=0.8)
            fig.add_hline(y=mean_rr + std_rr, line_dash="dot", line_color=COLORS['secondary'], 
                        line_width=1.5, opacity=0.6)
            fig.add_hline(y=mean_rr - std_rr, line_dash="dot", line_color=COLORS['secondary'], 
                        line_width=1.5, opacity=0.6)
            
            # Add integrated time domain metrics panel
            td_results = st.session_state.analyzer.results['time_domain']
            if 'error' not in td_results:
                metrics_text = "<b>Time Domain Metrics</b><br><br>"
                
                # Add recording window info
                if 'time_window' in st.session_state:
                    tw = st.session_state.time_window
                    metrics_text += f"Window: {tw['start_time']:.0f}-{tw['end_time']:.0f}s ({tw['duration']:.0f}s)<br><br>"
                else:
                    metrics_text += f"Window: Full Recording<br><br>"
                
                metrics_text += f"R-R Intervals: {len(rr_intervals_filtered)}<br><br>"
                metrics_text += f"HR: {td_results['hr']:.1f} BPM<br><br>"
                metrics_text += f"Mean RR: {td_results['mean_rr']:.1f} ms<br><br>"
                metrics_text += f"RMSSD: {td_results['rmssd']:.1f} ms<br><br>"
                metrics_text += f"SDNN: {td_results['sdnn']:.1f} ms<br><br>"
                metrics_text += f"SDSD: {td_results['sdsd']:.1f} ms<br><br>"
                metrics_text += f"pNN50: {td_results['pnn50']:.1f} %"

                fig.add_annotation(
                    x=1.02, y=0.5, xref="paper", yref="paper",
                    text=metrics_text, showarrow=False,
                    font=dict(family="Inter", size=13, color="black"),
                    align="left", bgcolor="rgba(255, 255, 255, 0.95)",
                    bordercolor="rgba(0, 0, 0, 0.2)", borderwidth=1, borderpad=20,
                    xanchor="left", yanchor="middle"
                )
            
            # Apply updated professional layout
            fig = apply_professional_layout(
                fig, 
                f'Interactive R-R Interval Tachogram with Time Domain Results',
                'Time (seconds)', 
                'RR Interval (ms)', 
                height=600
            )
            
            # Extend right margin to accommodate metrics
            fig.update_layout(margin=dict(l=60, r=250, t=60, b=60))
            
            st.plotly_chart(fig, use_container_width=True)
            close_plot_section()
            
            if "RRI Histogram" in st.session_state.selected_plots:
                with st.container(border=True):
                    
                    if hasattr(st.session_state.analyzer, 'ecg_data') and 'rr_intervals' in st.session_state.analyzer.ecg_data:
                        rr_intervals = st.session_state.analyzer.ecg_data['rr_intervals']
                        
                        # Calculate basic statistics
                        rr_mean = np.mean(rr_intervals)
                        rr_std = np.std(rr_intervals, ddof=1)
                        
                        # Create histogram using Plotly instead of matplotlib
                        fig = go.Figure()
                        
                        # Add histogram
                        fig.add_trace(go.Histogram(
                            x=rr_intervals,
                            nbinsx=30,
                            marker=dict(color='#3498db', opacity=0.7, line=dict(color='#2980b9', width=0.5)),
                            name='RR Intervals',
                            showlegend=False
                        ))
                        
                        # Add mean line using add_vline (simpler approach)
                        fig.add_vline(
                            x=rr_mean, 
                            line=dict(color='#e74c3c', width=2, dash='dash'),
                            name=f'Mean: {rr_mean:.1f} ± {rr_std:.1f} ms'
                        )
                        
                        # Apply same layout as tachogram  
                        fig = apply_professional_layout(
                            fig, 
                            f'RR Interval Distribution (n={len(rr_intervals)})',
                            'RR Interval (ms)', 
                            'Frequency', 
                            height=600
                        )
                        
                        # Add legend manually using annotation (positioned like original)
                        fig.add_annotation(
                            x=0.98, y=0.98, xref="paper", yref="paper",
                            text=f'<span style="color:#e74c3c">- - - -</span> Mean: {rr_mean:.1f} ± {rr_std:.1f} ms',
                            showarrow=False,
                            font=dict(family="Inter", size=14, color="black"),  # Bigger black text
                            bgcolor="rgba(255, 255, 255, 0.8)",
                            borderpad=10,  # More padding for bigger appearance
                            xanchor="right", yanchor="top"
)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        st.error("❌ No RR interval data available. Complete peak detection analysis first.")
                    
                    close_plot_section()

        
        if "Frequency Domain" in st.session_state.selected_plots:
            with st.container(border=True):
                
                freq_data = st.session_state.analyzer.results['frequency_domain']
                if 'error' not in freq_data:
                    fig = go.Figure()
                    
                    frequencies = np.asarray(freq_data['frequencies'], dtype=float)
                    psd = np.asarray(freq_data['psd'], dtype=float)

                    # Clean data (same as original)
                    good = np.isfinite(frequencies) & np.isfinite(psd) & (frequencies > 0)
                    frequencies, psd = frequencies[good], psd[good]
                    order = np.argsort(frequencies)
                    frequencies, psd = frequencies[order], psd[order]
                    frequencies, uniq_idx = np.unique(frequencies, return_index=True)
                    psd = psd[uniq_idx]

                    # Scale exactly like original
                    scale = 1e6
                    psd_scaled = psd * scale

                    # Create band masks exactly like original
                    df = np.median(np.diff(frequencies))
                    eps = float(df) * 0.51 if np.isfinite(df) and df > 0 else 1e-12

                    vlf_mask = (frequencies >= (0.003 - eps)) & (frequencies <= (0.04 + eps))
                    lf_mask = (frequencies >= (0.04 - eps)) & (frequencies <= (0.15 + eps))
                    hf_mask = (frequencies >= (0.15 - eps)) & (frequencies <= (0.40 + eps))

                    # Add band fills using same approach as original but in Plotly
                    if np.any(vlf_mask):
                        fig.add_trace(go.Scatter(
                            x=frequencies[vlf_mask], 
                            y=psd_scaled[vlf_mask],
                            fill='tozeroy', 
                            fillcolor='rgba(149, 165, 166, 0.4)',
                            mode='none',
                            line=dict(width=0),
                            name='VLF (0.003–0.04 Hz)',
                            showlegend=True
                        ))

                    if np.any(lf_mask):
                        fig.add_trace(go.Scatter(
                            x=frequencies[lf_mask], 
                            y=psd_scaled[lf_mask],
                            fill='tozeroy', 
                            fillcolor='rgba(52, 152, 219, 0.5)',
                            mode='none',
                            line=dict(width=0),
                            name='LF (0.04–0.15 Hz)',
                            showlegend=True
                        ))

                    if np.any(hf_mask):
                        fig.add_trace(go.Scatter(
                            x=frequencies[hf_mask], 
                            y=psd_scaled[hf_mask],
                            fill='tozeroy', 
                            fillcolor='rgba(231, 76, 60, 0.5)',
                            mode='none',
                            line=dict(width=0),
                            name='HF (0.15–0.40 Hz)',
                            showlegend=True
                        ))

                    # Main PSD curve (same as original)
                    fig.add_trace(go.Scatter(
                        x=frequencies, 
                        y=psd_scaled,
                        mode='lines',
                        line=dict(color='#2c3e50', width=2.5),
                        name='PSD',
                        showlegend=True
                    ))

                    # Add metrics panel
                    fd = freq_data
                    metrics_text = "<b>Frequency Domain Metrics</b><br><br>"
                    metrics_text += f"VLF Power: {fd.get('vlf_power', 0):.2f} ms²<br><br>"
                    metrics_text += f"LF Power: {fd.get('lf_power', 0):.2f} ms²<br><br>"
                    metrics_text += f"HF Power: {fd.get('hf_power', 0):.2f} ms²<br><br>"
                    metrics_text += f"Total Power: {fd.get('total_power', 0):.2f} ms²<br><br>"
                    metrics_text += f"LF/HF Ratio: {fd.get('lf_hf_ratio', 0):.2f}<br><br>"
                    metrics_text += f"LF n.u.: {fd.get('lf_nu', 0):.2f}<br><br>"
                    metrics_text += f"HF n.u.: {fd.get('hf_nu', 0):.2f}"

                    fig.add_annotation(
                        x=1.02, y=0.5, xref="paper", yref="paper",
                        text=metrics_text, showarrow=False,
                        font=dict(family="Inter", size=13, color="black"),
                        align="left", bgcolor="rgba(255, 255, 255, 0.95)",
                        bordercolor="rgba(0, 0, 0, 0.2)", borderwidth=1, borderpad=20,
                        xanchor="left", yanchor="middle"
                    )

                    # Apply layout
                    fig = apply_professional_layout(
                        fig, 
                        'Heart Rate Variability - Frequency Domain Analysis',
                        'Frequency (Hz)', 
                        'Power Spectral Density (ms²/Hz)', 
                        height=600
                    )

                    # Set limits and margins exactly like original
                    fig.update_xaxes(range=[0, 0.5])
                    fig.update_yaxes(tickformat='.1e')  # This will show scientific notation like 1.0e+10
                    fig.update_layout(
                        margin=dict(l=60, r=250, t=60, b=60),
                        showlegend=True,
                        legend=dict(
                            x=0.98, y=0.98, xanchor='right', yanchor='top',
                            bgcolor='rgba(255, 255, 255, 0.9)',
                            bordercolor='rgba(0, 0, 0, 0.2)', borderwidth=1
                        )
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Frequency domain analysis error: {freq_data['error']}")
                
                close_plot_section()


        if "Poincaré Plot" in st.session_state.selected_plots:
            with st.container(border=True):
                
                fig = go.Figure()
                
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
                fig.add_trace(go.Scatter(
                    x=np.delete(RRDistance_ms, -1),
                    y=RRIplusOne,
                    mode='markers',
                    marker=dict(size=6, color='#3498db', opacity=0.7, line=dict(color='white', width=0.5)),
                    name='RR Data Points',
                    showlegend=True
                ))
                
                # Identity line
                fig.add_trace(go.Scatter(
                    x=np.delete(RRDistance_ms, -1),
                    y=p(np.delete(RRDistance_ms, -1)),
                    mode='lines',
                    line=dict(color='#e74c3c', width=3),
                    name='Identity Line',
                    opacity=0.5,
                    showlegend=True
                ))
                
                # Get SD values and add ellipse and axis lines
                if 'time_domain' in st.session_state.analyzer.results:
                    td_results = st.session_state.analyzer.results['time_domain']
                    sd1 = td_results['sd1']
                    sd2 = td_results['sd2']
                    
                    # Create ellipse points manually
                    t = np.linspace(0, 2*np.pi, 100)
                    ellipse_x = sd2 * np.cos(t)
                    ellipse_y = sd1 * np.sin(t)
                    
                    # Rotate ellipse
                    cos_angle = np.cos(theta_rad)
                    sin_angle = np.sin(theta_rad)
                    ellipse_x_rot = ellipse_x * cos_angle - ellipse_y * sin_angle + EllipseCenterX
                    ellipse_y_rot = ellipse_x * sin_angle + ellipse_y * cos_angle + EllipseCenterY
                    
                    # Add ellipse as a line trace
                    fig.add_trace(go.Scatter(
                        x=ellipse_x_rot,
                        y=ellipse_y_rot,
                        mode='lines',
                        line=dict(color='#2c3e50', width=2.5),
                        name='Ellipse',
                        showlegend=True
                    ))
                    
                    # SD2 axis line
                    x_sd2 = [EllipseCenterX, EllipseCenterX + sd2 * np.cos(theta_rad)]
                    y_sd2 = [EllipseCenterY, EllipseCenterY + sd2 * np.sin(theta_rad)]
                    fig.add_trace(go.Scatter(
                        x=x_sd2, y=y_sd2,
                        mode='lines',
                        line=dict(color='#3498db', width=3.5),
                        name='SD2 (Long-term)',
                        showlegend=True
                    ))
                    
                    # SD1 axis line  
                    x_sd1 = [EllipseCenterX, EllipseCenterX - sd1 * np.sin(theta_rad)]
                    y_sd1 = [EllipseCenterY, EllipseCenterY + sd1 * np.cos(theta_rad)]
                    fig.add_trace(go.Scatter(
                        x=x_sd1, y=y_sd1,
                        mode='lines',
                        line=dict(color='#27ae60', width=3.5),
                        name='SD1 (Short-term)',
                        showlegend=True
                    ))
                
                # Add integrated nonlinear metrics panel
                td_results = st.session_state.analyzer.results['time_domain']
                if 'error' not in td_results:
                    metrics_text = "<b>Nonlinear Metrics</b><br><br>"
                    metrics_text += f"SD1: {td_results.get('sd1', 0):.1f} ms<br><br>"
                    metrics_text += f"SD2: {td_results.get('sd2', 0):.1f} ms<br><br>"
                    metrics_text += f"SD1/SD2: {td_results.get('sd1_sd2_ratio', 0):.3f}<br><br>"
                    metrics_text += f"Ellipse Area: {td_results.get('ellipse_area', 0):.1f} ms²<br><br>"
                    metrics_text += f"Sample Entropy: {td_results.get('sample_entropy', 0):.3f}"

                    fig.add_annotation(
                        x=1.02, y=0.5, xref="paper", yref="paper",
                        text=metrics_text, showarrow=False,
                        font=dict(family="Inter", size=13, color="black"),
                        align="left", bgcolor="rgba(255, 255, 255, 0.95)",
                        bordercolor="rgba(0, 0, 0, 0.2)", borderwidth=1, borderpad=20,
                        xanchor="left", yanchor="middle"
                    )
                
                # Apply same layout as other plots
                # Apply same layout as other plots
                fig = apply_professional_layout(
                    fig, 
                    'Poincaré Plot - Nonlinear HRV Analysis',
                    'RR Interval (ms)', 
                    'RR Interval + 1 (ms)', 
                    height=600
                )

                # Calculate ranges first
                rr_x_min = np.min(np.delete(RRDistance_ms, -1))
                rr_x_max = np.max(np.delete(RRDistance_ms, -1))
                rr_y_min = np.min(RRIplusOne)
                rr_y_max = np.max(RRIplusOne)

                # Set layout WITHOUT equal aspect ratio
                fig.update_layout(
                    margin=dict(l=60, r=250, t=60, b=60),
                    showlegend=True,
                    legend=dict(
                        x=0.02, y=0.98, xanchor='left', yanchor='top',
                        bgcolor='rgba(255, 255, 255, 0.9)',
                        bordercolor='rgba(0, 0, 0, 0.2)', borderwidth=1
                    )
                )

                # Set axis ranges without aspect ratio constraint
                fig.update_xaxes(range=[rr_x_min - 50, rr_x_max + 50])
                fig.update_yaxes(range=[rr_y_min - 50, rr_y_max + 50])

                st.plotly_chart(fig, use_container_width=True)
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
                                fillcolor="rgba(255, 193, 7, 0.2)", opacity=0.3,
                                annotation_text="Analysis Window", annotation_position="top left",
                                row=1, col=1
                            )
                            fig.add_vrect(
                                x0=tw['start_time'], x1=tw['end_time'],
                                fillcolor="rgba(255, 193, 7, 0.2)", opacity=0.3,
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
    <div class="preview-header">
        <h3 style="margin: 0;">Peak Detection Preview & Time Window Selection</h3>
        <p style="margin: 0.5rem 0 0 0;">Review detected peaks and selected analysis window. Adjust parameters in sidebar if needed.</p>
    </div>
    """, unsafe_allow_html=True)

    # Get peaks data safely
    peaks = st.session_state.analyzer.ecg_data.get('peaks', [])
    time_data = st.session_state.analyzer.ecg_data.get('time', [])
    if (len(peaks) > 1 and len(time_data) > 0) or (len(time_data) > 0):
    
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
                    fillcolor="rgba(255, 193, 7, 0.3)", opacity=0.3,
                    annotation_text=f"Analysis Window ({tw['duration']:.0f}s)", 
                    annotation_position="top left"
                )
            
            duration_min = time_data[-1] / 60 if len(time_data) > 0 else 0
            
            fig.update_layout(
                title=dict(
                    text=f'ECG Peak Detection - Full Recording ({duration_min:.1f} min) - {len(peaks)} R-peaks detected',
                    font=dict(size=18, color='black', family='Inter'),
                    x=0.5,
                    xanchor='center'
                ),
                xaxis=dict(
                    title=dict(text='Time (s)', font=dict(size=14, color='black', family='Inter')),
                    tickfont=dict(size=12, color='black', family='Inter'),
                    gridcolor='rgba(0,0,0,0.1)',
                    showgrid=True,
                    zeroline=False
                ),
                yaxis=dict(
                    title=dict(text='ECG (mV)', font=dict(size=14, color='black', family='Inter')),
                    tickfont=dict(size=12, color='black', family='Inter'),
                    gridcolor='rgba(0,0,0,0.1)',
                    showgrid=True,
                    zeroline=False
                ),
                height=400,
                showlegend=False,
                hovermode='x unified',
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=10, r=10, t=50, b=10),
                font=dict(family='Inter', size=12, color='black')
            )
            
            # Add after peak detection in your preview mode
            st.plotly_chart(fig, use_container_width=True)
        
        close_plot_section()

        # Pan-Tompkins validation results 
        if hasattr(st.session_state.analyzer, 'ecg_data') and 'pantompkins_validation' in st.session_state.analyzer.ecg_data:
            validation_data = st.session_state.analyzer.ecg_data['pantompkins_validation']
            missed_peaks = validation_data['missed_peaks']
            
            if len(missed_peaks) == 0:
                st.markdown("""
                <div class="window-info">
                    <strong>✅ Pan-Tompkins validation:</strong> No additional peaks detected
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-box">
                    <strong>⚠️ Pan-Tompkins validation:</strong> {len(missed_peaks)} potential missed R-peaks detected
                </div>
                """, unsafe_allow_html=True)
                
                # Show timestamps for investigation
                times_str = ", ".join([f"{peak['time_seconds']:.1f}s" for peak in missed_peaks[:5]])
                if len(missed_peaks) > 5:
                    times_str += f" (+{len(missed_peaks)-5} more)"
                
                st.info(f"🔍 **Investigate:** Potential R-peaks at {times_str}")
                st.info("💡 **Suggestion:** Consider lowering height threshold if visual inspection confirms these are valid R-peaks")
                
    # BP peak detection stats
    bp_peaks = st.session_state.analyzer.bp_data.get('peaks', [])
    bp_time_data = st.session_state.analyzer.bp_data.get('time', [])

    if (len(bp_peaks) > 1) or (len(bp_time_data) > 0):
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
                    fillcolor="rgba(255, 193, 7, 0.3)", opacity=0.3,
                    annotation_text=f"Analysis Window ({tw['duration']:.0f}s)", 
                    annotation_position="top left"
                )
            
            duration_min = bp_time_data[-1] / 60 if len(bp_time_data) > 0 else 0
            
            fig.update_layout(
                title=dict(
                    text=f'BP Peak Detection - Full Recording ({duration_min:.1f} min) - {len(bp_peaks)} systolic peaks detected',
                    font=dict(size=18, color='black', family='Inter'),
                    x=0.5,
                    xanchor='center'
                ),
                xaxis=dict(
                    title=dict(text='Time (s)', font=dict(size=14, color='black', family='Inter')),
                    tickfont=dict(size=12, color='black', family='Inter'),
                    gridcolor='rgba(0,0,0,0.1)',
                    showgrid=True,
                    zeroline=False
                ),
                yaxis=dict(
                    title=dict(text='Blood Pressure (mmHg)', font=dict(size=14, color='black', family='Inter')),
                    tickfont=dict(size=12, color='black', family='Inter'),
                    gridcolor='rgba(0,0,0,0.1)',
                    showgrid=True,
                    zeroline=False
                ),
                height=400,
                showlegend=False,
                hovermode='x unified',
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=10, r=10, t=50, b=10),
                font=dict(family='Inter', size=12, color='black')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        close_plot_section()

    # Enhanced RR Interval Tachogram Preview
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
            marker=dict(size=5, color='#8e44ad')
        ))
        
        # Highlight analysis window
        if 'time_window' in st.session_state:
            tw = st.session_state.time_window
            fig_tacho.add_vrect(
                x0=tw['start_time'], x1=tw['end_time'],
                fillcolor="rgba(255, 193, 7, 0.3)", opacity=0.3,
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
                        annotation_text=f"Mean: {mean_rr:.1f} ms",
                        annotation_font=dict(weight='bold', color='black', size=12))
        fig_tacho.add_hline(y=mean_rr + std_rr, line_dash="dot", line_color="#f39c12", line_width=2,
                        annotation_text=f"+1 SD: {mean_rr + std_rr:.1f} ms",
                        annotation_font=dict(weight='bold', color='black', size=12))
        fig_tacho.add_hline(y=mean_rr - std_rr, line_dash="dot", line_color="#f39c12", line_width=2,
                        annotation_text=f"-1 SD: {mean_rr - std_rr:.1f} ms",
                        annotation_font=dict(weight='bold', color='black', size=12))
        
        fig_tacho.update_layout(
                    title=dict(
                        text=f'RR Interval Tachogram - Full Recording (Range: {min_rr:.0f}-{max_rr:.0f} ms)',
                        font=dict(size=18, color='black', family='Inter'),
                        x=0.5,
                        xanchor='center'
                    ),
                    xaxis=dict(
                        title=dict(text='Time (s)', font=dict(size=14, color='black', family='Inter')),
                        tickfont=dict(size=12, color='black', family='Inter'),
                        gridcolor='rgba(0,0,0,0.1)',
                        showgrid=True,
                        zeroline=False
                    ),
                    yaxis=dict(
                        title=dict(text='RR Interval (ms)', font=dict(size=14, color='black', family='Inter')),
                        tickfont=dict(size=12, color='black', family='Inter'),
                        gridcolor='rgba(0,0,0,0.1)',
                        showgrid=True,
                        zeroline=False
                    ),
                    height=450,
                    hovermode='x unified',
                    showlegend=False,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(l=10, r=10, t=50, b=10),
                    font=dict(family='Inter', size=12, color='black')
                )
        
        st.plotly_chart(fig_tacho, use_container_width=True)

        # Window-specific statistics
        if 'time_window' in st.session_state:
            tw = st.session_state.time_window
            rr_intervals_np = np.array(rr_intervals)
            rr_time_points_np = np.array(rr_time_points)
            
            # Filter to window
            window_mask = (rr_time_points_np >= tw['start_time']) & (rr_time_points_np <= tw['end_time'])
            display_rr = rr_intervals_np[window_mask]
            display_count = len(display_rr)
        else:
            # No window selected, use all data
            display_rr = rr_intervals
            display_count = len(rr_intervals)

        if len(display_rr) > 0:
            display_mean = np.mean(display_rr)
            display_std = np.std(display_rr)
            display_cv = (display_std / display_mean) * 100
            display_hr = 60000 / display_mean
            
            # Show the 4 key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("RR Count", display_count, help="RR intervals in analysis window")
            with col2:
                st.metric("Mean RR", f"{display_mean:.1f} ms", help="Average RR interval")
            with col3:
                st.metric("RR CV%", f"{display_cv:.1f}%", help="Coefficient of variation")
            with col4:
                st.metric("Heart Rate", f"{display_hr:.1f} BPM", help="Estimated from mean RR")
            
            # Info message outside the columns
            if 'time_window' in st.session_state:
                tw = st.session_state.time_window
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"""
                    <div class="window-info" style="margin-bottom: 0;">
                        <strong>ℹ️ Analysis Preview:</strong> The selected {tw['duration']:.0f}-second window contains 
                        {len(display_rr)} RR intervals ready for comprehensive HRV analysis.
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown('<div style="padding-top: 8px;">', unsafe_allow_html=True)
                    if st.session_state.preview_mode and 'rr_intervals' in st.session_state.analyzer.ecg_data:
                        if st.button("Detect Ectopic Beats", type="secondary", use_container_width=True):
                            if 'rr_intervals' in st.session_state.analyzer.ecg_data:
                                current_rr = st.session_state.analyzer.ecg_data['rr_intervals']
                                ectopic_results = st.session_state.analyzer.detect_ectopic_beats(current_rr)
                                st.session_state.ectopic_results = ectopic_results
                            else:
                                st.warning("No RR intervals available. Please run peak detection first.")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-box">
                    <strong>⚠️ Warning:</strong> No RR intervals found in the selected time window. 
                    Please adjust the window or peak detection parameters.
                </div>
                """, unsafe_allow_html=True)

    close_plot_section()
        
    # Display results if available
    if 'ectopic_results' in st.session_state:
        results = st.session_state.ectopic_results
        
        if 'rr_intervals' in st.session_state.analyzer.ecg_data:
            rr_intervals = st.session_state.analyzer.ecg_data['rr_intervals']
        else:
            rr_intervals = []
        
        if results['total_flagged'] == 0:
            st.markdown(f"""
            <div class="window-info">
                <strong>✅ Quality Check:</strong> No ectopic beats detected in {len(rr_intervals)} RR intervals
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="window-info">
                <strong>⚠️ Quality Check:</strong> {results['total_flagged']} potential ectopic beats detected ({results['percentage_flagged']:.1f}%)
            </div>
            """, unsafe_allow_html=True)
                    
            # Manual review interface
            st.markdown("#### Manual Review")
            st.markdown("Review each flagged beat and decide whether to apply correction:")
            
            correction_decisions = {}
            td_peaks = st.session_state.analyzer.ecg_data.get('td_peaks', [])
            
            for i, (idx, reason) in enumerate(zip(results['flagged_indices'], results['flagged_reasons'])):
                col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 1])
                
                with col1:
                    st.write(f"**Beat {idx + 1}:**")
                with col2:
                    st.write(f"{rr_intervals[idx]:.0f} ms")
                with col3:
                    st.write(f"{reason}")
                with col4:
                    # Add timestamp if available
                    if idx < len(td_peaks):
                        st.write(f"Time: {td_peaks[idx]:.1f}s")
                    else:
                        st.write("Time: N/A")
                with col5:
                    correction_decisions[idx] = st.checkbox(
                        "Correct", 
                        key=f"correct_{idx}",
                        help=f"Apply linear interpolation to RR interval {idx + 1}"
                    )
            
            # Apply corrections button
            if st.button("Apply Selected Corrections", type="primary"):
                corrected_rr = st.session_state.analyzer.apply_ectopic_corrections(correction_decisions)
                
                approved_count = sum(correction_decisions.values())
                st.success(f"Applied {approved_count} corrections to RR intervals. Tachogram regenerated.")
                
                # Show before/after stats
                if approved_count > 0:
                    col1, col2 = st.columns(2)
                    with col1:
                        original_rr = st.session_state.analyzer.ectopic_correction_info['original_rr']
                        st.metric("Original Mean RR", f"{np.mean(original_rr):.1f} ms")
                        st.metric("Original RMSSD", f"{np.sqrt(np.mean(np.diff(original_rr)**2)):.1f} ms")
                    with col2:
                        st.metric("Corrected Mean RR", f"{np.mean(corrected_rr):.1f} ms")
                        st.metric("Corrected RMSSD", f"{np.sqrt(np.mean(np.diff(corrected_rr)**2)):.1f} ms")
                    
                    st.rerun()  # Auto-refresh immediately
                    st.info("RR intervals have been updated. You can now proceed with full analysis.")

    close_plot_section()

    # Enhanced action buttons
    st.markdown("---")
    st.markdown("### Next Steps")
    st.markdown("Choose your next action based on the preview results above:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
            if st.button("Accept & Run Full Analysis", use_container_width=True,
                        help="Proceed with analysis using current settings"):
                
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
                                st.success("Complete analysis finished successfully!")
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
        if st.button("Adjust Parameters", use_container_width=True,
                    help="Return to parameter adjustment mode"):
            st.session_state.preview_mode = False
            st.info("👈 Adjust parameters or time window in the sidebar and click 'Preview' again")
            st.rerun()
    
    with col3:
        if st.button("Use Default Settings", use_container_width=True,
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
                        st.success("Analysis with defaults completed!")
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
        <p style="margin: 0.5rem 0 0 0;">Upload an ACQ or EDF file using the sidebar to begin analysis of physiological signals</p>
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
                <li><strong>Channel Configuration:</strong> Flexible channel selection</li>
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
<div class="footer-section" style="text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(37, 99, 235, 0.95) 0%, rgba(79, 70, 229, 0.95) 100%); 
           border: 1px solid rgba(0, 0, 0, 0.7); border-radius: 10px; margin-top: 2rem; display: flex; align-items: center; justify-content: center;">
    <div style="display: flex; align-items: center; gap: 15px;">
        <img src="data:image/png;base64,{get_base64_of_image("logo.png")}" 
             style="width: 55px; height: 55px; object-fit: contain;" 
             alt="ChronOS Logo"/>
        <div>
            <p style="margin: 0; font-size: 0.9rem;">
                <strong>ChronOS v1.4</strong> | Professional HRV & BRS Analysis Platform<br>
                Built with Streamlit • Enhanced User Experience • Advanced Peak Detection • Time Window Selection
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)