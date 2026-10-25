"""
Main Application Entrypoint: ICU Risk Intelligence
Agentic AI-Assisted ICU Mortality Risk Prediction from the First 48 Hours.
"""

import streamlit as st
from pathlib import Path
import sys

# Ensure root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.components.sidebar import render_sidebar
from src.services.icu_workflow import ICUWorkflowService


def init_session_state():
    """Initializes global session state for patient isolation."""
    if "patient_result" not in st.session_state:
        st.session_state["patient_result"] = None
    if "patient_df" not in st.session_state:
        st.session_state["patient_df"] = None
    if "selected_patient_id" not in st.session_state:
        st.session_state["selected_patient_id"] = None
    if "prediction_horizon_hours" not in st.session_state:
        st.session_state["prediction_horizon_hours"] = 48


# Configure Streamlit page
st.set_page_config(
    page_title="ICU Risk Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom healthcare aesthetic CSS
st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 800;
            color: #0F172A;
            margin-bottom: 0px;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #475569;
            margin-top: 4px;
            margin-bottom: 24px;
        }
        .hero-card {
            background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
            color: white;
            padding: 24px;
            border-radius: 12px;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .feature-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            height: 100%;
        }
        .disclaimer-banner {
            background-color: #FEF2F2;
            border: 1px solid #FCA5A5;
            border-left: 5px solid #EF4444;
            padding: 14px 18px;
            border-radius: 6px;
            margin-bottom: 24px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

init_session_state()
render_sidebar()

# Main Landing Header
st.markdown('<div class="main-header">🏥 ICU Risk Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Agentic AI-Assisted In-Hospital Mortality Risk Prediction from the First 48 Hours of Intensive Care</div>',
    unsafe_allow_html=True,
)

# Mandatory Top Disclaimer Banner
st.markdown(
    """
    <div class="disclaimer-banner">
        <h4 style="color: #991B1B; margin: 0 0 6px 0; font-size: 0.95rem;">⚠️ Academic Research & Educational Prototype Only</h4>
        <p style="color: #B91C1C; margin: 0; font-size: 0.85rem; line-height: 1.4;">
            This system is an academic research platform designed for the retrospective study of the PhysioNet Challenge 2012 / MIMIC-II dataset. 
            All outputs are <b>model-estimated statistical probabilities</b> and <b>research workflow prioritization labels</b>. 
            The system is <b>NOT intended for autonomous clinical diagnosis, emergency triage, or treatment selection</b>.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Hero Banner
st.markdown(
    """
    <div class="hero-card">
        <h2 style="margin: 0; font-size: 1.7rem; font-weight: 700; color: white;">
            Multi-Agent Clinical Intelligence Architecture
        </h2>
        <p style="margin: 10px 0 0 0; font-size: 0.95rem; opacity: 0.92; line-height: 1.5;">
            An integrated agentic system orchestrating four specialized tiers: Data Quality Auditing (DQI), Dual Predictive Modeling (XGBoost Tabular + Multimodal Sequential LSTM), Tree SHAP & Temporal Explainability, and Transparent Research Reporting.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Quick Navigation Grid
st.markdown("### 🧭 System Navigation")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="feature-card">
            <h4 style="color: #1E3A8A; margin-top: 0;">📊 Dashboard Overview</h4>
            <p style="color: #475569; font-size: 0.88rem; line-height: 1.4;">
                Explore dataset characteristics across 12,000 PhysioNet records, benchmark performance metrics, and precomputed experimental results.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Go to Overview →", key="btn_nav_overview", use_container_width=True):
        st.switch_page("pages/1_Dashboard_Overview.py")

with col2:
    st.markdown(
        """
        <div class="feature-card">
            <h4 style="color: #1E3A8A; margin-top: 0;">🔬 Patient Analysis</h4>
            <p style="color: #475569; font-size: 0.88rem; line-height: 1.4;">
                Select a benchmark patient or upload raw telemetry to trigger live, end-to-end multi-agent risk evaluation and quality auditing.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Analyze Patient →", key="btn_nav_analysis", use_container_width=True):
        st.switch_page("pages/2_Patient_Analysis.py")

with col3:
    st.markdown(
        """
        <div class="feature-card">
            <h4 style="color: #1E3A8A; margin-top: 0;">📈 Model Benchmarks</h4>
            <p style="color: #475569; font-size: 0.88rem; line-height: 1.4;">
                Inspect rigorous held-out test evaluations comparing XGBoost (AUROC 0.860), 1D CNN, Simple RNN, and Multimodal LSTM.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("View Benchmarks →", key="btn_nav_benchmarks", use_container_width=True):
        st.switch_page("pages/6_Model_Performance.py")

st.divider()

# System Architectural Flow
st.markdown("### 🔄 Multi-Agent Pipeline Overview")

cols = st.columns(4)

with cols[0]:
    st.markdown(
        """
        <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-top: 4px solid #3B82F6; padding: 14px; border-radius: 6px;">
            <b style="color: #1E3A8A;">Tier 1: Data Quality</b>
            <p style="font-size: 0.82rem; color: #475569; margin-top: 6px; margin-bottom: 0;">
                Audits 37 clinical channels, screens sensor disconnection artifacts, and enforces DQI Gate (50.0).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with cols[1]:
    st.markdown(
        """
        <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-top: 4px solid #2563EB; padding: 14px; border-radius: 6px;">
            <b style="color: #1E3A8A;">Tier 2: Prediction</b>
            <p style="font-size: 0.82rem; color: #475569; margin-top: 6px; margin-bottom: 0;">
                Independent dual inference: primary XGBoost tabular (334 features) + secondary LSTM sequential (74 channels).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with cols[2]:
    st.markdown(
        """
        <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-top: 4px solid #4F46E5; padding: 14px; border-radius: 6px;">
            <b style="color: #1E3A8A;">Tier 3: Explanation</b>
            <p style="font-size: 0.82rem; color: #475569; margin-top: 6px; margin-bottom: 0;">
                Computes Tree SHAP attributions, temporal landmark trajectories, and LSTM occlusion quarter sensitivities.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with cols[3]:
    st.markdown(
        """
        <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-top: 4px solid #7C3AED; padding: 14px; border-radius: 6px;">
            <b style="color: #1E3A8A;">Tier 4: Reporting</b>
            <p style="font-size: 0.82rem; color: #475569; margin-top: 6px; margin-bottom: 0;">
                Evaluates model agreement, assigns deterministic priority (LOW/MOD/HIGH/CRITICAL), and exports JSON/Markdown.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
