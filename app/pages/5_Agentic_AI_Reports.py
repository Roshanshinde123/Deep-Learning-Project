"""
Page 5: Agentic AI Reports
Displays the final, quality-gated research review summary from the Reporting Agent.
Provides formatted ASCII view and direct downloads for JSON and Markdown deliverables.
"""

import streamlit as st
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.components.sidebar import render_sidebar
from app.components.report_viewer import render_report_view


st.set_page_config(page_title="Agentic AI Reports | ICU Risk Intelligence", page_icon="📑", layout="wide")
render_sidebar()

st.title("📑 Structured Multi-Agent Research Reports")
st.markdown("Automated synthesis produced by the capstone Reporting / Priority Agent.")

# Check session state
if "patient_result" not in st.session_state or st.session_state["patient_result"] is None:
    st.info("ℹ️ No patient has been analyzed in the current session. Please navigate to **Patient Analysis** to generate a report.")
    if st.button("Go to Patient Analysis →"):
        st.switch_page("pages/2_Patient_Analysis.py")
    st.stop()

res = st.session_state["patient_result"]
report_dict = res["report"]
human_report = res.get("human_readable_report", "")

# Render Report Viewer Component
render_report_view(report_dict=report_dict, human_report=human_report)

st.divider()

# Clinical Limitations Section
with st.expander("🛡️ Clinical Research Limitations & Regulatory Guardrails", expanded=False):
    limitations = report_dict.get("limitations", [])
    for idx, lim in enumerate(limitations, 1):
        st.markdown(f"{idx}. {lim}")
