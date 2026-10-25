"""
Page 3: Model Predictions
Displays independent dual model predictions (XGBoost Tabular vs Multimodal LSTM),
evaluates model agreement status, and displays quality-gated warnings without probability averaging.
"""

import streamlit as st
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.components.sidebar import render_sidebar
from app.components.metric_cards import render_prediction_cards, render_agreement_card, render_priority_card


st.set_page_config(page_title="Model Predictions | ICU Risk Intelligence", page_icon="🎯", layout="wide")
render_sidebar()

st.title("🎯 Dual Predictive Modeling & Model Agreement")
st.markdown("Independent tabular (XGBoost) and sequential (LSTM) mortality risk estimations.")

# Check for active session result
if "patient_result" not in st.session_state or st.session_state["patient_result"] is None:
    st.info("ℹ️ No patient has been analyzed in the current session. Please navigate to **Patient Analysis** to select or upload a record.")
    if st.button("Go to Patient Analysis →"):
        st.switch_page("pages/2_Patient_Analysis.py")
    st.stop()

res = st.session_state["patient_result"]
pid = res["patient_id"]
horizon = res["prediction_horizon_hours"]
dq = res["data_quality"]
pred_info = res["predictions"]
rep = res["report"]

st.markdown(f"### Patient #{pid} — {horizon}-Hour ICU Observation Horizon")

# 1. Telemetry Quality Gate Check
dqi = dq.get("dqi", 0.0)
status = dq.get("status", "UNKNOWN")

if dqi < 50.0 or pred_info.get("status") == "BLOCKED_BY_DATA_QUALITY":
    st.error(
        f"⛔ **PREDICTION INHIBITED BY QUALITY GATE:** The patient's Data Quality Index (DQI: {dqi:.1f}/100, Status: {status}) "
        "failed the minimum threshold of 50.0. In accordance with clinical AI safety standards, models are not executed on unverified or corrupted telemetry."
    )
    render_priority_card("CRITICAL_DATA_QUALITY", rep.get("rationale", ""))
    st.stop()

# 2. Dual Prediction Cards
st.subheader("1. Independent Model Estimations")
st.caption("Note: Model probabilities are derived from distinct mathematical representations and are strictly NOT averaged, blended, or synthesized into a fake consensus score.")

render_prediction_cards(pred_info, horizon_hours=horizon)

st.divider()

# 3. Model Agreement Evaluation
st.subheader("2. Model Agreement Analysis")
agreement_status = rep.get("agreement_status", "UNKNOWN")
agreement_meaning = rep.get("agreement_meaning", "")

render_agreement_card(agreement_status, agreement_meaning)

if agreement_status == "MODEL_DISCORDANT":
    st.warning(
        "⚠️ **METHODOLOGICAL DISCORDANCE DETECTED:** XGBoost and LSTM diverge across their validation thresholds (τ = 0.58). "
        "This divergence highlights architectural differences: XGBoost captures extreme static laboratory boundaries (e.g. extreme Bilirubin or BUN), "
        "whereas LSTM integrates dynamic rates of change and multivariable temporal stability over time. "
        "Examine the **Explainability** tab for individual feature and sequential attributions."
    )

# 4. Deterministic Research Review Priority
st.subheader("3. Deterministic Research Workflow Priority")
priority = rep.get("review_priority", "UNKNOWN")
rationale = rep.get("rationale", "")

render_priority_card(priority, rationale)

st.markdown(
    """
    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 12px; border-radius: 6px; font-size: 0.82rem; color: #64748B;">
        <b>Research Workflow Notice:</b> Review priority tiers (LOW, MODERATE, HIGH, CRITICAL_DATA_QUALITY) reflect structured algorithmic audit classifications. 
        They do not represent clinical triage emergency scores (such as ESI, START, or NEWS2).
    </div>
    """,
    unsafe_allow_html=True,
)
