"""
Page 4: Explainability
Displays local Tree SHAP attributions for XGBoost, sequential occlusion sensitivities for LSTM,
and physiological trajectory shift tables.
"""

import os
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.components.sidebar import render_sidebar


st.set_page_config(page_title="Explainability | ICU Risk Intelligence", page_icon="🔍", layout="wide")
render_sidebar()

st.title("🔍 Model Explainability & Feature Attribution")
st.markdown("Dual-model interpretability: Tree SHAP (XGBoost) and sequential perturbation sensitivities (LSTM).")

# Check session state
if "patient_result" not in st.session_state or st.session_state["patient_result"] is None:
    st.info("ℹ️ No patient has been analyzed in the current session. Please navigate to **Patient Analysis** to select or upload a record.")
    if st.button("Go to Patient Analysis →"):
        st.switch_page("pages/2_Patient_Analysis.py")
    st.stop()

res = st.session_state["patient_result"]
pid = res["patient_id"]
horizon = res["prediction_horizon_hours"]
expl = res["explanation"]
pred_status = res["predictions"].get("status")

st.markdown(f"### Patient #{pid} ({horizon}-Hour Observation Horizon)")

# Quality Gate Block Check
if pred_status == "BLOCKED_BY_DATA_QUALITY" or expl.get("status") == "BLOCKED_BY_DATA_QUALITY":
    st.error(
        "⛔ **EXPLANATION INHIBITED:** The patient's telemetry failed minimum clinical data quality standards (DQI < 50.0). "
        "To prevent misleading or fabricated clinical attributions, local SHAP explanations and perturbation matrices are not generated."
    )
    st.stop()

st.markdown(
    """
    <div style="background-color: #F0FDF4; border-left: 4px solid #16A34A; padding: 12px; border-radius: 4px; margin-bottom: 20px;">
        <p style="color: #166534; margin: 0; font-size: 0.85rem; font-weight: 600;">
            🔬 Clinical Research Interpretability Notice
        </p>
        <p style="color: #15803D; margin: 4px 0 0 0; font-size: 0.82rem; line-height: 1.4;">
            Feature attributions reflect statistical associations within the trained model weights, <b>not verified biological causation</b>. 
            All attributions should be contextualized alongside bedside clinical assessments.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs([
    "1. XGBoost Local SHAP (Tabular)",
    "2. LSTM Temporal Dynamics (Sequential)",
    "3. Longitudinal Physiological Trajectories",
])

# -----------------------------------------------------------------------------
# TAB 1: XGBOOST TREE SHAP
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("🌲 Tree SHAP Local Feature Attribution")
    st.markdown("Exact Shapley additive attributions decomposing the XGBoost prediction into log-odds shifts.")

    xgb_expl = expl.get("xgboost_explanation", {})
    top_pos = xgb_expl.get("top_positive_contributors", [])
    top_neg = xgb_expl.get("top_negative_contributors", [])
    plot_path = xgb_expl.get("waterfall_plot_path")

    # Display Waterfall Plot Image if generated
    if plot_path and os.path.exists(plot_path):
        st.markdown("##### 📊 SHAP Waterfall Plot")
        st.image(plot_path, caption=f"Tree SHAP Waterfall Plot — Patient #{pid} ({horizon}h)", use_container_width=True)

    col_pos, col_neg = st.columns(2)

    with col_pos:
        st.markdown("##### 🔴 Top Risk-Increasing Contributors (+SHAP)")
        if top_pos:
            df_pos = pd.DataFrame(top_pos)
            # Reorder / format columns
            cols = [c for c in ["feature", "actual_value", "shap_value"] if c in df_pos.columns]
            df_pos = df_pos[cols]
            df_pos.columns = ["Feature", "Patient Value", "SHAP Impact (Log-Odds)"]
            df_pos["SHAP Impact (Log-Odds)"] = df_pos["SHAP Impact (Log-Odds)"].map(lambda x: f"+{x:.4f}")
            st.dataframe(df_pos, use_container_width=True, hide_index=True)
        else:
            st.write("No strong risk-increasing factors above baseline.")

    with col_neg:
        st.markdown("##### 🟢 Top Protective Associations (-SHAP)")
        if top_neg:
            df_neg = pd.DataFrame(top_neg)
            cols = [c for c in ["feature", "actual_value", "shap_value"] if c in df_neg.columns]
            df_neg = df_neg[cols]
            df_neg.columns = ["Feature", "Patient Value", "SHAP Impact (Log-Odds)"]
            df_neg["SHAP Impact (Log-Odds)"] = df_neg["SHAP Impact (Log-Odds)"].map(lambda x: f"{x:.4f}")
            st.dataframe(df_neg, use_container_width=True, hide_index=True)
        else:
            st.write("No strong protective factors below baseline.")

# -----------------------------------------------------------------------------
# TAB 2: LSTM TEMPORAL DYNAMICS
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("⏱️ LSTM Temporal Occlusion & Window Sensitivity")
    st.markdown("Quantifies the marginal predictive shift when masking discrete time quarters or dynamic physiological channels.")

    lstm_expl = expl.get("lstm_explanation", {})
    periods = lstm_expl.get("important_time_periods", [])
    features = lstm_expl.get("important_features", [])

    col_per, col_feat = st.columns(2)

    with col_per:
        st.markdown("##### 🕒 Sequential Time Period Sensitivity")
        if periods:
            df_per = pd.DataFrame(periods)
            st.dataframe(df_per, use_container_width=True, hide_index=True)
        else:
            st.write("No quarter perturbation shifts available.")

    with col_feat:
        st.markdown("##### 📈 Top Dynamic Trajectory Influences")
        if features:
            df_feat = pd.DataFrame(features)
            st.dataframe(df_feat, use_container_width=True, hide_index=True)
        else:
            st.write("No dynamic channel perturbation shifts available.")

# -----------------------------------------------------------------------------
# TAB 3: LONGITUDINAL TRAJECTORIES
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("📉 Observed Physiological Parameter Transitions")
    st.markdown("Longitudinal shifts across key physiological markers with recorded observations.")

    trends = expl.get("temporal_changes", [])
    if trends:
        df_trends = pd.DataFrame(trends)
        st.dataframe(df_trends, use_container_width=True, hide_index=True)
    else:
        st.info("No longitudinal transition parameters recorded for this patient.")
