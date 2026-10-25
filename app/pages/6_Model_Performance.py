"""
Page 6: Model Performance
Comprehensive retrospective benchmark evaluations across XGBoost, 1D CNN, Simple RNN, and LSTM.
Strictly displays verified experimental metrics from frozen validation, Test B, and Test C sets.
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
from src.services.icu_workflow import ICUWorkflowService


st.set_page_config(page_title="Model Performance | ICU Risk Intelligence", page_icon="📈", layout="wide")
render_sidebar()

st.title("📈 Model Performance & Retrospective Benchmarks")
st.markdown("Official evaluations on frozen PhysioNet Challenge 2012 / MIMIC-II test cohorts (4,000 Test B + 4,000 Test C).")

st.markdown(
    """
    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 12px; border-radius: 6px; font-size: 0.85rem; color: #475569; margin-bottom: 20px;">
        <b>Rigorous Validation Protocol:</b> All deep learning and gradient boosted tree models were tuned strictly on training/validation partitions (Set A). 
        Early stopping and decision thresholds (τ) were frozen before evaluating held-out Test Set B and Test Set C.
    </div>
    """,
    unsafe_allow_html=True,
)

service = ICUWorkflowService(base_dir=ROOT_DIR)
benchmarks_df = service.get_stored_model_benchmarks()

# -----------------------------------------------------------------------------
# 1. OFFICIAL BENCHMARK METRICS TABLE
# -----------------------------------------------------------------------------
st.subheader("1. Benchmark Summary Table (Validation vs Test B vs Test C)")

if not benchmarks_df.empty:
    st.dataframe(benchmarks_df, use_container_width=True, hide_index=True)
else:
    st.warning("Could not load reports/model_comparison.csv.")

st.divider()

# -----------------------------------------------------------------------------
# 2. EXPERIMENTAL CURVES & CONFUSION MATRICES (STORED ARTIFACTS)
# -----------------------------------------------------------------------------
st.subheader("2. Experimental Diagnostic Visualizations")

tab_roc, tab_cm, tab_hist = st.tabs(["ROC & PR Curves", "Confusion Matrices", "Deep Learning Training History"])

with tab_roc:
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("##### 🌲 XGBoost Baseline ROC & PR Curves")
        xgb_roc_path = ROOT_DIR / "reports" / "xgboost_roc_pr_curves.png"
        if xgb_roc_path.exists():
            st.image(str(xgb_roc_path), use_container_width=True)
        else:
            st.info("XGBoost ROC/PR curve image not found.")

    with col_r2:
        st.markdown("##### 🧠 Deep Learning ROC Curves (CNN, RNN, LSTM)")
        dl_roc_path = ROOT_DIR / "reports" / "deep_learning_roc_curves.png"
        if dl_roc_path.exists():
            st.image(str(dl_roc_path), use_container_width=True)
        else:
            st.info("Deep learning ROC curve image not found.")

    st.markdown("##### 🎯 Deep Learning Precision-Recall Curves")
    dl_pr_path = ROOT_DIR / "reports" / "deep_learning_pr_curves.png"
    if dl_pr_path.exists():
        st.image(str(dl_pr_path), use_container_width=True)

with tab_cm:
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("##### 🌲 XGBoost Confusion Matrices (Val, Test B, Test C)")
        xgb_cm_path = ROOT_DIR / "reports" / "xgboost_confusion_matrices.png"
        if xgb_cm_path.exists():
            st.image(str(xgb_cm_path), use_container_width=True)

    with col_c2:
        st.markdown("##### 🧠 Deep Learning Confusion Matrices (48h & 24h)")
        dl_cm_path = ROOT_DIR / "reports" / "deep_learning_confusion_matrices.png"
        if dl_cm_path.exists():
            st.image(str(dl_cm_path), use_container_width=True)

with tab_hist:
    st.markdown("##### 📉 Training & Validation Loss/AUROC History")
    hist_path = ROOT_DIR / "reports" / "training_history.png"
    if hist_path.exists():
        st.image(str(hist_path), use_container_width=True)
    else:
        st.info("Training history plot not found.")

st.divider()

# -----------------------------------------------------------------------------
# 3. XGBOOST GLOBAL FEATURE IMPORTANCE & SHAP SUMMARY
# -----------------------------------------------------------------------------
st.subheader("3. Global Tabular Feature Attribution")
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.markdown("##### 📊 Top 20 Tabular Feature Gain Importances")
    fi_path = ROOT_DIR / "reports" / "xgboost_feature_importance.png"
    if fi_path.exists():
        st.image(str(fi_path), use_container_width=True)

with col_g2:
    st.markdown("##### 🌈 Global Tree SHAP Summary (Beeswarm Distribution)")
    shap_sum_path = ROOT_DIR / "reports" / "xgboost_shap_summary.png"
    if shap_sum_path.exists():
        st.image(str(shap_sum_path), use_container_width=True)
