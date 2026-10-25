"""
Page 1: Dashboard Overview
Displays high-level cohort telemetry metrics, model comparisons,
and official experimental benchmark metrics from saved reports.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.components.sidebar import render_sidebar
from app.components.charts import plot_benchmark_comparison
from src.services.icu_workflow import ICUWorkflowService


st.set_page_config(page_title="Dashboard Overview | ICU Risk Intelligence", page_icon="📊", layout="wide")
render_sidebar()

st.title("📊 Project Overview & Benchmark Performance")
st.markdown("High-level summary of dataset cohorts, model architectures, and validation-frozen metrics.")

# 1. Project-level Metric Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Cohort Records", "12,000", help="4,000 Set A (training/validation), 4,000 Set B (test), 4,000 Set C (test)")

with col2:
    st.metric("Prediction Models", "2 Independent", help="Primary: XGBoost (334 tabular features) | Secondary: Multimodal LSTM (74 sequential channels)")

with col3:
    st.metric("Autonomous Agents", "4 Functional", help="Data Quality Agent, Prediction Agent, Explanation Agent, Reporting Agent")

with col4:
    st.metric("Development Phases", "7 / 7 Complete", help="Phases 1 through 7 completed and audited")

st.divider()

# 2. Benchmark Comparisons from Saved Reports
workflow = ICUWorkflowService(base_dir=ROOT_DIR)
benchmarks_df = workflow.get_stored_model_benchmarks()
imp_df = workflow.get_stored_imputation_comparison()

st.subheader("🏆 Model Performance Comparison (Saved Retrospective Evaluations)")
st.markdown(
    """
    Performance across primary tabular (XGBoost) and deep learning architectures (1D CNN, Simple RNN, LSTM) 
    frozen during Phase 4 and Phase 5 evaluations. *Evaluation metrics are strictly loaded from verified experimental artifacts.*
    """
)

if not benchmarks_df.empty:
    tab1, tab2, tab3 = st.tabs(["Test C (Unseen Evaluation)", "Test B (Held-Out Evaluation)", "Validation Tuning"])

    with tab1:
        st.markdown("##### 📌 Test Set C Official Metrics")
        st.plotly_chart(plot_benchmark_comparison(benchmarks_df, metric_set="TestC"), use_container_width=True)
        
        # Display clean summary table for Test C
        testc_cols = ["Model", "Model_Type", "Horizon", "Optimal_Threshold", "TestC_AUROC", "TestC_AUPRC", "TestC_F1", "TestC_Sens", "TestC_Spec"]
        display_c = benchmarks_df[[c for c in testc_cols if c in benchmarks_df.columns]].copy()
        display_c.columns = [c.replace("TestC_", "") for c in display_c.columns]
        st.dataframe(display_c, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("##### 📌 Test Set B Held-Out Metrics")
        st.plotly_chart(plot_benchmark_comparison(benchmarks_df, metric_set="TestB"), use_container_width=True)
        
        testb_cols = ["Model", "Model_Type", "Horizon", "Optimal_Threshold", "TestB_AUROC", "TestB_AUPRC", "TestB_F1", "TestB_Sens", "TestB_Spec"]
        display_b = benchmarks_df[[c for c in testb_cols if c in benchmarks_df.columns]].copy()
        display_b.columns = [c.replace("TestB_", "") for c in display_b.columns]
        st.dataframe(display_b, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("##### 📌 Validation Set (Early Stopping & Threshold Tuning)")
        val_cols = ["Model", "Model_Type", "Horizon", "Optimal_Threshold", "Val_AUROC", "Val_AUPRC", "Val_F1", "Val_Sens", "Val_Spec"]
        display_val = benchmarks_df[[c for c in val_cols if c in benchmarks_df.columns]].copy()
        display_val.columns = [c.replace("Val_", "") for c in display_val.columns]
        st.dataframe(display_val, use_container_width=True, hide_index=True)
else:
    st.warning("Benchmark metrics file 'reports/model_comparison.csv' not found.")

st.divider()

# 3. Methodological Comparisons: 24h vs 48h & Imputation
st.subheader("🔬 Methodological Insights")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### ⏱️ 24-Hour vs 48-Hour Horizon Comparison")
    st.markdown(
        """
        - **48-Hour Observation (Standard):** Delivers superior discriminatory power (XGBoost Test C AUROC: **0.8602**, LSTM AUROC: **0.8187**).
        - **24-Hour Observation (Early Triage):** Achieves strong early discrimination (LSTM Test C AUROC: **0.7986**, AUPRC: **0.4101**) enabling rapid triage of deteriorating patients 24 hours earlier.
        - **Clinical Utility:** 24h prediction serves as an early warning trigger; 48h provides a comprehensive diagnostic synthesis.
        """
    )

with col_b:
    st.markdown("#### 🩹 Imputation Strategy Impact (Method A vs Method B)")
    st.markdown(
        """
        - **Method A (Zero Imputation):** Substantially degrades temporal continuous modeling in recurrent architectures due to dense artificial zero values.
        - **Method B (Causal Forward-Fill + Training Medians):** Maintains natural physiological continuity without lookahead leakage.
        - **Mask Integration:** Dual-channel representation (37 values + 37 observation masks) preserved for temporal deep learning.
        """
    )
    if not imp_df.empty:
        st.dataframe(imp_df, use_container_width=True, hide_index=True)
