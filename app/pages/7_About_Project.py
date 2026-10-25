"""
Page 7: About Project
Background documentation, multi-agent architecture breakdown,
academic citations, and safety guidelines.
"""

import streamlit as st
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.components.sidebar import render_sidebar


st.set_page_config(page_title="About Project | ICU Risk Intelligence", page_icon="📖", layout="wide")
render_sidebar()

st.title("📖 About ICU Risk Intelligence")
st.markdown("Agentic AI-Assisted In-Hospital Mortality Risk Prediction from the First 48 Hours of Intensive Care.")

# Project Mission
st.markdown(
    """
    ### 🎯 Research Objective & Scope
    In critical care medicine, early identification of physiological deterioration during the first 24 to 48 hours 
    of ICU admission is vital for prioritizing clinical attention. Traditional risk-scoring systems (such as SAPS-II, APACHE-IV, and SOFA) 
    rely on static worst-value aggregations and do not fully leverage granular, time-varying trajectory shifts or observation missingness patterns.

    **ICU Risk Intelligence** bridges this gap through a modular **4-tier Agentic AI system** that couples:
    1. **Data Quality Auditing (Tier 1):** Screening EHR telemetry for transducer disconnection artifacts and observational missingness before inference.
    2. **Dual Model Inference (Tier 2):** Combining gradient-boosted tree models on tabular aggregations (XGBoost) with deep sequential recurrent networks (LSTM).
    3. **Multi-Faceted Explainability (Tier 3):** Decomposing predictions into exact Tree SHAP attributions and sequential temporal occlusion sensitivities.
    4. **Quality-Gated Reporting (Tier 4):** Synthesizing agreement status and research review priorities under strict clinical research guardrails.
    """
)

st.divider()

# Multi-Agent Architecture
st.subheader("🤖 The 4 Autonomous Agents")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        #### 1. Data Quality Agent (`DataQualityAgent`)
        - **Role:** First line of defense and automated quality gatekeeper.
        - **DQI (0-100):** Research-defined index penalizing unmonitored essential vitals, sensor disconnection bounds, and observation sparsity.
        - **Quality Gate:** Any record with $DQI < 50.0$ triggers `PREDICTION_BLOCKED`, strictly inhibiting downstream modeling.

        #### 2. Prediction Agent (`PredictionAgent`)
        - **Role:** Autonomous dual-model inference engine.
        - **XGBoost:** Evaluates 334 patient-level tabular features (min, max, mean, std, trend, counts).
        - **LSTM:** Processes dynamic 74-channel multimodal temporal tensors (37 clinical values + 37 observation masks).
        - **Independence:** Probabilities are never averaged or blended.
        """
    )

with col2:
    st.markdown(
        """
        #### 3. Explanation Agent (`ExplanationAgent`)
        - **Role:** Interpretability and clinical transparency.
        - **Tree SHAP:** Computes exact local Shapley values separating risk-increasing (+SHAP) from protective (-SHAP) features.
        - **Temporal Occlusion:** Systematically perturbs 12-hour time quarters (Q1-Q4) and dynamic channels to measure marginal impact.
        - **Trajectory Landmarks:** Extracts net deltas and trend direction for key laboratory and vital markers.

        #### 4. Reporting / Priority Agent (`ReportingAgent`)
        - **Role:** Orchestration, agreement analysis, and report generation.
        - **Concordance:** Evaluates `MODEL_CONCORDANT_LOW`, `MODEL_CONCORDANT_HIGH`, and `MODEL_DISCORDANT`.
        - **Deterministic Priority:** Assigns research review tiers (`LOW`, `MODERATE`, `HIGH`, `CRITICAL_DATA_QUALITY`).
        """
    )

st.divider()

# Dataset & Benchmark Provenance
st.subheader("📚 Dataset & Validation Provenance")
st.markdown(
    """
    - **Cohort:** PhysioNet Challenge 2012 / MIMIC-II (Retrospective intensive care admissions).
    - **Dataset Size:** 12,000 adult patient admissions across medical, surgical, cardiac, and trauma ICUs.
      - **Set A (4,000 records):** Used for EDA, feature engineering, cross-validation, and threshold selection.
      - **Set B (4,000 records):** Frozen held-out test evaluation.
      - **Set C (4,000 records):** Unseen secondary test evaluation.
    - **Target Outcome:** In-Hospital Mortality (`In-hospital_death`), with an overall cohort mortality rate of ~14.1%.
    """
)

st.divider()

# Regulatory & Safety Disclaimers
st.markdown(
    """
    ### 🛡️ Clinical Research Safety & Boundaries
    - **Non-Diagnostic:** This application is strictly an educational prototype and research platform.
    - **No Treatment Directives:** The system never provides medication prescriptions, ventilator adjustments, or intervention orders.
    - **Retrospective Validation:** All reported metrics are derived from retrospective MIMIC-II observational records; prospective clinical trials have not been conducted.
    - **Physician Oversight:** All predictions must be interpreted by qualified healthcare providers alongside real-time bedside clinical assessment.
    """
)
