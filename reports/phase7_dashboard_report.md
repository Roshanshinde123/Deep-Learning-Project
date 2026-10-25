# Phase 7 — Streamlit Dashboard & System Integration Report

**Project Title:** Agentic AI-Assisted ICU Mortality Risk Prediction from the First 48 Hours  
**Execution Timestamp:** 2026-10-24  
**Dashboard Framework:** Streamlit (v1.61.1) with Plotly (v6.3.0) and Matplotlib (v3.10.8)  
**Backend & Modeling:** Python 3.13, XGBoost (v3.4.1), Keras (v3.15.1, Torch Backend), SHAP (v0.54.1), Scikit-Learn (v1.8.0)  

---

## 1. Executive Summary

Phase 7 delivers a comprehensive, interactive, healthcare-grade clinical research dashboard titled **"ICU Risk Intelligence: Agentic AI-Assisted ICU Mortality Prediction from the First 48 Hours"**. 

The dashboard provides a complete web user interface that orchestrates all four intelligent agents developed in Phase 6:
1. **Data Quality Agent:** Evaluates raw telemetry streams, flags sensor artifacts, checks missing descriptors, and calculates the Data Quality Index (DQI).
2. **Prediction Agent:** Runs independent inference through frozen XGBoost and LSTM architectures, comparing predictions against validation-tuned decision thresholds.
3. **Explanation Agent:** Derives local feature attributions via Tree SHAP for tabular features and temporal occlusion perturbations for longitudinal sequences.
4. **Reporting Agent:** Synthesizes upstream findings into deterministic research review priorities, comprehensive markdown reports, and schema-compliant JSON documents.

All core models and pipelines remain intact, with zero modifications to frozen checkpoints, training procedures, or ground truth datasets.

---

## 2. System Architecture & Component Design

The application structure adheres strictly to the modular architectural blueprint specified in Part 12:

```text
D:\ICU_Prediction\
├── app\
│   ├── streamlit_app.py               # Main Application Landing Page & Routing Hub
│   ├── pages\
│   │   ├── 1_Dashboard_Overview.py    # Retrospective Cohort & Comparative Analytics
│   │   ├── 2_Patient_Analysis.py      # Telemetry Ingestion, DQI Audit & Pipeline Trigger
│   │   ├── 3_Model_Predictions.py     # Independent XGBoost & LSTM Risk Cards & Concordance
│   │   ├── 4_Explainability.py        # Local Tree SHAP Waterfalls & Longitudinal Dynamics
│   │   ├── 5_Agentic_AI_Reports.py    # Structured Research Review Synthesis & Exports
│   │   ├── 6_Model_Performance.py     # Retrospective Validation/Test Performance Benchmarks
│   │   └── 7_About_Project.py         # Clinical Context, Pipeline Architecture & Governance
│   └── components\
│       ├── sidebar.py                 # Persistent Patient Status, Configuration & Safety Disclaimer
│       ├── metric_cards.py            # Healthcare-styled KPI Metric Cards
│       ├── charts.py                  # Interactive Plotly Trajectories & Visualizations
│       └── report_viewer.py           # Markdown & Machine-Readable JSON Export Viewers
├── src\
│   ├── services\
│   │   ├── __init__.py
│   │   └── icu_workflow.py            # Master Orchestration Service (Stateless Agent Adapter)
│   ├── agents\                        # Core Intelligent Agent Implementations
│   └── preprocessing\                 # Frozen Preprocessing & Feature Engineering Pipelines
├── models\                            # Frozen Machine Learning Checkpoints & Scalers
├── reports\                           # Retrospective Experiment Reports & Performance Tables
└── tests\
    └── test_dashboard_integration.py  # 15-Point End-to-End System Integration Suite
```

---

## 3. Detailed Page Functionality & Implementation

### 3.1 Main Landing Page (`streamlit_app.py`)
- **Visual Design:** Healthcare-inspired dark-mode styling with subtle slate accents and card elevation.
- **Hero Banner:** Displays the project title, subtitle, and an interactive navigation grid.
- **Workflow Overview:** High-level visualization of the 4-tier pipeline flow (Ingestion $\rightarrow$ Quality Gate $\rightarrow$ Dual Model Inference $\rightarrow$ Explanation Engine $\rightarrow$ Research Synthesis).
- **Persistent Safety Disclaimer:** Prominently highlights the prototype status: *"Educational and research prototype only. Not for clinical diagnosis or treatment decisions."*

### 3.2 Dashboard Overview (`1_Dashboard_Overview.py`)
- **Cohort Summary Cards:** Summarizes total available records (4,000 Set A records), 4 functional agents, 2 primary production models (XGBoost & LSTM), and 7 completed development phases.
- **Interactive Visualizations:**
  - In-hospital mortality outcome distribution (13.85% positive class balance).
  - Validation vs. held-out test discrimination (AUROC & AUPRC grouped bar comparisons).
  - Imputation strategy impact (forward-fill Method B vs. mean imputation).
  - 24-hour vs. 48-hour observation window comparison showing discrimination gain over time.

### 3.3 Patient Analysis & Telemetry Audit (`2_Patient_Analysis.py`)
- **Dual Ingestion Modalities:**
  - **Option A:** Upload custom PhysioNet `.txt` patient file.
  - **Option B:** Select from permitted local demonstration cohort (e.g., Patient #132539, #132551, #132588, #132601).
- **Observation Horizon Control:** Supports toggle between First 24 Hours and First 48 Hours.
- **Streamlit Status Container:** Uses `st.status` for real-time visual progress as each agent executes sequentially without fake progress bars or simulated pauses.
- **Data Quality Dashboard:**
  - Prominent DQI metric card with color-coded quality grade (Excellent, Good, Fair, Critical).
  - Quality gate banner indicating whether the record is cleared for machine learning inference.
  - Missing descriptor audit (Height, Weight, Age, Gender, ICU Type).
  - Sensor artifact log separating genuine extreme physiological values from impossible sensor readings.
  - Interactive multi-parameter vital sign trajectory chart (Plotly) and observation density bar chart.

### 3.4 Model Predictions & Concordance (`3_Model_Predictions.py`)
- **Dual Risk Prediction Cards:**
  - **XGBoost:** Probability estimate (e.g., 0.7325), validation threshold (0.2855), and above/below status.
  - **LSTM:** Probability estimate (e.g., 0.7573), validation threshold (0.3540), and above/below status.
- **Strict Independence Policy:** Explicitly prohibits averaging model probabilities. Prohibits labeling model probabilities as "clinical confidence scores".
- **Concordance Analysis:**
  - Categorizes status into `MODEL_CONCORDANT_HIGH`, `MODEL_CONCORDANT_LOW`, or `MODEL_DISCORDANT`.
  - Emits clear amber warnings during discordance, detailing architectural hypotheses (e.g., early static risk vs. acute trajectory worsening).
- **Deterministic Priority Badge:** Displays the assigned research review priority (`HIGH`, `MODERATE`, `LOW`, or `CRITICAL_DATA_QUALITY`).

### 3.5 Explainability Engine (`4_Explainability.py`)
- **XGBoost Tree SHAP Attribution:**
  - High-resolution SHAP waterfall plots for local patient prediction breakdown.
  - Top 5 positive risk factors (elevating mortality probability) and top 5 protective factors (lowering probability).
  - Real observed patient values displayed alongside feature attribution values.
- **LSTM Temporal Occlusion & Trends:**
  - Longitudinal feature importance ranking computed via temporal masking.
  - Trajectory trend classification (Worsening, Improving, Stable, Insufficient Data) across 9 key clinical parameters: Heart Rate, Temperature, GCS, BUN, Urine, Respiratory Rate, Sodium, Lactate, and pH.
  - Explicit advisory stating explanations represent statistical associations within models, not clinical causality.

### 3.6 Agentic AI Reports (`5_Agentic_AI_Reports.py`)
- **Unified Review Document:** Renders the complete, professional ASCII review document generated by `ReportingAgent`.
- **Export Facilities:**
  - Download full research report as a structured Markdown (`.md`) file.
  - Download full report as a machine-readable JSON (`.json`) conforming to the project schema.
- **Clinical Limitations Section:** Automatically populates limitations regarding retrospective PhysioNet 2012 cohort constraints, missingness patterns, and institutional calibration requirements.

### 3.7 Retrospective Model Performance (`6_Model_Performance.py`)
- **Retrospective Benchmark Table:** Full comparison of XGBoost, 1D CNN, Simple RNN, and LSTM across Validation, Test Set B, and Test Set C (AUROC, AUPRC, F1, Sensitivity, Specificity, Accuracy).
- **ROC and Precision-Recall Visualizations:** Displays precomputed discrimination curves.
- **Confusion Matrices & Training History:** Displays classification trade-offs and convergence curves from Phase 5 experiments.
- **Integrity Guarantee:** Uses precomputed values from saved files; patient uploads never mutate model performance tables.

### 3.8 About Project & Governance (`7_About_Project.py`)
- **Research Background:** Details the clinical challenge of early ICU mortality prediction.
- **4-Tier Agent Pipeline Diagram:** Comprehensive Mermaid flowchart illustrating agent handoffs, quality gates, and output synthesis.
- **Dataset Provenance:** Full documentation of PhysioNet/Computing in Cardiology Challenge 2012 data provenance.
- **Ethical Guardrails:** Clear statement on algorithmic bias, fairness, transparency, and intended use as a decision-support prototype.

---

## 4. Verification & Testing Results

### 4.1 Automated Integration Tests
The dedicated integration suite in `tests/test_dashboard_integration.py` was executed alongside the workspace test suite:
- **Total Tests Executed:** 69 tests across 7 test files.
- **Tests Passed:** 69 (100%).
- **Failures / Errors:** 0.
- **Execution Time:** 19.44 seconds.

### 4.2 Browser Subagent UI Validation
A headless browser subagent automated an end-to-end verification session of the live dashboard at `http://localhost:8501`:
- **Navigation:** All 7 multi-page links navigated successfully without 404s or blank screens.
- **Console Errors:** 0 JavaScript runtime errors, 0 Streamlit exceptions.
- **Interactive Controls:** Horizon selection, sample selection, and Plotly interactive zooming functioned smoothly.
- **Artifacts Recorded:** 
  - Subagent recording session saved to WebP video artifact.
  - High-resolution page screenshots captured for each major application view.

---

## 5. Deployment & Execution Instructions

### 5.1 Environment Requirements
- Python 3.10 – 3.13
- Dependencies specified in `requirements.txt`:
  ```text
  streamlit>=1.35.0,<=1.61.1
  xgboost>=2.0.0,<=3.4.1
  torch>=2.0.0
  keras>=3.0.0,<=3.15.1
  shap>=0.45.0,<=0.54.1
  pandas>=2.0.0,<=2.3.3
  numpy>=1.26.0,<=2.3.5
  scikit-learn>=1.4.0,<=1.8.0
  joblib>=1.3.0,<=1.5.2
  plotly>=5.20.0,<=6.3.0
  matplotlib>=3.8.0,<=3.10.8
  ```

### 5.2 Launch Command
To launch the dashboard locally:
```bash
streamlit run app/streamlit_app.py
```
Default local access URL: `http://localhost:8501`

---

## 6. Conclusion

Phase 7 is **100% complete**. The dashboard establishes a standard for transparent, explainable, and multi-agent clinical machine learning prototypes.
