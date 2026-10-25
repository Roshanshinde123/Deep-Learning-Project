# Phase 6.2 — Prediction Agent Live Demonstration & Clinical Scenario Audit

**Date of Execution:** 2026-10-23  
**Project:** Agentic AI-Assisted ICU Mortality Risk Prediction from the First 48 Hours  
**Primary Model:** XGBoost Tabular Baseline (334 Features; `best_xgboost_forward_fill_48h.json`)  
**Secondary Model:** Bidirectional LSTM Sequential Network (74 Channels; `lstm_48h_forward_fill.keras`)  
**Data Quality Connection:** Autonomous Clinical Data Quality Agent (Phase 6.1 Refined)  

---

## 1. Executive Summary & Architecture Overview

The **Prediction Agent** serves as the central analytical engine in the ICU Decision Support Framework.
It is engineered with rigorous architectural isolation:
1. **Automated Quality Gatekeeper:** Evaluates telemetry via `DataQualityAgent`. If the research-defined Data Quality Index (DQI) falls below 50.0, automated predictions are **strictly blocked**, preventing the fabrication of ungrounded probabilities on degraded signals.
2. **Deterministic Dual-Model Inference:** Executes both the **Primary Model** (XGBoost, exploiting 334 hand-crafted tabular statistics, observation counts, and trajectory slopes) and the **Secondary Model** (LSTM, modeling continuous temporal dynamics across a 74-channel multimodal tensor).
3. **Threshold Calibration:** Compares predicted probabilities against research decision thresholds optimized exclusively on the retrospective Validation set ($F_1$-score maximization):
   - **48-Hour Horizon:** XGBoost = 0.58 | LSTM = 0.58
   - **24-Hour Horizon:** XGBoost = 0.57 | LSTM = 0.66
4. **Clinical Safety Safeguard:** Models are never averaged, and consensus is not claimed without a validated prospective ensemble. Model outputs are clearly labeled as research-only predictions.

---

## 2. Real-Patient Clinical Case Evaluations (48-Hour Horizon)

### Patient Record #132539 (48-Hour Horizon)
- **Official Outcome (Ground Truth):** **Survivor (In-hospital death = 0)**
- **Prediction Status:** `SUCCESS`
- **Data Quality Index (DQI):** **93.0/100** (`EXCELLENT` | Status: `PASS`)

| Predictive Model | Model Role | Architecture | Predicted Mortality Probability | Validation Threshold | Decision vs Threshold |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **XGBoost** | `PRIMARY` | 334 Tabular Features (Hist GBDT) | **0.0356** | 0.58 | **`BELOW_THRESHOLD`** |
| **LSTM** | `SECONDARY` | Sequential (48h x 74 channels + 9 static) | **0.0144** | 0.58 | **`BELOW_THRESHOLD`** |

### Patient Record #132540 (48-Hour Horizon)
- **Official Outcome (Ground Truth):** **Survivor (In-hospital death = 0)**
- **Prediction Status:** `SUCCESS`
- **Data Quality Index (DQI):** **90.0/100** (`EXCELLENT` | Status: `PASS`)

| Predictive Model | Model Role | Architecture | Predicted Mortality Probability | Validation Threshold | Decision vs Threshold |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **XGBoost** | `PRIMARY` | 334 Tabular Features (Hist GBDT) | **0.0209** | 0.58 | **`BELOW_THRESHOLD`** |
| **LSTM** | `SECONDARY` | Sequential (48h x 74 channels + 9 static) | **0.0063** | 0.58 | **`BELOW_THRESHOLD`** |

### Patient Record #132543 (48-Hour Horizon)
- **Official Outcome (Ground Truth):** **Survivor (In-hospital death = 0)**
- **Prediction Status:** `SUCCESS`
- **Data Quality Index (DQI):** **100.0/100** (`EXCELLENT` | Status: `PASS`)

| Predictive Model | Model Role | Architecture | Predicted Mortality Probability | Validation Threshold | Decision vs Threshold |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **XGBoost** | `PRIMARY` | 334 Tabular Features (Hist GBDT) | **0.0603** | 0.58 | **`BELOW_THRESHOLD`** |
| **LSTM** | `SECONDARY` | Sequential (48h x 74 channels + 9 static) | **0.0085** | 0.58 | **`BELOW_THRESHOLD`** |

### Patient Record #132551 (48-Hour Horizon)
- **Official Outcome (Ground Truth):** **Deceased (In-hospital death = 1)**
- **Prediction Status:** `SUCCESS`
- **Data Quality Index (DQI):** **90.0/100** (`EXCELLENT` | Status: `PASS`)

| Predictive Model | Model Role | Architecture | Predicted Mortality Probability | Validation Threshold | Decision vs Threshold |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **XGBoost** | `PRIMARY` | 334 Tabular Features (Hist GBDT) | **0.7325** | 0.58 | **`ABOVE_THRESHOLD`** |
| **LSTM** | `SECONDARY` | Sequential (48h x 74 channels + 9 static) | **0.7573** | 0.58 | **`ABOVE_THRESHOLD`** |

### Patient Record #132588 (48-Hour Horizon)
- **Official Outcome (Ground Truth):** **Deceased (In-hospital death = 1)**
- **Prediction Status:** `SUCCESS`
- **Data Quality Index (DQI):** **100.0/100** (`EXCELLENT` | Status: `PASS`)

| Predictive Model | Model Role | Architecture | Predicted Mortality Probability | Validation Threshold | Decision vs Threshold |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **XGBoost** | `PRIMARY` | 334 Tabular Features (Hist GBDT) | **0.7625** | 0.58 | **`ABOVE_THRESHOLD`** |
| **LSTM** | `SECONDARY` | Sequential (48h x 74 channels + 9 static) | **0.5622** | 0.58 | **`BELOW_THRESHOLD`** |

**Active Clinical & Data Warnings:**
- ⚠️ MODEL_DISCORDANCE: Primary XGBoost (ABOVE_THRESHOLD, p=0.7625) and Secondary LSTM (BELOW_THRESHOLD, p=0.5622) diverge across decision thresholds. Examine underlying physiological trajectories and tabular feature importance.



---

## 3. Early Triage Temporal Comparison: 24 Hours vs 48 Hours

Evaluating **Patient #132551** (Elderly Medical ICU patient with progressive multiorgan failure):

### Patient Record #132551 (24-Hour Horizon)
- **Official Outcome (Ground Truth):** **Deceased (In-hospital death = 1)**
- **Prediction Status:** `SUCCESS`
- **Data Quality Index (DQI):** **90.0/100** (`EXCELLENT` | Status: `PASS`)

| Predictive Model | Model Role | Architecture | Predicted Mortality Probability | Validation Threshold | Decision vs Threshold |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **XGBoost** | `PRIMARY` | 334 Tabular Features (Hist GBDT) | **0.7046** | 0.57 | **`ABOVE_THRESHOLD`** |
| **LSTM** | `SECONDARY` | Sequential (24h x 74 channels + 9 static) | **0.7433** | 0.66 | **`ABOVE_THRESHOLD`** |



### Temporal Trajectory Analysis (Patient #132551)
| Evaluation Window | Primary Model (XGBoost) | Secondary Model (LSTM) | Clinical Interpretation |
| :---: | :---: | :---: | :--- |
| **First 24 Hours** | **p = 0.7046** (`ABOVE_THRESHOLD`) | **p = 0.7433** (`ABOVE_THRESHOLD`) | Early septic shock detected; patient crosses 24h risk threshold (0.57). |
| **First 48 Hours** | **p = 0.7325** (`ABOVE_THRESHOLD`) | **p = 0.7573** (`ABOVE_THRESHOLD`) | Progressive hemodynamic decline and escalating lactate confirm persistent extreme risk. |

**Zero Future Leakage Guarantee:** The 24-hour evaluation strictly ignored all observations logged at $t > 1440$ minutes. Dynamic ventilation and hourly measurements past 24h were completely excluded from the 24h temporal grid and static vectors.

---

## 4. Quality Failure & Protection Against Fabricated Scores

Demonstration of automated blocking on a corrupted patient record (all descriptors missing, sparse vitals):

### Patient Record #999999 (48-Hour Horizon)
- **Official Outcome (Ground Truth):** **Withheld / Unknown**
- **Prediction Status:** `BLOCKED_BY_DATA_QUALITY`
- **Data Quality Index (DQI):** **3.0/100** (`UNRELIABLE_CRITICAL` | Status: `FAIL`)

> [!CAUTION]
> **AUTOMATED PREDICTION BLOCKED:** Input telemetry failed minimum clinical data quality standards (DQI < 50.0).
> No artificial risk probability was fabricated.
>
> **Audit Recommendations:**
> - Data quality audit FAILED. Severe missingness or sensor corruption prevents reliable AI inference.

**Active Clinical & Data Warnings:**
- ⚠️ Automated prediction blocked due to critical data quality failure.
- ⚠️ MISSING_AGE
- ⚠️ MISSING_GENDER
- ⚠️ MISSING_ICU_TYPE
- ⚠️ DQI score 3.0 is below the critical threshold (50.0).
- ⚠️ Missing descriptors: Age (sentinel -1), Gender (sentinel -1), Height (sentinel -1), Weight (sentinel -1), ICUType (sentinel -1)



---

## 5. Summary of Prediction Agent Safety Guarantees

1. **Non-Fabrication Guarantee:** When data quality is insufficient (DQI < 50.0), the Prediction Agent refuses to produce risk probabilities, returning an empty prediction dictionary with clear clinical remediation instructions.
2. **Model Independence:** Primary (XGBoost) and Secondary (LSTM) probabilities are displayed side-by-side without artificial blending or heuristic averaging.
3. **Validation Threshold Preservation:** Thresholds reflect empirical $F_1$-optimal cutoffs derived on the Validation cohort (Set A 20%), preventing subjective clinician bias while clearly disclaiming that they do not represent clinical treatment boundaries.
4. **Leakage Prevention:** 24-hour mode strictly masks and excludes all subsequent observations, preserving temporal validity for early triage use cases.

---
*Report automatically generated by `scripts/demo_prediction_agent.py`.*
