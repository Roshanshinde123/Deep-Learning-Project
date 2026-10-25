# Reporting & Research Priority Agent Demonstration (Phase 6.4)
**Project:** Agentic AI-Assisted ICU Mortality Risk Prediction from the First 48 Hours  
**Execution Environment:** Dual Model Architecture (XGBoost 48h Tabular + LSTM 48h Multimodal Sequential)  
**Agent Role:** Multi-Agent Orchestration, Model Agreement Synthesis, Deterministic Priority Assignment, and Report Generation

---

## Executive Summary

Phase 6.4 establishes the **Reporting / Priority Agent**, the orchestration and synthesis tier of the 4-agent clinical AI architecture. Rather than operating as a chatbot or conversational text wrapper, the Reporting Agent actively coordinates and ingests concrete outputs from:
1. **Data Quality Agent (Phase 6.1):** Rigorous telemetry audit, sensor artifact screening, observation density quantification, and Data Quality Index (DQI) evaluation.
2. **Prediction Agent (Phase 6.2):** Dual-model inference producing calibrated model-estimated mortality probabilities and validation-derived threshold classifications for both primary tabular (XGBoost) and multimodal temporal (LSTM) models.
3. **Explanation Agent (Phase 6.3):** Tree SHAP feature attribution, temporal landmark trajectory shifts, and physiological trend analysis.

### Core Capabilities Verified:
- **Non-Redundant Computation:** Directly accepts and reuses precomputed artifacts from upstream agents without redundant re-inference or SHAP re-computation.
- **Model Agreement Evaluation:** Transparently evaluates concordant low-risk, concordant high-risk, discordant, or data-blocked states without averaging probabilities or constructing synthetic consensus scores.
- **Deterministic Research Review Priority:** Transparently classifies records into `LOW`, `MODERATE`, `HIGH`, or `CRITICAL_DATA_QUALITY` using strictly documented, reproducible algorithmic rules.
- **Dual Deliverables:** Generates structured machine-readable JSON payloads conforming to standard clinical data schemas alongside human-readable ASCII/Markdown risk review reports.
- **Rigorous Safety Protocol:** Uses calibrated confidence language ("model-estimated mortality probability", "research review priority"), prohibits clinical treatment/medication recommendations, and enforces safety gates.

---

## Research-Only Priority Assignment Rules

The Reporting Agent evaluates risk and uncertainty using a 4-tier deterministic rule hierarchy:

| Priority Category | Trigger Conditions | Model Agreement Status | Primary Research Action |
| :--- | :--- | :--- | :--- |
| **`CRITICAL_DATA_QUALITY`** | Telemetry failed minimum data quality ($DQI < 50.0$) OR upstream processing blocked. | `PREDICTION_BLOCKED` | **Prediction Inhibited.** Technical audit required to address transducer disconnection, sensor artifacts, or massive missingness. |
| **`HIGH`** | Both models concordant high risk ($P_{xgb} \ge 0.58$ and $P_{lstm} \ge 0.58$), with adequate data quality ($DQI \ge 50.0$). | `MODEL_CONCORDANT_HIGH` | **Immediate Research Protocol Review.** Elevated risk supported by both tabular extremity and temporal dynamics. |
| **`MODERATE`** | Model discordance (one model $\ge 0.58$, one $< 0.58$) OR borderline probability ($0.48 \le P < 0.58$) OR suboptimal data quality ($50.0 \le DQI < 70.0$). | `MODEL_DISCORDANT` or `MODEL_CONCORDANT_*` | **Secondary Investigator Scrutiny.** In-depth inspection of discordance rationale (e.g. static acuity vs temporal stability) or data quality limitations. |
| **`LOW`** | Both models concordant low risk ($P_{xgb} < 0.58$ and $P_{lstm} < 0.58$), with acceptable/excellent data quality ($DQI \ge 70.0$). | `MODEL_CONCORDANT_LOW` | **Standard Surveillance Protocol.** Minimal acute deterioration indicators detected across 48-hour window. |

> [!IMPORTANT]
> **Research-Only Boundary:** These priority categories represent internal research workflow sorting tiers. They are **NOT** clinical triage categories (such as ESI, START, or NEWS2) and must never be used to direct patient admissions, medication administration, or withdrawal of life support.

---

## Detailed Demonstration Scenarios

### Case 1: Concordant Low-Risk Profile: Patient #132539 (48-Hour ICU Horizon)

**Clinical Context & Outcome:** 54-year-old female post-cardiac surgery. Clinically stable recovery, survived ICU admission (In-Hospital Death = 0).

- **Data Quality Index (DQI):** **93.0/100** (`EXCELLENT` | Status: `PASS`)
- **Model Agreement Status:** `MODEL_CONCORDANT_LOW`
- **Research Review Priority:** **`LOW`**
- **Priority Rationale:** *LOW: Both independent models estimate low mortality risk below validation cutoffs (XGBoost: 0.0356 < 0.58, LSTM: 0.0144 < 0.58) supported by reliable telemetry (DQI=93.0/100, MODEL_CONCORDANT_LOW).*

#### 1. Human-Readable Clinical Review Report
```text
===============================================================================
                   ICU MORTALITY RISK REVIEW SUMMARY
               Academic Research Prototype — Not for Clinical Use
===============================================================================
Patient ID:             #132539
Prediction Horizon:     48 Hours from ICU Admission
Data Source:            PhysioNet Challenge 2012 / MIMIC-II Dataset
Data Quality Index:     93.0/100 [EXCELLENT] (Status: PASS)
-------------------------------------------------------------------------------
1. MODEL PREDICTION SUMMARY
-------------------------------------------------------------------------------
Primary Model (XGBoost):  Probability = 0.0356  |  Threshold = 0.58  |  Decision = BELOW_THRESHOLD
Secondary Model (LSTM):   Probability = 0.0144  |  Threshold = 0.58  |  Decision = BELOW_THRESHOLD

Note: Model probabilities are independent and are NOT averaged or blended.
-------------------------------------------------------------------------------
2. MODEL AGREEMENT STATUS: MODEL_CONCORDANT_LOW
-------------------------------------------------------------------------------
Both independent models estimate low mortality risk below validation cutoffs (XGBoost: 0.0356 < 0.58, LSTM: 0.0144 < 0.58).
-------------------------------------------------------------------------------
3. TOP MODEL FEATURE CONTRIBUTIONS (XGBoost Tree SHAP)
-------------------------------------------------------------------------------
Risk-Increasing Contributions (Positive SHAP):
  1. GCS_trend (Patient Value: 0.0, SHAP: +0.0625)
  2. GCS_std (Patient Value: 0.0726, SHAP: +0.0365)
  3. Creatinine_std (Patient Value: 0.0535, SHAP: +0.0356)
Protective Associations (Negative SHAP):
  1. GCS_last (Patient Value: 0.8357, SHAP: -0.6524)
  2. Age_scaled (Patient Value: -0.5962, SHAP: -0.2306)
  3. BUN_last (Patient Value: -0.8448, SHAP: -0.1723)
-------------------------------------------------------------------------------
4. IMPORTANT TEMPORAL PATTERNS (LSTM Perturbation & Trajectories)
-------------------------------------------------------------------------------
Sequential Time Period Sensitivity:
  1. Q4 (Hours 36-48) (Impact Delta P: -0.0166, Sensitivity: RISK_DECREASING)
  2. Q3 (Hours 24-36) (Impact Delta P: -0.0040, Sensitivity: RISK_DECREASING)
Key Dynamic Trajectory Influences:
  1. Temp (Occlusion Delta P: -0.0106, Direction: RISK_DECREASING)
  2. GCS (Occlusion Delta P: -0.0069, Direction: RISK_DECREASING)
  3. BUN (Occlusion Delta P: -0.0061, Direction: RISK_DECREASING)
-------------------------------------------------------------------------------
5. DATA QUALITY AUDIT & SENSOR WARNINGS
-------------------------------------------------------------------------------
  1. HIGH_IMPUTATION_WARNING: Tabular feature 'Creatinine_std' (SHAP: +0.0356) was derived from 'Creatinine', which had 95.8% missingness (2 recorded observations). Interpret attribution with caution.
  2. HIGH_IMPUTATION_WARNING: Tabular feature 'BUN_last' (SHAP: -0.1723) was derived from 'BUN', which had 95.8% missingness (2 recorded observations). Interpret attribution with caution.
-------------------------------------------------------------------------------
6. RESEARCH REVIEW PRIORITY:  [ LOW ]
-------------------------------------------------------------------------------
RATIONALE: LOW: Both independent models estimate low mortality risk below validation cutoffs (XGBoost: 0.0356 < 0.58, LSTM: 0.0144 < 0.58) supported by reliable telemetry (DQI=93.0/100, MODEL_CONCORDANT_LOW).
-------------------------------------------------------------------------------
7. CLINICAL SAFETY LIMITATIONS & RESEARCH BOUNDARIES
-------------------------------------------------------------------------------
  1. Research prototype only; not intended for autonomous clinical diagnosis, triage, or treatment guidance.
  2. Decision thresholds (0.58 / 0.58) are validation-selected cutoffs optimizing F1-score, not clinical severity limits.
  3. Priority levels reflect internal research workflow prioritization, NOT clinical urgency scores.
  4. Predictions and attributions are derived from retrospective MIMIC-II ICU data with prevalent clinical missingness.

"These outputs are research-model estimates and must not be used to make clinical treatment decisions."
===============================================================================
```

#### 2. Machine-Readable Structured JSON Output
```json
{
  "patient_id": "132539",
  "horizon_hours": 48,
  "data_quality": {
    "dqi": 93.0,
    "category": "EXCELLENT",
    "status": "PASS",
    "confidence_label": "EXCELLENT_DATA_QUALITY",
    "missing_descriptors": [],
    "sensor_artifacts": 0,
    "observation_density": 0
  },
  "prediction": {
    "status": "SUCCESS",
    "xgboost": {
      "predicted_probability": 0.0356,
      "threshold": 0.58,
      "decision": "BELOW_THRESHOLD"
    },
    "lstm": {
      "predicted_probability": 0.0144,
      "threshold": 0.58,
      "decision": "BELOW_THRESHOLD"
    }
  },
  "agreement_status": "MODEL_CONCORDANT_LOW",
  "agreement_meaning": "Both independent models estimate low mortality risk below validation cutoffs (XGBoost: 0.0356 < 0.58, LSTM: 0.0144 < 0.58).",
  "explanation": {
    "top_xgboost_risk_contributors": [
      {
        "feature": "GCS_trend",
        "actual_value": 0.0,
        "shap_value": 0.0625,
        "abs_shap": 0.06252928078174591
      },
      {
        "feature": "GCS_std",
        "actual_value": 0.0726,
        "shap_value": 0.0365,
        "abs_shap": 0.03649460896849632
      },
      {
        "feature": "Creatinine_std",
        "actual_value": 0.0535,
        "shap_value": 0.0356,
        "abs_shap": 0.03558900207281113
      },
      {
        "feature": "Temp_min",
        "actual_value": -2.173,
        "shap_value": 0.0305,
        "abs_shap": 0.03054846078157425
      },
      {
        "feature": "NIDiasABP_mean",
        "actual_value": -0.5917,
        "shap_value": 0.0254,
        "abs_shap": 0.025421220809221268
      },
      {
        "feature": "NISysABP_first",
        "actual_value": 1.7551,
        "shap_value": 0.0142,
        "abs_shap": 0.014150455594062805
      },
      {
        "feature": "Glucose_mean",
        "actual_value": 0.4977,
        "shap_value": 0.014,
        "abs_shap": 0.014041977003216743
      },
      {
        "feature": "Urine_trend",
        "actual_value": -0.0246,
        "shap_value": 0.0133,
        "abs_shap": 0.013305830769240856
      },
      {
        "feature": "FiO2_trend",
        "actual_value": 0.0,
        "shap_value": 0.0129,
        "abs_shap": 0.012895787134766579
      },
      {
        "feature": "Mg_std",
        "actual_value": 0.7872,
        "shap_value": 0.0125,
        "abs_shap": 0.01245490275323391
      }
    ],
    "top_xgboost_protective_contributors": [
      {
        "feature": "GCS_last",
        "actual_value": 0.8357,
        "shap_value": -0.6524,
        "abs_shap": 0.6524184942245483
      },
      {
        "feature": "Age_scaled",
        "actual_value": -0.5962,
        "shap_value": -0.2306,
        "abs_shap": 0.23058819770812988
      },
      {
        "feature": "BUN_last",
        "actual_value": -0.8448,
        "shap_value": -0.1723,
        "abs_shap": 0.1723058670759201
      },
      {
        "feature": "RespRate_min",
        "actual_value": -2.6063,
        "shap_value": -0.171,
        "abs_shap": 0.17104750871658325
      },
      {
        "feature": "Urine_mean",
        "actual_value": 0.1987,
        "shap_value": -0.1591,
        "abs_shap": 0.15909768640995026
      },
      {
        "feature": "GCS_mean",
        "actual_value": 0.8156,
        "shap_value": -0.1541,
        "abs_shap": 0.15412771701812744
      },
      {
        "feature": "BUN_mean",
        "actual_value": -0.7196,
        "shap_value": -0.1444,
        "abs_shap": 0.14435966312885284
      },
      {
        "feature": "BUN_min",
        "actual_value": -0.8448,
        "shap_value": -0.1124,
        "abs_shap": 0.11241698265075684
      },
      {
        "feature": "Temp_last",
        "actual_value": 0.972,
        "shap_value": -0.0914,
        "abs_shap": 0.09143343567848206
      },
      {
        "feature": "HR_first",
        "actual_value": -0.6682,
        "shap_value": -0.0869,
        "abs_shap": 0.08686968684196472
      }
    ],
    "top_lstm_temporal_features": [
      {
        "feature": "Temp",
        "delta_probability": -0.0106,
        "perturbed_probability": 0.025,
        "impact_direction": "RISK_DECREASING",
        "abs_delta": 0.010573195046186448
      },
      {
        "feature": "GCS",
        "delta_probability": -0.0069,
        "perturbed_probability": 0.0213,
        "impact_direction": "RISK_DECREASING",
        "abs_delta": 0.006855116909742356
      },
      {
        "feature": "BUN",
        "delta_probability": -0.0061,
        "perturbed_probability": 0.0205,
        "impact_direction": "RISK_DECREASING",
        "abs_delta": 0.006079418337345124
      },
      {
        "feature": "Urine",
        "delta_probability": -0.0042,
        "perturbed_probability": 0.0186,
        "impact_direction": "RISK_DECREASING",
        "abs_delta": 0.004151278859376908
      },
      {
        "feature": "NIDiasABP",
        "delta_probability": 0.0039,
        "perturbed_probability": 0.0105,
        "impact_direction": "RISK_INCREASING",
        "abs_delta": 0.0038545515999197956
      },
      {
        "feature": "HCO3",
        "delta_probability": 0.0038,
        "perturbed_probability": 0.0106,
        "impact_direction": "RISK_INCREASING",
        "abs_delta": 0.0037673714250326153
      },
      {
        "feature": "NISysABP",
        "delta_probability": 0.0035,
        "perturbed_probability": 0.0109,
        "impact_direction": "RISK_INCREASING",
        "abs_delta": 0.003470530980825424
      },
      {
        "feature": "Na",
        "delta_probability": -0.0025,
        "perturbed_probability": 0.0169,
        "impact_direction": "RISK_DECREASING",
        "abs_delta": 0.0024670173734426502
      },
      {
        "feature": "HR",
        "delta_probability": -0.0022,
        "perturbed_probability": 0.0166,
        "impact_direction": "RISK_DECREASING",
        "abs_delta": 0.002223480245471001
      },
      {
        "feature": "Age_scaled",
        "delta_probability": -0.0016,
        "perturbed_probability": 0.016,
        "impact_direction": "RISK_DECREASING",
        "abs_delta": 0.001559158539772034
      }
    ],
    "important_time_periods": [
      {
        "time_period": "Q4 (Hours 36-48)",
        "delta_probability": -0.0166,
        "perturbed_probability": 0.031,
        "impact_direction": "RISK_DECREASING",
        "abs_delta": 0.016630265614390374
      },
      {
        "time_period": "Q3 (Hours 24-36)",
        "delta_probability": -0.004,
        "perturbed_probability": 0.0184,
        "impact_direction": "RISK_DECREASING",
        "abs_delta": 0.003971891230344773
      },
      {
        "time_period": "Q2 (Hours 12-24)",
        "delta_probability": 0.0003,
        "perturbed_probability": 0.0141,
        "impact_direction": "NEUTRAL",
        "abs_delta": 0.0002606196939945217
      },
      {
        "time_period": "Q1 (Hours 00-12)",
        "delta_probability": -0.0001,
        "perturbed_probability": 0.0145,
        "impact_direction": "NEUTRAL",
        "abs_delta": 0.0001150078460574154
      }
    ],
    "physiological_trends": [
      {
        "parameter": "HR",
        "first_value": 73.0,
        "first_time": "00:07",
        "last_value": 86.0,
        "last_time": "47:37",
        "delta": 13.0,
        "trend": "INCREASING",
        "observation_count": 37,
        "missingness_percentage": 25.0
      },
      {
        "parameter": "Temp",
        "first_value": 35.1,
        "first_time": "00:07",
        "last_value": 37.8,
        "last_time": "47:37",
        "delta": 2.7,
        "trend": "INCREASING",
        "observation_count": 14,
        "missingness_percentage": 72.9
      },
      {
        "parameter": "GCS",
        "first_value": 15.0,
        "first_time": "00:07",
        "last_value": 15.0,
        "last_time": "47:37",
        "delta": 0.0,
        "trend": "STABLE",
        "observation_count": 13,
        "missingness_percentage": 72.9
      },
      {
        "parameter": "BUN",
        "first_value": 13.0,
        "first_time": "10:37",
        "last_value": 8.0,
        "last_time": "33:07",
        "delta": -5.0,
        "trend": "DECREASING",
        "observation_count": 2,
        "missingness_percentage": 95.8
      },
      {
        "parameter": "Urine",
        "first_value": 900.0,
        "first_time": "00:07",
        "last_value": 280.0,
        "last_time": "47:37",
        "delta": -620.0,
        "trend": "DECREASING",
        "observation_count": 38,
        "missingness_percentage": 25.0
      },
      {
        "parameter": "RespRate",
        "first_value": 19.0,
        "first_time": "00:07",
        "last_value": 23.0,
        "last_time": "47:37",
        "delta": 4.0,
        "trend": "INCREASING",
        "observation_count": 42,
        "missingness_percentage": 14.6
      },
      {
        "parameter": "Na",
        "first_value": 137.0,
        "first_time": "10:37",
        "last_value": 136.0,
        "last_time": "33:07",
        "delta": -1.0,
        "trend": "STABLE",
        "observation_count": 2,
        "missingness_percentage": 95.8
      },
      {
        "parameter": "Creatinine",
        "first_value": 0.8,
        "first_time": "10:37",
        "last_value": 0.7,
        "last_time": "33:07",
        "delta": -0.1,
        "trend": "STABLE",
        "observation_count": 2,
        "missingness_percentage": 95.8
      },
      {
        "parameter": "Glucose",
        "first_value": 205.0,
        "first_time": "10:37",
        "last_value": 115.0,
        "last_time": "33:07",
        "delta": -90.0,
        "trend": "DECREASING",
        "observation_count": 2,
        "missingness_percentage": 95.8
      },
      {
        "parameter": "WBC",
        "first_value": 11.2,
        "first_time": "10:37",
        "last_value": 9.4,
        "last_time": "33:07",
        "delta": -1.8,
        "trend": "DECREASING",
        "observation_count": 2,
        "missingness_percentage": 95.8
      },
      {
        "parameter": "Platelets",
        "first_value": 221.0,
        "first_time": "10:37",
        "last_value": 185.0,
        "last_time": "33:07",
        "delta": -36.0,
        "trend": "DECREASING",
        "observation_count": 2,
        "missingness_percentage": 95.8
      },
      {
        "parameter": "K",
        "first_value": 4.4,
        "first_time": "10:37",
        "last_value": 4.0,
        "last_time": "33:07",
        "delta": -0.4,
        "trend": "STABLE",
        "observation_count": 2,
        "missingness_percentage": 95.8
      }
    ]
  },
  "review_priority": "LOW",
  "rationale": "LOW: Both independent models estimate low mortality risk below validation cutoffs (XGBoost: 0.0356 < 0.58, LSTM: 0.0144 < 0.58) supported by reliable telemetry (DQI=93.0/100, MODEL_CONCORDANT_LOW).",
  "limitations": [
    "Research prototype only; not intended for autonomous clinical diagnosis, triage, or treatment guidance.",
    "Decision thresholds (0.58 / 0.58) are validation-selected cutoffs optimizing F1-score, not clinical severity limits.",
    "Priority levels reflect internal research workflow prioritization, NOT clinical urgency scores.",
    "Predictions and attributions are derived from retrospective MIMIC-II ICU data with prevalent clinical missingness.",
    "Research-only risk predictions; not approved for autonomous clinical diagnosis or treatment guidance.",
    "Decision thresholds (XGBoost 48h: 0.58, LSTM 48h: 0.58) are validation-selected research cutoffs optimizing F1-score, not clinical risk categories.",
    "XGBoost serves as the primary tabular predictive model; LSTM provides an independent sequential trajectory comparison.",
    "Predictions from XGBoost and LSTM must NOT be averaged without a validated prospective ensemble.",
    "Evaluated retrospectively on the PhysioNet 2012 / MIMIC-II cohort; prospective generalizability has not been established.",
    "HIGH_IMPUTATION_WARNING: Tabular feature 'Creatinine_std' (SHAP: +0.0356) was derived from 'Creatinine', which had 95.8% missingness (2 recorded observations). Interpret attribution with caution.",
    "HIGH_IMPUTATION_WARNING: Tabular feature 'BUN_last' (SHAP: -0.1723) was derived from 'BUN', which had 95.8% missingness (2 recorded observations). Interpret attribution with caution.",
    "SHAP values and occlusion perturbations reflect model-specific statistical sensitivities, not clinical etiology.",
    "Feature attributions are derived from retrospective MIMIC-II observational data with prevalent clinical missingness."
  ],
  "research_only": true
}
```

### Case 2: Concordant High-Risk Profile: Patient #132551 (48-Hour ICU Horizon)

**Clinical Context & Outcome:** 77-year-old male with septic shock and escalating vasopressor requirement. Deceased during ICU hospitalization (In-Hospital Death = 1).

- **Data Quality Index (DQI):** **90.0/100** (`EXCELLENT` | Status: `PASS`)
- **Model Agreement Status:** `MODEL_CONCORDANT_HIGH`
- **Research Review Priority:** **`HIGH`**
- **Priority Rationale:** *HIGH: Both primary tabular XGBoost (p=0.7325 >= 0.58) and secondary sequential LSTM (p=0.7573 >= 0.58) independently cross validation-derived decision thresholds under adequate telemetry quality (DQI=90.0/100).*

#### 1. Human-Readable Clinical Review Report
```text
===============================================================================
                   ICU MORTALITY RISK REVIEW SUMMARY
               Academic Research Prototype — Not for Clinical Use
===============================================================================
Patient ID:             #132551
Prediction Horizon:     48 Hours from ICU Admission
Data Source:            PhysioNet Challenge 2012 / MIMIC-II Dataset
Data Quality Index:     90.0/100 [EXCELLENT] (Status: PASS)
-------------------------------------------------------------------------------
1. MODEL PREDICTION SUMMARY
-------------------------------------------------------------------------------
Primary Model (XGBoost):  Probability = 0.7325  |  Threshold = 0.58  |  Decision = ABOVE_THRESHOLD
Secondary Model (LSTM):   Probability = 0.7573  |  Threshold = 0.58  |  Decision = ABOVE_THRESHOLD

Note: Model probabilities are independent and are NOT averaged or blended.
-------------------------------------------------------------------------------
2. MODEL AGREEMENT STATUS: MODEL_CONCORDANT_HIGH
-------------------------------------------------------------------------------
Both independent models estimate elevated mortality risk above validation cutoffs (XGBoost: 0.7325 >= 0.58, LSTM: 0.7573 >= 0.58).
-------------------------------------------------------------------------------
3. TOP MODEL FEATURE CONTRIBUTIONS (XGBoost Tree SHAP)
-------------------------------------------------------------------------------
Risk-Increasing Contributions (Positive SHAP):
  1. Urine_mean (Patient Value: -0.4821, SHAP: +0.3196)
  2. PaCO2_mean (Patient Value: -1.2973, SHAP: +0.2560)
  3. GCS_last (Patient Value: -0.735, SHAP: +0.1884)
Protective Associations (Negative SHAP):
  1. Na_trend (Patient Value: -0.0313, SHAP: -0.1196)
  2. Mg_std (Patient Value: 3.6323, SHAP: -0.1140)
  3. Temp_mean (Patient Value: -0.0919, SHAP: -0.0935)
-------------------------------------------------------------------------------
4. IMPORTANT TEMPORAL PATTERNS (LSTM Perturbation & Trajectories)
-------------------------------------------------------------------------------
Sequential Time Period Sensitivity:
  1. Q4 (Hours 36-48) (Impact Delta P: +0.5118, Sensitivity: RISK_INCREASING)
  2. Q3 (Hours 24-36) (Impact Delta P: +0.0454, Sensitivity: RISK_INCREASING)
Key Dynamic Trajectory Influences:
  1. HCO3 (Occlusion Delta P: +0.1411, Direction: RISK_INCREASING)
  2. DiasABP (Occlusion Delta P: +0.0965, Direction: RISK_INCREASING)
  3. BUN (Occlusion Delta P: +0.0849, Direction: RISK_INCREASING)
-------------------------------------------------------------------------------
5. DATA QUALITY AUDIT & SENSOR WARNINGS
-------------------------------------------------------------------------------
  1. HIGH_IMPUTATION_WARNING: Tabular feature 'BUN_last' (SHAP: +0.1830) was derived from 'BUN', which had 89.6% missingness (5 recorded observations). Interpret attribution with caution.
  2. HIGH_IMPUTATION_WARNING: Tabular feature 'Na_trend' (SHAP: -0.1196) was derived from 'Na', which had 89.6% missingness (5 recorded observations). Interpret attribution with caution.
  3. HIGH_IMPUTATION_WARNING: Tabular feature 'Creatinine_std' (SHAP: -0.0715) was derived from 'Creatinine', which had 89.6% missingness (5 recorded observations). Interpret attribution with caution.
-------------------------------------------------------------------------------
6. RESEARCH REVIEW PRIORITY:  [ HIGH ]
-------------------------------------------------------------------------------
RATIONALE: HIGH: Both primary tabular XGBoost (p=0.7325 >= 0.58) and secondary sequential LSTM (p=0.7573 >= 0.58) independently cross validation-derived decision thresholds under adequate telemetry quality (DQI=90.0/100).
-------------------------------------------------------------------------------
7. CLINICAL SAFETY LIMITATIONS & RESEARCH BOUNDARIES
-------------------------------------------------------------------------------
  1. Research prototype only; not intended for autonomous clinical diagnosis, triage, or treatment guidance.
  2. Decision thresholds (0.58 / 0.58) are validation-selected cutoffs optimizing F1-score, not clinical severity limits.
  3. Priority levels reflect internal research workflow prioritization, NOT clinical urgency scores.
  4. Predictions and attributions are derived from retrospective MIMIC-II ICU data with prevalent clinical missingness.

"These outputs are research-model estimates and must not be used to make clinical treatment decisions."
===============================================================================
```

#### 2. Machine-Readable Structured JSON Output
```json
{
  "patient_id": "132551",
  "horizon_hours": 48,
  "data_quality": {
    "dqi": 90.0,
    "category": "EXCELLENT",
    "status": "PASS",
    "confidence_label": "EXCELLENT_DATA_QUALITY",
    "missing_descriptors": [],
    "sensor_artifacts": 0,
    "observation_density": 0
  },
  "prediction": {
    "status": "SUCCESS",
    "xgboost": {
      "predicted_probability": 0.7325,
      "threshold": 0.58,
      "decision": "ABOVE_THRESHOLD"
    },
    "lstm": {
      "predicted_probability": 0.7573,
      "threshold": 0.58,
      "decision": "ABOVE_THRESHOLD"
    }
  },
  "agreement_status": "MODEL_CONCORDANT_HIGH",
  "agreement_meaning": "Both independent models estimate elevated mortality risk above validation cutoffs (XGBoost: 0.7325 >= 0.58, LSTM: 0.7573 >= 0.58).",
  "explanation": {
    "top_xgboost_risk_contributors": [
      {
        "feature": "Urine_mean",
        "actual_value": -0.4821,
        "shap_value": 0.3196,
        "abs_shap": 0.3195672035217285
      },
      {
        "feature": "PaCO2_mean",
        "actual_value": -1.2973,
        "shap_value": 0.256,
        "abs_shap": 0.2559632956981659
      },
      {
        "feature": "GCS_last",
        "actual_value": -0.735,
        "shap_value": 0.1884,
        "abs_shap": 0.18844744563102722
      },
      {
        "feature": "BUN_last",
        "actual_value": 1.6599,
        "shap_value": 0.183,
        "abs_shap": 0.18298181891441345
      },
      {
        "feature": "GCS_trend",
        "actual_value": -0.0341,
        "shap_value": 0.1203,
        "abs_shap": 0.12029849737882614
      },
      {
        "feature": "HCO3_mean",
        "actual_value": -2.4172,
        "shap_value": 0.1161,
        "abs_shap": 0.11605053395032883
      },
      {
        "feature": "WBC_last",
        "actual_value": 1.6593,
        "shap_value": 0.1111,
        "abs_shap": 0.11108182370662689
      },
      {
        "feature": "Weight_admission_scaled",
        "actual_value": -1.4205,
        "shap_value": 0.0998,
        "abs_shap": 0.09982212632894516
      },
      {
        "feature": "Age_scaled",
        "actual_value": 0.7761,
        "shap_value": 0.077,
        "abs_shap": 0.07698480039834976
      },
      {
        "feature": "Temp_min",
        "actual_value": -1.852,
        "shap_value": 0.0737,
        "abs_shap": 0.07365933805704117
      }
    ],
    "top_xgboost_protective_contributors": [
      {
        "feature": "Na_trend",
        "actual_value": -0.0313,
        "shap_value": -0.1196,
        "abs_shap": 0.11956948041915894
      },
      {
        "feature": "Mg_std",
        "actual_value": 3.6323,
        "shap_value": -0.114,
        "abs_shap": 0.11403650045394897
      },
      {
        "feature": "Temp_mean",
        "actual_value": -0.0919,
        "shap_value": -0.0935,
        "abs_shap": 0.09350904822349548
      },
      {
        "feature": "GCS_mean",
        "actual_value": 0.0101,
        "shap_value": -0.0723,
        "abs_shap": 0.07230861485004425
      },
      {
        "feature": "Creatinine_std",
        "actual_value": 0.0986,
        "shap_value": -0.0715,
        "abs_shap": 0.07151226699352264
      },
      {
        "feature": "Creatinine_trend",
        "actual_value": -0.0073,
        "shap_value": -0.0705,
        "abs_shap": 0.0705171599984169
      },
      {
        "feature": "FiO2_trend",
        "actual_value": -0.1046,
        "shap_value": -0.064,
        "abs_shap": 0.06398740410804749
      },
      {
        "feature": "HR_last",
        "actual_value": -1.6478,
        "shap_value": -0.0631,
        "abs_shap": 0.0630895122885704
      },
      {
        "feature": "Lactate_min",
        "actual_value": -0.8041,
        "shap_value": -0.0589,
        "abs_shap": 0.05887206271290779
      },
      {
        "feature": "ALP_first",
        "actual_value": -0.673,
        "shap_value": -0.0576,
        "abs_shap": 0.05760882794857025
      }
    ],
    "top_lstm_temporal_features": [
      {
        "feature": "HCO3",
        "delta_probability": 0.1411,
        "perturbed_probability": 0.6162,
        "impact_direction": "RISK_INCREASING",
        "abs_delta": 0.14113007049560544
      },
      {
        "feature": "DiasABP",
        "delta_probability": 0.0965,
        "perturbed_probability": 0.6608,
        "impact_direction": "RISK_INCREASING",
        "abs_delta": 0.0964536474227905
      },
      {
        "feature": "BUN",
        "delta_probability": 0.0849,
        "perturbed_probability": 0.6724,
        "impact_direction": "RISK_INCREASING",
        "abs_delta": 0.08488849859237668
      },
      {
        "feature": "GCS",
        "delta_probability": 0.0546,
        "perturbed_probability": 0.7027,
        "impact_direction": "RISK_INCREASING",
        "abs_delta": 0.054627816486358616
      },
      {
        "feature": "PaCO2",
        "delta_probability": -0.0546,
        "perturbed_probability": 0.8119,
        "impact_direction": "RISK_DECREASING",
        "abs_delta": 0.054610092830657986
      },
      {
        "feature": "Mg",
        "delta_probability": -0.0392,
        "perturbed_probability": 0.7965,
        "impact_direction": "RISK_DECREASING",
        "abs_delta": 0.03924043912887576
      },
      {
        "feature": "Urine",
        "delta_probability": 0.0358,
        "perturbed_probability": 0.7215,
        "impact_direction": "RISK_INCREASING",
        "abs_delta": 0.03584537963867185
      },
      {
        "feature": "NIMAP",
        "delta_probability": -0.0348,
        "perturbed_probability": 0.7921,
        "impact_direction": "RISK_DECREASING",
        "abs_delta": 0.0348218276023865
      },
      {
        "feature": "HR",
        "delta_probability": -0.0348,
        "perturbed_probability": 0.7921,
        "impact_direction": "RISK_DECREASING",
        "abs_delta": 0.034821529579162624
      },
      {
        "feature": "Age_scaled",
        "delta_probability": 0.0296,
        "perturbed_probability": 0.7277,
        "impact_direction": "RISK_INCREASING",
        "abs_delta": 0.029602925586700413
      }
    ],
    "important_time_periods": [
      {
        "time_period": "Q4 (Hours 36-48)",
        "delta_probability": 0.5118,
        "perturbed_probability": 0.2455,
        "impact_direction": "RISK_INCREASING",
        "abs_delta": 0.5118376867055893
      },
      {
        "time_period": "Q3 (Hours 24-36)",
        "delta_probability": 0.0454,
        "perturbed_probability": 0.7119,
        "impact_direction": "RISK_INCREASING",
        "abs_delta": 0.04542867403030393
      },
      {
        "time_period": "Q2 (Hours 12-24)",
        "delta_probability": -0.0033,
        "perturbed_probability": 0.7606,
        "impact_direction": "RISK_DECREASING",
        "abs_delta": 0.003336150836944607
      },
      {
        "time_period": "Q1 (Hours 00-12)",
        "delta_probability": -0.001,
        "perturbed_probability": 0.7583,
        "impact_direction": "RISK_DECREASING",
        "abs_delta": 0.0009651972770691186
      }
    ],
    "physiological_trends": [
      {
        "parameter": "HR",
        "first_value": 111.0,
        "first_time": "00:00",
        "last_value": 58.0,
        "last_time": "47:40",
        "delta": -53.0,
        "trend": "DECREASING",
        "observation_count": 73,
        "missingness_percentage": 10.4
      },
      {
        "parameter": "Temp",
        "first_value": 38.0,
        "first_time": "00:00",
        "last_value": 36.6,
        "last_time": "47:40",
        "delta": -1.4,
        "trend": "STABLE",
        "observation_count": 36,
        "missingness_percentage": 52.1
      },
      {
        "parameter": "GCS",
        "first_value": 15.0,
        "first_time": "00:10",
        "last_value": 9.0,
        "last_time": "46:40",
        "delta": -6.0,
        "trend": "DECREASING",
        "observation_count": 13,
        "missingness_percentage": 72.9
      },
      {
        "parameter": "BUN",
        "first_value": 81.0,
        "first_time": "00:50",
        "last_value": 58.0,
        "last_time": "31:40",
        "delta": -23.0,
        "trend": "DECREASING",
        "observation_count": 5,
        "missingness_percentage": 89.6
      },
      {
        "parameter": "Urine",
        "first_value": 120.0,
        "first_time": "01:40",
        "last_value": 40.0,
        "last_time": "46:40",
        "delta": -80.0,
        "trend": "DECREASING",
        "observation_count": 37,
        "missingness_percentage": 22.9
      },
      {
        "parameter": "PaCO2",
        "first_value": 30.0,
        "first_time": "00:38",
        "last_value": 24.0,
        "last_time": "47:16",
        "delta": -6.0,
        "trend": "DECREASING",
        "observation_count": 15,
        "missingness_percentage": 68.8
      },
      {
        "parameter": "Na",
        "first_value": 141.0,
        "first_time": "00:50",
        "last_value": 137.0,
        "last_time": "31:40",
        "delta": -4.0,
        "trend": "STABLE",
        "observation_count": 5,
        "missingness_percentage": 89.6
      },
      {
        "parameter": "pH",
        "first_value": 7.4,
        "first_time": "00:38",
        "last_value": 7.37,
        "last_time": "47:16",
        "delta": -0.03,
        "trend": "STABLE",
        "observation_count": 15,
        "missingness_percentage": 68.8
      },
      {
        "parameter": "Lactate",
        "first_value": 1.4,
        "first_time": "00:38",
        "last_value": 1.8,
        "last_time": "32:09",
        "delta": 0.4,
        "trend": "STABLE",
        "observation_count": 8,
        "missingness_percentage": 83.3
      },
      {
        "parameter": "MAP",
        "first_value": 83.0,
        "first_time": "00:00",
        "last_value": 61.0,
        "last_time": "47:40",
        "delta": -22.0,
        "trend": "DECREASING",
        "observation_count": 60,
        "missingness_percentage": 20.8
      },
      {
        "parameter": "Creatinine",
        "first_value": 0.9,
        "first_time": "00:50",
        "last_value": 0.6,
        "last_time": "31:40",
        "delta": -0.3,
        "trend": "STABLE",
        "observation_count": 5,
        "missingness_percentage": 89.6
      },
      {
        "parameter": "Glucose",
        "first_value": 132.0,
        "first_time": "00:50",
        "last_value": 116.0,
        "last_time": "31:40",
        "delta": -16.0,
        "trend": "DECREASING",
        "observation_count": 5,
        "missingness_percentage": 89.6
      },
      {
        "parameter": "WBC",
        "first_value": 16.1,
        "first_time": "00:50",
        "last_value": 23.5,
        "last_time": "31:40",
        "delta": 7.4,
        "trend": "INCREASING",
        "observation_count": 3,
        "missingness_percentage": 93.8
      },
      {
        "parameter": "Platelets",
        "first_value": 91.0,
        "first_time": "00:50",
        "last_value": 102.0,
        "last_time": "31:40",
        "delta": 11.0,
        "trend": "INCREASING",
        "observation_count": 3,
        "missingness_percentage": 93.8
      },
      {
        "parameter": "SysABP",
        "first_value": 103.0,
        "first_time": "00:00",
        "last_value": 126.0,
        "last_time": "47:40",
        "delta": 23.0,
        "trend": "INCREASING",
        "observation_count": 60,
        "missingness_percentage": 20.8
      },
      {
        "parameter": "DiasABP",
        "first_value": 50.0,
        "first_time": "00:00",
        "last_value": 35.0,
        "last_time": "47:40",
        "delta": -15.0,
        "trend": "DECREASING",
        "observation_count": 60,
        "missingness_percentage": 20.8
      },
      {
        "parameter": "SaO2",
        "first_value": 97.0,
        "first_time": "06:51",
        "last_value": 96.0,
        "last_time": "47:16",
        "delta": -1.0,
        "trend": "STABLE",
        "observation_count": 5,
        "missingness_percentage": 89.6
      },
      {
        "parameter": "FiO2",
        "first_value": 1.0,
        "first_time": "18:10",
        "last_value": 0.4,
        "last_time": "46:40",
        "delta": -0.6,
        "trend": "DECREASING",
        "observation_count": 11,
        "missingness_percentage": 79.2
      },
      {
        "parameter": "K",
        "first_value": 4.2,
        "first_time": "00:50",
        "last_value": 4.5,
        "last_time": "31:40",
        "delta": 0.3,
        "trend": "STABLE",
        "observation_count": 5,
        "missingness_percentage": 89.6
      }
    ]
  },
  "review_priority": "HIGH",
  "rationale": "HIGH: Both primary tabular XGBoost (p=0.7325 >= 0.58) and secondary sequential LSTM (p=0.7573 >= 0.58) independently cross validation-derived decision thresholds under adequate telemetry quality (DQI=90.0/100).",
  "limitations": [
    "Research prototype only; not intended for autonomous clinical diagnosis, triage, or treatment guidance.",
    "Decision thresholds (0.58 / 0.58) are validation-selected cutoffs optimizing F1-score, not clinical severity limits.",
    "Priority levels reflect internal research workflow prioritization, NOT clinical urgency scores.",
    "Predictions and attributions are derived from retrospective MIMIC-II ICU data with prevalent clinical missingness.",
    "Research-only risk predictions; not approved for autonomous clinical diagnosis or treatment guidance.",
    "Decision thresholds (XGBoost 48h: 0.58, LSTM 48h: 0.58) are validation-selected research cutoffs optimizing F1-score, not clinical risk categories.",
    "XGBoost serves as the primary tabular predictive model; LSTM provides an independent sequential trajectory comparison.",
    "Predictions from XGBoost and LSTM must NOT be averaged without a validated prospective ensemble.",
    "Evaluated retrospectively on the PhysioNet 2012 / MIMIC-II cohort; prospective generalizability has not been established.",
    "HIGH_IMPUTATION_WARNING: Tabular feature 'BUN_last' (SHAP: +0.1830) was derived from 'BUN', which had 89.6% missingness (5 recorded observations). Interpret attribution with caution.",
    "HIGH_IMPUTATION_WARNING: Tabular feature 'Na_trend' (SHAP: -0.1196) was derived from 'Na', which had 89.6% missingness (5 recorded observations). Interpret attribution with caution.",
    "HIGH_IMPUTATION_WARNING: Tabular feature 'Creatinine_std' (SHAP: -0.0715) was derived from 'Creatinine', which had 89.6% missingness (5 recorded observations). Interpret attribution with caution.",
    "SHAP values and occlusion perturbations reflect model-specific statistical sensitivities, not clinical etiology.",
    "Feature attributions are derived from retrospective MIMIC-II observational data with prevalent clinical missingness."
  ],
  "research_only": true
}
```

### Case 3: Model-Discordant Profile: Patient #132588 (48-Hour ICU Horizon)

**Clinical Context & Outcome:** 85-year-old female with multiorgan dysfunction. Deceased (In-Hospital Death = 1). Tabular feature extremity triggers high XGBoost risk, whereas temporal LSTM reflects delayed deterioration.

- **Data Quality Index (DQI):** **100.0/100** (`EXCELLENT` | Status: `PASS`)
- **Model Agreement Status:** `MODEL_DISCORDANT`
- **Research Review Priority:** **`MODERATE`**
- **Priority Rationale:** *MODERATE: Assigned MODERATE review priority due to model discordance: XGBoost (p=0.7625) and LSTM (p=0.5622) diverge across decision thresholds. Tabular aggregations and sequential trajectories warrant individual inspection.*

#### 1. Human-Readable Clinical Review Report
```text
===============================================================================
                   ICU MORTALITY RISK REVIEW SUMMARY
               Academic Research Prototype — Not for Clinical Use
===============================================================================
Patient ID:             #132588
Prediction Horizon:     48 Hours from ICU Admission
Data Source:            PhysioNet Challenge 2012 / MIMIC-II Dataset
Data Quality Index:     100.0/100 [EXCELLENT] (Status: PASS)
-------------------------------------------------------------------------------
1. MODEL PREDICTION SUMMARY
-------------------------------------------------------------------------------
Primary Model (XGBoost):  Probability = 0.7625  |  Threshold = 0.58  |  Decision = ABOVE_THRESHOLD
Secondary Model (LSTM):   Probability = 0.5622  |  Threshold = 0.58  |  Decision = BELOW_THRESHOLD

Note: Model probabilities are independent and are NOT averaged or blended.
-------------------------------------------------------------------------------
2. MODEL AGREEMENT STATUS: MODEL_DISCORDANT
-------------------------------------------------------------------------------
Models diverge across decision thresholds: XGBoost=ABOVE_THRESHOLD (p=0.7625, th=0.58), LSTM=BELOW_THRESHOLD (p=0.5622, th=0.58). Multidisciplinary clinical trajectory review recommended.
-------------------------------------------------------------------------------
3. TOP MODEL FEATURE CONTRIBUTIONS (XGBoost Tree SHAP)
-------------------------------------------------------------------------------
Risk-Increasing Contributions (Positive SHAP):
  1. Bilirubin_min (Patient Value: 2.0428, SHAP: +0.2907)
  2. NISysABP_last (Patient Value: -1.5849, SHAP: +0.2760)
  3. Albumin_mean (Patient Value: -2.5646, SHAP: +0.1611)
Protective Associations (Negative SHAP):
  1. GCS_last (Patient Value: 0.8357, SHAP: -0.3538)
  2. Age_scaled (Patient Value: -0.9393, SHAP: -0.1332)
  3. BUN_last (Patient Value: -0.9951, SHAP: -0.1241)
-------------------------------------------------------------------------------
4. IMPORTANT TEMPORAL PATTERNS (LSTM Perturbation & Trajectories)
-------------------------------------------------------------------------------
Sequential Time Period Sensitivity:
  1. Q4 (Hours 36-48) (Impact Delta P: +0.3170, Sensitivity: RISK_INCREASING)
  2. Q3 (Hours 24-36) (Impact Delta P: +0.0743, Sensitivity: RISK_INCREASING)
Key Dynamic Trajectory Influences:
  1. Bilirubin (Occlusion Delta P: +0.3749, Direction: RISK_INCREASING)
  2. Albumin (Occlusion Delta P: +0.2584, Direction: RISK_INCREASING)
  3. GCS (Occlusion Delta P: -0.1440, Direction: RISK_DECREASING)
-------------------------------------------------------------------------------
5. DATA QUALITY AUDIT & SENSOR WARNINGS
-------------------------------------------------------------------------------
  1. MODEL_DISCORDANCE: Primary XGBoost (ABOVE_THRESHOLD, p=0.7625) and Secondary LSTM (BELOW_THRESHOLD, p=0.5622) diverge across decision thresholds. Examine underlying physiological trajectories and tabular feature importance.
  2. HIGH_IMPUTATION_WARNING: Tabular feature 'Glucose_trend' (SHAP: +0.1325) was derived from 'Glucose', which had 95.8% missingness (2 recorded observations). Interpret attribution with caution.
  3. HIGH_IMPUTATION_WARNING: Tabular feature 'GCS_last' (SHAP: -0.3538) was derived from 'GCS', which had 83.3% missingness (8 recorded observations). Interpret attribution with caution.
  4. HIGH_IMPUTATION_WARNING: Tabular feature 'BUN_last' (SHAP: -0.1241) was derived from 'BUN', which had 95.8% missingness (2 recorded observations). Interpret attribution with caution.
  5. HIGH_IMPUTATION_WARNING: Tabular feature 'GCS_mean' (SHAP: -0.1007) was derived from 'GCS', which had 83.3% missingness (8 recorded observations). Interpret attribution with caution.
  6. HIGH_IMPUTATION_WARNING: Tabular feature 'BUN_min' (SHAP: -0.0976) was derived from 'BUN', which had 95.8% missingness (2 recorded observations). Interpret attribution with caution.
-------------------------------------------------------------------------------
6. RESEARCH REVIEW PRIORITY:  [ MODERATE ]
-------------------------------------------------------------------------------
RATIONALE: MODERATE: Assigned MODERATE review priority due to model discordance: XGBoost (p=0.7625) and LSTM (p=0.5622) diverge across decision thresholds. Tabular aggregations and sequential trajectories warrant individual inspection.
-------------------------------------------------------------------------------
7. CLINICAL SAFETY LIMITATIONS & RESEARCH BOUNDARIES
-------------------------------------------------------------------------------
  1. Research prototype only; not intended for autonomous clinical diagnosis, triage, or treatment guidance.
  2. Decision thresholds (0.58 / 0.58) are validation-selected cutoffs optimizing F1-score, not clinical severity limits.
  3. Priority levels reflect internal research workflow prioritization, NOT clinical urgency scores.
  4. Predictions and attributions are derived from retrospective MIMIC-II ICU data with prevalent clinical missingness.

"These outputs are research-model estimates and must not be used to make clinical treatment decisions."
===============================================================================
```

#### 2. Machine-Readable Structured JSON Output
```json
{
  "patient_id": "132588",
  "horizon_hours": 48,
  "data_quality": {
    "dqi": 100.0,
    "category": "EXCELLENT",
    "status": "PASS",
    "confidence_label": "EXCELLENT_DATA_QUALITY",
    "missing_descriptors": [],
    "sensor_artifacts": 0,
    "observation_density": 0
  },
  "prediction": {
    "status": "SUCCESS",
    "xgboost": {
      "predicted_probability": 0.7625,
      "threshold": 0.58,
      "decision": "ABOVE_THRESHOLD"
    },
    "lstm": {
      "predicted_probability": 0.5622,
      "threshold": 0.58,
      "decision": "BELOW_THRESHOLD"
    }
  },
  "agreement_status": "MODEL_DISCORDANT",
  "agreement_meaning": "Models diverge across decision thresholds: XGBoost=ABOVE_THRESHOLD (p=0.7625, th=0.58), LSTM=BELOW_THRESHOLD (p=0.5622, th=0.58). Multidisciplinary clinical trajectory review recommended.",
  "explanation": {
    "top_xgboost_risk_contributors": [
      {
        "feature": "Bilirubin_min",
        "actual_value": 2.0428,
        "shap_value": 0.2907,
        "abs_shap": 0.2906950116157532
      },
      {
        "feature": "NISysABP_last",
        "actual_value": -1.5849,
        "shap_value": 0.276,
        "abs_shap": 0.2759722173213959
      },
      {
        "feature": "Albumin_mean",
        "actual_value": -2.5646,
        "shap_value": 0.1611,
        "abs_shap": 0.16105781495571136
      },
      {
        "feature": "Bilirubin_last",
        "actual_value": 2.4844,
        "shap_value": 0.1359,
        "abs_shap": 0.13589343428611755
      },
      {
        "feature": "Glucose_trend",
        "actual_value": 0.0351,
        "shap_value": 0.1325,
        "abs_shap": 0.13254891335964203
      },
      {
        "feature": "HR_first",
        "actual_value": 1.608,
        "shap_value": 0.1017,
        "abs_shap": 0.1016717180609703
      },
      {
        "feature": "ALP_mean",
        "actual_value": 1.4303,
        "shap_value": 0.099,
        "abs_shap": 0.0990162268280983
      },
      {
        "feature": "Weight_admission_scaled",
        "actual_value": -1.6818,
        "shap_value": 0.0953,
        "abs_shap": 0.09525923430919647
      },
      {
        "feature": "NISysABP_min",
        "actual_value": -1.8932,
        "shap_value": 0.09,
        "abs_shap": 0.09002848714590073
      },
      {
        "feature": "Bilirubin_mean",
        "actual_value": 2.374,
        "shap_value": 0.0868,
        "abs_shap": 0.0868237242102623
      }
    ],
    "top_xgboost_protective_contributors": [
      {
        "feature": "GCS_last",
        "actual_value": 0.8357,
        "shap_value": -0.3538,
        "abs_shap": 0.35379108786582947
      },
      {
        "feature": "Age_scaled",
        "actual_value": -0.9393,
        "shap_value": -0.1332,
        "abs_shap": 0.1331857591867447
      },
      {
        "feature": "BUN_last",
        "actual_value": -0.9951,
        "shap_value": -0.1241,
        "abs_shap": 0.12406463176012039
      },
      {
        "feature": "GCS_mean",
        "actual_value": 0.8357,
        "shap_value": -0.1007,
        "abs_shap": 0.10070142894983292
      },
      {
        "feature": "BUN_min",
        "actual_value": -0.9951,
        "shap_value": -0.0976,
        "abs_shap": 0.09760497510433197
      },
      {
        "feature": "Temp_last",
        "actual_value": -0.0549,
        "shap_value": -0.0748,
        "abs_shap": 0.07476607710123062
      },
      {
        "feature": "Na_trend",
        "actual_value": -0.0628,
        "shap_value": -0.0742,
        "abs_shap": 0.07420244812965393
      },
      {
        "feature": "PaCO2_mean",
        "actual_value": -0.1049,
        "shap_value": -0.0589,
        "abs_shap": 0.05891106650233269
      },
      {
        "feature": "BUN_mean",
        "actual_value": -0.945,
        "shap_value": -0.0541,
        "abs_shap": 0.054064057767391205
      },
      {
        "feature": "RespRate_min",
        "actual_value": -3.6989,
        "shap_value": -0.0517,
        "abs_shap": 0.051704950630664825
      }
    ],
    "top_lstm_temporal_features": [
      {
        "feature": "Bilirubin",
        "delta_probability": 0.3749,
        "perturbed_probability": 0.1873,
        "impact_direction": "RISK_INCREASING",
        "abs_delta": 0.3748630932092667
      },
      {
        "feature": "Albumin",
        "delta_probability": 0.2584,
        "perturbed_probability": 0.3038,
        "impact_direction": "RISK_INCREASING",
        "abs_delta": 0.25843237848281864
      },
      {
        "feature": "GCS",
        "delta_probability": -0.144,
        "perturbed_probability": 0.7062,
        "impact_direction": "RISK_DECREASING",
        "abs_delta": 0.14402032880783078
      },
      {
        "feature": "NISysABP",
        "delta_probability": 0.1344,
        "perturbed_probability": 0.4278,
        "impact_direction": "RISK_INCREASING",
        "abs_delta": 0.134390135717392
      },
      {
        "feature": "NIMAP",
        "delta_probability": 0.1251,
        "perturbed_probability": 0.4371,
        "impact_direction": "RISK_INCREASING",
        "abs_delta": 0.12507491321563724
      },
      {
        "feature": "WBC",
        "delta_probability": 0.1225,
        "perturbed_probability": 0.4397,
        "impact_direction": "RISK_INCREASING",
        "abs_delta": 0.12250797958374027
      },
      {
        "feature": "Weight",
        "delta_probability": 0.12,
        "perturbed_probability": 0.4422,
        "impact_direction": "RISK_INCREASING",
        "abs_delta": 0.12004261226654056
      },
      {
        "feature": "ALP",
        "delta_probability": -0.0945,
        "perturbed_probability": 0.6567,
        "impact_direction": "RISK_DECREASING",
        "abs_delta": 0.09448022632598874
      },
      {
        "feature": "NIDiasABP",
        "delta_probability": 0.088,
        "perturbed_probability": 0.4742,
        "impact_direction": "RISK_INCREASING",
        "abs_delta": 0.08801331133842472
      },
      {
        "feature": "BUN",
        "delta_probability": -0.0713,
        "perturbed_probability": 0.6335,
        "impact_direction": "RISK_DECREASING",
        "abs_delta": 0.07133312015533444
      }
    ],
    "important_time_periods": [
      {
        "time_period": "Q4 (Hours 36-48)",
        "delta_probability": 0.317,
        "perturbed_probability": 0.2452,
        "impact_direction": "RISK_INCREASING",
        "abs_delta": 0.3170153549432755
      },
      {
        "time_period": "Q3 (Hours 24-36)",
        "delta_probability": 0.0743,
        "perturbed_probability": 0.4879,
        "impact_direction": "RISK_INCREASING",
        "abs_delta": 0.07425682754516605
      },
      {
        "time_period": "Q2 (Hours 12-24)",
        "delta_probability": -0.0069,
        "perturbed_probability": 0.5691,
        "impact_direction": "RISK_DECREASING",
        "abs_delta": 0.006868133831024137
      },
      {
        "time_period": "Q1 (Hours 00-12)",
        "delta_probability": -0.0009,
        "perturbed_probability": 0.5631,
        "impact_direction": "RISK_DECREASING",
        "abs_delta": 0.0009089210510253576
      }
    ],
    "physiological_trends": [
      {
        "parameter": "HR",
        "first_value": 115.0,
        "first_time": "01:09",
        "last_value": 78.0,
        "last_time": "47:39",
        "delta": -37.0,
        "trend": "DECREASING",
        "observation_count": 40,
        "missingness_percentage": 22.9
      },
      {
        "parameter": "Temp",
        "first_value": 38.4,
        "first_time": "01:09",
        "last_value": 37.0,
        "last_time": "41:39",
        "delta": -1.4,
        "trend": "STABLE",
        "observation_count": 8,
        "missingness_percentage": 83.3
      },
      {
        "parameter": "GCS",
        "first_value": 15.0,
        "first_time": "01:09",
        "last_value": 15.0,
        "last_time": "39:39",
        "delta": 0.0,
        "trend": "STABLE",
        "observation_count": 8,
        "missingness_percentage": 83.3
      },
      {
        "parameter": "BUN",
        "first_value": 7.0,
        "first_time": "04:22",
        "last_value": 5.0,
        "last_time": "31:12",
        "delta": -2.0,
        "trend": "DECREASING",
        "observation_count": 2,
        "missingness_percentage": 95.8
      },
      {
        "parameter": "RespRate",
        "first_value": 17.0,
        "first_time": "01:39",
        "last_value": 9.0,
        "last_time": "47:39",
        "delta": -8.0,
        "trend": "DECREASING",
        "observation_count": 38,
        "missingness_percentage": 25.0
      },
      {
        "parameter": "Na",
        "first_value": 140.0,
        "first_time": "04:22",
        "last_value": 133.0,
        "last_time": "31:12",
        "delta": -7.0,
        "trend": "STABLE",
        "observation_count": 2,
        "missingness_percentage": 95.8
      },
      {
        "parameter": "Creatinine",
        "first_value": 3.5,
        "first_time": "04:22",
        "last_value": 2.2,
        "last_time": "31:12",
        "delta": -1.3,
        "trend": "DECREASING",
        "observation_count": 2,
        "missingness_percentage": 95.8
      },
      {
        "parameter": "Glucose",
        "first_value": 63.0,
        "first_time": "04:22",
        "last_value": 110.0,
        "last_time": "31:12",
        "delta": 47.0,
        "trend": "INCREASING",
        "observation_count": 2,
        "missingness_percentage": 95.8
      },
      {
        "parameter": "WBC",
        "first_value": 7.0,
        "first_time": "04:22",
        "last_value": 6.5,
        "last_time": "47:17",
        "delta": -0.5,
        "trend": "STABLE",
        "observation_count": 4,
        "missingness_percentage": 91.7
      },
      {
        "parameter": "Platelets",
        "first_value": 152.0,
        "first_time": "04:22",
        "last_value": 130.0,
        "last_time": "47:17",
        "delta": -22.0,
        "trend": "DECREASING",
        "observation_count": 4,
        "missingness_percentage": 91.7
      },
      {
        "parameter": "K",
        "first_value": 3.5,
        "first_time": "04:22",
        "last_value": 3.8,
        "last_time": "31:12",
        "delta": 0.3,
        "trend": "STABLE",
        "observation_count": 2,
        "missingness_percentage": 95.8
      }
    ]
  },
  "review_priority": "MODERATE",
  "rationale": "MODERATE: Assigned MODERATE review priority due to model discordance: XGBoost (p=0.7625) and LSTM (p=0.5622) diverge across decision thresholds. Tabular aggregations and sequential trajectories warrant individual inspection.",
  "limitations": [
    "Research prototype only; not intended for autonomous clinical diagnosis, triage, or treatment guidance.",
    "Decision thresholds (0.58 / 0.58) are validation-selected cutoffs optimizing F1-score, not clinical severity limits.",
    "Priority levels reflect internal research workflow prioritization, NOT clinical urgency scores.",
    "Predictions and attributions are derived from retrospective MIMIC-II ICU data with prevalent clinical missingness.",
    "Research-only risk predictions; not approved for autonomous clinical diagnosis or treatment guidance.",
    "Decision thresholds (XGBoost 48h: 0.58, LSTM 48h: 0.58) are validation-selected research cutoffs optimizing F1-score, not clinical risk categories.",
    "XGBoost serves as the primary tabular predictive model; LSTM provides an independent sequential trajectory comparison.",
    "Predictions from XGBoost and LSTM must NOT be averaged without a validated prospective ensemble.",
    "Evaluated retrospectively on the PhysioNet 2012 / MIMIC-II cohort; prospective generalizability has not been established.",
    "HIGH_IMPUTATION_WARNING: Tabular feature 'Glucose_trend' (SHAP: +0.1325) was derived from 'Glucose', which had 95.8% missingness (2 recorded observations). Interpret attribution with caution.",
    "HIGH_IMPUTATION_WARNING: Tabular feature 'GCS_last' (SHAP: -0.3538) was derived from 'GCS', which had 83.3% missingness (8 recorded observations). Interpret attribution with caution.",
    "HIGH_IMPUTATION_WARNING: Tabular feature 'BUN_last' (SHAP: -0.1241) was derived from 'BUN', which had 95.8% missingness (2 recorded observations). Interpret attribution with caution.",
    "HIGH_IMPUTATION_WARNING: Tabular feature 'GCS_mean' (SHAP: -0.1007) was derived from 'GCS', which had 83.3% missingness (8 recorded observations). Interpret attribution with caution.",
    "HIGH_IMPUTATION_WARNING: Tabular feature 'BUN_min' (SHAP: -0.0976) was derived from 'BUN', which had 95.8% missingness (2 recorded observations). Interpret attribution with caution.",
    "SHAP values and occlusion perturbations reflect model-specific statistical sensitivities, not clinical etiology.",
    "Feature attributions are derived from retrospective MIMIC-II observational data with prevalent clinical missingness."
  ],
  "research_only": true
}
```

### Case 4: Critical Data Quality Failure (Gate Block): Patient #999999 (48-Hour ICU Horizon)

**Clinical Context & Outcome:** Synthetic telemetry corruption simulating severe transducer disconnection, extreme physiologically impossible artifacts, and massive observation dropout.

- **Data Quality Index (DQI):** **6.0/100** (`UNRELIABLE_CRITICAL` | Status: `FAIL`)
- **Model Agreement Status:** `PREDICTION_BLOCKED`
- **Research Review Priority:** **`CRITICAL_DATA_QUALITY`**
- **Priority Rationale:** *CRITICAL_DATA_QUALITY: Automated risk scoring is blocked because the Data Quality Index (DQI=6.0/100) is below the critical threshold (50.0). Telemetry requires audit and potential re-acquisition before research interpretation.*

#### 1. Human-Readable Clinical Review Report
```text
===============================================================================
                   ICU MORTALITY RISK REVIEW SUMMARY
               Academic Research Prototype — Not for Clinical Use
===============================================================================
Patient ID:             #999999
Prediction Horizon:     48 Hours from ICU Admission
Data Source:            PhysioNet Challenge 2012 / MIMIC-II Dataset
Data Quality Index:     6.0/100 [UNRELIABLE_CRITICAL] (Status: FAIL)
-------------------------------------------------------------------------------
1. MODEL PREDICTION SUMMARY
-------------------------------------------------------------------------------
Primary Model (XGBoost):  Probability = N/A  |  Threshold = N/A  |  Decision = N/A
Secondary Model (LSTM):   Probability = N/A  |  Threshold = N/A  |  Decision = N/A

Note: Model probabilities are independent and are NOT averaged or blended.
-------------------------------------------------------------------------------
2. MODEL AGREEMENT STATUS: PREDICTION_BLOCKED
-------------------------------------------------------------------------------
Automated risk predictions are blocked due to telemetry quality failure. No consensus evaluated.
-------------------------------------------------------------------------------
3. TOP MODEL FEATURE CONTRIBUTIONS (XGBoost Tree SHAP)
-------------------------------------------------------------------------------
  No strong positive contributors above baseline.
  No strong protective contributors below baseline.
-------------------------------------------------------------------------------
4. IMPORTANT TEMPORAL PATTERNS (LSTM Perturbation & Trajectories)
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
5. DATA QUALITY AUDIT & SENSOR WARNINGS
-------------------------------------------------------------------------------
  1. Automated prediction blocked due to critical telemetry quality failure (DQI < 50.0).
-------------------------------------------------------------------------------
6. RESEARCH REVIEW PRIORITY:  [ CRITICAL_DATA_QUALITY ]
-------------------------------------------------------------------------------
RATIONALE: CRITICAL_DATA_QUALITY: Automated risk scoring is blocked because the Data Quality Index (DQI=6.0/100) is below the critical threshold (50.0). Telemetry requires audit and potential re-acquisition before research interpretation.
-------------------------------------------------------------------------------
7. CLINICAL SAFETY LIMITATIONS & RESEARCH BOUNDARIES
-------------------------------------------------------------------------------
  1. Research prototype only; not intended for autonomous clinical diagnosis, triage, or treatment guidance.
  2. Decision thresholds (0.58 / 0.58) are validation-selected cutoffs optimizing F1-score, not clinical severity limits.
  3. Priority levels reflect internal research workflow prioritization, NOT clinical urgency scores.
  4. Predictions and attributions are derived from retrospective MIMIC-II ICU data with prevalent clinical missingness.

"These outputs are research-model estimates and must not be used to make clinical treatment decisions."
===============================================================================
```

#### 2. Machine-Readable Structured JSON Output
```json
{
  "patient_id": "999999",
  "horizon_hours": 48,
  "data_quality": {
    "dqi": 6.0,
    "category": "UNRELIABLE_CRITICAL",
    "status": "FAIL",
    "confidence_label": "PREDICTION_BLOCKED",
    "missing_descriptors": [],
    "sensor_artifacts": 0,
    "observation_density": 0
  },
  "prediction": {
    "status": "BLOCKED_BY_DATA_QUALITY",
    "xgboost": {
      "predicted_probability": null,
      "threshold": 0.58,
      "decision": null
    },
    "lstm": {
      "predicted_probability": null,
      "threshold": 0.58,
      "decision": null
    }
  },
  "agreement_status": "PREDICTION_BLOCKED",
  "agreement_meaning": "Automated risk predictions are blocked due to telemetry quality failure. No consensus evaluated.",
  "explanation": {
    "top_xgboost_risk_contributors": [],
    "top_xgboost_protective_contributors": [],
    "top_lstm_temporal_features": [],
    "important_time_periods": [],
    "physiological_trends": []
  },
  "review_priority": "CRITICAL_DATA_QUALITY",
  "rationale": "CRITICAL_DATA_QUALITY: Automated risk scoring is blocked because the Data Quality Index (DQI=6.0/100) is below the critical threshold (50.0). Telemetry requires audit and potential re-acquisition before research interpretation.",
  "limitations": [
    "Research prototype only; not intended for autonomous clinical diagnosis, triage, or treatment guidance.",
    "Decision thresholds (0.58 / 0.58) are validation-selected cutoffs optimizing F1-score, not clinical severity limits.",
    "Priority levels reflect internal research workflow prioritization, NOT clinical urgency scores.",
    "Predictions and attributions are derived from retrospective MIMIC-II ICU data with prevalent clinical missingness.",
    "Explanations are inhibited when input telemetry fails data quality thresholds.",
    "Automated mortality risk prediction is inhibited when input telemetry fails minimum clinical data quality standards.",
    "Research-only risk predictions; not approved for autonomous clinical diagnosis or treatment guidance.",
    "Decision thresholds (XGBoost 48h: 0.58, LSTM 48h: 0.58) are validation-selected research cutoffs optimizing F1-score, not clinical risk categories.",
    "XGBoost serves as the primary tabular predictive model; LSTM provides an independent sequential trajectory comparison.",
    "Predictions from XGBoost and LSTM must NOT be averaged without a validated prospective ensemble.",
    "Evaluated retrospectively on the PhysioNet 2012 / MIMIC-II cohort; prospective generalizability has not been established."
  ],
  "research_only": true
}
```


---

## Methodological Summary & Technical Audit

| Patient ID | DQI | XGBoost Prob ($P_{xgb}$) | LSTM Prob ($P_{lstm}$) | Agreement Status | Review Priority | Safety Gate Triggered |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **132539** | 93.0 | 0.0356 | 0.0144 | `MODEL_CONCORDANT_LOW` | **`LOW`** | None |
| **132551** | 90.0 | 0.7325 | 0.7573 | `MODEL_CONCORDANT_HIGH` | **`HIGH`** | None |
| **132588** | 100.0 | 0.7625 | 0.5622 | `MODEL_DISCORDANT` | **`MODERATE`** | None |
| **999999** | 6.0 | N/A | N/A | `PREDICTION_BLOCKED` | **`CRITICAL_DATA_QUALITY`** | DQI Gate Inhibited |

### Key Methodological Observations:
1. **Exact Upstream Integrity:** The Reporting Agent preserves the exact outputs produced by the Data Quality Agent, Prediction Agent, and Explanation Agent with zero recalculation or probability modification.
2. **Discordance Transparency (Patient 132588):** Rather than masking divergence behind an averaged probability, the Reporting Agent clearly flagged `MODEL_DISCORDANT`, preserved both distinct probabilities, highlighted the relevant feature domains (tabular lab extremity vs temporal trajectories), and assigned `MODERATE` review priority.
3. **Data Quality Gate Enforcement (Patient 999999):** When severe sensor artifacts and extreme missingness dropped DQI to 6.0, the agent automatically inhibited downstream predictions and explanations, returning `PREDICTION_BLOCKED` and `CRITICAL_DATA_QUALITY` without generating speculative numbers.
4. **Reproducibility:** Re-running the report across identical patient records yields deterministic, bit-for-bit identical JSON schemas and priority assignments.
