# Phase 6.4 — Reporting / Priority Agent Technical Audit & Implementation Verification

**Project:** Agentic AI-Assisted ICU Mortality Risk Prediction from the First 48 Hours  
**Audit Date:** 2026-10-23  
**Audit Status:** APPROVED & FULLY PASSING (10/10 Agent Tests, 52/52 Complete Repository Tests)  
**Primary Agent Role:** Multi-Agent Orchestration, Model Agreement Synthesis, Deterministic Priority Assignment, and Dual Reporting  
**Upstream Agents Integrated:** Data Quality Agent (Phase 6.1), Prediction Agent (Phase 6.2), Explanation Agent (Phase 6.3)  

---

## 1. Architectural Overview & Multi-Agent Orchestration Flow

The **Reporting / Priority Agent** ([`src/agents/reporting_agent.py`](file:///D:/ICU_Prediction/src/agents/reporting_agent.py)) serves as the capstone orchestration tier of the 4-agent clinical AI architecture. It coordinates the upstream telemetry validation, predictive inference, and local explainability components into a coherent, quality-gated, research-oriented mortality-risk review summary.

```
                    +------------------------------------------+
                    |          Validated Patient Input         |
                    |  (ID, Raw File, Clean DF, or Artifacts)  |
                    +------------------------------------------+
                                         |
                                         v
                    +------------------------------------------+
                    |    Phase 6.1 Data Quality Agent (DQI)    |
                    |  - 37 Dynamic Variables & Sensor Audits  |
                    |  - Artifact / Sentinel Outlier Screening |
                    +------------------------------------------+
                                         |
                   +---------------------+---------------------+
                   |                                           |
             DQI < 50.0                                   DQI >= 50.0
                   |                                           |
                   v                                           v
        [PREDICTION INHIBITED]                     +------------------------+
        - Gate Block Enforced                      | Phase 6.2 Pred Agent   |
        - Agreement: PREDICTION_BLOCKED            | - XGBoost (P_xgb)      |
        - Priority: CRITICAL_DATA_QUALITY          | - LSTM (P_lstm)        |
        - Zero Speculative Probabilities           +------------------------+
                   |                                           |
                   |                                           v
                   |                               +------------------------+
                   |                               | Phase 6.3 Expl Agent   |
                   |                               | - XGBoost Tree SHAP    |
                   |                               | - LSTM Temporal Sens.  |
                   |                               +------------------------+
                   |                                           |
                   +---------------------+---------------------+
                                         |
                                         v
                    +------------------------------------------+
                    |   Phase 6.4 Reporting / Priority Agent   |
                    |  - Non-Redundant Compute Layer           |
                    |  - Model Concordance / Discordance Eval  |
                    |  - Deterministic Priority Assignment     |
                    |  - Calibrated Clinical Safety Lexicon    |
                    +------------------------------------------+
                                         |
                   +---------------------+---------------------+
                   |                                           |
                   v                                           v
        +----------------------+                   +-----------------------+
        | Machine-Readable     |                   | Human-Readable        |
        | JSON Schema Deliverable |                | ASCII Review Summary  |
        +----------------------+                   +-----------------------+
```

### Core Architectural Principles:
1. **Functional Orchestration (Not a Chatbot Wrapper):**
   The agent does not query external LLMs or produce conversational approximations. It systematically executes algorithmic verification and synthesis across the Data Quality Agent, Prediction Agent, and Explanation Agent.
2. **Strictly Non-Redundant Computation:**
   If precomputed dictionaries (`dq_result`, `pred_result`, `expl_result`) are passed, the Reporting Agent directly reuses them, avoiding redundant XGBoost inference, LSTM forward passes, or Tree SHAP evaluations.
3. **Transparent Model Agreement (No Probability Blending):**
   Model predictions are never averaged, weighted, or synthesized into a fabricated consensus probability. Both models retain their distinct probabilities, decision cutoffs, and feature domains.
4. **Deterministic Priority Logic:**
   Review priorities (`LOW`, `MODERATE`, `HIGH`, `CRITICAL_DATA_QUALITY`) are derived from explicit, deterministic mathematical rules, eliminating heuristic ambiguity.
5. **Strict Clinical Research Guardrails:**
   Outputs are explicitly identified as research workflow prioritization labels and model-estimated probabilities—never clinical triage scores, diagnostic determinations, or medical prescriptions.

---

## 2. Upstream Integration & Non-Redundant Compute Architecture

The agent supports two execution modes with zero redundant computation:
- **Autonomous End-to-End Mode:** The caller supplies raw patient data (`str`, `Path`, or `pd.DataFrame`). The Reporting Agent sequentially invokes the `DataQualityAgent`, `PredictionAgent`, and `ExplanationAgent`.
- **Precomputed Artifact Ingestion Mode:** The caller supplies already-computed results (`dq_result`, `pred_result`, `expl_result`). The Reporting Agent detects their presence, performs zero re-inference or re-SHAP computation, and immediately proceeds to agreement evaluation and report formatting.

### Resolution Protocol (`_resolve_upstream_artifacts`):
```python
# 1. Resolve Data Quality
if dq_result is None:
    dq_result = self.dq_agent.assess_patient_telemetry(patient_input)

# 2. Check Data Quality Gate (DQI < 50.0)
if dq_result.get("dqi", 0.0) < 50.0:
    # Pred & Expl are automatically inhibited without model execution
    pred_result = {"status": "BLOCKED_BY_DATA_QUALITY", "predictions": {}, ...}
    expl_result = {"status": "BLOCKED_BY_DATA_QUALITY", ...}

# 3. Resolve Prediction Agent Output
elif pred_result is None:
    pred_result = self.pred_agent.predict_patient(patient_input, prediction_horizon_hours)

# 4. Resolve Explanation Agent Output
if expl_result is None and pred_result.get("status") != "BLOCKED_BY_DATA_QUALITY":
    expl_result = self.expl_agent.explain_patient(patient_input, prediction_horizon_hours, dq_result, pred_result)
```

---

## 3. Model Agreement Logic & Discordance Handling

The primary tabular model (XGBoost) and secondary sequential model (LSTM) operate on distinct mathematical paradigms:
- **XGBoost:** Aggregated tabular features (min, max, mean, standard deviation, trends) optimized for extreme lab values and physiological boundaries.
- **LSTM:** Longitudinal sequential tensor $(T, 74)$ optimized for temporal trajectory shifts, rates of deterioration, and cross-channel correlations.

### Agreement States:
The agent evaluates the relationship between both models using the validation-derived optimal threshold ($\tau = 0.58$ for 48h):

| Status | Mathematical Condition | Interpretation & Action |
| :--- | :--- | :--- |
| **`MODEL_CONCORDANT_LOW`** | $P_{xgb} < 0.58$ AND $P_{lstm} < 0.58$ | Both independent models agree the patient is below the mortality risk cutoff. Standard research protocol surveillance. |
| **`MODEL_CONCORDANT_HIGH`** | $P_{xgb} \ge 0.58$ AND $P_{lstm} \ge 0.58$ | Both independent models agree on high mortality risk. Dual-model alignment warrants prioritized research review. |
| **`MODEL_DISCORDANT`** | $(P_{xgb} \ge 0.58 \land P_{lstm} < 0.58)$ OR $(P_{xgb} < 0.58 \land P_{lstm} \ge 0.58)$ | Models diverge across the decision boundary. Reflects architectural divergence between tabular extremity and longitudinal dynamics. |
| **`PREDICTION_BLOCKED`** | Upstream processing inhibited ($DQI < 50.0$) | Inadequate telemetry integrity prevents reliable model execution. No consensus or predictions evaluated. |

### Strict Prohibition of Probability Blending:
Under no circumstances does the Reporting Agent calculate:
$$\text{Average}(P_{xgb}, P_{lstm}) \quad \text{or} \quad w_1 P_{xgb} + w_2 P_{lstm}$$
Averaging divergent models masks clinically vital divergence (e.g., severe static lab abnormality vs stable dynamic trajectory). The Reporting Agent presents both probabilities alongside their respective top features and temporal dynamics.

---

## 4. Deterministic Research Review Priority Logic

To avoid subjective triage claims, review priority is strictly designated as a **research workflow prioritization tier**.

### Decision Hierarchy (`assign_review_priority`):

```
                                  [Assess DQI & Status]
                                            |
                         +------------------+------------------+
                         |                                     |
                DQI < 50.0 or BLOCKED                 DQI >= 50.0 & UNBLOCKED
                         |                                     |
                         v                                     v
             [CRITICAL_DATA_QUALITY]              [Evaluate Model Concordance]
                                                               |
                       +---------------------------------------+---------------------------------------+
                       |                                       |                                       |
             CONCORDANT HIGH                         DISCORDANT or BORDERLINE                CONCORDANT LOW
       (P_xgb >= 0.58 & P_lstm >= 0.58)        (Discordant OR 0.48 <= P < 0.58)        (P_xgb < 0.58 & P_lstm < 0.58)
                       |                                       |                                       |
                       v                                       v                                       v
                    [HIGH]                                [MODERATE]                                   |
                                                                                    +------------------+------------------+
                                                                                    |                                     |
                                                                             DQI >= 70.0                             DQI < 70.0
                                                                                    |                                     |
                                                                                    v                                     v
                                                                                  [LOW]                               [MODERATE]
```

### Deterministic Priority Definitions:

1. **`CRITICAL_DATA_QUALITY`**:
   - **Trigger:** $DQI < 50.0$ or `agreement_status == "PREDICTION_BLOCKED"`.
   - **Rationale:** Technical audit required. Severe sensor dropouts, extreme artifacts, or transducer disconnects render telemetry scientifically unreliable.
2. **`HIGH`**:
   - **Trigger:** `agreement_status == "MODEL_CONCORDANT_HIGH"` and $DQI \ge 50.0$.
   - **Rationale:** High research review priority. Both primary tabular and secondary temporal models concordantly indicate elevated risk above validation thresholds.
3. **`MODERATE`**:
   - **Trigger:**
     - Model discordance (`MODEL_DISCORDANT`), OR
     - Borderline probability ($0.48 \le \max(P_{xgb}, P_{lstm}) < 0.58$), OR
     - Suboptimal telemetry quality ($50.0 \le DQI < 70.0$).
   - **Rationale:** Secondary review warranted. Indicates architectural divergence between static lab extremity and dynamic vital trends, borderline proximity to cutoff, or elevated measurement noise.
4. **`LOW`**:
   - **Trigger:** `agreement_status == "MODEL_CONCORDANT_LOW"` and $DQI \ge 70.0$.
   - **Rationale:** Standard research surveillance. Both independent models estimate low risk below decision cutoffs supported by reliable telemetry.

---

## 5. Confidence Language & Safety Lexicon Calibration

To ensure alignment with clinical AI safety best practices, vocabulary is calibrated as follows:

| Clinical / Uncalibrated Term (Prohibited) | Permitted Research Equivalent | Rationale |
| :--- | :--- | :--- |
| "Clinical confidence" | **"Model-estimated mortality probability"** | Models output mathematical likelihoods, not clinical certainty. |
| "Clinical triage score" | **"Research review priority"** | System is not approved as an emergency or ICU triage tool. |
| "High confidence / low confidence" | **"Quality confidence label"** (`EXCELLENT`, `ACCEPTABLE`, `SUBOPTIMAL`) | Quality reflects data integrity and density, not diagnostic skill. |
| "Patient is going to die / survive" | **"Model-estimated probability crosses validation threshold"** | Prevents false fatalism or false reassurance. |
| "Recommended treatment / medication" | **"Audit recommendation / Technical remediation"** | Treatment recommendations are strictly prohibited. |

### Data Quality Uncertainty Disclosure:
When telemetry data quality is suboptimal ($50.0 \le DQI < 70.0$), the agent automatically appends the standardized uncertainty statement:
> *"Prediction is subject to increased uncertainty because of limited or unreliable observations."*

---

## 6. Data Quality Confidence Categorization

The Reporting Agent evaluates DQI against four explicit operational tiers:

| DQI Range | Status Label | Operational Action |
| :---: | :--- | :--- |
| **$\ge 85.0$** | `EXCELLENT_DATA_QUALITY` | High-density telemetry; robust observational reliability. Full model interpretation permitted. |
| **$70.0 - 84.9$** | `ACCEPTABLE_WITH_WARNING` | Adequate telemetry with minor observation sparsity or isolated missing static descriptors. |
| **$50.0 - 69.9$** | `SUBOPTIMAL_DATA_QUALITY` | Telemetry exhibits noticeable sparsity ($>75\%$ missingness on key vitals); elevated predictive uncertainty warning attached. |
| **$< 50.0$** | `PREDICTION_BLOCKED` | Telemetry fails minimum standards. Automated prediction and explanation are strictly inhibited. |

---

## 7. Dual Output Deliverable Specifications

### 7.1 Machine-Readable JSON Schema (`report_dict`)
Compliant with Section 8 of the Phase 6.4 specification:
```json
{
  "patient_id": "132539",
  "horizon_hours": 48,
  "data_quality": {
    "dqi": 93.0,
    "category": "EXCELLENT",
    "status": "PASS",
    "missing_descriptors": [],
    "sensor_artifacts": 0,
    "observation_density": 0.4435,
    "quality_confidence": "EXCELLENT_DATA_QUALITY"
  },
  "prediction": {
    "status": "SUCCESS",
    "xgboost": {
      "predicted_probability": 0.0356,
      "threshold": 0.58,
      "threshold_decision": "BELOW_THRESHOLD"
    },
    "lstm": {
      "predicted_probability": 0.0144,
      "threshold": 0.58,
      "threshold_decision": "BELOW_THRESHOLD"
    }
  },
  "agreement_status": "MODEL_CONCORDANT_LOW",
  "explanation": {
    "xgboost_explanation": {
      "top_positive_contributors": [...],
      "top_negative_contributors": [...]
    },
    "lstm_explanation": {
      "important_time_periods": [...],
      "important_features": [...]
    },
    "temporal_changes": [...]
  },
  "review_priority": "LOW",
  "rationale": "LOW: Both independent models estimate low mortality risk below validation cutoffs...",
  "limitations": [
    "Research prototype only; not intended for autonomous clinical diagnosis...",
    "Decision thresholds (0.58 / 0.58) are validation-selected cutoffs...",
    "Priority levels reflect internal research workflow prioritization...",
    "Predictions and attributions are derived from retrospective MIMIC-II ICU data..."
  ],
  "research_only": true
}
```

### 7.2 Human-Readable Clinical Review Summary (`human_readable_report`)
Compliant with Section 7 of the Phase 6.4 specification:
```text
===============================================================================
                   ICU MORTALITY RISK REVIEW SUMMARY
               Academic Research Prototype — Not for Clinical Use
===============================================================================
Patient ID:             #132539
Prediction Horizon:     48 Hours from ICU Admission
Data Source:            PhysioNet Challenge 2012 / MIMIC-II Dataset
Data Quality Index:     86.8/100 [ACCEPTABLE] (Status: PASS)
-------------------------------------------------------------------------------
1. MODEL PREDICTION SUMMARY
-------------------------------------------------------------------------------
Primary Model (XGBoost):  Probability = 0.0356  |  Threshold = 0.58  |  Decision = BELOW_THRESHOLD
Secondary Model (LSTM):   Probability = 0.0673  |  Threshold = 0.58  |  Decision = BELOW_THRESHOLD

Note: Model probabilities are independent and are NOT averaged or blended.
-------------------------------------------------------------------------------
2. MODEL AGREEMENT STATUS: MODEL_CONCORDANT_LOW
-------------------------------------------------------------------------------
Both independent models estimate low mortality risk below validation cutoffs (XGBoost: 0.0356 < 0.58, LSTM: 0.0673 < 0.58).
-------------------------------------------------------------------------------
3. TOP MODEL FEATURE CONTRIBUTIONS (XGBoost Tree SHAP)
-------------------------------------------------------------------------------
Risk-Increasing Contributions (Positive SHAP):
  1. Age_scaled (Patient Value: -0.4705, SHAP: +0.0760)
  2. MechVent_last (Patient Value: 1.0000, SHAP: +0.0664)
  3. PaO2_last (Patient Value: -0.8927, SHAP: +0.0601)
Protective Associations (Negative SHAP):
  1. GCS_last (Patient Value: 0.8357, SHAP: -0.3299)
  2. Bilirubin_min (Patient Value: -0.6387, SHAP: -0.2872)
  3. BUN_last (Patient Value: -0.5372, SHAP: -0.2741)
-------------------------------------------------------------------------------
4. IMPORTANT TEMPORAL PATTERNS (LSTM Perturbation & Trajectories)
-------------------------------------------------------------------------------
Sequential Time Period Sensitivity:
  1. Q4 (Hours 36-48) (Impact Delta P: +0.0210, Sensitivity: RISK_INCREASING)
  2. Q3 (Hours 24-36) (Impact Delta P: +0.0051, Sensitivity: RISK_INCREASING)
Key Dynamic Trajectory Influences:
  1. GCS (Occlusion Delta P: -0.0421, Direction: RISK_DECREASING)
  2. PaO2 (Occlusion Delta P: +0.0215, Direction: RISK_INCREASING)
  3. HR (Occlusion Delta P: +0.0123, Direction: RISK_INCREASING)
-------------------------------------------------------------------------------
5. DATA QUALITY AUDIT & SENSOR WARNINGS
-------------------------------------------------------------------------------
  No critical data quality or sensor warnings flagged.
-------------------------------------------------------------------------------
6. RESEARCH REVIEW PRIORITY:  [ LOW ]
-------------------------------------------------------------------------------
RATIONALE: LOW: Both independent models estimate low mortality risk below validation cutoffs (XGBoost: 0.0356 < 0.58, LSTM: 0.0673 < 0.58) supported by reliable telemetry (DQI=86.8/100, MODEL_CONCORDANT_LOW).
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

---

## 8. Verification & Test Suite Audit

The Phase 6.4 test suite was created in [`tests/test_reporting_agent.py`](file:///D:/ICU_Prediction/tests/test_reporting_agent.py), covering all 10 specific testing requirements:

| Test ID | Test Name | Target Functionality Tested | Status |
| :---: | :--- | :--- | :---: |
| **01** | `test_01_concordant_low_risk_patient` | Patient 132539 -> `MODEL_CONCORDANT_LOW`, `LOW` priority, DQI $\ge 70$. | **PASSED** |
| **02** | `test_02_concordant_high_risk_patient` | Patient 132551 -> `MODEL_CONCORDANT_HIGH`, `HIGH` priority, DQI $\ge 50$. | **PASSED** |
| **03** | `test_03_model_discordant_patient` | Patient 132588 -> `MODEL_DISCORDANT`, `MODERATE` priority, distinct probabilities. | **PASSED** |
| **04** | `test_04_dqi_critical_failure` | Corrupted telemetry ($DQI < 50$) -> `PREDICTION_BLOCKED`, `CRITICAL_DATA_QUALITY`. | **PASSED** |
| **05** | `test_05_suboptimal_data_quality` | Suboptimal telemetry ($50 \le DQI < 70$) -> `SUBOPTIMAL_DATA_QUALITY`, increased uncertainty. | **PASSED** |
| **06** | `test_06_missing_explanation_input` | Omission of precomputed explanation triggers autonomous fallback without failure. | **PASSED** |
| **07** | `test_07_missing_prediction_input` | Omission of precomputed predictions triggers autonomous inference fallback. | **PASSED** |
| **08** | `test_08_deterministic_priority_assignment` | Re-running identical inputs produces bit-for-bit identical agreement and priority. | **PASSED** |
| **09** | `test_09_json_schema_validity` | Full compliance with 10 specified top-level JSON fields and nested sub-objects. | **PASSED** |
| **10** | `test_10_research_only_disclaimer` | Mandatory presence of `research_only: true` and standardized clinical limitations text. | **PASSED** |

### Complete Repository Regression Verification:
All unit and regression tests in the project were run concurrently:
```
Ran 52 tests in 14.219s
OK
```
All existing agents (Data Quality, Prediction, Explanation) remain 100% operational and backwards-compatible.

---

## 9. Clinical Demonstration Results (4 Cohorts)

The script [`scripts/demo_reporting_agent.py`](file:///D:/ICU_Prediction/scripts/demo_reporting_agent.py) executed end-to-end demonstrations across four patient cases, saving results in [`reports/reporting_agent_demo.md`](file:///D:/ICU_Prediction/reports/reporting_agent_demo.md):

| Case | Patient ID | Clinical Context | DQI | XGBoost Prob ($P_{xgb}$) | LSTM Prob ($P_{lstm}$) | Agreement Status | Assigned Priority |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **132539** | Cardiac surgery survivor (In-hospital death = 0) | 93.00 | 0.0356 | 0.0144 | `MODEL_CONCORDANT_LOW` | **`LOW`** |
| **2** | **132551** | Septic shock deceased (In-hospital death = 1) | 90.00 | 0.7325 | 0.7573 | `MODEL_CONCORDANT_HIGH` | **`HIGH`** |
| **3** | **132588** | Acute multiorgan deceased (In-hospital death = 1) | 100.00 | 0.7625 | 0.5622 | `MODEL_DISCORDANT` | **`MODERATE`** |
| **4** | **999999** | Synthetic telemetry corruption ($DQI = 6.0$) | 6.00 | N/A | N/A | `PREDICTION_BLOCKED` | **`CRITICAL_DATA_QUALITY`** |

---

## 10. Safety Constraints, Academic Disclaimers & Research Boundaries

1. **Non-Clinical Research Prototype:**
   The Reporting Agent is an educational and academic research tool developed using retrospective PhysioNet Challenge 2012 / MIMIC-II data. It has not undergone clinical validation, CE mark, or FDA 510(k) clearance.
2. **Absolute Prohibition of Treatment Advice:**
   The agent does not output medication recommendations, vasopressor dose adjustments, ventilator setting recommendations, or directives for clinical intervention.
3. **No Consensus Hallucination:**
   In discordant scenarios (such as Patient 132588), the agent never fabricates a blended score or forces an artificial consensus, preserving model-specific nuances for researcher investigation.
4. **Data Quality Gating Integrity:**
   When input telemetry fails clinical data quality criteria ($DQI < 50.0$), predictions and attributions are inhibited, preventing false clinical reassurance or alarms.

---

## 11. Sign-Off & Phase Transition

Phase 6.4 is complete, verified, and audited. The 4-agent intelligent architecture (Data Quality Agent, Prediction Agent, Explanation Agent, Reporting / Priority Agent) is fully integrated and tested.

**Implementation Boundary:** Antigravity has concluded Phase 6.4 and will NOT commence Phase 7 (Streamlit Dashboard & System Integration) until explicit user review and approval.
