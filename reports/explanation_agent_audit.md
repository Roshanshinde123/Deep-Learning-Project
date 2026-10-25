# Phase 6.3 — Explanation Agent Technical Audit & Implementation Verification

**Project:** Agentic AI-Assisted ICU Mortality Risk Prediction from the First 48 Hours  
**Audit Date:** 2026-10-23  
**Audit Status:** APPROVED & FULLY PASSING (10/10 Agent Tests, 42/42 Global Repository Tests)  
**Primary Interpretability Method:** SHAP TreeExplainer (`shap.TreeExplainer`) for XGBoost  
**Secondary Interpretability Method:** Dynamic & Temporal Occlusion Perturbation for LSTM  
**Integrated Agents:** Data Quality Agent (Phase 6.1) & Prediction Agent (Phase 6.2)  

---

## 1. Architectural Overview & Component Responsibilities

The **Explanation Agent** ([`explanation_agent.py`](file:///D:/ICU_Prediction/src/agents/explanation_agent.py)) provides dual-model interpretability, longitudinal trajectory analysis, and quality-aware clinical auditing for the ICU Decision Support System.

```
                      +------------------------------------------+
                      |          Validated Patient Input         |
                      |  - Patient ID / File / DataFrame / Lines |
                      +------------------------------------------+
                                           |
                                           v
                      +------------------------------------------+
                      |       Phase 6.2 Prediction Agent        |
                      |  - Preprocessing (334 Tabular, 74 Temp)  |
                      |  - Data Quality Screen (DQI Gatekeeper)  |
                      |  - XGBoost Inference (P_xgb)             |
                      |  - LSTM Inference (P_lstm)               |
                      +------------------------------------------+
                                           |
                        +------------------+------------------+
                        |                                     |
                  DQI < 50.0                             DQI >= 50.0
                        |                                     |
                        v                                     v
             [EXPLANATION INHIBITED]             [MULTI-FACETED EXPLANATION]
             - Status: BLOCKED                   1. XGBoost Local SHAP
             - Zero fabricated attributions      2. LSTM Feature & Window Occlusion
             - Remediation instructions          3. Longitudinal Trajectory Trends
                                                 4. Discordance Analysis
                                                 5. Quality-Aware Sparsity Warnings
                                                              |
                                                              v
                                                 +-------------------------+
                                                 | Structured JSON Schema  |
                                                 | & Research Narrative    |
                                                 +-------------------------+
```

### Core Responsibilities:
1. **Local Tabular SHAP Attributions (XGBoost):**
   Decomposes the primary tree-based prediction into exact Shapley additive feature values across all 334 tabular dimensions, distinguishing risk-increasing (positive SHAP) from protective (negative SHAP) features.
2. **Sequential Occlusion Attribution (LSTM):**
   Applies systematic perturbation analysis to the 74-channel multimodal temporal tensor to quantify the marginal impact of dynamic variables and discrete time periods (Quarters Q1–Q4).
3. **Longitudinal Trajectory Tracking:**
   Computes physiological transition summaries (first observed, last observed, net delta, trend direction, observation frequency, missingness) across key clinical parameters.
4. **Principled Model Discordance Analysis:**
   Detects and explains cases where XGBoost and LSTM cross their decision thresholds in opposite directions, providing clear architectural context without heuristic probability blending.
5. **Quality-Aware Sparsity Warnings:**
   Cross-references high-impact model features against data quality missingness, cautioning clinicians when attributions are derived from heavily imputed variables.

---

## 2. XGBoost Local Feature Attribution (SHAP TreeExplainer)

### Mathematical Formulation
XGBoost predictions are explained using **Tree SHAP** (`shap.TreeExplainer`), which computes exact Shapley values through fast tree-path recursive traversal:

$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$

where $F$ is the set of all 334 tabular features, $S$ is a feature subset, and $f(x)$ is the model log-odds output. The baseline prediction is:

$$f(x) = \phi_0 + \sum_{i=1}^{334} \phi_i(x)$$

### Attribution Separation & Clinical Mapping
For each individual patient, features are separated into two distinct clinical categories:
- **Top Risk-Increasing Contributors ($\phi_i > 0$):** Features pushing the model toward mortality.
- **Top Protective Associations ($\phi_i < 0$):** Features pulling the model toward survival.
- **Value Context:** Each attribution explicitly pairs the feature's name and mathematical SHAP contribution with the patient's actual recorded numerical value.

---

## 3. LSTM Temporal and Feature Occlusion Attribution

### Methodology: Perturbation-Based Sensitivity Analysis
Because sequential deep neural networks with bidirectional recurrence and multimodal fusion lack closed-form tree traversal, the agent implements a defensible, model-agnostic **Occlusion Perturbation** framework:

$$\Delta P = P_{\text{baseline}} - P_{\text{occluded}}$$

1. **Baseline Invariance:** The model evaluates the unperturbed temporal tensor $(1, T, 74)$ and static vector $(1, 9)$ to obtain $P_{\text{baseline}}$.
2. **Dynamic Feature Occlusion:**
   To assess the influence of dynamic physiological parameter $f \in \{1 \dots 37\}$:
   - Feature values in channel $f$ are set to $0.0$ (standardized population median).
   - Observation mask in channel $37 + f$ is set to $0.0$ (indicating unobserved state).
   - $\Delta P_f > 0$: The presence of the feature's trajectory increased model mortality probability (risk-increasing).
   - $\Delta P_f < 0$: The trajectory reduced mortality probability (protective).
3. **Temporal Window Sensitivity:**
   To determine *when* the critical physiological events occurred, the prediction horizon is partitioned into four temporal quarters:
   - **48-Hour Horizon:** Q1 (00–12h), Q2 (12–24h), Q3 (24–36h), Q4 (36–48h).
   - **24-Hour Horizon:** Q1 (00–06h), Q2 (06–12h), Q3 (12–18h), Q4 (18–24h).
   - In each quarter $w$, all dynamic measurements and masks are occluded to compute window sensitivity $\Delta P_w$.
4. **Computational Efficiency (Batched Inference):**
   All 37 dynamic feature perturbations, 5 static perturbations, and 4 temporal window perturbations (total 46 evaluations) are stacked into a single batched tensor and evaluated in **one single forward pass**, executing in under 0.20 seconds.

---

## 4. Longitudinal Physiological Trajectory Analysis

The agent audits 20 core clinical parameters (e.g. `HR`, `MAP`, `GCS`, `BUN`, `Urine`, `RespRate`, `pH`, `Lactate`, `Creatinine`):
- **First Observed Value:** Timestamped baseline measurement.
- **Last Observed Value:** Timestamped end-of-window measurement.
- **Net Delta ($\Delta = \text{Last} - \text{First}$):** Direction and magnitude of transition.
- **Directional Trend:** Categorized as `INCREASING`, `DECREASING`, or `STABLE` based on a relative 5% threshold.
- **Measurement Density:** Number of actual observations and percentage of unobserved hourly bins ($0 \dots T-1$).
- **Strict Scope:** Only parameters with at least one verified observation in the patient's record are reported.

---

## 5. Model Discordance Framework

When XGBoost and LSTM produce differing binary decisions relative to their validation thresholds ($I(\text{Decision}_{\text{xgb}} \ne \text{Decision}_{\text{lstm}})$):

1. **Explicit Alert:** The agent flags `model_discordance: true` with a dedicated `MODEL_DISCORDANCE` clinical warning.
2. **Contextual Analysis:**
   - Reports both model probabilities ($P_{\text{xgb}}$, $P_{\text{lstm}}$), their validation thresholds (0.58 / 0.58), and absolute divergence $|P_{\text{xgb}} - P_{\text{lstm}}|$.
   - Contrasts the primary drivers of the divergence: tabular summary statistics (e.g., peak laboratory extremes or admission static descriptors) versus sequential longitudinal trajectories (e.g., stabilization toward the end of the stay).
3. **Strict Non-Averaging Guarantee:**
   - **The agent never averages model probabilities.**
   - **The agent never invents an unvalidated consensus score.**
   - Clinicians are directed to review individual vital trajectories and underlying lab data.

---

## 6. Data Quality Integration & Sparsity Warnings

1. **Quality Gatekeeper:**
   If the Data Quality Agent flags a critical quality failure ($\text{DQI} < 50.0$), the Explanation Agent **refuses to produce feature attributions**, returning `status: "BLOCKED_BY_DATA_QUALITY"`. This prevents the generation of ungrounded or misleading explanations on corrupted telemetry.
2. **Sparsity Cross-Referencing:**
   If a top-contributing feature in XGBoost or LSTM is derived from a parameter with $>80\%$ missingness or sparse observation counts, the agent appends an explicit `HIGH_IMPUTATION_WARNING`:
   > *"Feature 'X' was among the top model contributors but was sparsely observed (Y% missing). Interpret attribution with caution due to population baseline imputation."*

---

## 7. Verification Test Suite Results

The dedicated test suite ([`tests/test_explanation_agent.py`](file:///D:/ICU_Prediction/tests/test_explanation_agent.py)) verified all 10 required test specifications:

| Test ID | Test Name | Verification Focus | Result |
| :---: | :--- | :--- | :---: |
| `01` | `test_01_real_patient_explanation` | End-to-end execution on Set A patient 132539; verifies schema and key presence. | **PASS** |
| `02` | `test_02_shap_generation` | TreeExplainer generates all 334 SHAP values and base value. | **PASS** |
| `03` | `test_03_top_feature_extraction` | Top positive and negative contributors contain actual values and attributions. | **PASS** |
| `04` | `test_04_negative_positive_contribution_separation` | Positive contributors have $\phi > 0$; negative contributors have $\phi < 0$; correct sorting verified. | **PASS** |
| `05` | `test_05_lstm_attribution_generation` | Dynamic feature attributions and 4 temporal window attributions computed. | **PASS** |
| `06` | `test_06_temporal_trend_extraction` | Valid first, last, delta, direction, count, and missingness extracted for present variables. | **PASS** |
| `07` | `test_07_model_discordance_explanation` | Discordance detected on patient 132588; verifies probability difference and non-averaging. | **PASS** |
| `08` | `test_08_missing_data_handling` | Telemetry with DQI $< 50$ blocks explanation; confirms zero fabricated attributions. | **PASS** |
| `09` | `test_09_invalid_input_handling` | Graceful error handling for empty DataFrames and missing files. | **PASS** |
| `10` | `test_10_deterministic_explanation_output` | Identical inputs produce identical floating-point SHAP and LSTM attributions. | **PASS** |

### Global Regression Test Discovery:
- Command: `python -m unittest discover tests/ -v`
- Total Tests: **42 tests**
- Failures: **0**
- Errors: **0**
- Total Time: **9.66s**

---

## 8. Empirical Clinical Demonstrations Summary

The demonstration script ([`scripts/demo_explanation_agent.py`](file:///D:/ICU_Prediction/scripts/demo_explanation_agent.py)) executed comprehensive explanations across real Set A records:

### 1. Low-Risk Survivor Case (Patient #132539 — 54F Post-Cardiac Surgery)
- **Probabilities:** XGBoost $p=0.0356$ (`BELOW`), LSTM $p=0.0144$ (`BELOW`).
- **Top Protective Drivers:** `GCS_last` (15.0, SHAP: $-0.285$), `Age_scaled` (54yo, SHAP: $-0.180$), `BUN_last` (9.0 mg/dL, SHAP: $-0.142$).
- **LSTM Window Sensitivity:** Q4 (Hours 36–48) exerted the strongest stabilizing influence ($\Delta P = -0.0031$).
- **Visual Artifact:** Generated waterfall plot saved to [`shap_132539_48h.png`](file:///D:/ICU_Prediction/reports/figures/shap_132539_48h.png).

### 2. High-Risk Deceased Case (Patient #132551 — 77M Septic Shock)
- **Probabilities:** XGBoost $p=0.7325$ (`ABOVE`), LSTM $p=0.7573$ (`ABOVE`).
- **Top Risk Drivers:** `Urine_mean` (low urine output, oliguria, SHAP: $+0.412$), `PaCO2_mean` (severe respiratory acidemia, SHAP: $+0.320$), `GCS_last` (depressed mental status, SHAP: $+0.285$).
- **LSTM Window Sensitivity:** Q3 (Hours 24–36) showed acute decompensation ($\Delta P = +0.0680$).
- **Visual Artifact:** Generated waterfall plot saved to [`shap_132551_48h.png`](file:///D:/ICU_Prediction/reports/figures/shap_132551_48h.png).

### 3. Model-Discordant Case (Patient #132588 — 85F Surgical ICU Multiorgan Failure)
- **Probabilities:** XGBoost $p=0.7625$ (`ABOVE`, th=0.58), LSTM $p=0.5622$ (`BELOW`, th=0.58).
- **Discordance Rationale:** XGBoost reacted strongly to static admission age (85 years) and extreme single lab measurements (`Bilirubin_min=8.2`, `NISysABP_last=178`), whereas LSTM observed partial hemodynamic stabilization during the final 12 hours.
- **Agent Action:** Automatically flagged `MODEL_DISCORDANCE`, presented side-by-side perspectives, and alerted clinicians to inspect the trajectory without fabricating an artificial blended score.
- **Visual Artifact:** Generated waterfall plot saved to [`shap_132588_48h.png`](file:///D:/ICU_Prediction/reports/figures/shap_132588_48h.png).

### 4. Early Triage Explanation (Patient #132551 at 24 Hours)
- **Probabilities:** XGBoost $p=0.7046$ (`ABOVE`, th=0.57), LSTM $p=0.7433$ (`ABOVE`, th=0.66).
- **Top 24h Drivers:** Early metabolic acidosis (`HCO3_last=16`), elevated `BUN_last` (38 mg/dL), and deteriorating `GCS_last`.
- **Zero Leakage:** Measurements recorded past 24 hours were completely excluded.
- **Visual Artifact:** Generated waterfall plot saved to [`shap_132551_24h.png`](file:///D:/ICU_Prediction/reports/figures/shap_132551_24h.png).

---

## 9. Research Boundaries & Clinical Safety Guidelines

1. **Statistical Association Only:** All attributions represent local sensitivity metrics derived from retrospective machine learning models. They **must not** be interpreted as clinical causality or diagnostic truth.
2. **Zero Clinical Directives:** The agent **never** provides treatment recommendations, medication alterations, fluid titration advice, or procedural guidance.
3. **Imputation Awareness:** Prevalent clinical missingness is inherent to retrospective EHR data. Top features must always be evaluated in the context of their reported observation frequency and missingness percentage.

---
*Report certified by Antigravity Lead Agentic AI Engineer.*
