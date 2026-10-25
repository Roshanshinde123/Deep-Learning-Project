# Phase 6.3 — Explanation Agent Live Demonstration & Clinical Interpretability Audit

**Date of Execution:** 2026-10-23  
**Project:** Agentic AI-Assisted ICU Mortality Risk Prediction from the First 48 Hours  
**Interpretability Framework:** Local SHAP TreeExplainer (XGBoost) + Occlusion Perturbation Attribution (LSTM)  
**Agent Component:** [`ExplanationAgent`](file:///D:/ICU_Prediction/src/agents/explanation_agent.py)  

---

## 1. Executive Summary

The **Explanation Agent** bridges the critical gap between complex machine learning inferences and clinical transparency.
Rather than presenting clinicians with opaque risk probabilities, the agent decomposes model outputs across multiple orthogonal dimensions:
1. **Local Tabular SHAP Attributions (XGBoost):** Pinpoints the precise hand-crafted summary statistics (mean, min, max, count, slope) driving tree splits, separated into risk-increasing (positive SHAP) and protective (negative SHAP) contributions.
2. **Sequential Occlusion Attribution (LSTM):** Identifies dynamic physiological variables and discrete temporal windows (Quarters Q1–Q4) that most strongly influenced the neural recurrent trajectory.
3. **Longitudinal Trajectory Tracking:** Extracts first, last, delta, and directional trends across key verified ICU measurements.
4. **Principled Model Discordance Analysis:** Automatically flags and contextualizes instances where XGBoost and LSTM diverge across decision thresholds, explaining the architectural divergence without heuristic probability averaging.
5. **Quality-Aware Sparsity Warnings:** Cross-references top contributors with telemetry missingness, cautioning clinicians when a high-impact feature was derived from heavily imputed variables.

---

## 2. Clinical Scenario Demonstrations

### Patient Record #132539 (48-Hour ICU Horizon)
- **Ground Truth Outcome:** **Survivor (In-hospital death = 0)**
- **Audit Status:** `SUCCESS`
- **Data Quality Index (DQI):** **93.0/100** (`EXCELLENT` | Status: `PASS`)

#### 1. Predictive Model Results
| Predictive Model | Role | Model-Estimated Mortality Probability | Validation Threshold | Decision vs Threshold |
| :--- | :--- | :---: | :---: | :---: |
| **XGBoost** | `PRIMARY` | **0.0356** | 0.58 | **`BELOW_THRESHOLD`** |
| **LSTM** | `SECONDARY` | **0.0144** | 0.58 | **`BELOW_THRESHOLD`** |

#### 2. Local Tabular Attribution (XGBoost Tree SHAP)
**Top Risk-Increasing Model Contributors (Positive SHAP):**
| Rank | Feature Name | Patient Value | SHAP Contribution (Delta Log-Odds) | Clinical Domain |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `GCS_trend` | 0.0 | **+0.0625** | Physiological Dynamic |
| 2 | `GCS_std` | 0.0726 | **+0.0365** | Physiological Dynamic |
| 3 | `Creatinine_std` | 0.0535 | **+0.0356** | Physiological Dynamic |
| 4 | `Temp_min` | -2.173 | **+0.0305** | Physiological Dynamic |
| 5 | `NIDiasABP_mean` | -0.5917 | **+0.0254** | Physiological Dynamic |

**Top Protective Associations (Negative SHAP):**
| Rank | Feature Name | Patient Value | SHAP Contribution (Delta Log-Odds) | Clinical Domain |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `GCS_last` | 0.8357 | **-0.6524** | Physiological Dynamic |
| 2 | `Age_scaled` | -0.5962 | **-0.2306** | Admission Descriptor |
| 3 | `BUN_last` | -0.8448 | **-0.1723** | Physiological Dynamic |
| 4 | `RespRate_min` | -2.6063 | **-0.1710** | Physiological Dynamic |
| 5 | `Urine_mean` | 0.1987 | **-0.1591** | Physiological Dynamic |

**SHAP Waterfall Visualization:**

![SHAP Waterfall Plot for Patient #132539](/D:/ICU_Prediction/reports/figures/shap_132539_48h.png)

#### 3. Temporal Sequence Attribution (LSTM Perturbation)
*Method: `Temporal and feature occlusion perturbation (Delta P = P_baseline - P_occluded)`*

**Top Dynamic Features Impacting LSTM Probability:**
| Rank | Dynamic Feature | Occlusion Delta P | Impact Direction | Interpretation |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `Temp` | **-0.0106** | `RISK_DECREASING` | Trajectory presence altered risk |
| 2 | `GCS` | **-0.0069** | `RISK_DECREASING` | Trajectory presence altered risk |
| 3 | `BUN` | **-0.0061** | `RISK_DECREASING` | Trajectory presence altered risk |
| 4 | `Urine` | **-0.0042** | `RISK_DECREASING` | Trajectory presence altered risk |
| 5 | `NIDiasABP` | **+0.0039** | `RISK_INCREASING` | Trajectory presence altered risk |

**Temporal Window Sensitivity (Time Period Importance):**
| Rank | Temporal Period | Window Occlusion Delta P | Window Sensitivity |
| :---: | :--- | :---: | :--- |
| 1 | **Q4 (Hours 36-48)** | **-0.0166** | `RISK_DECREASING` |
| 2 | **Q3 (Hours 24-36)** | **-0.0040** | `RISK_DECREASING` |
| 3 | **Q2 (Hours 12-24)** | **+0.0003** | `NEUTRAL` |
| 4 | **Q1 (Hours 00-12)** | **-0.0001** | `NEUTRAL` |

#### 4. Verified Longitudinal Trajectories
| Measurement | First Observed | Last Observed | Net Change (Delta) | Trajectory Trend | Observations | Missingness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HR** | 73.0 (00:07) | 86.0 (47:37) | **+13.00** | `INCREASING` | 37 | 25.0% |
| **Temp** | 35.1 (00:07) | 37.8 (47:37) | **+2.70** | `INCREASING` | 14 | 72.9% |
| **GCS** | 15.0 (00:07) | 15.0 (47:37) | **+0.00** | `STABLE` | 13 | 72.9% |
| **BUN** | 13.0 (10:37) | 8.0 (33:07) | **-5.00** | `DECREASING` | 2 | 95.8% |
| **Urine** | 900.0 (00:07) | 280.0 (47:37) | **-620.00** | `DECREASING` | 38 | 25.0% |
| **RespRate** | 19.0 (00:07) | 23.0 (47:37) | **+4.00** | `INCREASING` | 42 | 14.6% |
| **Na** | 137.0 (10:37) | 136.0 (33:07) | **-1.00** | `STABLE` | 2 | 95.8% |
| **Creatinine** | 0.8 (10:37) | 0.7 (33:07) | **-0.10** | `STABLE` | 2 | 95.8% |

#### 5. Clinician-Oriented Research Synthesis
```text
Prediction Summary for Patient #132539 (48-Hour ICU Horizon):
Primary XGBoost assigned a model-estimated mortality probability of 0.0356 (BELOW_THRESHOLD validation threshold 0.58). Secondary LSTM assigned a model-estimated mortality probability of 0.0144 (BELOW_THRESHOLD validation threshold 0.58).

Feature Attribution (XGBoost SHAP):
Strongest model contributors increasing predicted risk: GCS_trend (val: 0.0, SHAP: +0.062), GCS_std (val: 0.0726, SHAP: +0.036), Creatinine_std (val: 0.0535, SHAP: +0.036).
Strongest protective associations reducing predicted risk: GCS_last (val: 0.8357, SHAP: -0.652), Age_scaled (val: -0.5962, SHAP: -0.231), BUN_last (val: -0.8448, SHAP: -0.172).

Temporal Dynamics & Trajectory Analysis:
Sequential LSTM attribution indicated that Q4 (Hours 36-48) exerted the highest dynamic impact (Delta P = -0.0166) on temporal risk encoding.
Longitudinal physiological trends observed: HR transitioned from 73.0 (00:07) to 86.0 (47:37) [INCREASING]; Temp transitioned from 35.1 (00:07) to 37.8 (47:37) [INCREASING]; GCS transitioned from 15.0 (00:07) to 15.0 (47:37) [STABLE]; BUN transitioned from 13.0 (10:37) to 8.0 (33:07) [DECREASING].

Model Concordance Assessment:
MODEL CONCORDANCE: Both XGBoost (BELOW_THRESHOLD, p=0.0356) and LSTM (BELOW_THRESHOLD, p=0.0144) reached concordant risk classifications relative to their validation-selected thresholds. The models exhibit an absolute probability difference of 0.0212.

Clinical Safety & Research Boundary:
The attributions above reflect statistical model associations derived from retrospective EHR data and must NOT be interpreted as causal medical conclusions, physiological certainties, or clinical treatment recommendations.
```

---
### Patient Record #132551 (48-Hour ICU Horizon)
- **Ground Truth Outcome:** **Deceased (In-hospital death = 1)**
- **Audit Status:** `SUCCESS`
- **Data Quality Index (DQI):** **90.0/100** (`EXCELLENT` | Status: `PASS`)

#### 1. Predictive Model Results
| Predictive Model | Role | Model-Estimated Mortality Probability | Validation Threshold | Decision vs Threshold |
| :--- | :--- | :---: | :---: | :---: |
| **XGBoost** | `PRIMARY` | **0.7325** | 0.58 | **`ABOVE_THRESHOLD`** |
| **LSTM** | `SECONDARY` | **0.7573** | 0.58 | **`ABOVE_THRESHOLD`** |

#### 2. Local Tabular Attribution (XGBoost Tree SHAP)
**Top Risk-Increasing Model Contributors (Positive SHAP):**
| Rank | Feature Name | Patient Value | SHAP Contribution (Delta Log-Odds) | Clinical Domain |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `Urine_mean` | -0.4821 | **+0.3196** | Physiological Dynamic |
| 2 | `PaCO2_mean` | -1.2973 | **+0.2560** | Physiological Dynamic |
| 3 | `GCS_last` | -0.735 | **+0.1884** | Physiological Dynamic |
| 4 | `BUN_last` | 1.6599 | **+0.1830** | Physiological Dynamic |
| 5 | `GCS_trend` | -0.0341 | **+0.1203** | Physiological Dynamic |

**Top Protective Associations (Negative SHAP):**
| Rank | Feature Name | Patient Value | SHAP Contribution (Delta Log-Odds) | Clinical Domain |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `Na_trend` | -0.0313 | **-0.1196** | Physiological Dynamic |
| 2 | `Mg_std` | 3.6323 | **-0.1140** | Physiological Dynamic |
| 3 | `Temp_mean` | -0.0919 | **-0.0935** | Physiological Dynamic |
| 4 | `GCS_mean` | 0.0101 | **-0.0723** | Physiological Dynamic |
| 5 | `Creatinine_std` | 0.0986 | **-0.0715** | Physiological Dynamic |

**SHAP Waterfall Visualization:**

![SHAP Waterfall Plot for Patient #132551](/D:/ICU_Prediction/reports/figures/shap_132551_48h.png)

#### 3. Temporal Sequence Attribution (LSTM Perturbation)
*Method: `Temporal and feature occlusion perturbation (Delta P = P_baseline - P_occluded)`*

**Top Dynamic Features Impacting LSTM Probability:**
| Rank | Dynamic Feature | Occlusion Delta P | Impact Direction | Interpretation |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `HCO3` | **+0.1411** | `RISK_INCREASING` | Trajectory presence altered risk |
| 2 | `DiasABP` | **+0.0965** | `RISK_INCREASING` | Trajectory presence altered risk |
| 3 | `BUN` | **+0.0849** | `RISK_INCREASING` | Trajectory presence altered risk |
| 4 | `GCS` | **+0.0546** | `RISK_INCREASING` | Trajectory presence altered risk |
| 5 | `PaCO2` | **-0.0546** | `RISK_DECREASING` | Trajectory presence altered risk |

**Temporal Window Sensitivity (Time Period Importance):**
| Rank | Temporal Period | Window Occlusion Delta P | Window Sensitivity |
| :---: | :--- | :---: | :--- |
| 1 | **Q4 (Hours 36-48)** | **+0.5118** | `RISK_INCREASING` |
| 2 | **Q3 (Hours 24-36)** | **+0.0454** | `RISK_INCREASING` |
| 3 | **Q2 (Hours 12-24)** | **-0.0033** | `RISK_DECREASING` |
| 4 | **Q1 (Hours 00-12)** | **-0.0010** | `RISK_DECREASING` |

#### 4. Verified Longitudinal Trajectories
| Measurement | First Observed | Last Observed | Net Change (Delta) | Trajectory Trend | Observations | Missingness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HR** | 111.0 (00:00) | 58.0 (47:40) | **-53.00** | `DECREASING` | 73 | 10.4% |
| **Temp** | 38.0 (00:00) | 36.6 (47:40) | **-1.40** | `STABLE` | 36 | 52.1% |
| **GCS** | 15.0 (00:10) | 9.0 (46:40) | **-6.00** | `DECREASING` | 13 | 72.9% |
| **BUN** | 81.0 (00:50) | 58.0 (31:40) | **-23.00** | `DECREASING` | 5 | 89.6% |
| **Urine** | 120.0 (01:40) | 40.0 (46:40) | **-80.00** | `DECREASING` | 37 | 22.9% |
| **PaCO2** | 30.0 (00:38) | 24.0 (47:16) | **-6.00** | `DECREASING` | 15 | 68.8% |
| **Na** | 141.0 (00:50) | 137.0 (31:40) | **-4.00** | `STABLE` | 5 | 89.6% |
| **pH** | 7.4 (00:38) | 7.37 (47:16) | **-0.03** | `STABLE` | 15 | 68.8% |

#### 5. Clinician-Oriented Research Synthesis
```text
Prediction Summary for Patient #132551 (48-Hour ICU Horizon):
Primary XGBoost assigned a model-estimated mortality probability of 0.7325 (ABOVE_THRESHOLD validation threshold 0.58). Secondary LSTM assigned a model-estimated mortality probability of 0.7573 (ABOVE_THRESHOLD validation threshold 0.58).

Feature Attribution (XGBoost SHAP):
Strongest model contributors increasing predicted risk: Urine_mean (val: -0.4821, SHAP: +0.320), PaCO2_mean (val: -1.2973, SHAP: +0.256), GCS_last (val: -0.735, SHAP: +0.188).
Strongest protective associations reducing predicted risk: Na_trend (val: -0.0313, SHAP: -0.120), Mg_std (val: 3.6323, SHAP: -0.114), Temp_mean (val: -0.0919, SHAP: -0.093).

Temporal Dynamics & Trajectory Analysis:
Sequential LSTM attribution indicated that Q4 (Hours 36-48) exerted the highest dynamic impact (Delta P = +0.5118) on temporal risk encoding.
Longitudinal physiological trends observed: HR transitioned from 111.0 (00:00) to 58.0 (47:40) [DECREASING]; Temp transitioned from 38.0 (00:00) to 36.6 (47:40) [STABLE]; GCS transitioned from 15.0 (00:10) to 9.0 (46:40) [DECREASING]; BUN transitioned from 81.0 (00:50) to 58.0 (31:40) [DECREASING].

Model Concordance Assessment:
MODEL CONCORDANCE: Both XGBoost (ABOVE_THRESHOLD, p=0.7325) and LSTM (ABOVE_THRESHOLD, p=0.7573) reached concordant risk classifications relative to their validation-selected thresholds. The models exhibit an absolute probability difference of 0.0248.

Clinical Safety & Research Boundary:
The attributions above reflect statistical model associations derived from retrospective EHR data and must NOT be interpreted as causal medical conclusions, physiological certainties, or clinical treatment recommendations.
```

---
### Patient Record #132588 (48-Hour ICU Horizon)
- **Ground Truth Outcome:** **Deceased (In-hospital death = 1)**
- **Audit Status:** `SUCCESS`
- **Data Quality Index (DQI):** **100.0/100** (`EXCELLENT` | Status: `PASS`)

#### 1. Predictive Model Results
| Predictive Model | Role | Model-Estimated Mortality Probability | Validation Threshold | Decision vs Threshold |
| :--- | :--- | :---: | :---: | :---: |
| **XGBoost** | `PRIMARY` | **0.7625** | 0.58 | **`ABOVE_THRESHOLD`** |
| **LSTM** | `SECONDARY` | **0.5622** | 0.58 | **`BELOW_THRESHOLD`** |

> [!WARNING]
> **MODEL DISCORDANCE DETECTED:** MODEL DISCORDANCE DETECTED: Primary XGBoost evaluated the patient as ABOVE_THRESHOLD (p=0.7625, threshold=0.58), whereas Secondary LSTM evaluated the patient as BELOW_THRESHOLD (p=0.5622, threshold=0.58). The absolute probability discrepancy is 0.2003. This divergence occurs because XGBoost is a tree-based tabular model heavily influenced by aggregate summary statistics and extreme static/lab values (Bilirubin_min, NISysABP_last, Albumin_mean), whereas LSTM processes sequential longitudinal trajectories (Bilirubin, Albumin, GCS) and may capture temporal stability or late decompensation differently. Clinicians must review individual vital trajectories rather than relying on a heuristic consensus.

#### 2. Local Tabular Attribution (XGBoost Tree SHAP)
**Top Risk-Increasing Model Contributors (Positive SHAP):**
| Rank | Feature Name | Patient Value | SHAP Contribution (Delta Log-Odds) | Clinical Domain |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `Bilirubin_min` | 2.0428 | **+0.2907** | Physiological Dynamic |
| 2 | `NISysABP_last` | -1.5849 | **+0.2760** | Physiological Dynamic |
| 3 | `Albumin_mean` | -2.5646 | **+0.1611** | Physiological Dynamic |
| 4 | `Bilirubin_last` | 2.4844 | **+0.1359** | Physiological Dynamic |
| 5 | `Glucose_trend` | 0.0351 | **+0.1325** | Physiological Dynamic |

**Top Protective Associations (Negative SHAP):**
| Rank | Feature Name | Patient Value | SHAP Contribution (Delta Log-Odds) | Clinical Domain |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `GCS_last` | 0.8357 | **-0.3538** | Physiological Dynamic |
| 2 | `Age_scaled` | -0.9393 | **-0.1332** | Admission Descriptor |
| 3 | `BUN_last` | -0.9951 | **-0.1241** | Physiological Dynamic |
| 4 | `GCS_mean` | 0.8357 | **-0.1007** | Physiological Dynamic |
| 5 | `BUN_min` | -0.9951 | **-0.0976** | Physiological Dynamic |

**SHAP Waterfall Visualization:**

![SHAP Waterfall Plot for Patient #132588](/D:/ICU_Prediction/reports/figures/shap_132588_48h.png)

#### 3. Temporal Sequence Attribution (LSTM Perturbation)
*Method: `Temporal and feature occlusion perturbation (Delta P = P_baseline - P_occluded)`*

**Top Dynamic Features Impacting LSTM Probability:**
| Rank | Dynamic Feature | Occlusion Delta P | Impact Direction | Interpretation |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `Bilirubin` | **+0.3749** | `RISK_INCREASING` | Trajectory presence altered risk |
| 2 | `Albumin` | **+0.2584** | `RISK_INCREASING` | Trajectory presence altered risk |
| 3 | `GCS` | **-0.1440** | `RISK_DECREASING` | Trajectory presence altered risk |
| 4 | `NISysABP` | **+0.1344** | `RISK_INCREASING` | Trajectory presence altered risk |
| 5 | `NIMAP` | **+0.1251** | `RISK_INCREASING` | Trajectory presence altered risk |

**Temporal Window Sensitivity (Time Period Importance):**
| Rank | Temporal Period | Window Occlusion Delta P | Window Sensitivity |
| :---: | :--- | :---: | :--- |
| 1 | **Q4 (Hours 36-48)** | **+0.3170** | `RISK_INCREASING` |
| 2 | **Q3 (Hours 24-36)** | **+0.0743** | `RISK_INCREASING` |
| 3 | **Q2 (Hours 12-24)** | **-0.0069** | `RISK_DECREASING` |
| 4 | **Q1 (Hours 00-12)** | **-0.0009** | `RISK_DECREASING` |

#### 4. Verified Longitudinal Trajectories
| Measurement | First Observed | Last Observed | Net Change (Delta) | Trajectory Trend | Observations | Missingness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HR** | 115.0 (01:09) | 78.0 (47:39) | **-37.00** | `DECREASING` | 40 | 22.9% |
| **Temp** | 38.4 (01:09) | 37.0 (41:39) | **-1.40** | `STABLE` | 8 | 83.3% |
| **GCS** | 15.0 (01:09) | 15.0 (39:39) | **+0.00** | `STABLE` | 8 | 83.3% |
| **BUN** | 7.0 (04:22) | 5.0 (31:12) | **-2.00** | `DECREASING` | 2 | 95.8% |
| **RespRate** | 17.0 (01:39) | 9.0 (47:39) | **-8.00** | `DECREASING` | 38 | 25.0% |
| **Na** | 140.0 (04:22) | 133.0 (31:12) | **-7.00** | `STABLE` | 2 | 95.8% |
| **Creatinine** | 3.5 (04:22) | 2.2 (31:12) | **-1.30** | `DECREASING` | 2 | 95.8% |
| **Glucose** | 63.0 (04:22) | 110.0 (31:12) | **+47.00** | `INCREASING` | 2 | 95.8% |

#### 5. Clinician-Oriented Research Synthesis
```text
Prediction Summary for Patient #132588 (48-Hour ICU Horizon):
Primary XGBoost assigned a model-estimated mortality probability of 0.7625 (ABOVE_THRESHOLD validation threshold 0.58). Secondary LSTM assigned a model-estimated mortality probability of 0.5622 (BELOW_THRESHOLD validation threshold 0.58).

Feature Attribution (XGBoost SHAP):
Strongest model contributors increasing predicted risk: Bilirubin_min (val: 2.0428, SHAP: +0.291), NISysABP_last (val: -1.5849, SHAP: +0.276), Albumin_mean (val: -2.5646, SHAP: +0.161).
Strongest protective associations reducing predicted risk: GCS_last (val: 0.8357, SHAP: -0.354), Age_scaled (val: -0.9393, SHAP: -0.133), BUN_last (val: -0.9951, SHAP: -0.124).

Temporal Dynamics & Trajectory Analysis:
Sequential LSTM attribution indicated that Q4 (Hours 36-48) exerted the highest dynamic impact (Delta P = +0.3170) on temporal risk encoding.
Longitudinal physiological trends observed: HR transitioned from 115.0 (01:09) to 78.0 (47:39) [DECREASING]; Temp transitioned from 38.4 (01:09) to 37.0 (41:39) [STABLE]; GCS transitioned from 15.0 (01:09) to 15.0 (39:39) [STABLE]; BUN transitioned from 7.0 (04:22) to 5.0 (31:12) [DECREASING].

Model Concordance Assessment:
MODEL DISCORDANCE DETECTED: Primary XGBoost evaluated the patient as ABOVE_THRESHOLD (p=0.7625, threshold=0.58), whereas Secondary LSTM evaluated the patient as BELOW_THRESHOLD (p=0.5622, threshold=0.58). The absolute probability discrepancy is 0.2003. This divergence occurs because XGBoost is a tree-based tabular model heavily influenced by aggregate summary statistics and extreme static/lab values (Bilirubin_min, NISysABP_last, Albumin_mean), whereas LSTM processes sequential longitudinal trajectories (Bilirubin, Albumin, GCS) and may capture temporal stability or late decompensation differently. Clinicians must review individual vital trajectories rather than relying on a heuristic consensus.

Clinical Safety & Research Boundary:
The attributions above reflect statistical model associations derived from retrospective EHR data and must NOT be interpreted as causal medical conclusions, physiological certainties, or clinical treatment recommendations.
```

---
### Patient Record #132551 (24-Hour ICU Horizon)
- **Ground Truth Outcome:** **Deceased (In-hospital death = 1)**
- **Audit Status:** `SUCCESS`
- **Data Quality Index (DQI):** **90.0/100** (`EXCELLENT` | Status: `PASS`)

#### 1. Predictive Model Results
| Predictive Model | Role | Model-Estimated Mortality Probability | Validation Threshold | Decision vs Threshold |
| :--- | :--- | :---: | :---: | :---: |
| **XGBoost** | `PRIMARY` | **0.7046** | 0.57 | **`ABOVE_THRESHOLD`** |
| **LSTM** | `SECONDARY` | **0.7433** | 0.66 | **`ABOVE_THRESHOLD`** |

#### 2. Local Tabular Attribution (XGBoost Tree SHAP)
**Top Risk-Increasing Model Contributors (Positive SHAP):**
| Rank | Feature Name | Patient Value | SHAP Contribution (Delta Log-Odds) | Clinical Domain |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `GCS_last` | -0.9967 | **+0.1660** | Physiological Dynamic |
| 2 | `HCO3_last` | -3.0509 | **+0.1572** | Physiological Dynamic |
| 3 | `BUN_last` | 1.71 | **+0.1103** | Physiological Dynamic |
| 4 | `BUN_min` | 1.71 | **+0.1012** | Physiological Dynamic |
| 5 | `Urine_mean` | -0.4393 | **+0.0772** | Physiological Dynamic |

**Top Protective Associations (Negative SHAP):**
| Rank | Feature Name | Patient Value | SHAP Contribution (Delta Log-Odds) | Clinical Domain |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `BUN_trend` | -0.0525 | **-0.0718** | Physiological Dynamic |
| 2 | `Lactate_first` | -0.6209 | **-0.0634** | Physiological Dynamic |
| 3 | `Mg_std` | 3.7391 | **-0.0456** | Physiological Dynamic |
| 4 | `Lactate_min` | -0.8041 | **-0.0434** | Physiological Dynamic |
| 5 | `Mg_trend` | 0.8813 | **-0.0383** | Physiological Dynamic |

**SHAP Waterfall Visualization:**

![SHAP Waterfall Plot for Patient #132551](/D:/ICU_Prediction/reports/figures/shap_132551_24h.png)

#### 3. Temporal Sequence Attribution (LSTM Perturbation)
*Method: `Temporal and feature occlusion perturbation (Delta P = P_baseline - P_occluded)`*

**Top Dynamic Features Impacting LSTM Probability:**
| Rank | Dynamic Feature | Occlusion Delta P | Impact Direction | Interpretation |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `NIDiasABP` | **+0.1888** | `RISK_INCREASING` | Trajectory presence altered risk |
| 2 | `BUN` | **+0.1457** | `RISK_INCREASING` | Trajectory presence altered risk |
| 3 | `DiasABP` | **+0.0903** | `RISK_INCREASING` | Trajectory presence altered risk |
| 4 | `Mg` | **-0.0786** | `RISK_DECREASING` | Trajectory presence altered risk |
| 5 | `Weight` | **+0.0712** | `RISK_INCREASING` | Trajectory presence altered risk |

**Temporal Window Sensitivity (Time Period Importance):**
| Rank | Temporal Period | Window Occlusion Delta P | Window Sensitivity |
| :---: | :--- | :---: | :--- |
| 1 | **Q4 (Hours 18-24)** | **+0.0914** | `RISK_INCREASING` |
| 2 | **Q2 (Hours 06-12)** | **+0.0370** | `RISK_INCREASING` |
| 3 | **Q1 (Hours 00-06)** | **-0.0300** | `RISK_DECREASING` |
| 4 | **Q3 (Hours 12-18)** | **+0.0229** | `RISK_INCREASING` |

#### 4. Verified Longitudinal Trajectories
| Measurement | First Observed | Last Observed | Net Change (Delta) | Trajectory Trend | Observations | Missingness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HR** | 111.0 (00:00) | 69.0 (23:40) | **-42.00** | `DECREASING` | 32 | 16.7% |
| **Temp** | 38.0 (00:00) | 36.7 (22:40) | **-1.30** | `STABLE` | 6 | 75.0% |
| **GCS** | 15.0 (00:10) | 8.0 (22:40) | **-7.00** | `DECREASING` | 8 | 66.7% |
| **BUN** | 81.0 (00:50) | 59.0 (21:54) | **-22.00** | `DECREASING` | 3 | 87.5% |
| **Urine** | 120.0 (01:40) | 24.0 (23:40) | **-96.00** | `DECREASING` | 19 | 20.8% |
| **PaCO2** | 30.0 (00:38) | 28.0 (21:55) | **-2.00** | `DECREASING` | 9 | 62.5% |
| **Na** | 141.0 (00:50) | 141.0 (21:54) | **+0.00** | `STABLE` | 3 | 87.5% |
| **pH** | 7.4 (00:38) | 7.24 (21:55) | **-0.16** | `STABLE` | 9 | 62.5% |

#### 5. Clinician-Oriented Research Synthesis
```text
Prediction Summary for Patient #132551 (24-Hour ICU Horizon):
Primary XGBoost assigned a model-estimated mortality probability of 0.7046 (ABOVE_THRESHOLD validation threshold 0.57). Secondary LSTM assigned a model-estimated mortality probability of 0.7433 (ABOVE_THRESHOLD validation threshold 0.66).

Feature Attribution (XGBoost SHAP):
Strongest model contributors increasing predicted risk: GCS_last (val: -0.9967, SHAP: +0.166), HCO3_last (val: -3.0509, SHAP: +0.157), BUN_last (val: 1.71, SHAP: +0.110).
Strongest protective associations reducing predicted risk: BUN_trend (val: -0.0525, SHAP: -0.072), Lactate_first (val: -0.6209, SHAP: -0.063), Mg_std (val: 3.7391, SHAP: -0.046).

Temporal Dynamics & Trajectory Analysis:
Sequential LSTM attribution indicated that Q4 (Hours 18-24) exerted the highest dynamic impact (Delta P = +0.0914) on temporal risk encoding.
Longitudinal physiological trends observed: HR transitioned from 111.0 (00:00) to 69.0 (23:40) [DECREASING]; Temp transitioned from 38.0 (00:00) to 36.7 (22:40) [STABLE]; GCS transitioned from 15.0 (00:10) to 8.0 (22:40) [DECREASING]; BUN transitioned from 81.0 (00:50) to 59.0 (21:54) [DECREASING].

Model Concordance Assessment:
MODEL CONCORDANCE: Both XGBoost (ABOVE_THRESHOLD, p=0.7046) and LSTM (ABOVE_THRESHOLD, p=0.7433) reached concordant risk classifications relative to their validation-selected thresholds. The models exhibit an absolute probability difference of 0.0387.

Clinical Safety & Research Boundary:
The attributions above reflect statistical model associations derived from retrospective EHR data and must NOT be interpreted as causal medical conclusions, physiological certainties, or clinical treatment recommendations.
```

---
### Patient Record #999999 (48-Hour ICU Horizon)
- **Ground Truth Outcome:** **Synthetic Corrupted Record**
- **Audit Status:** `BLOCKED_BY_DATA_QUALITY`
- **Data Quality Index (DQI):** **3.0/100** (`UNRELIABLE_CRITICAL` | Status: `FAIL`)

> [!CAUTION]
> **MODEL EXPLANATION INHIBITED:** Input telemetry failed minimum clinical data quality standards (DQI < 50.0).
> To prevent clinical misinformation, no feature attributions or synthetic SHAP explanations were generated.
>
> **Audit Recommendations:**
> - Data quality audit FAILED. Severe missingness or sensor corruption prevents reliable AI inference.



---

## 3. Methodological Summary & Interpretability Guarantees

| Analysis Dimension | Methodological Implementation | Mathematical Formulation / Criteria | Clinical Guarantee |
| :--- | :--- | :--- | :--- |
| **XGBoost Local Feature Attribution** | SHAP TreeExplainer (`shap.TreeExplainer`) | Shapley additive feature attributions (Tree SHAP exact formula) | Additive feature attribution; exact log-odds contribution. Never interpreted as clinical causation. |
| **LSTM Feature Attribution** | Dynamic Feature Occlusion Perturbation | Delta P_f = P_baseline - P_occluded(f) (Values=0.0, Mask=0.0) | Quantifies direct impact of each dynamic variable on neural sequential output. |
| **LSTM Temporal Window Sensitivity** | Time Period Occlusion Perturbation | Delta P_w = P_baseline - P_occluded(w) (w in Q1, Q2, Q3, Q4) | Evaluates whether early resuscitation vs late stay deterioration drove temporal risk encoding. |
| **Model Discordance Detection** | Validation Threshold Comparison | Indicator(Decision_xgb != Decision_lstm) | Flags divergence without creating an unvalidated blended consensus score. |
| **Data Quality Gatekeeper** | Telemetry Pre-Screening via DQI | DQI < 50.0 -> Block | Strictly inhibits explanations on corrupted data to prevent clinical misinformation. |

---

## 4. Research Language & Safety Safeguards

1. **Association vs Causation:** All outputs emphasize statistical model associations and explicitly disclaim clinical medical causation.
2. **Zero Treatment Guidance:** The agent strictly refrains from suggesting medications, fluid interventions, ventilator adjustments, or clinical procedures.
3. **Threshold Context:** All decisions are reported relative to empirical $F_1$-optimal research thresholds derived from retrospective Validation data, not clinical intervention triggers.

---
*Report automatically compiled and certified by `scripts/demo_explanation_agent.py`.*
