# ICU Mortality Prediction: Dataset Management & Setup Guide

This directory contains configuration, scalers, and documentation for the retrospective ICU patient telemetry dataset used across the project.

---

## 1. Dataset Overview

The project utilizes the **PhysioNet / Computing in Cardiology Challenge 2012** benchmark dataset (*"Predicting Mortality of an In-Hospital ICU Population"*).

- **Patient Population:** 12,000 adult intensive care unit (ICU) admissions across diverse ICU types (Coronary Care, Cardiac Surgery Recovery, Medical ICU, Surgical ICU).
- **Observation Window:** The first 48 hours of ICU admission.
- **Primary Clinical Target:** Retrospective In-Hospital Mortality (`In-hospital_death` $\in \{0, 1\}$).
- **Total Raw Size:** ~1.8 GB uncompressed across 12,000 individual patient records.

> **Data Stewardship & Privacy Policy:**  
> In compliance with GitHub repository best practices and clinical data stewardship guidelines, the full raw dataset (>1.5 GB, 17,500+ files) and large preprocessed NumPy array binaries (>1.4 GB) are excluded from version control via `.gitignore`. Essential model inference scalers, feature configurations, and metadata are retained in the repository so inference runs out-of-the-box.

---

## 2. Dataset Cohorts & Partitions

The challenge dataset is divided into three distinct cohorts:

| Cohort | Record Count | Record ID Range | Outcome File | Project Role |
| :--- | :---: | :---: | :---: | :--- |
| **Set A** | 4,000 records | 132539 – 142673 | `Outcomes-a.txt` | Model Training & 5-Fold Stratified Cross-Validation |
| **Set B** | 4,000 records | 142675 – 152864 | `Outcomes-b.txt` | Held-Out Retrospective Validation Benchmark |
| **Set C** | 4,000 records | 152871 – 163037 | `Outcomes-c.txt` | Held-Out External Generalization Benchmark |

---

## 3. How to Obtain the Dataset from PhysioNet

The full dataset is openly accessible from PhysioNet under the Open Data Commons Public Domain Dedication and License v1.0.

### Step 1: Download from PhysioNet
Download the files from the official PhysioNet archive:
- **Archive URL:** [https://physionet.org/content/challenge-2012/1.0.0/](https://physionet.org/content/challenge-2012/1.0.0/)
- **Direct Files:**
  - `set-a.tar.gz` (or `set-a.zip`) — 4,000 training records
  - `set-b.tar.gz` (or `set-b.zip`) — 4,000 validation records
  - `set-c.tar.gz` — 4,000 test records
  - `Outcomes-a.txt`, `Outcomes-b.txt`, `Outcomes-c.txt` — Mortality labels and SAPS-I scores

### Step 2: Placement & Directory Structure
Extract the downloaded archives into the project root directory under a `Dataset/` folder:

```text
d:\ICU_Prediction\
├── Dataset/
│   ├── set-a/                     # 4,000 patient records (132539.txt to 142673.txt)
│   ├── set-b/                     # 4,000 patient records (142675.txt to 152864.txt)
│   ├── set-c/                     # 4,000 patient records (152871.txt to 163037.txt)
│   ├── Outcomes-a.txt             # Ground truth outcomes for Set A
│   ├── Outcomes-b.txt             # Ground truth outcomes for Set B
│   └── Outcomes-c.txt             # Ground truth outcomes for Set C
├── data/
│   ├── README.md                  # This file
│   └── processed/
│       ├── feature_names.json     # Feature name catalog (334 features)
│       ├── forward_fill/
│       │   ├── scaler_static.joblib      # StandardScaler for demographic descriptors
│       │   └── scaler_temporal.joblib    # StandardScaler for time-series variables
│       ├── simple_imputation/
│       │   ├── scaler_static.joblib
│       │   └── scaler_temporal.joblib
│       └── tabular/
│           └── tabular_feature_metadata.json
```

---

## 4. Telemetry Record Format

Each patient file is a comma-separated flat text file (`<RecordID>.txt`) with the following format:

```text
Time,Parameter,Value
00:00,RecordID,132539
00:00,Age,54
00:00,Gender,0
00:00,Height,-1
00:00,Weight,-1
00:00,ICUType,2
00:07,HR,73
00:07,NIDiasABP,65
00:07,NIMAP,78
00:07,NISysABP,106
00:07,RespRate,19
00:07,SaO2,97
...
```

### Static General Descriptors:
- `RecordID`: Unique integer patient record identifier.
- `Age`: Patient age in years (values $\ge 90$ are mapped to 91.4 in compliance with HIPAA).
- `Gender`: Biological sex (0 = Female, 1 = Male, -1 = Missing).
- `Height`: Height in centimeters (-1 if unrecorded).
- `Weight`: Weight in kilograms (-1 if unrecorded).
- `ICUType`: 1 = Coronary Care Unit, 2 = Cardiac Surgery Recovery Unit, 3 = Medical ICU, 4 = Surgical ICU.

### Longitudinal Time-Series Variables (37 Physiological Biomarkers):
- **Vital Signs:** Heart Rate (`HR`), Invasive/Non-invasive Blood Pressure (`SysABP`, `DiasABP`, `MAP`, `NISysABP`, `NIDiasABP`, `NIMAP`), Respiratory Rate (`RespRate`), Core Temperature (`Temp`), Oxygen Saturation (`SaO2`).
- **Neurological:** Glasgow Coma Scale (`GCS`).
- **Renal & Metabolic:** Urine Output (`Urine`), Serum Creatinine (`Creatinine`), Blood Urea Nitrogen (`BUN`), Glucose, Sodium (`Na`), Potassium (`K`).
- **Hematology:** White Blood Cell Count (`WBC`), Platelets, Hematocrit (`HCT`).
- **Arterial Blood Gas & Ventilation:** Arterial pH (`pH`), Arterial $\text{PaO}_2$, Arterial $\text{PaCO}_2$, $\text{FiO}_2$, Serum Lactate, Serum Bicarbonate (`HCO3`).

---

## 5. Preprocessing & Feature Engineering Execution

If you wish to re-generate the full tabular and longitudinal dataset matrices from the raw files:

```bash
# 1. Run full dataset inspection and audit
python scripts/inspect_dataset.py

# 2. Run Phase 2 preprocessing and tensor extraction
python scripts/run_phase2_preprocessing.py

# 3. Verify output tensor shapes and scaler integrity
python scripts/verify_phase2_outputs.py
```

### Generated Artifacts in `data/processed/`:
1. **Longitudinal Tensors:** `(N, 48, 37)` or `(N, 24, 37)` tensors with associated binary observation masks.
2. **Tabular Feature Matrix:** `(N, 334)` matrix containing first, last, min, max, mean, standard deviation, and trajectory slope features across all dynamic variables.
3. **Fitted Scalers:** Pre-trained zero-leakage `StandardScaler` instances fit exclusively on training folds.
