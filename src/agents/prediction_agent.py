"""
Prediction Agent (Phase 6.2)
Agentic AI-Assisted ICU Mortality Risk Prediction from the First 48 Hours.

Responsibilities:
1. Receive validated patient data from multiple input modalities (Patient ID, file path, DataFrame, raw text).
2. Connect directly to the Data Quality Agent (Phase 6.1) to audit telemetry integrity and DQI.
3. Enforce quality-driven routing rules:
   - DQI >= 85.0: Continue with prediction.
   - 70.0 <= DQI < 85.0: Continue with telemetry quality warning.
   - 50.0 <= DQI < 70.0: Flag substantial uncertainty due to sparse or degraded observations.
   - DQI < 50.0: Block automated prediction; refuse to output fabricated risk scores.
4. Execute exact, reproducible preprocessing transformations matching Phase 2-5 pipelines:
   - Replicate temporal tensor construction (1, T, 74) with historical forward-fill and training-set medians.
   - Replicate static descriptor encoding and scaling (1, 9) with training-set imputation.
   - Replicate 334 patient-level tabular features in identical order.
   - Use saved training-fitted StandardScaler artifacts (zero refitting on new data).
5. Load and execute actual trained machine learning models:
   - Primary Model: XGBoost (Tabular Gradient Boosted Decision Trees).
   - Secondary Model: LSTM (Bidirectional Sequential Neural Network).
   - Comparative Models: 1D CNN, Simple RNN (available on-demand).
6. Support dual prediction horizons:
   - 48-Hour Standard ICU Horizon (observations <= 48:00).
   - 24-Hour Early Triage Horizon (observations <= 24:00, strictly preventing future leakage).
7. Apply validation-selected decision thresholds:
   - 48-Hour: XGBoost = 0.58, LSTM = 0.58.
   - 24-Hour: XGBoost = 0.57, LSTM = 0.66.
8. Return structured, auditable prediction schemas with explicit limitations and clinical warnings.

RESEARCH DISCLAIMER:
Predictions and thresholds are validation-selected research metrics developed on the retrospective
PhysioNet 2012 / MIMIC-II cohort. They are intended strictly for clinical decision support research
and MUST NOT be used as autonomous diagnostic or treatment criteria.
"""

import os
import sys
import glob
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

# Set environment variables for PyTorch / Keras backend before imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import keras

# Project internal imports
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import DYNAMIC_PARAMS, STATIC_PARAMS, parse_time_to_minutes
from src.preprocessing import VALID_RANGES, clean_observation, get_hour_bin
from src.feature_engineering import extract_patient_tabular_features
from src.agents.data_quality_agent import DataQualityAgent

logger = logging.getLogger("PredictionAgent")


class PredictionAgent:
    """
    Autonomous Prediction Agent for ICU Mortality Risk Assessment.
    Executes trained XGBoost and LSTM models on preprocessed patient telemetry
    after rigorous Data Quality verification.
    """

    # Static admission descriptor training statistics (from Phase 2 / Set A Training split)
    STATIC_IMPUTE_DEFAULTS = {
        "Age": 64.426875,
        "Height": 169.82228125,
        "Weight_admission": 81.56295313,
        "Gender": 1.0,  # Mode: Male (1)
        "ICUType": 3.0,  # Mode: MedICU (3)
    }

    # Validation-selected decision thresholds (optimized for F1-score on Validation Cohort)
    DEFAULT_THRESHOLDS = {
        48: {
            "xgboost": 0.58,
            "lstm": 0.58,
        },
        24: {
            "xgboost": 0.57,
            "lstm": 0.66,
        },
    }

    # Standard model and artifact relative paths
    MODEL_PATHS = {
        48: {
            "xgboost": os.path.join("models", "xgboost", "best_xgboost_forward_fill_48h.json"),
            "lstm": os.path.join("models", "deep_learning", "lstm_48h_forward_fill.keras"),
        },
        24: {
            "xgboost": os.path.join("models", "xgboost", "best_xgboost_forward_fill_24h.json"),
            "lstm": os.path.join("models", "deep_learning", "lstm_24h_forward_fill.keras"),
        },
    }

    # Scaler artifact paths
    SCALER_PATHS = {
        "temporal": os.path.join("data", "processed", "forward_fill", "scaler_temporal.joblib"),
        "static": os.path.join("data", "processed", "forward_fill", "scaler_static.joblib"),
        "imputation_medians": os.path.join("reports", "imputation_comparison.csv"),
    }

    def __init__(
        self,
        base_dir: Optional[str] = None,
        data_quality_agent: Optional[DataQualityAgent] = None,
        preload_models: bool = False,
    ):
        """
        Initializes the Prediction Agent.

        Parameters:
            base_dir: Root directory of the project (default: auto-detected project root).
            data_quality_agent: Optional pre-instantiated DataQualityAgent instance.
            preload_models: If True, eagerly loads 48h and 24h models and scalers into memory.
        """
        self.base_dir = str(base_dir or ROOT_DIR)
        self.dq_agent = data_quality_agent or DataQualityAgent()

        # Cache for loaded models and preprocessing artifacts
        self._models: Dict[str, Any] = {}
        self._scalers: Dict[str, Any] = {}
        self._train_medians: Optional[Dict[str, float]] = None
        self._thresholds: Dict[int, Dict[str, float]] = {}

        # Initialize threshold configurations from disk or defaults
        self._load_threshold_configurations()

        if preload_models:
            self._preload_all_artifacts()

    # =========================================================================
    # 1. ARTIFACT & MODEL MANAGEMENT
    # =========================================================================

    def _resolve_path(self, rel_path: str) -> str:
        """Resolves relative path against project base directory."""
        if os.path.isabs(rel_path):
            return rel_path
        return os.path.join(self.base_dir, rel_path)

    def _load_threshold_configurations(self):
        """Loads optimal decision thresholds from reports or falls back to validated defaults."""
        self._thresholds = {
            48: dict(self.DEFAULT_THRESHOLDS[48]),
            24: dict(self.DEFAULT_THRESHOLDS[24]),
        }

        # Attempt to load exact thresholds from saved JSON reports if available
        xgb_report_path = self._resolve_path(os.path.join("reports", "xgboost_results.json"))
        if os.path.exists(xgb_report_path):
            try:
                with open(xgb_report_path, "r", encoding="utf-8") as f:
                    xgb_res = json.load(f)
                th_48 = (
                    xgb_res.get("primary_model_performance", {})
                    .get("validation", {})
                    .get("optimal_threshold", {})
                    .get("threshold")
                )
                if th_48 is not None:
                    self._thresholds[48]["xgboost"] = float(th_48)
            except Exception as e:
                logger.warning(f"Could not parse XGBoost results report: {e}")

        dl_report_path = self._resolve_path(os.path.join("reports", "deep_learning_results.json"))
        if os.path.exists(dl_report_path):
            try:
                with open(dl_report_path, "r", encoding="utf-8") as f:
                    dl_res = json.load(f)
                exps = dl_res.get("experiments", {})
                if "LSTM (48h FF)" in exps and "optimal_threshold" in exps["LSTM (48h FF)"]:
                    self._thresholds[48]["lstm"] = float(exps["LSTM (48h FF)"]["optimal_threshold"])
                if "LSTM (24h FF)" in exps and "optimal_threshold" in exps["LSTM (24h FF)"]:
                    self._thresholds[24]["lstm"] = float(exps["LSTM (24h FF)"]["optimal_threshold"])
            except Exception as e:
                logger.warning(f"Could not parse Deep Learning results report: {e}")

    def _load_training_medians(self) -> Dict[str, float]:
        """Loads the training-set medians used for leading missing value imputation."""
        if self._train_medians is not None:
            return self._train_medians

        csv_path = self._resolve_path(self.SCALER_PATHS["imputation_medians"])
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Training medians file not found at: {csv_path}. "
                "Ensure Phase 2 preprocessing report 'reports/imputation_comparison.csv' exists."
            )

        df_imp = pd.read_csv(csv_path)
        medians = dict(zip(df_imp["Feature"], df_imp["Train_Median"]))

        # Confirm all 37 dynamic parameters are present
        for param in DYNAMIC_PARAMS:
            if param not in medians:
                if param == "MechVent":
                    medians[param] = 0.0
                else:
                    raise KeyError(f"Missing required training median for parameter: {param}")

        self._train_medians = medians
        return self._train_medians

    def _load_scalers(self) -> Tuple[Any, Any]:
        """Loads fitted StandardScaler objects for temporal and static features."""
        if "temporal" in self._scalers and "static" in self._scalers:
            return self._scalers["temporal"], self._scalers["static"]

        temp_path = self._resolve_path(self.SCALER_PATHS["temporal"])
        stat_path = self._resolve_path(self.SCALER_PATHS["static"])

        if not os.path.exists(temp_path):
            raise FileNotFoundError(
                f"Temporal scaler not found at: {temp_path}. "
                "Ensure Phase 2 preprocessing artifact 'scaler_temporal.joblib' exists."
            )
        if not os.path.exists(stat_path):
            raise FileNotFoundError(
                f"Static scaler not found at: {stat_path}. "
                "Ensure Phase 2 preprocessing artifact 'scaler_static.joblib' exists."
            )

        self._scalers["temporal"] = joblib.load(temp_path)
        self._scalers["static"] = joblib.load(stat_path)
        return self._scalers["temporal"], self._scalers["static"]

    def _load_model(self, model_type: str, horizon_hours: int) -> Any:
        """
        Loads a specific trained model (XGBoost or LSTM) for the requested horizon.
        Caches models in memory for fast subsequent inferences.
        """
        cache_key = f"{model_type}_{horizon_hours}h"
        if cache_key in self._models:
            return self._models[cache_key]

        if horizon_hours not in self.MODEL_PATHS:
            raise ValueError(f"Unsupported prediction horizon: {horizon_hours} hours. Supported: [24, 48].")

        rel_path = self.MODEL_PATHS[horizon_hours].get(model_type)
        if not rel_path:
            raise ValueError(f"Unknown model type: '{model_type}'. Supported: ['xgboost', 'lstm'].")

        full_path = self._resolve_path(rel_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(
                f"Trained {model_type.upper()} model file missing at: {full_path}. "
                f"Verify that model checkpoint exists before running predictions."
            )

        if model_type == "xgboost":
            model = xgb.Booster()
            model.load_model(full_path)
            self._models[cache_key] = model
            return model

        elif model_type == "lstm":
            # Load Keras model with Torch backend
            model = keras.models.load_model(full_path)
            self._models[cache_key] = model
            return model

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _preload_all_artifacts(self):
        """Preloads all models, scalers, and imputation medians."""
        self._load_training_medians()
        self._load_scalers()
        for h in [48, 24]:
            self._load_model("xgboost", h)
            self._load_model("lstm", h)

    # =========================================================================
    # 2. PATIENT DATA INGESTION & PARSING
    # =========================================================================

    def _locate_patient_file(self, patient_id: Union[int, str]) -> str:
        """
        Locates a PhysioNet patient record file by RecordID across available partitions.
        Searches Set A, Set B, and Set C.
        """
        pid_str = str(patient_id).strip()
        candidates = [
            os.path.join(self.base_dir, "Dataset", "set-a", f"{pid_str}.txt"),
            os.path.join(self.base_dir, "Dataset", "set-b", f"{pid_str}.txt"),
            os.path.join(self.base_dir, "Dataset", "set-c", f"{pid_str}.txt"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            f"Patient record '{pid_str}.txt' could not be located in Dataset/set-a, set-b, or set-c."
        )

    def parse_patient_input(
        self, patient_input: Union[int, str, pd.DataFrame, List[str]]
    ) -> Tuple[str, pd.DataFrame]:
        """
        Normalizes diverse patient input representations into a standard DataFrame:
        Columns: ['Time', 'Parameter', 'Value']

        Supported inputs:
        A. Patient RecordID (int or numeric string) -> resolved from dataset directory.
        B. Path to patient .txt file.
        C. Raw CSV/text content string.
        D. List of raw file lines.
        E. Pre-validated pandas DataFrame.

        Returns:
            record_id_str: Extracted patient ID string.
            df_patient: Parsed DataFrame with ['Time', 'Parameter', 'Value'].
        """
        # Case 1: In-memory pandas DataFrame
        if isinstance(patient_input, pd.DataFrame):
            df = patient_input.copy()
            req_cols = {"Time", "Parameter", "Value"}
            if not req_cols.issubset(set(df.columns)):
                raise ValueError(
                    f"Invalid patient DataFrame format. Required columns: {req_cols}, Found: {list(df.columns)}"
                )
            # Extract record_id
            m_rec = df[df["Parameter"] == "RecordID"]
            if not m_rec.empty:
                rec_id = str(int(m_rec.iloc[0]["Value"]))
            else:
                rec_id = "UNKNOWN_PATIENT"
            return rec_id, df

        # Case 2: Numeric patient ID
        if isinstance(patient_input, int) or (isinstance(patient_input, str) and patient_input.strip().isdigit()):
            file_path = self._locate_patient_file(patient_input)
            return self._parse_file(file_path)

        # Case 3: File path string
        if isinstance(patient_input, str):
            candidate_path = self._resolve_path(patient_input)
            if os.path.exists(candidate_path) and os.path.isfile(candidate_path):
                return self._parse_file(candidate_path)
            # If not a file on disk, attempt parsing as raw multiline string
            if "\n" in patient_input or "," in patient_input:
                lines = patient_input.strip().splitlines()
                return self._parse_lines(lines, source_name="Raw_String_Input")
            raise FileNotFoundError(f"Input file path does not exist: {patient_input}")

        # Case 4: List of lines
        if isinstance(patient_input, list):
            return self._parse_lines(patient_input, source_name="Lines_Input")

        raise TypeError(f"Unsupported patient input type: {type(patient_input)}")

    def _parse_file(self, file_path: str) -> Tuple[str, pd.DataFrame]:
        """Reads and parses an official PhysioNet patient record text file."""
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        rec_id_fallback = os.path.basename(file_path).replace(".txt", "")
        return self._parse_lines(lines, source_name=rec_id_fallback)

    def _parse_lines(self, lines: List[str], source_name: str) -> Tuple[str, pd.DataFrame]:
        """Parses raw patient text lines into a structured DataFrame."""
        records = []
        rec_id = source_name

        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 3:
                continue

            t_str, param, val_str = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if not param or (line_idx == 0 and t_str.lower() == "time"):
                continue

            try:
                val = float(val_str)
            except ValueError:
                val = np.nan

            if param == "RecordID" and not np.isnan(val) and val > 0:
                rec_id = str(int(val))

            records.append({"Time": t_str, "Parameter": param, "Value": val})

        df = pd.DataFrame(records)
        return rec_id, df

    # =========================================================================
    # 3. REPRODUCIBLE PREPROCESSING PIPELINE
    # =========================================================================

    def preprocess_patient(
        self, df_patient: pd.DataFrame, horizon_hours: int = 48
    ) -> Dict[str, Any]:
        """
        Executes exact preprocessing transformations matching Phase 2-5 pipelines:
        1. Filters observations strictly to horizon_hours (24h or 48h; zero temporal leakage).
        2. Bins observations into discrete hourly grid [0..T-1] using Phase 2 conventions.
        3. Averages continuous measurements; max-aggregates MechVent (1.0 or 0.0).
        4. Imputes missing observations via Historical Forward-Fill (LOCF):
           - MechVent is unforward-filled (unrecorded hours remain 0.0 with mask 0.0).
           - Continuous leading missing hours filled with training medians.
        5. Scales continuous temporal signals using saved training StandardScaler.
        6. Constructs integrated temporal tensor: (1, T, 74) [37 values + 37 observation masks].
        7. Extracts and scales static descriptors: (1, 9) [3 scaled continuous + 6 binary].
        8. Extracts 334 patient-level tabular features for XGBoost in exact training order.

        Returns:
            Dict containing:
                'temporal_tensor': np.ndarray of shape (1, T, 74)
                'static_tensor': np.ndarray of shape (1, 9)
                'tabular_features': np.ndarray of shape (1, 334)
                'feature_names': list of 334 feature names
                'cleaning_log': dict of cleaning and outlier filtering counts
                'ever_mechvent': float (1.0 or 0.0)
        """
        if horizon_hours not in [24, 48]:
            raise ValueError(f"Unsupported prediction horizon: {horizon_hours}h. Supported: [24, 48].")

        max_minutes = horizon_hours * 60
        num_features = len(DYNAMIC_PARAMS)
        param_to_idx = {p: i for i, p in enumerate(DYNAMIC_PARAMS)}
        mechvent_idx = param_to_idx["MechVent"]
        cont_indices = [i for i in range(num_features) if i != mechvent_idx]

        train_medians = self._load_training_medians()
        scaler_temp, scaler_stat = self._load_scalers()

        cleaning_log = {
            "outliers_filtered": {p: 0 for p in DYNAMIC_PARAMS},
            "invalid_timestamps_dropped": 0,
        }

        # -------------------------------------------------------------
        # A. EXTRACT STATIC ADMISSION DESCRIPTORS
        # -------------------------------------------------------------
        static_raw = {}
        for p in ["Age", "Gender", "Height", "ICUType"]:
            m = df_patient[df_patient["Parameter"] == p]
            if not m.empty:
                val = m.iloc[0]["Value"]
                # -1 represents standard PhysioNet missing sentinel
                static_raw[p] = np.nan if (val == -1 or val < 0 or np.isnan(val)) else float(val)
            else:
                static_raw[p] = np.nan

        # Weight admission: look for Weight at 00:00 or first available positive weight
        m_w = df_patient[df_patient["Parameter"] == "Weight"]
        w_adm = np.nan
        if not m_w.empty:
            m_w0 = m_w[m_w["Time"] == "00:00"]
            if not m_w0.empty and m_w0.iloc[0]["Value"] > 0:
                w_adm = float(m_w0.iloc[0]["Value"])
            else:
                valid_weights = m_w[m_w["Value"] > 0]
                if not valid_weights.empty:
                    w_adm = float(valid_weights.iloc[0]["Value"])
        static_raw["Weight_admission"] = w_adm

        # Clean static continuous variables with VALID_RANGES
        for p in ["Age", "Height", "Weight_admission"]:
            if not np.isnan(static_raw[p]):
                static_raw[p] = clean_observation(p, static_raw[p], cleaning_log)

        # -------------------------------------------------------------
        # B. BUILD HOURLY GRID & OBSERVATION MASKS
        # -------------------------------------------------------------
        hourly_buckets: Dict[Tuple[int, int], List[float]] = {}
        ever_mv = 0.0

        for _, row in df_patient.iterrows():
            param = row["Parameter"]
            val = row["Value"]
            t_str = str(row["Time"]).strip()

            try:
                minutes = parse_time_to_minutes(t_str)
            except Exception:
                cleaning_log["invalid_timestamps_dropped"] += 1
                continue

            # Strict horizon enforcement: zero future leakage!
            if minutes < 0 or minutes > max_minutes:
                continue

            # Check patient-level MechVent indicator within horizon
            if param == "MechVent" and val == 1.0:
                ever_mv = 1.0

            if param not in param_to_idx:
                continue

            clean_val = clean_observation(param, val, cleaning_log)
            if np.isnan(clean_val):
                continue

            hour = get_hour_bin(minutes)
            if hour is None or hour >= horizon_hours:
                continue

            p_idx = param_to_idx[param]
            if (hour, p_idx) not in hourly_buckets:
                hourly_buckets[(hour, p_idx)] = []
            hourly_buckets[(hour, p_idx)].append(clean_val)

        grid = np.full((1, horizon_hours, num_features), np.nan, dtype=np.float32)
        mask = np.zeros((1, horizon_hours, num_features), dtype=np.float32)

        for (hour, p_idx), val_list in hourly_buckets.items():
            param_name = DYNAMIC_PARAMS[p_idx]
            if param_name == "MechVent":
                grid[0, hour, p_idx] = float(np.max(val_list))
            else:
                grid[0, hour, p_idx] = float(np.mean(val_list))
            mask[0, hour, p_idx] = 1.0

        # -------------------------------------------------------------
        # C. HISTORICAL FORWARD-FILL (LOCF) IMPUTATION
        # -------------------------------------------------------------
        imputed = grid.copy()
        for f in range(num_features):
            param_name = DYNAMIC_PARAMS[f]
            if f == mechvent_idx:
                # MechVent: unobserved hours remain 0.0 with mask 0.0 (no forward fill)
                nan_m = np.isnan(imputed[0, :, f])
                imputed[0, nan_m, f] = 0.0
                continue

            last_known = np.nan
            for t in range(horizon_hours):
                val = imputed[0, t, f]
                if not np.isnan(val):
                    last_known = val
                else:
                    if not np.isnan(last_known):
                        imputed[0, t, f] = last_known
                    else:
                        # Leading missing value -> fill with training-set population median
                        imputed[0, t, f] = train_medians[param_name]

        # -------------------------------------------------------------
        # D. TEMPORAL SCALING & TENSOR FORMATION
        # -------------------------------------------------------------
        # Standardize 36 continuous temporal features (MechVent remains binary {0, 1})
        t_cont = imputed[:, :, cont_indices].reshape(-1, len(cont_indices))
        t_cont_scaled = scaler_temp.transform(t_cont).reshape(1, horizon_hours, len(cont_indices))

        X_temp_37 = np.zeros_like(imputed, dtype=np.float32)
        X_temp_37[:, :, cont_indices] = t_cont_scaled
        X_temp_37[:, :, [mechvent_idx]] = imputed[:, :, [mechvent_idx]]

        # Integrated temporal tensor: (1, T, 74)
        temporal_tensor = np.concatenate([X_temp_37, mask], axis=-1).astype(np.float32)

        # -------------------------------------------------------------
        # E. STATIC DESCRIPTOR SCALING & ENCODING (1, 9)
        # -------------------------------------------------------------
        # Impute missing static descriptors with training defaults
        age_imp = static_raw["Age"] if not np.isnan(static_raw["Age"]) else self.STATIC_IMPUTE_DEFAULTS["Age"]
        hgt_imp = (
            static_raw["Height"]
            if not np.isnan(static_raw["Height"])
            else self.STATIC_IMPUTE_DEFAULTS["Height"]
        )
        wgt_imp = (
            static_raw["Weight_admission"]
            if not np.isnan(static_raw["Weight_admission"])
            else self.STATIC_IMPUTE_DEFAULTS["Weight_admission"]
        )

        cont_stat_scaled = scaler_stat.transform([[age_imp, hgt_imp, wgt_imp]])  # (1, 3)

        # Binary Gender
        gender_val = (
            static_raw["Gender"]
            if not np.isnan(static_raw["Gender"])
            else self.STATIC_IMPUTE_DEFAULTS["Gender"]
        )
        gender_arr = np.array([[gender_val]], dtype=np.float32)  # (1, 1)

        # Binary MechVent_ever_recorded
        mv_ever_arr = np.array([[ever_mv]], dtype=np.float32)  # (1, 1)

        # One-Hot Encoded ICUType: 1=CCU, 2=CSRU, 3=MedICU, 4=SurgICU
        icu_val = (
            int(static_raw["ICUType"])
            if not np.isnan(static_raw["ICUType"])
            else int(self.STATIC_IMPUTE_DEFAULTS["ICUType"])
        )
        icu_ohe = np.zeros((1, 4), dtype=np.float32)
        if 1 <= icu_val <= 4:
            icu_ohe[0, icu_val - 1] = 1.0
        else:
            icu_ohe[0, 2] = 1.0  # Default to MedICU (3)

        # Full static array: (1, 9)
        static_tensor = np.hstack([cont_stat_scaled, gender_arr, mv_ever_arr, icu_ohe]).astype(np.float32)

        # -------------------------------------------------------------
        # F. TABULAR FEATURE EXTRACTION FOR XGBOOST (1, 334)
        # -------------------------------------------------------------
        X_tabular, feat_names = extract_patient_tabular_features(X_temp_37, mask, static_tensor)

        return {
            "temporal_tensor": temporal_tensor,
            "static_tensor": static_tensor,
            "tabular_features": X_tabular,
            "feature_names": feat_names,
            "cleaning_log": cleaning_log,
            "ever_mechvent": ever_mv,
            "descriptors_raw": static_raw,
        }

    # =========================================================================
    # 4. PREDICTION EXECUTION & ROUTING LOGIC
    # =========================================================================

    def predict_patient(
        self,
        patient_input: Union[int, str, pd.DataFrame, List[str]],
        prediction_horizon_hours: int = 48,
    ) -> Dict[str, Any]:
        """
        End-to-End Prediction Workflow:
        Step 1: Ingest and parse patient telemetry.
        Step 2: Execute Data Quality Agent audit and verify DQI.
        Step 3: Apply quality routing rules (block prediction if DQI < 50).
        Step 4: Execute standardized preprocessing (temporal, static, tabular).
        Step 5: Generate XGBoost prediction (Primary Model).
        Step 6: Generate LSTM prediction (Secondary Model).
        Step 7: Compare against validation-selected classification thresholds.
        Step 8: Return structured diagnostic results with explicit limitations.

        Parameters:
            patient_input: Patient ID, file path, DataFrame, or raw record lines.
            prediction_horizon_hours: 48 (default) or 24 hours.

        Returns:
            Structured prediction dictionary matching project schema.
        """
        warnings_list: List[str] = []
        limitations_list: List[str] = [
            "Research-only risk predictions; not approved for autonomous clinical diagnosis or treatment guidance.",
            f"Decision thresholds (XGBoost {prediction_horizon_hours}h: {self._thresholds.get(prediction_horizon_hours, {}).get('xgboost', 0.58):.2f}, "
            f"LSTM {prediction_horizon_hours}h: {self._thresholds.get(prediction_horizon_hours, {}).get('lstm', 0.58):.2f}) "
            "are validation-selected research cutoffs optimizing F1-score, not clinical risk categories.",
            "XGBoost serves as the primary tabular predictive model; LSTM provides an independent sequential trajectory comparison.",
            "Predictions from XGBoost and LSTM must NOT be averaged without a validated prospective ensemble.",
            "Evaluated retrospectively on the PhysioNet 2012 / MIMIC-II cohort; prospective generalizability has not been established.",
        ]

        # ---------------------------------------------------------
        # STEP 1: PARSE PATIENT INPUT
        # ---------------------------------------------------------
        try:
            record_id, df_patient = self.parse_patient_input(patient_input)
        except Exception as e:
            return {
                "patient_id": "UNKNOWN",
                "status": "ERROR_INPUT_PARSING",
                "prediction_horizon_hours": prediction_horizon_hours,
                "data_quality": {
                    "dqi": 0.0,
                    "category": "UNRELIABLE_CRITICAL",
                    "status": "FAIL",
                },
                "predictions": {},
                "warnings": [f"Input parsing failed: {str(e)}"],
                "limitations": limitations_list,
            }

        if df_patient.empty:
            return {
                "patient_id": record_id,
                "status": "ERROR_EMPTY_RECORD",
                "prediction_horizon_hours": prediction_horizon_hours,
                "data_quality": {
                    "dqi": 0.0,
                    "category": "UNRELIABLE_CRITICAL",
                    "status": "FAIL",
                },
                "predictions": {},
                "warnings": ["Patient record is completely empty. No observations available."],
                "limitations": limitations_list,
            }

        # ---------------------------------------------------------
        # STEP 2 & 3: CALL DATA QUALITY AGENT & ENFORCE ROUTING RULES
        # ---------------------------------------------------------
        try:
            dq_audit = self.dq_agent.audit_dataframe(
                df_patient,
                file_name=f"{record_id}.txt",
                prediction_horizon_hours=prediction_horizon_hours,
            )
        except Exception as e:
            return {
                "patient_id": record_id,
                "status": "ERROR_DATA_QUALITY_AUDIT",
                "prediction_horizon_hours": prediction_horizon_hours,
                "data_quality": {
                    "dqi": 0.0,
                    "category": "UNRELIABLE_CRITICAL",
                    "status": "FAIL",
                },
                "predictions": {},
                "warnings": [f"Data Quality Agent crashed during execution: {str(e)}"],
                "limitations": limitations_list,
            }

        dqi = float(dq_audit.get("data_quality_index", 0.0))
        dq_category = dq_audit.get("quality_grade", "UNKNOWN")
        dq_status = dq_audit.get("status", "FAIL")

        # Routing Rule: DQI < 50.0 -> BLOCK PREDICTION
        if dqi < 50.0 or dq_status == "CRITICAL_FAIL":
            reasons = dq_audit.get("descriptor_issues", []) + [
                f"DQI score {dqi:.1f} is below the critical threshold (50.0)."
            ]
            if dq_audit.get("missing_descriptors"):
                reasons.append(f"Missing descriptors: {', '.join(dq_audit['missing_descriptors'])}")
            recs = dq_audit.get("recommendations", ["Re-fetch telemetry."])

            return {
                "patient_id": record_id,
                "status": "BLOCKED_BY_DATA_QUALITY",
                "prediction_horizon_hours": prediction_horizon_hours,
                "data_quality": {
                    "dqi": round(dqi, 2),
                    "category": dq_category,
                    "status": dq_status,
                },
                "predictions": {},
                "warnings": [
                    "Automated prediction blocked due to critical data quality failure.",
                    *reasons,
                ],
                "limitations": [
                    "Automated mortality risk prediction is inhibited when input telemetry fails minimum clinical data quality standards.",
                    *limitations_list,
                ],
                "audit_recommendations": recs,
            }

        # Routing Rule: 50.0 <= DQI < 70.0 -> Flag substantial uncertainty
        elif 50.0 <= dqi < 70.0:
            warnings_list.append(
                f"SUBOPTIMAL_DATA_QUALITY: DQI={dqi:.1f} ({dq_category}). "
                "Substantial prediction uncertainty exists due to sparse observations, missing vital domains, or degraded telemetry."
            )
            if dq_audit.get("missing_descriptors"):
                warnings_list.append(f"Missing descriptors: {', '.join(dq_audit['missing_descriptors'])}")

        # Routing Rule: 70.0 <= DQI < 85.0 -> Data quality warning
        elif 70.0 <= dqi < 85.0:
            warnings_list.append(
                f"ACCEPTABLE_WITH_WARNING: DQI={dqi:.1f} ({dq_category}). "
                "Some non-critical descriptors or physiological measurements are absent or sparse."
            )
            if dq_audit.get("missing_descriptors"):
                warnings_list.append(f"Missing descriptors: {', '.join(dq_audit['missing_descriptors'])}")

        # Include any extreme pathological alerts from Data Quality Agent
        pathological_alerts = dq_audit.get("pathological_alerts", [])
        if pathological_alerts:
            for alert in pathological_alerts[:3]:  # Top 3 most critical
                warnings_list.append(
                    f"CRITICAL_PHYSIOLOGY: {alert.get('parameter')}={alert.get('value')} ({alert.get('description')})"
                )

        # ---------------------------------------------------------
        # STEP 4: PREPROCESSING
        # ---------------------------------------------------------
        try:
            prep_data = self.preprocess_patient(df_patient, horizon_hours=prediction_horizon_hours)
        except Exception as e:
            return {
                "patient_id": record_id,
                "status": "ERROR_PREPROCESSING",
                "prediction_horizon_hours": prediction_horizon_hours,
                "data_quality": {
                    "dqi": round(dqi, 2),
                    "category": dq_category,
                    "status": dq_status,
                },
                "predictions": {},
                "warnings": [f"Preprocessing failed: {str(e)}", *warnings_list],
                "limitations": limitations_list,
            }

        temporal_tensor = prep_data["temporal_tensor"]
        static_tensor = prep_data["static_tensor"]
        tabular_features = prep_data["tabular_features"]

        # Validate tensor shapes
        expected_temporal_shape = (1, prediction_horizon_hours, 74)
        expected_static_shape = (1, 9)
        expected_tabular_shape = (1, 334)

        if temporal_tensor.shape != expected_temporal_shape:
            raise ValueError(
                f"Incorrect temporal tensor shape: {temporal_tensor.shape}, expected: {expected_temporal_shape}"
            )
        if static_tensor.shape != expected_static_shape:
            raise ValueError(
                f"Incorrect static tensor shape: {static_tensor.shape}, expected: {expected_static_shape}"
            )
        if tabular_features.shape != expected_tabular_shape:
            raise ValueError(
                f"Incorrect tabular features shape: {tabular_features.shape}, expected: {expected_tabular_shape}"
            )

        # ---------------------------------------------------------
        # STEP 5: PRIMARY MODEL PREDICTION (XGBOOST)
        # ---------------------------------------------------------
        try:
            xgb_model = self._load_model("xgboost", prediction_horizon_hours)
            dmatrix = xgb.DMatrix(tabular_features)
            p_xgb = float(xgb_model.predict(dmatrix)[0])
        except Exception as e:
            return {
                "patient_id": record_id,
                "status": "ERROR_XGBOOST_INFERENCE",
                "prediction_horizon_hours": prediction_horizon_hours,
                "data_quality": {
                    "dqi": round(dqi, 2),
                    "category": dq_category,
                    "status": dq_status,
                },
                "predictions": {},
                "warnings": [f"XGBoost model execution failed: {str(e)}", *warnings_list],
                "limitations": limitations_list,
            }

        # ---------------------------------------------------------
        # STEP 6: SECONDARY MODEL PREDICTION (LSTM)
        # ---------------------------------------------------------
        try:
            lstm_model = self._load_model("lstm", prediction_horizon_hours)
            p_lstm = float(lstm_model.predict([temporal_tensor, static_tensor], verbose=0)[0][0])
        except Exception as e:
            return {
                "patient_id": record_id,
                "status": "ERROR_LSTM_INFERENCE",
                "prediction_horizon_hours": prediction_horizon_hours,
                "data_quality": {
                    "dqi": round(dqi, 2),
                    "category": dq_category,
                    "status": dq_status,
                },
                "predictions": {},
                "warnings": [f"LSTM model execution failed: {str(e)}", *warnings_list],
                "limitations": limitations_list,
            }

        # ---------------------------------------------------------
        # STEP 7: THRESHOLD RETRIEVAL & CLASSIFICATION
        # ---------------------------------------------------------
        th_xgb = self._thresholds.get(prediction_horizon_hours, {}).get("xgboost", 0.58)
        th_lstm = self._thresholds.get(prediction_horizon_hours, {}).get("lstm", 0.58)

        decision_xgb = "ABOVE_THRESHOLD" if p_xgb >= th_xgb else "BELOW_THRESHOLD"
        decision_lstm = "ABOVE_THRESHOLD" if p_lstm >= th_lstm else "BELOW_THRESHOLD"

        # Check for model divergence (e.g. one above threshold, one below)
        if decision_xgb != decision_lstm:
            warnings_list.append(
                f"MODEL_DISCORDANCE: Primary XGBoost ({decision_xgb}, p={p_xgb:.4f}) and Secondary LSTM "
                f"({decision_lstm}, p={p_lstm:.4f}) diverge across decision thresholds. "
                "Examine underlying physiological trajectories and tabular feature importance."
            )

        # ---------------------------------------------------------
        # STEP 8: ASSEMBLE STRUCTURED PREDICTION RESPONSE
        # ---------------------------------------------------------
        response = {
            "patient_id": str(record_id),
            "status": "SUCCESS",
            "prediction_horizon_hours": prediction_horizon_hours,
            "data_quality": {
                "dqi": round(dqi, 2),
                "category": dq_category,
                "status": dq_status,
            },
            "predictions": {
                "xgboost": {
                    "model_role": "PRIMARY",
                    "model_architecture": "Gradient Boosted Decision Trees (334 Tabular Features)",
                    "predicted_probability": round(p_xgb, 4),
                    "threshold": round(th_xgb, 4),
                    "threshold_decision": decision_xgb,
                },
                "lstm": {
                    "model_role": "SECONDARY",
                    "model_architecture": f"Bidirectional LSTM ({prediction_horizon_hours}h x 74 channels + 9 static)",
                    "predicted_probability": round(p_lstm, 4),
                    "threshold": round(th_lstm, 4),
                    "threshold_decision": decision_lstm,
                },
            },
            "warnings": warnings_list,
            "limitations": limitations_list,
        }

        return response
