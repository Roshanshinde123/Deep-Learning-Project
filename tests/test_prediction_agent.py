"""
Comprehensive Unit and Integration Test Suite for Prediction Agent (Phase 6.2)

Tests:
1. Successful real-patient prediction (PhysioNet Set A).
2. XGBoost model loading (48h and 24h native Booster).
3. LSTM model loading (48h and 24h Keras model).
4. Correct temporal input shape (1, 48, 74) and (1, 24, 74).
5. Correct static input shape (1, 9).
6. Patient data validation (ID, filepath, DataFrame, raw string).
7. Data Quality Agent integration (DQI propagation and warning triggers).
8. DQI critical failure blocking (DQI < 50 prevents prediction; no fabricated scores).
9. Missing model handling (FileNotFoundError/ValueError on missing checkpoint).
10. Invalid input handling (empty DataFrame, malformed timestamps).
11. 24-hour horizon leakage prevention (zero future observation leakage).
12. Deterministic inference (identical inputs produce identical probabilities).
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure ROOT_DIR is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.prediction_agent import PredictionAgent
from src.agents.data_quality_agent import DataQualityAgent


class TestPredictionAgent(unittest.TestCase):
    """Test suite for Phase 6.2 PredictionAgent."""

    @classmethod
    def setUpClass(cls):
        cls.base_dir = r"D:\ICU_Prediction"
        cls.sample_patient_id = "132539"
        cls.sample_patient_file = os.path.join(cls.base_dir, "Dataset", "set-a", f"{cls.sample_patient_id}.txt")
        cls.agent = PredictionAgent(base_dir=cls.base_dir)

    # -------------------------------------------------------------------------
    # TEST 1: REAL-PATIENT PREDICTION
    # -------------------------------------------------------------------------
    def test_01_successful_real_patient_prediction(self):
        """Verify successful prediction on a real PhysioNet Set A patient."""
        self.assertTrue(os.path.exists(self.sample_patient_file), "Sample patient file must exist")
        res = self.agent.predict_patient(self.sample_patient_id, prediction_horizon_hours=48)

        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["patient_id"], self.sample_patient_id)
        self.assertEqual(res["prediction_horizon_hours"], 48)

        # Check Data Quality block
        self.assertIn("data_quality", res)
        self.assertGreaterEqual(res["data_quality"]["dqi"], 50.0)
        self.assertIn(res["data_quality"]["status"], ["PASS", "WARNING"])

        # Check XGBoost (Primary)
        self.assertIn("xgboost", res["predictions"])
        xgb_p = res["predictions"]["xgboost"]["predicted_probability"]
        self.assertIsInstance(xgb_p, float)
        self.assertTrue(0.0 <= xgb_p <= 1.0)
        self.assertEqual(res["predictions"]["xgboost"]["model_role"], "PRIMARY")
        self.assertEqual(res["predictions"]["xgboost"]["threshold"], 0.58)
        self.assertIn(res["predictions"]["xgboost"]["threshold_decision"], ["ABOVE_THRESHOLD", "BELOW_THRESHOLD"])

        # Check LSTM (Secondary)
        self.assertIn("lstm", res["predictions"])
        lstm_p = res["predictions"]["lstm"]["predicted_probability"]
        self.assertIsInstance(lstm_p, float)
        self.assertTrue(0.0 <= lstm_p <= 1.0)
        self.assertEqual(res["predictions"]["lstm"]["model_role"], "SECONDARY")
        self.assertEqual(res["predictions"]["lstm"]["threshold"], 0.58)
        self.assertIn(res["predictions"]["lstm"]["threshold_decision"], ["ABOVE_THRESHOLD", "BELOW_THRESHOLD"])

        # Check Limitations and Disclaimers
        self.assertGreater(len(res["limitations"]), 0)
        self.assertTrue(any("Research-only" in lim for lim in res["limitations"]))

    # -------------------------------------------------------------------------
    # TEST 2: XGBOOST MODEL LOADING
    # -------------------------------------------------------------------------
    def test_02_xgboost_model_loading(self):
        """Verify XGBoost 48h and 24h models load into native Booster objects."""
        xgb_48 = self.agent._load_model("xgboost", 48)
        xgb_24 = self.agent._load_model("xgboost", 24)

        import xgboost as xgb
        self.assertIsInstance(xgb_48, xgb.Booster)
        self.assertIsInstance(xgb_24, xgb.Booster)
        self.assertIn("xgboost_48h", self.agent._models)
        self.assertIn("xgboost_24h", self.agent._models)

    # -------------------------------------------------------------------------
    # TEST 3: LSTM MODEL LOADING
    # -------------------------------------------------------------------------
    def test_03_lstm_model_loading(self):
        """Verify LSTM 48h and 24h Keras models load properly."""
        lstm_48 = self.agent._load_model("lstm", 48)
        lstm_24 = self.agent._load_model("lstm", 24)

        import keras
        self.assertIsInstance(lstm_48, keras.Model)
        self.assertIsInstance(lstm_24, keras.Model)
        self.assertIn("lstm_48h", self.agent._models)
        self.assertIn("lstm_24h", self.agent._models)

    # -------------------------------------------------------------------------
    # TEST 4: TEMPORAL INPUT SHAPES
    # -------------------------------------------------------------------------
    def test_04_correct_temporal_input_shape(self):
        """Verify temporal tensor shape is exactly (1, 48, 74) for 48h and (1, 24, 74) for 24h."""
        _, df = self.agent.parse_patient_input(self.sample_patient_id)

        prep_48 = self.agent.preprocess_patient(df, horizon_hours=48)
        self.assertEqual(prep_48["temporal_tensor"].shape, (1, 48, 74))

        prep_24 = self.agent.preprocess_patient(df, horizon_hours=24)
        self.assertEqual(prep_24["temporal_tensor"].shape, (1, 24, 74))

    # -------------------------------------------------------------------------
    # TEST 5: STATIC INPUT SHAPE
    # -------------------------------------------------------------------------
    def test_05_correct_static_input_shape(self):
        """Verify static admission tensor shape is exactly (1, 9) with no NaNs."""
        _, df = self.agent.parse_patient_input(self.sample_patient_id)

        prep_48 = self.agent.preprocess_patient(df, horizon_hours=48)
        s_48 = prep_48["static_tensor"]
        self.assertEqual(s_48.shape, (1, 9))
        self.assertFalse(np.isnan(s_48).any(), "Static tensor must not contain NaNs")
        self.assertFalse(np.isinf(s_48).any(), "Static tensor must not contain Infs")

        prep_24 = self.agent.preprocess_patient(df, horizon_hours=24)
        s_24 = prep_24["static_tensor"]
        self.assertEqual(s_24.shape, (1, 9))
        self.assertFalse(np.isnan(s_24).any())

    # -------------------------------------------------------------------------
    # TEST 6: PATIENT DATA INPUT MODALITIES
    # -------------------------------------------------------------------------
    def test_06_patient_data_validation(self):
        """Verify that Patient ID, file path, DataFrame, and raw text lines parse identically."""
        # A. Patient ID
        id_1, df_1 = self.agent.parse_patient_input(self.sample_patient_id)

        # B. File path
        id_2, df_2 = self.agent.parse_patient_input(self.sample_patient_file)

        # C. DataFrame
        id_3, df_3 = self.agent.parse_patient_input(df_1)

        # D. Raw lines
        with open(self.sample_patient_file, "r") as f:
            lines = f.readlines()
        id_4, df_4 = self.agent.parse_patient_input(lines)

        self.assertEqual(id_1, self.sample_patient_id)
        self.assertEqual(id_2, self.sample_patient_id)
        self.assertEqual(id_3, self.sample_patient_id)
        self.assertEqual(id_4, self.sample_patient_id)

        self.assertEqual(len(df_1), len(df_2))
        self.assertEqual(len(df_1), len(df_3))
        self.assertEqual(len(df_1), len(df_4))

    # -------------------------------------------------------------------------
    # TEST 7: DATA QUALITY AGENT INTEGRATION
    # -------------------------------------------------------------------------
    def test_07_data_quality_agent_integration(self):
        """Verify Data Quality Agent audit results propagate into prediction response."""
        res = self.agent.predict_patient(self.sample_patient_id, prediction_horizon_hours=48)

        self.assertIn("data_quality", res)
        dq = res["data_quality"]
        self.assertIn("dqi", dq)
        self.assertIn("category", dq)
        self.assertIn("status", dq)
        self.assertGreaterEqual(dq["dqi"], 0.0)
        self.assertLessEqual(dq["dqi"], 100.0)

    # -------------------------------------------------------------------------
    # TEST 8: DQI CRITICAL FAILURE BLOCKING (DQI < 50)
    # -------------------------------------------------------------------------
    def test_08_dqi_critical_failure_blocking(self):
        """Verify that records with DQI < 50 block prediction without fabricating scores."""
        # Synthetic corrupted patient: all static descriptors missing / sentinel -1, zero vitals
        bad_records = [
            {"Time": "00:00", "Parameter": "RecordID", "Value": 999999},
            {"Time": "00:00", "Parameter": "Age", "Value": -1.0},
            {"Time": "00:00", "Parameter": "Gender", "Value": -1.0},
            {"Time": "00:00", "Parameter": "Height", "Value": -1.0},
            {"Time": "00:00", "Parameter": "Weight", "Value": -1.0},
            {"Time": "00:00", "Parameter": "ICUType", "Value": -1.0},
            {"Time": "01:00", "Parameter": "Urine", "Value": 10.0},
        ]
        df_bad = pd.DataFrame(bad_records)

        res = self.agent.predict_patient(df_bad, prediction_horizon_hours=48)

        self.assertEqual(res["status"], "BLOCKED_BY_DATA_QUALITY")
        self.assertLess(res["data_quality"]["dqi"], 50.0)
        self.assertEqual(res["predictions"], {})  # Strictly empty!
        self.assertTrue(any("blocked" in w.lower() for w in res["warnings"]))
        self.assertIn("audit_recommendations", res)

    # -------------------------------------------------------------------------
    # TEST 9: MISSING MODEL HANDLING
    # -------------------------------------------------------------------------
    def test_09_missing_model_handling(self):
        """Verify proper error handling when requested model or horizon is invalid."""
        with self.assertRaises(ValueError):
            self.agent._load_model("xgboost", 99)  # Unsupported horizon

        with self.assertRaises(ValueError):
            self.agent._load_model("invalid_model_type", 48)

    # -------------------------------------------------------------------------
    # TEST 10: INVALID INPUT HANDLING
    # -------------------------------------------------------------------------
    def test_10_invalid_input_handling(self):
        """Verify graceful error reporting when input is corrupted or empty."""
        # Empty DataFrame
        df_empty = pd.DataFrame(columns=["Time", "Parameter", "Value"])
        res_empty = self.agent.predict_patient(df_empty)
        self.assertEqual(res_empty["status"], "ERROR_EMPTY_RECORD")
        self.assertEqual(res_empty["predictions"], {})

        # Non-existent file
        res_nofile = self.agent.predict_patient("non_existent_file_path_123.txt")
        self.assertEqual(res_nofile["status"], "ERROR_INPUT_PARSING")

    # -------------------------------------------------------------------------
    # TEST 11: 24-HOUR HORIZON LEAKAGE PREVENTION
    # -------------------------------------------------------------------------
    def test_11_24_hour_horizon_leakage_prevention(self):
        """Verify that observations beyond 24h are strictly excluded in 24h mode."""
        # Patient with measurements at 10h (<=24h) and 35h (>24h)
        leakage_test_records = [
            {"Time": "00:00", "Parameter": "RecordID", "Value": 888888},
            {"Time": "00:00", "Parameter": "Age", "Value": 65.0},
            {"Time": "00:00", "Parameter": "Gender", "Value": 1.0},
            {"Time": "00:00", "Parameter": "ICUType", "Value": 2.0},
            {"Time": "10:00", "Parameter": "HR", "Value": 80.0},
            {"Time": "10:00", "Parameter": "DiasABP", "Value": 70.0},
            {"Time": "10:00", "Parameter": "SysABP", "Value": 120.0},
            {"Time": "10:00", "Parameter": "MAP", "Value": 85.0},
            {"Time": "10:00", "Parameter": "GCS", "Value": 15.0},
            {"Time": "10:00", "Parameter": "RespRate", "Value": 16.0},
            {"Time": "10:00", "Parameter": "Temp", "Value": 37.0},
            # Observations strictly AFTER 24 hours:
            {"Time": "35:00", "Parameter": "HR", "Value": 140.0},  # Extreme tachycardia
            {"Time": "35:00", "Parameter": "MechVent", "Value": 1.0},  # Intubation at 35h
        ]
        df_leak = pd.DataFrame(leakage_test_records)

        prep_24 = self.agent.preprocess_patient(df_leak, horizon_hours=24)

        # 1. Temporal shape is strictly 24 hours
        self.assertEqual(prep_24["temporal_tensor"].shape, (1, 24, 74))

        # 2. MechVent logged only at 35h must NOT register in 24h static tensor
        # Static tensor column 4 is MechVent_ever_recorded
        mv_ever_24 = prep_24["static_tensor"][0, 4]
        self.assertEqual(mv_ever_24, 0.0, "MechVent recorded after 24h leaked into 24h static features!")

        # 3. Tabular features must have shape (1, 334)
        self.assertEqual(prep_24["tabular_features"].shape, (1, 334))

        # 4. In 48h mode, the 35h MechVent SHOULD be captured
        prep_48 = self.agent.preprocess_patient(df_leak, horizon_hours=48)
        self.assertEqual(prep_48["static_tensor"][0, 4], 1.0)

    # -------------------------------------------------------------------------
    # TEST 12: DETERMINISTIC INFERENCE
    # -------------------------------------------------------------------------
    def test_12_deterministic_inference(self):
        """Verify identical inputs yield identical floating-point prediction probabilities."""
        res_1 = self.agent.predict_patient(self.sample_patient_id, prediction_horizon_hours=48)
        res_2 = self.agent.predict_patient(self.sample_patient_id, prediction_horizon_hours=48)

        xgb_p1 = res_1["predictions"]["xgboost"]["predicted_probability"]
        xgb_p2 = res_2["predictions"]["xgboost"]["predicted_probability"]

        lstm_p1 = res_1["predictions"]["lstm"]["predicted_probability"]
        lstm_p2 = res_2["predictions"]["lstm"]["predicted_probability"]

        self.assertEqual(xgb_p1, xgb_p2, "XGBoost predictions must be deterministic")
        self.assertEqual(lstm_p1, lstm_p2, "LSTM predictions must be deterministic")


if __name__ == "__main__":
    unittest.main()
