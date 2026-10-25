"""
Comprehensive Unit and Integration Test Suite for Explanation Agent (Phase 6.3)

Tests:
1. Real patient explanation execution (Set A patient 132539).
2. XGBoost SHAP generation and base value verification.
3. Top feature extraction (schema integrity and values).
4. Negative and positive contribution separation.
5. LSTM perturbation attribution (features and temporal windows).
6. Temporal trend extraction across available clinical parameters.
7. Model discordance detection and explanation (Patient 132588).
8. Missing data and DQI < 50 blocking protection.
9. Invalid input handling (empty DataFrame, invalid path).
10. Deterministic explanation outputs.
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

from src.agents.explanation_agent import ExplanationAgent


class TestExplanationAgent(unittest.TestCase):
    """Test suite for Phase 6.3 ExplanationAgent."""

    @classmethod
    def setUpClass(cls):
        cls.base_dir = r"D:\ICU_Prediction"
        cls.sample_patient_id = "132539"
        cls.discordant_patient_id = "132588"
        cls.agent = ExplanationAgent(base_dir=cls.base_dir)

    # -------------------------------------------------------------------------
    # TEST 1: REAL PATIENT EXPLANATION EXECUTION
    # -------------------------------------------------------------------------
    def test_01_real_patient_explanation(self):
        """Verify successful end-to-end explanation on a real Set A patient."""
        res = self.agent.explain_patient(self.sample_patient_id, prediction_horizon_hours=48)

        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["patient_id"], self.sample_patient_id)
        self.assertEqual(res["prediction_horizon"], 48)

        # Confirm all required keys exist
        self.assertIn("model_outputs", res)
        self.assertIn("xgboost_explanation", res)
        self.assertIn("lstm_explanation", res)
        self.assertIn("temporal_changes", res)
        self.assertIn("model_discordance", res)
        self.assertIn("human_readable_summary", res)
        self.assertIn("data_quality", res)
        self.assertIn("limitations", res)

    # -------------------------------------------------------------------------
    # TEST 2: SHAP GENERATION
    # -------------------------------------------------------------------------
    def test_02_shap_generation(self):
        """Verify SHAP TreeExplainer generates values for all 334 tabular features."""
        res = self.agent.explain_patient(self.sample_patient_id, prediction_horizon_hours=48)
        xgb_expl = res["xgboost_explanation"]

        self.assertIn("shap_values", xgb_expl)
        self.assertEqual(len(xgb_expl["shap_values"]), 334)
        self.assertIsInstance(xgb_expl["base_value"], float)

    # -------------------------------------------------------------------------
    # TEST 3: TOP FEATURE EXTRACTION
    # -------------------------------------------------------------------------
    def test_03_top_feature_extraction(self):
        """Verify top positive and negative features contain expected fields and ranking."""
        res = self.agent.explain_patient(self.sample_patient_id, prediction_horizon_hours=48, top_n=5)
        top_pos = res["xgboost_explanation"]["top_positive_contributors"]

        self.assertGreater(len(top_pos), 0)
        self.assertLessEqual(len(top_pos), 5)

        first = top_pos[0]
        self.assertIn("feature", first)
        self.assertIn("actual_value", first)
        self.assertIn("shap_value", first)
        self.assertIsInstance(first["actual_value"], float)
        self.assertIsInstance(first["shap_value"], float)

    # -------------------------------------------------------------------------
    # TEST 4: NEGATIVE AND POSITIVE CONTRIBUTION SEPARATION
    # -------------------------------------------------------------------------
    def test_04_negative_positive_contribution_separation(self):
        """Verify strictly positive SHAP values for positive contributors and negative for negative."""
        res = self.agent.explain_patient(self.sample_patient_id, prediction_horizon_hours=48)
        top_pos = res["xgboost_explanation"]["top_positive_contributors"]
        top_neg = res["xgboost_explanation"]["top_negative_contributors"]

        for item in top_pos:
            self.assertGreater(item["shap_value"], 0.0, "Positive contributor must have SHAP > 0")

        for item in top_neg:
            self.assertLess(item["shap_value"], 0.0, "Negative contributor must have SHAP < 0")

        # Verify sorting order: positive is descending, negative is ascending
        if len(top_pos) >= 2:
            self.assertGreaterEqual(top_pos[0]["shap_value"], top_pos[1]["shap_value"])
        if len(top_neg) >= 2:
            self.assertLessEqual(top_neg[0]["shap_value"], top_neg[1]["shap_value"])

    # -------------------------------------------------------------------------
    # TEST 5: LSTM PERTURBATION ATTRIBUTION
    # -------------------------------------------------------------------------
    def test_05_lstm_attribution_generation(self):
        """Verify LSTM occlusion attributions are computed for features and time windows."""
        res = self.agent.explain_patient(self.sample_patient_id, prediction_horizon_hours=48)
        lstm_expl = res["lstm_explanation"]

        self.assertIn("important_features", lstm_expl)
        self.assertIn("important_time_periods", lstm_expl)
        self.assertIn("attribution_method", lstm_expl)

        self.assertGreater(len(lstm_expl["important_features"]), 0)
        self.assertEqual(len(lstm_expl["important_time_periods"]), 4)  # 4 quarters

        first_period = lstm_expl["important_time_periods"][0]
        self.assertIn("time_period", first_period)
        self.assertIn("delta_probability", first_period)
        self.assertIn("impact_direction", first_period)

    # -------------------------------------------------------------------------
    # TEST 6: TEMPORAL TREND EXTRACTION
    # -------------------------------------------------------------------------
    def test_06_temporal_trend_extraction(self):
        """Verify temporal changes extract valid first, last, delta, and trend directions."""
        res = self.agent.explain_patient(self.sample_patient_id, prediction_horizon_hours=48)
        trends = res["temporal_changes"]

        self.assertIsInstance(trends, list)
        self.assertGreater(len(trends), 0)

        for tr in trends:
            self.assertIn("parameter", tr)
            self.assertIn("first_value", tr)
            self.assertIn("last_value", tr)
            self.assertIn("delta", tr)
            self.assertIn(tr["trend"], ["INCREASING", "DECREASING", "STABLE"])
            self.assertGreater(tr["observation_count"], 0)
            self.assertTrue(0.0 <= tr["missingness_percentage"] <= 100.0)

    # -------------------------------------------------------------------------
    # TEST 7: MODEL DISCORDANCE EXPLANATION
    # -------------------------------------------------------------------------
    def test_07_model_discordance_explanation(self):
        """Verify discordant predictions are explicitly flagged and detailed without averaging."""
        # Patient 132588 is known to produce discordant classifications (XGBoost ABOVE, LSTM BELOW)
        res = self.agent.explain_patient(self.discordant_patient_id, prediction_horizon_hours=48)

        self.assertTrue(res["model_discordance"])
        self.assertIn("MODEL DISCORDANCE DETECTED", res["discordance_explanation"])
        self.assertIn("XGBoost", res["discordance_explanation"])
        self.assertIn("LSTM", res["discordance_explanation"])
        # Confirm no consensus or average probability was fabricated
        self.assertNotIn("average_probability", res["model_outputs"])

    # -------------------------------------------------------------------------
    # TEST 8: MISSING DATA & QUALITY GATE BLOCKING
    # -------------------------------------------------------------------------
    def test_08_missing_data_handling(self):
        """Verify corrupted records with DQI < 50 block explanation without fabricating attributions."""
        bad_df = pd.DataFrame([
            {"Time": "00:00", "Parameter": "RecordID", "Value": 999999},
            {"Time": "00:00", "Parameter": "Age", "Value": -1.0},
            {"Time": "00:00", "Parameter": "Gender", "Value": -1.0},
            {"Time": "00:00", "Parameter": "Height", "Value": -1.0},
            {"Time": "00:00", "Parameter": "Weight", "Value": -1.0},
            {"Time": "00:00", "Parameter": "ICUType", "Value": -1.0},
            {"Time": "01:00", "Parameter": "Urine", "Value": 10.0},
        ])

        res = self.agent.explain_patient(bad_df, prediction_horizon_hours=48)

        self.assertEqual(res["status"], "BLOCKED_BY_DATA_QUALITY")
        self.assertEqual(res["xgboost_explanation"]["top_positive_contributors"], [])
        self.assertEqual(res["lstm_explanation"]["important_features"], [])
        self.assertIn("blocked", res["human_readable_summary"].lower())

    # -------------------------------------------------------------------------
    # TEST 9: INVALID INPUT HANDLING
    # -------------------------------------------------------------------------
    def test_09_invalid_input_handling(self):
        """Verify graceful error reporting when input is corrupted or non-existent."""
        res_empty = self.agent.explain_patient(pd.DataFrame(columns=["Time", "Parameter", "Value"]))
        self.assertEqual(res_empty["status"], "ERROR_EMPTY_RECORD")

        res_none = self.agent.explain_patient("non_existent_file_xyz.txt")
        self.assertEqual(res_none["status"], "ERROR_INPUT_PARSING")

    # -------------------------------------------------------------------------
    # TEST 10: DETERMINISTIC EXPLANATION OUTPUT
    # -------------------------------------------------------------------------
    def test_10_deterministic_explanation_output(self):
        """Verify identical inputs yield identical SHAP and attribution values."""
        res_1 = self.agent.explain_patient(self.sample_patient_id, prediction_horizon_hours=48)
        res_2 = self.agent.explain_patient(self.sample_patient_id, prediction_horizon_hours=48)

        shap_1 = res_1["xgboost_explanation"]["shap_values"]
        shap_2 = res_2["xgboost_explanation"]["shap_values"]

        for k in list(shap_1.keys())[:20]:
            self.assertEqual(shap_1[k], shap_2[k], f"SHAP value for {k} must be deterministic")

        lstm_d1 = res_1["lstm_explanation"]["important_features"][0]["delta_probability"]
        lstm_d2 = res_2["lstm_explanation"]["important_features"][0]["delta_probability"]
        self.assertEqual(lstm_d1, lstm_d2, "LSTM delta probability must be deterministic")


if __name__ == "__main__":
    unittest.main()
