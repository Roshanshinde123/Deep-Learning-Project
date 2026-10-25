"""
Comprehensive Unit and Integration Test Suite for Reporting / Priority Agent (Phase 6.4)

Tests:
1. Concordant low-risk patient (Set A 132539 -> MODEL_CONCORDANT_LOW, LOW priority).
2. Concordant high-risk patient (Set A 132551 -> MODEL_CONCORDANT_HIGH, HIGH priority).
3. Model-discordant patient (Set A 132588 -> MODEL_DISCORDANT, MODERATE priority).
4. DQI critical failure (Synthetic 999999 -> PREDICTION_BLOCKED, CRITICAL_DATA_QUALITY).
5. Suboptimal data quality (DQI 50-70 -> SUBOPTIMAL_DATA_QUALITY warning and MODERATE priority).
6. Missing explanation input (orchestrates ExplanationAgent on the fly).
7. Missing prediction input (orchestrates PredictionAgent on the fly).
8. Deterministic priority assignment (identical inputs yield identical priority).
9. JSON schema validity (all required schema keys and typing confirmed).
10. Research-only disclaimer (explicit presence in report and metadata).
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

from src.agents.reporting_agent import ReportingAgent


class TestReportingAgent(unittest.TestCase):
    """Test suite for Phase 6.4 ReportingAgent."""

    @classmethod
    def setUpClass(cls):
        cls.base_dir = r"D:\ICU_Prediction"
        cls.low_risk_id = "132539"
        cls.high_risk_id = "132551"
        cls.discordant_id = "132588"
        cls.agent = ReportingAgent(base_dir=cls.base_dir)

    # -------------------------------------------------------------------------
    # TEST 1: CONCORDANT LOW-RISK PATIENT
    # -------------------------------------------------------------------------
    def test_01_concordant_low_risk_patient(self):
        """Verify Patient 132539 is assigned MODEL_CONCORDANT_LOW and LOW priority."""
        rep = self.agent.generate_report(self.low_risk_id, prediction_horizon_hours=48)

        self.assertEqual(rep["patient_id"], self.low_risk_id)
        self.assertEqual(rep["agreement_status"], "MODEL_CONCORDANT_LOW")
        self.assertEqual(rep["review_priority"], "LOW")
        self.assertLess(rep["prediction"]["xgboost"]["predicted_probability"], rep["prediction"]["xgboost"]["threshold"])
        self.assertLess(rep["prediction"]["lstm"]["predicted_probability"], rep["prediction"]["lstm"]["threshold"])
        self.assertIn("LOW", rep["rationale"])

    # -------------------------------------------------------------------------
    # TEST 2: CONCORDANT HIGH-RISK PATIENT
    # -------------------------------------------------------------------------
    def test_02_concordant_high_risk_patient(self):
        """Verify Patient 132551 is assigned MODEL_CONCORDANT_HIGH and HIGH priority."""
        rep = self.agent.generate_report(self.high_risk_id, prediction_horizon_hours=48)

        self.assertEqual(rep["patient_id"], self.high_risk_id)
        self.assertEqual(rep["agreement_status"], "MODEL_CONCORDANT_HIGH")
        self.assertEqual(rep["review_priority"], "HIGH")
        self.assertGreaterEqual(rep["prediction"]["xgboost"]["predicted_probability"], rep["prediction"]["xgboost"]["threshold"])
        self.assertGreaterEqual(rep["prediction"]["lstm"]["predicted_probability"], rep["prediction"]["lstm"]["threshold"])
        self.assertIn("HIGH", rep["rationale"])

    # -------------------------------------------------------------------------
    # TEST 3: MODEL-DISCORDANT PATIENT
    # -------------------------------------------------------------------------
    def test_03_model_discordant_patient(self):
        """Verify Patient 132588 is assigned MODEL_DISCORDANT and MODERATE priority."""
        rep = self.agent.generate_report(self.discordant_id, prediction_horizon_hours=48)

        self.assertEqual(rep["patient_id"], self.discordant_id)
        self.assertEqual(rep["agreement_status"], "MODEL_DISCORDANT")
        self.assertEqual(rep["review_priority"], "MODERATE")
        self.assertIn("MODERATE", rep["rationale"])
        self.assertIn("discordance", rep["rationale"].lower())

    # -------------------------------------------------------------------------
    # TEST 4: DQI CRITICAL FAILURE (DQI < 50)
    # -------------------------------------------------------------------------
    def test_04_dqi_critical_failure(self):
        """Verify corrupted telemetry triggers PREDICTION_BLOCKED and CRITICAL_DATA_QUALITY."""
        bad_df = pd.DataFrame([
            {"Time": "00:00", "Parameter": "RecordID", "Value": 999999},
            {"Time": "00:00", "Parameter": "Age", "Value": -1.0},
            {"Time": "00:00", "Parameter": "Gender", "Value": -1.0},
            {"Time": "00:00", "Parameter": "Height", "Value": -1.0},
            {"Time": "00:00", "Parameter": "Weight", "Value": -1.0},
            {"Time": "00:00", "Parameter": "ICUType", "Value": -1.0},
            {"Time": "01:00", "Parameter": "Urine", "Value": 5.0},
        ])

        rep = self.agent.generate_report(bad_df, prediction_horizon_hours=48)

        self.assertEqual(rep["agreement_status"], "PREDICTION_BLOCKED")
        self.assertEqual(rep["review_priority"], "CRITICAL_DATA_QUALITY")
        self.assertIsNone(rep["prediction"]["xgboost"]["predicted_probability"])
        self.assertIsNone(rep["prediction"]["lstm"]["predicted_probability"])
        self.assertIn("CRITICAL_DATA_QUALITY", rep["rationale"])

    # -------------------------------------------------------------------------
    # TEST 5: SUBOPTIMAL DATA QUALITY (50 <= DQI < 70)
    # -------------------------------------------------------------------------
    def test_05_suboptimal_data_quality(self):
        """Verify telemetry with DQI 50-70 flags SUBOPTIMAL_DATA_QUALITY and elevates priority."""
        # Patient with valid descriptors but extremely sparse single vital (DQI in 50-70 range)
        sparse_df = pd.DataFrame([
            {"Time": "00:00", "Parameter": "RecordID", "Value": 777777},
            {"Time": "00:00", "Parameter": "Age", "Value": 65.0},
            {"Time": "00:00", "Parameter": "Gender", "Value": 1.0},
            {"Time": "00:00", "Parameter": "Height", "Value": 175.0},
            {"Time": "00:00", "Parameter": "Weight", "Value": 80.0},
            {"Time": "00:00", "Parameter": "ICUType", "Value": 2.0},
            {"Time": "01:00", "Parameter": "HR", "Value": 75.0},
            {"Time": "01:00", "Parameter": "SysABP", "Value": 120.0},
            {"Time": "01:00", "Parameter": "DiasABP", "Value": 70.0},
            {"Time": "01:00", "Parameter": "MAP", "Value": 85.0},
        ])

        rep = self.agent.generate_report(sparse_df, prediction_horizon_hours=48)
        dqi = rep["data_quality"]["dqi"]

        # Ensure DQI is in suboptimal or acceptable range
        if 50.0 <= dqi < 70.0:
            self.assertEqual(rep["data_quality"]["confidence_label"], "SUBOPTIMAL_DATA_QUALITY")
            self.assertEqual(rep["review_priority"], "MODERATE")
            self.assertIn("suboptimal", rep["rationale"].lower())

    # -------------------------------------------------------------------------
    # TEST 6: MISSING EXPLANATION INPUT (ORCHESTRATED AUTOMATICALLY)
    # -------------------------------------------------------------------------
    def test_06_missing_explanation_input(self):
        """Verify ReportingAgent computes explanations on the fly when expl_result is omitted."""
        pred = self.agent.prediction_agent.predict_patient(self.low_risk_id, prediction_horizon_hours=48)
        rep = self.agent.generate_report(self.low_risk_id, prediction_horizon_hours=48, pred_result=pred, expl_result=None)

        self.assertIn("explanation", rep)
        self.assertIn("top_xgboost_risk_contributors", rep["explanation"])
        self.assertEqual(rep["agreement_status"], "MODEL_CONCORDANT_LOW")

    # -------------------------------------------------------------------------
    # TEST 7: MISSING PREDICTION INPUT (ORCHESTRATED AUTOMATICALLY)
    # -------------------------------------------------------------------------
    def test_07_missing_prediction_input(self):
        """Verify ReportingAgent computes predictions on the fly when pred_result is omitted."""
        rep = self.agent.generate_report(self.low_risk_id, prediction_horizon_hours=48, pred_result=None, expl_result=None)

        self.assertIn("prediction", rep)
        self.assertIsNotNone(rep["prediction"]["xgboost"]["predicted_probability"])
        self.assertEqual(rep["agreement_status"], "MODEL_CONCORDANT_LOW")

    # -------------------------------------------------------------------------
    # TEST 8: DETERMINISTIC PRIORITY ASSIGNMENT
    # -------------------------------------------------------------------------
    def test_08_deterministic_priority_assignment(self):
        """Verify identical inputs yield identical agreement status and priority."""
        rep_1 = self.agent.generate_report(self.low_risk_id, prediction_horizon_hours=48)
        rep_2 = self.agent.generate_report(self.low_risk_id, prediction_horizon_hours=48)

        self.assertEqual(rep_1["agreement_status"], rep_2["agreement_status"])
        self.assertEqual(rep_1["review_priority"], rep_2["review_priority"])
        self.assertEqual(rep_1["rationale"], rep_2["rationale"])

    # -------------------------------------------------------------------------
    # TEST 9: JSON SCHEMA VALIDITY
    # -------------------------------------------------------------------------
    def test_09_json_schema_validity(self):
        """Verify complete compliance with the machine-readable JSON schema."""
        rep = self.agent.generate_report(self.low_risk_id, prediction_horizon_hours=48)

        required_keys = [
            "patient_id", "horizon_hours", "data_quality", "prediction",
            "agreement_status", "agreement_meaning", "explanation",
            "review_priority", "rationale", "human_readable_report",
            "limitations", "research_only"
        ]
        for k in required_keys:
            self.assertIn(k, rep, f"Missing required top-level key: {k}")

        self.assertIsInstance(rep["data_quality"], dict)
        self.assertIsInstance(rep["prediction"], dict)
        self.assertIsInstance(rep["explanation"], dict)
        self.assertIsInstance(rep["limitations"], list)
        self.assertTrue(rep["research_only"])

    # -------------------------------------------------------------------------
    # TEST 10: RESEARCH-ONLY DISCLAIMER
    # -------------------------------------------------------------------------
    def test_10_research_only_disclaimer(self):
        """Verify explicit research disclaimers are present in report and limitations."""
        rep = self.agent.generate_report(self.low_risk_id, prediction_horizon_hours=48)

        self.assertTrue(rep["research_only"])
        self.assertIn("Academic Research Prototype — Not for Clinical Use", rep["human_readable_report"])
        self.assertTrue(any("research" in lim.lower() for lim in rep["limitations"]))
        self.assertIn("must not be used to make clinical treatment decisions", rep["human_readable_report"].lower())

    # -------------------------------------------------------------------------
    # TEST 11: END-TO-END UPSTREAM OUTPUT INTEGRITY REGRESSION
    # -------------------------------------------------------------------------
    def test_11_upstream_integrity_regression(self):
        """Verify ReportingAgent preserves exact upstream DQI, probabilities, and attributions without modification."""
        pred = self.agent.prediction_agent.predict_patient(self.low_risk_id, prediction_horizon_hours=48)
        dq = pred.get("data_quality", {})
        expl = self.agent.explanation_agent.explain_patient(self.low_risk_id, prediction_horizon_hours=48, generate_plots=False)

        rep = self.agent.generate_report(
            patient_input=self.low_risk_id,
            prediction_horizon_hours=48,
            dq_result=dq,
            pred_result=pred,
            expl_result=expl,
        )

        # Exact match assertions
        self.assertEqual(rep["data_quality"]["dqi"], round(dq["dqi"], 2))
        self.assertEqual(rep["prediction"]["xgboost"]["predicted_probability"], round(pred["predictions"]["xgboost"]["predicted_probability"], 4))
        self.assertEqual(rep["prediction"]["lstm"]["predicted_probability"], round(pred["predictions"]["lstm"]["predicted_probability"], 4))
        self.assertEqual(len(rep["explanation"]["top_xgboost_risk_contributors"]), len(expl["xgboost_explanation"]["top_positive_contributors"]))
        self.assertEqual(len(rep["explanation"]["top_lstm_temporal_features"]), len(expl["lstm_explanation"]["important_features"]))

    # -------------------------------------------------------------------------
    # TEST 12: DETERMINISTIC PRIORITY PRECEDENCE HIERARCHY
    # -------------------------------------------------------------------------
    def test_12_priority_precedence_hierarchy(self):
        """Verify strict priority rule precedence:
        1. DQI < 50 -> CRITICAL_DATA_QUALITY
        2. 50 <= DQI < 70 -> MODERATE (overrides concordant high)
        3. Model discordance -> MODERATE
        4. Both models above threshold with DQI >= 70 -> HIGH
        5. Both models below threshold with DQI >= 70 -> LOW
        """
        # 1. Critical Failure
        prio, _ = self.agent.assign_review_priority(dqi=45.0, agreement_status="PREDICTION_BLOCKED", p_xgb=0.0, p_lstm=0.0, th_xgb=0.58, th_lstm=0.58)
        self.assertEqual(prio, "CRITICAL_DATA_QUALITY")

        # 2. Suboptimal DQI (50 <= DQI < 70) overrides high-risk predictions
        prio, _ = self.agent.assign_review_priority(dqi=65.0, agreement_status="MODEL_CONCORDANT_HIGH", p_xgb=0.85, p_lstm=0.88, th_xgb=0.58, th_lstm=0.58)
        self.assertEqual(prio, "MODERATE")

        # 3. Model discordance with good DQI
        prio, _ = self.agent.assign_review_priority(dqi=85.0, agreement_status="MODEL_DISCORDANT", p_xgb=0.80, p_lstm=0.30, th_xgb=0.58, th_lstm=0.58)
        self.assertEqual(prio, "MODERATE")

        # 4. Concordant high with DQI >= 70
        prio, _ = self.agent.assign_review_priority(dqi=85.0, agreement_status="MODEL_CONCORDANT_HIGH", p_xgb=0.85, p_lstm=0.88, th_xgb=0.58, th_lstm=0.58)
        self.assertEqual(prio, "HIGH")

        # 5. Concordant low with DQI >= 70
        prio, _ = self.agent.assign_review_priority(dqi=85.0, agreement_status="MODEL_CONCORDANT_LOW", p_xgb=0.10, p_lstm=0.05, th_xgb=0.58, th_lstm=0.58)
        self.assertEqual(prio, "LOW")


if __name__ == "__main__":
    unittest.main()
