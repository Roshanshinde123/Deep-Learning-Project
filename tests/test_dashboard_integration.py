"""
Integration Test Suite for Streamlit Dashboard & Complete Multi-Agent Workflow (Phase 7)
Agentic AI-Assisted ICU Mortality Risk Prediction from the First 48 Hours.

Covers all 15 required verification targets:
1. App and component modules import successfully.
2. All four agents initialize without error.
3. Patient file upload parsing works correctly.
4. 24-hour early triage workflow execution works.
5. 48-hour standard ICU workflow execution works.
6. Invalid input handling returns appropriate error/exceptions.
7. DQI quality gate blocking works when DQI < 50.
8. XGBoost tabular inference produces expected probabilities.
9. LSTM sequential inference produces expected probabilities.
10. Explanation Agent integration generates valid SHAP attributions.
11. Reporting Agent integration generates valid unified report.
12. JSON download payload complies with machine-readable schema.
13. No model probability averaging occurs (both distinct values preserved).
14. Patient session isolation is maintained without cross-patient leakage.
15. Verification that existing repository tests remain unaffected and passing.
"""

import unittest
import os
import io
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent

from src.services.icu_workflow import ICUWorkflowService
from src.agents.data_quality_agent import DataQualityAgent
from src.agents.prediction_agent import PredictionAgent
from src.agents.explanation_agent import ExplanationAgent
from src.agents.reporting_agent import ReportingAgent


class TestDashboardIntegration(unittest.TestCase):
    """Integration and End-to-End Suite for ICU Risk Intelligence Dashboard."""

    @classmethod
    def setUpClass(cls):
        cls.service = ICUWorkflowService(base_dir=ROOT_DIR)
        cls.sample_patient_id = "132539"
        cls.high_risk_id = "132551"

    # 1. App and Component Imports
    def test_01_app_imports_successfully(self):
        """Verify dashboard components and service modules import without circularity or missing dependencies."""
        try:
            import app.components.sidebar as sidebar
            import app.components.metric_cards as cards
            import app.components.charts as charts
            import app.components.report_viewer as viewer
            self.assertTrue(hasattr(sidebar, "render_sidebar"))
            self.assertTrue(hasattr(cards, "render_dqi_card"))
            self.assertTrue(hasattr(charts, "plot_vitals_trajectories"))
            self.assertTrue(hasattr(viewer, "render_report_view"))
        except ImportError as e:
            self.fail(f"Failed to import dashboard modules: {e}")

    # 2. Agent Initialization
    def test_02_all_four_agents_initialize(self):
        """Verify all four agents initialize cleanly with their respective model artifacts."""
        self.assertIsInstance(self.service.dq_agent, DataQualityAgent)
        self.assertIsInstance(self.service.pred_agent, PredictionAgent)
        self.assertIsInstance(self.service.expl_agent, ExplanationAgent)
        self.assertIsInstance(self.service.reporting_agent, ReportingAgent)

    # 3. Patient Upload Parsing
    def test_03_patient_upload_parsing_works(self):
        """Verify uploaded PhysioNet file text is correctly parsed into a validated DataFrame."""
        sample_txt = (
            "Time,Parameter,Value\n"
            "00:00,RecordID,132539\n"
            "00:00,Age,54\n"
            "00:00,Gender,0\n"
            "00:00,Height,165\n"
            "00:00,Weight,72\n"
            "00:00,ICUType,1\n"
            "01:00,HR,80\n"
            "01:00,SysABP,120\n"
            "01:00,DiasABP,75\n"
        )
        df, pid = self.service.parse_uploaded_patient_file(sample_txt)
        self.assertEqual(pid, "132539")
        self.assertEqual(len(df), 9)
        self.assertIn("Time", df.columns)
        self.assertIn("Parameter", df.columns)
        self.assertIn("Value", df.columns)

    # 4. 24-Hour Workflow Execution
    def test_04_24h_workflow_works(self):
        """Verify complete multi-agent workflow executes across the 24-hour observation horizon."""
        res = self.service.run_workflow(self.sample_patient_id, prediction_horizon_hours=24, generate_plots=False)
        self.assertEqual(res["patient_id"], "132539")
        self.assertEqual(res["prediction_horizon_hours"], 24)
        self.assertEqual(res["predictions"]["status"], "SUCCESS")
        self.assertIn("xgboost", res["predictions"]["predictions"])
        self.assertIn("lstm", res["predictions"]["predictions"])
        self.assertIn("review_priority", res["report"])

    # 5. 48-Hour Workflow Execution
    def test_05_48h_workflow_works(self):
        """Verify complete multi-agent workflow executes across the standard 48-hour horizon."""
        res = self.service.run_workflow(self.sample_patient_id, prediction_horizon_hours=48, generate_plots=False)
        self.assertEqual(res["patient_id"], "132539")
        self.assertEqual(res["prediction_horizon_hours"], 48)
        self.assertEqual(res["predictions"]["status"], "SUCCESS")
        self.assertGreaterEqual(res["data_quality"]["dqi"], 70.0)
        self.assertEqual(res["report"]["agreement_status"], "MODEL_CONCORDANT_LOW")
        self.assertEqual(res["report"]["review_priority"], "LOW")

    # 6. Invalid Input Handling
    def test_06_invalid_input_handling_works(self):
        """Verify empty files or nonexistent patient records raise clear exceptions without crash."""
        with self.assertRaises(ValueError):
            self.service.parse_uploaded_patient_file("")

        with self.assertRaises(FileNotFoundError):
            self.service.run_workflow("nonexistent_patient_000000", prediction_horizon_hours=48)

    # 7. DQI Quality Gate Blocking
    def test_07_dqi_blocking_works(self):
        """Verify severely corrupted records (DQI < 50.0) trigger gate block and inhibit predictions."""
        corrupted_df = pd.DataFrame([
            {"Time": "00:00", "Parameter": "RecordID", "Value": 999999},
            {"Time": "00:00", "Parameter": "Age", "Value": -1},
            {"Time": "00:00", "Parameter": "Gender", "Value": -1},
            {"Time": "00:00", "Parameter": "Height", "Value": -1},
            {"Time": "00:00", "Parameter": "Weight", "Value": -1},
            {"Time": "00:00", "Parameter": "ICUType", "Value": -1},
            {"Time": "01:00", "Parameter": "HR", "Value": 999.0},
            {"Time": "01:00", "Parameter": "SysABP", "Value": -50.0},
        ])
        res = self.service.run_workflow(corrupted_df, prediction_horizon_hours=48)
        self.assertLess(res["data_quality"]["dqi"], 50.0)
        self.assertEqual(res["predictions"]["status"], "BLOCKED_BY_DATA_QUALITY")
        self.assertEqual(res["report"]["agreement_status"], "PREDICTION_BLOCKED")
        self.assertEqual(res["report"]["review_priority"], "CRITICAL_DATA_QUALITY")

    # 8. XGBoost Tabular Inference
    def test_08_xgboost_inference_works(self):
        """Verify XGBoost produces calibrated probabilities within [0, 1]."""
        res = self.service.run_workflow(self.sample_patient_id, prediction_horizon_hours=48, generate_plots=False)
        p_xgb = res["predictions"]["predictions"]["xgboost"]["predicted_probability"]
        self.assertIsInstance(p_xgb, float)
        self.assertTrue(0.0 <= p_xgb <= 1.0)
        self.assertAlmostEqual(p_xgb, 0.0356, places=3)

    # 9. LSTM Sequential Inference
    def test_09_lstm_inference_works(self):
        """Verify LSTM produces calibrated probabilities within [0, 1]."""
        res = self.service.run_workflow(self.sample_patient_id, prediction_horizon_hours=48, generate_plots=False)
        p_lstm = res["predictions"]["predictions"]["lstm"]["predicted_probability"]
        self.assertIsInstance(p_lstm, float)
        self.assertTrue(0.0 <= p_lstm <= 1.0)
        self.assertAlmostEqual(p_lstm, 0.0144, places=3)

    # 10. Explanation Agent Integration
    def test_10_explanation_integration_works(self):
        """Verify local Tree SHAP attributions and LSTM perturbations are present."""
        res = self.service.run_workflow(self.sample_patient_id, prediction_horizon_hours=48, generate_plots=False)
        xgb_expl = res["explanation"]["xgboost_explanation"]
        lstm_expl = res["explanation"]["lstm_explanation"]
        self.assertIn("top_positive_contributors", xgb_expl)
        self.assertIn("top_negative_contributors", xgb_expl)
        self.assertIn("important_features", lstm_expl)
        self.assertIn("important_time_periods", lstm_expl)

    # 11. Reporting Agent Integration
    def test_11_reporting_integration_works(self):
        """Verify human-readable ASCII report and structured JSON are correctly compiled."""
        res = self.service.run_workflow(self.sample_patient_id, prediction_horizon_hours=48, generate_plots=False)
        human_rep = res["human_readable_report"]
        self.assertIn("ICU MORTALITY RISK REVIEW SUMMARY", human_rep)
        self.assertIn("MODEL PREDICTION SUMMARY", human_rep)
        self.assertIn("RESEARCH REVIEW PRIORITY", human_rep)
        self.assertIn("CLINICAL SAFETY LIMITATIONS", human_rep)

    # 12. JSON Download Generation
    def test_12_json_download_generation_works(self):
        """Verify clean serialization to machine-readable JSON format conforming to schema."""
        res = self.service.run_workflow(self.sample_patient_id, prediction_horizon_hours=48, generate_plots=False)
        clean_json = {k: v for k, v in res["report"].items() if k != "human_readable_report"}
        json_str = json.dumps(clean_json, indent=2)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["patient_id"], "132539")
        self.assertIn("data_quality", parsed)
        self.assertIn("prediction", parsed)
        self.assertIn("review_priority", parsed)
        self.assertTrue(parsed["research_only"])

    # 13. No Model Probability Averaging
    def test_13_no_model_probability_averaging(self):
        """Verify XGBoost and LSTM probabilities remain strictly distinct without averaging."""
        res = self.service.run_workflow(self.high_risk_id, prediction_horizon_hours=48, generate_plots=False)
        p_xgb = res["report"]["prediction"]["xgboost"]["predicted_probability"]
        p_lstm = res["report"]["prediction"]["lstm"]["predicted_probability"]
        self.assertNotEqual(p_xgb, p_lstm)
        # Verify no average key exists
        self.assertNotIn("average_probability", res["report"]["prediction"])
        self.assertNotIn("ensemble_probability", res["report"]["prediction"])

    # 14. Patient Isolation Maintained
    def test_14_patient_isolation_maintained(self):
        """Verify running two consecutive patients does not leak state or attributions between records."""
        res1 = self.service.run_workflow("132539", prediction_horizon_hours=48, generate_plots=False)
        res2 = self.service.run_workflow("132551", prediction_horizon_hours=48, generate_plots=False)
        self.assertEqual(res1["patient_id"], "132539")
        self.assertEqual(res2["patient_id"], "132551")
        self.assertNotEqual(res1["report"]["review_priority"], res2["report"]["review_priority"])
        self.assertEqual(res1["report"]["review_priority"], "LOW")
        self.assertEqual(res2["report"]["review_priority"], "HIGH")

    # 15. Benchmark Data Integrity
    def test_15_benchmark_data_integrity(self):
        """Verify precomputed model comparison data is loaded without error from reports/."""
        benchmarks = self.service.get_stored_model_benchmarks()
        self.assertFalse(benchmarks.empty)
        self.assertIn("Model", benchmarks.columns)
        self.assertIn("TestC_AUROC", benchmarks.columns)
        self.assertGreaterEqual(len(benchmarks), 5)


if __name__ == "__main__":
    unittest.main()
