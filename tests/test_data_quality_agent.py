"""
Comprehensive Unit and Integration Test Suite for Data Quality Agent (Phase 6.1 Refined)
Tests:
1. Real patient file parsing and quality scoring from PhysioNet Set A.
2. Missing Height and Weight sentinels (-1 classified as missing, not valid).
3. Valid alternative blood-pressure measurements (Invasive vs Non-Invasive NIBP).
4. Missing critical measurements (zero HR, zero BP, zero GCS).
5. Impossible physiological values (sensor artifacts vs genuine pathology).
6. Empty patient records.
7. DQI category boundaries (EXCELLENT >= 85, ACCEPTABLE 70-84, SUBOPTIMAL 50-69, UNRELIABLE < 50).
"""

import os
import unittest
import pandas as pd
from src.agents.data_quality_agent import DataQualityAgent


class TestDataQualityAgentRefined(unittest.TestCase):
    """Test suite for refined DataQualityAgent."""

    @classmethod
    def setUpClass(cls):
        cls.agent = DataQualityAgent()
        cls.base_dir = r"D:\ICU_Prediction"
        cls.sample_patient_path = os.path.join(cls.base_dir, "Dataset", "set-a", "132539.txt")

    def test_real_patient_audit_and_sentinel_recognition(self):
        """Verify audit on patient 132539.txt where Height and Weight are -1."""
        self.assertTrue(os.path.exists(self.sample_patient_path), "Sample patient record missing")
        audit = self.agent.audit_patient_file(self.sample_patient_path, prediction_horizon_hours=48)

        self.assertEqual(audit["status"], "PASS")
        self.assertIn(audit["quality_grade"], ["EXCELLENT", "ACCEPTABLE"])
        self.assertGreaterEqual(audit["data_quality_index"], 70.0)

        # Confirm sentinels -1 are classified as missing, not valid measurements
        self.assertIsNone(audit["descriptors_cleaned"]["Height"])
        self.assertIsNone(audit["descriptors_cleaned"]["Weight"])
        self.assertTrue(any("Height" in m for m in audit["missing_descriptors"]))
        self.assertTrue(any("Weight" in m for m in audit["missing_descriptors"]))

        # Confirm Blood Pressure is recognized as sufficient via non-invasive cuff
        self.assertEqual(audit["critical_vitals_summary"]["BloodPressure"]["status"], "SUFFICIENT")
        self.assertGreater(audit["blood_pressure_summary"]["noninvasive_count"], 0)

        # Markdown report generation
        md = self.agent.generate_markdown_report(audit)
        self.assertIn("Missing (sentinel -1)", md)
        self.assertIn("Data Quality Index (DQI)", md)

    def test_missing_height_and_weight_sentinels(self):
        """Verify that -1 sentinels are explicitly flagged and penalized in DQI."""
        data = [
            {"Time": "00:00", "Parameter": "RecordID", "Value": 100001},
            {"Time": "00:00", "Parameter": "Age", "Value": 60.0},
            {"Time": "00:00", "Parameter": "Gender", "Value": 1},
            {"Time": "00:00", "Parameter": "ICUType", "Value": 1},
            {"Time": "00:00", "Parameter": "Height", "Value": -1.0},  # Sentinel
            {"Time": "00:00", "Parameter": "Weight", "Value": -1.0},  # Sentinel
        ]
        # Add sufficient vitals
        for h in range(10):
            t_str = f"{h:02d}:00"
            data.extend([
                {"Time": t_str, "Parameter": "HR", "Value": 80.0},
                {"Time": t_str, "Parameter": "MAP", "Value": 75.0},
                {"Time": t_str, "Parameter": "GCS", "Value": 15.0},
                {"Time": t_str, "Parameter": "RespRate", "Value": 16.0},
                {"Time": t_str, "Parameter": "Temp", "Value": 37.0},
            ])

        df = pd.DataFrame(data)
        audit = self.agent.audit_dataframe(df, file_name="patient_sentinels.txt")

        self.assertIsNone(audit["descriptors_cleaned"]["Height"])
        self.assertIsNone(audit["descriptors_cleaned"]["Weight"])
        self.assertEqual(audit["raw_descriptors"]["Height"] if "raw_descriptors" in audit else audit["descriptors_raw"]["Height"], -1.0)
        self.assertIn("Height (sentinel -1)", audit["missing_descriptors"])
        self.assertIn("Weight (sentinel -1)", audit["missing_descriptors"])
        self.assertEqual(audit["data_quality_index"], 93.0)  # 100 - 2 (Height) - 5 (Weight) = 93.0

    def test_valid_alternative_blood_pressure_measurements(self):
        """Verify that missing invasive MAP is NOT flagged when non-invasive cuff measurements exist."""
        data = [
            {"Time": "00:00", "Parameter": "RecordID", "Value": 100002},
            {"Time": "00:00", "Parameter": "Age", "Value": 55.0},
            {"Time": "00:00", "Parameter": "Gender", "Value": 0},
            {"Time": "00:00", "Parameter": "ICUType", "Value": 3},
            {"Time": "00:00", "Parameter": "Height", "Value": 165.0},
            {"Time": "00:00", "Parameter": "Weight", "Value": 70.0},
        ]
        # Patient has ZERO invasive MAP/SysABP/DiasABP, but has non-invasive NISysABP and NIDiasABP
        for h in range(8):
            t_str = f"{h:02d}:00"
            data.extend([
                {"Time": t_str, "Parameter": "HR", "Value": 75.0},
                {"Time": t_str, "Parameter": "NISysABP", "Value": 120.0},
                {"Time": t_str, "Parameter": "NIDiasABP", "Value": 80.0},
                {"Time": t_str, "Parameter": "NIMAP", "Value": 93.0},
                {"Time": t_str, "Parameter": "GCS", "Value": 15.0},
                {"Time": t_str, "Parameter": "RespRate", "Value": 18.0},
                {"Time": t_str, "Parameter": "Temp", "Value": 36.8},
            ])

        df = pd.DataFrame(data)
        audit = self.agent.audit_dataframe(df, file_name="patient_nibp.txt")

        # Must recognize Blood Pressure as SUFFICIENT
        self.assertEqual(audit["critical_vitals_summary"]["BloodPressure"]["status"], "SUFFICIENT")
        self.assertEqual(audit["blood_pressure_summary"]["invasive_count"], 0)
        self.assertEqual(audit["blood_pressure_summary"]["noninvasive_count"], 24)
        self.assertEqual(audit["blood_pressure_summary"]["modality"], "INTERMITTENT_NONINVASIVE_CUFF")
        # No deductions for blood pressure
        self.assertEqual(audit["data_quality_index"], 100.0)

    def test_missing_critical_measurements(self):
        """Verify penalty when critical vital sign domains are completely unmonitored."""
        data = [
            {"Time": "00:00", "Parameter": "RecordID", "Value": 100003},
            {"Time": "00:00", "Parameter": "Age", "Value": 70.0},
            {"Time": "00:00", "Parameter": "Gender", "Value": 1},
            {"Time": "00:00", "Parameter": "ICUType", "Value": 2},
            {"Time": "00:00", "Parameter": "Height", "Value": 175.0},
            {"Time": "00:00", "Parameter": "Weight", "Value": 80.0},
        ]
        # Only Heart Rate and Temp monitored; BloodPressure, GCS, and RespRate are ABSENT (0 obs)
        for h in range(10):
            t_str = f"{h:02d}:00"
            data.extend([
                {"Time": t_str, "Parameter": "HR", "Value": 85.0},
                {"Time": t_str, "Parameter": "Temp", "Value": 37.2},
            ])

        df = pd.DataFrame(data)
        audit = self.agent.audit_dataframe(df, file_name="patient_missing_vitals.txt")

        self.assertEqual(audit["critical_vitals_summary"]["BloodPressure"]["status"], "ABSENT")
        self.assertEqual(audit["critical_vitals_summary"]["GCS"]["status"], "ABSENT")
        self.assertEqual(audit["critical_vitals_summary"]["RespRate"]["status"], "ABSENT")
        # 3 missing domains = 3 * -10 = -30 deduction (100 - 30 = 70.0)
        self.assertLessEqual(audit["data_quality_index"], 70.0)

    def test_sensor_artifacts_vs_pathological_extremes(self):
        """Verify that impossible values are flagged as artifacts, while extreme genuine values are preserved."""
        data = [
            {"Time": "00:00", "Parameter": "RecordID", "Value": 100004},
            {"Time": "00:00", "Parameter": "Age", "Value": 65.0},
            {"Time": "00:00", "Parameter": "Gender", "Value": 1},
            {"Time": "00:00", "Parameter": "ICUType", "Value": 1},
            {"Time": "00:00", "Parameter": "Height", "Value": 170.0},
            {"Time": "00:00", "Parameter": "Weight", "Value": 75.0},
            # Sensor Artifacts (Biological Impossibilities)
            {"Time": "01:00", "Parameter": "HR", "Value": 450.0},     # Impossible HR
            {"Time": "02:00", "Parameter": "pH", "Value": 3.1},      # Impossible arterial pH
            # Genuine Pathological Extremes (Must NOT be flagged as artifacts)
            {"Time": "03:00", "Parameter": "Lactate", "Value": 8.5},  # Severe septic shock
            {"Time": "04:00", "Parameter": "GCS", "Value": 4.0},      # Deep coma
            {"Time": "05:00", "Parameter": "K", "Value": 6.8},        # Dangerous hyperkalemia
        ]
        # Add basic vitals
        for h in range(6, 12):
            t_str = f"{h:02d}:00"
            data.extend([
                {"Time": t_str, "Parameter": "HR", "Value": 90.0},
                {"Time": t_str, "Parameter": "MAP", "Value": 80.0},
                {"Time": t_str, "Parameter": "GCS", "Value": 14.0},
                {"Time": t_str, "Parameter": "RespRate", "Value": 18.0},
                {"Time": t_str, "Parameter": "Temp", "Value": 37.0},
            ])

        df = pd.DataFrame(data)
        audit = self.agent.audit_dataframe(df, file_name="patient_artifacts.txt")

        # Exactly 2 sensor artifacts
        self.assertEqual(audit["sensor_artifacts_detected"], 2)
        # At least 3 genuine pathological alerts
        self.assertGreaterEqual(audit["extreme_pathological_events"], 3)
        # Check that Lactate=8.5 and GCS=4 are in extreme pathological details, NOT sensor artifacts
        path_params = [alert["parameter"] for alert in audit["extreme_pathological_details"]]
        self.assertIn("Lactate", path_params)
        self.assertIn("GCS", path_params)

    def test_empty_record_rejection(self):
        """Verify handling of completely empty patient records."""
        empty_df = pd.DataFrame()
        audit = self.agent.audit_dataframe(empty_df, file_name="empty.txt")
        self.assertEqual(audit["status"], "FAIL")
        self.assertEqual(audit["quality_grade"], "UNRELIABLE_CRITICAL")
        self.assertEqual(audit["data_quality_index"], 0.0)

    def test_dqi_category_boundaries(self):
        """Verify exact DQI quality grade boundaries (>=85, 70-84, 50-69, <50)."""
        agent = self.agent

        # 1. EXCELLENT (>= 85)
        # Perfect record
        d1 = [{"Time": "00:00", "Parameter": "RecordID", "Value": 1}, {"Time": "00:00", "Parameter": "Age", "Value": 50}, {"Time": "00:00", "Parameter": "Gender", "Value": 1}, {"Time": "00:00", "Parameter": "ICUType", "Value": 1}, {"Time": "00:00", "Parameter": "Height", "Value": 170}, {"Time": "00:00", "Parameter": "Weight", "Value": 70}]
        for h in range(12):
            t = f"{h:02d}:00"
            d1.extend([{"Time": t, "Parameter": "HR", "Value": 80}, {"Time": t, "Parameter": "MAP", "Value": 80}, {"Time": t, "Parameter": "GCS", "Value": 15}, {"Time": t, "Parameter": "RespRate", "Value": 16}, {"Time": t, "Parameter": "Temp", "Value": 37}])
        a1 = agent.audit_dataframe(pd.DataFrame(d1))
        self.assertGreaterEqual(a1["data_quality_index"], 85.0)
        self.assertEqual(a1["quality_grade"], "EXCELLENT")
        self.assertEqual(a1["status"], "PASS")

        # 2. ACCEPTABLE (70 - 84)
        # Missing Height (-2), Missing Weight (-5), Sparse RespRate (< 4 obs, -4), Sparse Temp (-4) -> 100 - 15 = 85.0 -> with 1 more missing descriptor -> 80.0
        d2 = list(d1[:4])  # Missing Height and Weight
        for h in range(10):
            t = f"{h:02d}:00"
            d2.extend([{"Time": t, "Parameter": "HR", "Value": 80}, {"Time": t, "Parameter": "MAP", "Value": 80}, {"Time": t, "Parameter": "GCS", "Value": 15}])
        # RespRate only 2 obs, Temp 2 obs
        d2.extend([{"Time": "01:00", "Parameter": "RespRate", "Value": 16}, {"Time": "02:00", "Parameter": "RespRate", "Value": 16}, {"Time": "01:00", "Parameter": "Temp", "Value": 37}, {"Time": "02:00", "Parameter": "Temp", "Value": 37}])
        a2 = agent.audit_dataframe(pd.DataFrame(d2))
        self.assertGreaterEqual(a2["data_quality_index"], 70.0)
        self.assertLess(a2["data_quality_index"], 85.0)
        self.assertEqual(a2["quality_grade"], "ACCEPTABLE")
        self.assertEqual(a2["status"], "PASS")

        # 3. SUBOPTIMAL_WARNING (50 - 69)
        # Missing Age (-15), Missing Weight (-5), Missing Height (-2), Absent GCS (-10), Absent Temp (-10) -> 100 - 42 = 58.0
        d3 = [{"Time": "00:00", "Parameter": "RecordID", "Value": 1}, {"Time": "00:00", "Parameter": "Gender", "Value": 1}, {"Time": "00:00", "Parameter": "ICUType", "Value": 1}]
        for h in range(10):
            t = f"{h:02d}:00"
            d3.extend([{"Time": t, "Parameter": "HR", "Value": 80}, {"Time": t, "Parameter": "MAP", "Value": 80}, {"Time": t, "Parameter": "RespRate", "Value": 16}])
        a3 = agent.audit_dataframe(pd.DataFrame(d3))
        self.assertGreaterEqual(a3["data_quality_index"], 50.0)
        self.assertLess(a3["data_quality_index"], 70.0)
        self.assertEqual(a3["quality_grade"], "SUBOPTIMAL_WARNING")
        self.assertEqual(a3["status"], "WARNING")

        # 4. UNRELIABLE_CRITICAL (< 50)
        # Only 2 observations total, missing all descriptors
        d4 = [{"Time": "01:00", "Parameter": "HR", "Value": 80}, {"Time": "02:00", "Parameter": "HR", "Value": 80}]
        a4 = agent.audit_dataframe(pd.DataFrame(d4))
        self.assertLess(a4["data_quality_index"], 50.0)
        self.assertEqual(a4["quality_grade"], "UNRELIABLE_CRITICAL")
        self.assertEqual(a4["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
