"""
Validation and Leakage Verification Suite for Phase 3 and Phase 4
Tests patient separation, shape integrity, non-leakage, reproducibility,
and metric consistency across all tabular datasets and models.
"""

import json
import os
import unittest
import numpy as np
import pandas as pd


class TestPhase3Phase4Validation(unittest.TestCase):
    """Rigorous technical tests for Phase 3 Feature Engineering and Phase 4 XGBoost."""

    @classmethod
    def setUpClass(cls):
        cls.base_dir = r"D:\ICU_Prediction"
        cls.tabular_dir = os.path.join(cls.base_dir, "data", "processed", "tabular")
        cls.reports_dir = os.path.join(cls.base_dir, "reports")
        cls.models_dir = os.path.join(cls.base_dir, "models", "xgboost")

        cls.ff_48h = np.load(os.path.join(cls.tabular_dir, "tabular_forward_fill_48h.npz"))
        cls.ff_24h = np.load(os.path.join(cls.tabular_dir, "tabular_forward_fill_24h.npz"))
        cls.si_48h = np.load(os.path.join(cls.tabular_dir, "tabular_simple_imputation_48h.npz"))
        cls.si_24h = np.load(os.path.join(cls.tabular_dir, "tabular_simple_imputation_24h.npz"))

        with open(os.path.join(cls.reports_dir, "xgboost_results.json"), "r") as f:
            cls.results = json.load(f)

        cls.metrics_df = pd.read_csv(os.path.join(cls.reports_dir, "xgboost_metrics.csv"))

    def test_cohort_separation_and_record_counts(self):
        """Verify strict sample counts: Train 3200, Val 800, Test B 4000, Test C 4000."""
        self.assertEqual(self.ff_48h["X_train"].shape[0], 3200)
        self.assertEqual(self.ff_48h["y_train"].shape[0], 3200)
        self.assertEqual(self.ff_48h["X_val"].shape[0], 800)
        self.assertEqual(self.ff_48h["y_val"].shape[0], 800)
        self.assertEqual(self.ff_48h["X_test_b"].shape[0], 4000)
        self.assertEqual(self.ff_48h["y_test_b"].shape[0], 4000)
        self.assertEqual(self.ff_48h["X_test_c"].shape[0], 4000)
        self.assertEqual(self.ff_48h["y_test_c"].shape[0], 4000)

    def test_patient_id_disjointness(self):
        """Verify zero overlap between patient IDs across splits."""
        manifest = pd.read_csv(os.path.join(self.base_dir, "data", "processed", "patient_split_manifest.csv"))
        train_ids = set(manifest[manifest["Partition"] == "train"]["RecordID"])
        val_ids = set(manifest[manifest["Partition"] == "val"]["RecordID"])
        test_b_ids = set(manifest[manifest["Partition"] == "test_b"]["RecordID"])
        test_c_ids = set(manifest[manifest["Partition"] == "test_c"]["RecordID"])

        self.assertEqual(len(train_ids), 3200)
        self.assertEqual(len(val_ids), 800)
        self.assertEqual(len(test_b_ids), 4000)
        self.assertEqual(len(test_c_ids), 4000)

        self.assertEqual(len(train_ids.intersection(val_ids)), 0, "Train and Val IDs overlap!")
        self.assertEqual(len(train_ids.intersection(test_b_ids)), 0, "Train and Test B IDs overlap!")
        self.assertEqual(len(train_ids.intersection(test_c_ids)), 0, "Train and Test C IDs overlap!")
        self.assertEqual(len(val_ids.intersection(test_b_ids)), 0, "Val and Test B IDs overlap!")
        self.assertEqual(len(val_ids.intersection(test_c_ids)), 0, "Val and Test C IDs overlap!")
        self.assertEqual(len(test_b_ids.intersection(test_c_ids)), 0, "Test B and Test C IDs overlap!")

    def test_feature_count_and_null_absence(self):
        """Verify 334 features and zero NaN/Inf values across all tabular datasets."""
        for name, data in [
            ("ff_48h", self.ff_48h),
            ("ff_24h", self.ff_24h),
            ("si_48h", self.si_48h),
            ("si_24h", self.si_24h),
        ]:
            for split in ["X_train", "X_val", "X_test_b", "X_test_c"]:
                X = data[split]
                self.assertEqual(X.shape[1], 334, f"{name} {split} does not have 334 features.")
                self.assertFalse(np.isnan(X).any(), f"NaN detected in {name} {split}")
                self.assertFalse(np.isinf(X).any(), f"Inf detected in {name} {split}")

    def test_target_leakage_absence(self):
        """Confirm forbidden variables (RecordID, Survival, Length_of_stay, SAPS-I, SOFA) are absent."""
        feature_names = [str(f) for f in self.ff_48h["feature_names"]]
        forbidden = ["RecordID", "Survival", "Length_of_stay", "SAPS-I", "SOFA", "SAPS", "In-hospital_death"]
        for feat in feature_names:
            for forb in forbidden:
                self.assertNotEqual(feat.lower(), forb.lower(), f"Forbidden target/metadata feature detected: {feat}")

    def test_scale_pos_weight_exact_calculation(self):
        """Verify scale_pos_weight was computed strictly from training labels."""
        y_train = self.ff_48h["y_train"]
        neg = np.sum(y_train == 0)
        pos = np.sum(y_train == 1)
        expected_weight = neg / pos
        saved_weight = self.results["primary_model_hyperparameters"]["scale_pos_weight"]
        self.assertAlmostEqual(expected_weight, saved_weight, places=3)

    def test_model_files_exist(self):
        """Verify all three fitted XGBoost models were serialized to JSON format."""
        expected_models = [
            "best_xgboost_forward_fill_48h.json",
            "best_xgboost_simple_imputation_48h.json",
            "best_xgboost_forward_fill_24h.json",
        ]
        for model_file in expected_models:
            full_path = os.path.join(self.models_dir, model_file)
            self.assertTrue(os.path.exists(full_path), f"Model file missing: {full_path}")
            self.assertGreater(os.path.getsize(full_path), 1000, f"Model file too small: {full_path}")

    def test_validation_threshold_optimization(self):
        """Verify threshold optimization improved or matched F1 on validation set."""
        val_default = self.metrics_df[
            (self.metrics_df["Model"] == "XGBoost") &
            (self.metrics_df["Cohort"] == "Validation") &
            (self.metrics_df["Threshold_Type"] == "Default 0.50")
        ].iloc[0]

        val_optimal = self.metrics_df[
            (self.metrics_df["Model"] == "XGBoost") &
            (self.metrics_df["Cohort"] == "Validation") &
            (self.metrics_df["Threshold_Type"] == "Optimal Tuned")
        ].iloc[0]

        self.assertGreaterEqual(
            val_optimal["f1_score"],
            val_default["f1_score"],
            "Optimal threshold failed to maximize validation F1-score"
        )


if __name__ == "__main__":
    unittest.main()
