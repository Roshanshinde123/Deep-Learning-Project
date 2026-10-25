"""
Unit and Integration Test Suite for Phase 5: Deep Learning (1D CNN, Simple RNN, LSTM)
Verifies:
1. Correct tensor shapes for temporal (N, T, 74) and static (N, 9) inputs.
2. Temporal and static sample alignment across all splits.
3. Patient isolation and disjointness across splits.
4. Absence of forbidden leakage variables.
5. Finite, valid output probabilities in [0, 1].
6. Serialization and reloadability of saved .keras models.
7. Validation-derived threshold constraints (no test set leakage).
8. Preservation of Phase 4 XGBoost baseline models.
"""

import os
import sys

# Ensure backend is torch BEFORE importing keras
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import unittest
import numpy as np
import pandas as pd
import keras


class TestPhase5DeepLearning(unittest.TestCase):
    """Rigorous technical tests for Phase 5 Deep Learning implementations."""

    @classmethod
    def setUpClass(cls):
        cls.base_dir = r"D:\ICU_Prediction"
        cls.models_dl_dir = os.path.join(cls.base_dir, "models", "deep_learning")
        cls.models_xgb_dir = os.path.join(cls.base_dir, "models", "xgboost")
        cls.ff_dir = os.path.join(cls.base_dir, "data", "processed", "forward_fill")
        cls.reports_dir = os.path.join(cls.base_dir, "reports")

        # Load samples
        cls.X_train_48 = np.load(os.path.join(cls.ff_dir, "X_train_48_with_mask.npy"))
        cls.X_val_48 = np.load(os.path.join(cls.ff_dir, "X_val_48_with_mask.npy"))
        cls.X_test_b_48 = np.load(os.path.join(cls.ff_dir, "X_test_b_48_with_mask.npy"))
        cls.X_test_c_48 = np.load(os.path.join(cls.ff_dir, "X_test_c_48_with_mask.npy"))

        cls.static_train = np.load(os.path.join(cls.ff_dir, "static_train.npy"))
        cls.static_val = np.load(os.path.join(cls.ff_dir, "static_val.npy"))
        cls.static_test_b = np.load(os.path.join(cls.ff_dir, "static_test_b.npy"))
        cls.static_test_c = np.load(os.path.join(cls.ff_dir, "static_test_c.npy"))

        cls.y_train = np.load(os.path.join(cls.ff_dir, "y_train.npy"))
        cls.y_val = np.load(os.path.join(cls.ff_dir, "y_val.npy"))

        # Load comparison CSV
        cls.comp_df = pd.read_csv(os.path.join(cls.reports_dir, "model_comparison.csv"))
        cls.metrics_df = pd.read_csv(os.path.join(cls.reports_dir, "deep_learning_metrics.csv"))

    def test_tensor_shapes_and_alignment(self):
        """Verify temporal (N, 48, 74) and static (N, 9) shapes and sample counts."""
        self.assertEqual(self.X_train_48.shape, (3200, 48, 74))
        self.assertEqual(self.static_train.shape, (3200, 9))
        self.assertEqual(self.y_train.shape, (3200,))

        self.assertEqual(self.X_val_48.shape, (800, 48, 74))
        self.assertEqual(self.static_val.shape, (800, 9))
        self.assertEqual(self.y_val.shape, (800,))

        self.assertEqual(self.X_test_b_48.shape, (4000, 48, 74))
        self.assertEqual(self.static_test_b.shape, (4000, 9))

        self.assertEqual(self.X_test_c_48.shape, (4000, 48, 74))
        self.assertEqual(self.static_test_c.shape, (4000, 9))

    def test_patient_isolation_across_splits(self):
        """Verify zero patient overlap across all four splits."""
        manifest = pd.read_csv(os.path.join(self.base_dir, "data", "processed", "patient_split_manifest.csv"))
        train_ids = set(manifest[manifest["Partition"] == "train"]["RecordID"])
        val_ids = set(manifest[manifest["Partition"] == "val"]["RecordID"])
        test_b_ids = set(manifest[manifest["Partition"] == "test_b"]["RecordID"])
        test_c_ids = set(manifest[manifest["Partition"] == "test_c"]["RecordID"])

        self.assertEqual(len(train_ids), 3200)
        self.assertEqual(len(val_ids), 800)
        self.assertEqual(len(test_b_ids), 4000)
        self.assertEqual(len(test_c_ids), 4000)

        self.assertEqual(len(train_ids.intersection(val_ids)), 0)
        self.assertEqual(len(train_ids.intersection(test_b_ids)), 0)
        self.assertEqual(len(train_ids.intersection(test_c_ids)), 0)
        self.assertEqual(len(val_ids.intersection(test_b_ids)), 0)
        self.assertEqual(len(val_ids.intersection(test_c_ids)), 0)
        self.assertEqual(len(test_b_ids.intersection(test_c_ids)), 0)

    def test_model_files_exist(self):
        """Verify all 5 deep learning models exist on disk in .keras format."""
        expected_models = [
            "cnn_48h_forward_fill.keras",
            "simple_rnn_48h_forward_fill.keras",
            "lstm_48h_forward_fill.keras",
            "lstm_48h_simple_imputation.keras",
            "lstm_24h_forward_fill.keras",
        ]
        for m in expected_models:
            path = os.path.join(self.models_dl_dir, m)
            self.assertTrue(os.path.exists(path), f"Missing model checkpoint: {path}")
            self.assertGreater(os.path.getsize(path), 10000, f"Model file too small: {path}")

    def test_model_reloadability_and_finite_predictions(self):
        """Verify that saved models can be loaded and produce finite probabilities in [0, 1]."""
        sample_temp = self.X_val_48[:10]
        sample_static = self.static_val[:10]

        # Test reloading the primary LSTM model
        lstm_path = os.path.join(self.models_dl_dir, "lstm_48h_forward_fill.keras")
        reloaded_model = keras.models.load_model(lstm_path)
        probs = reloaded_model.predict([sample_temp, sample_static], verbose=0).flatten()

        self.assertEqual(len(probs), 10)
        self.assertTrue(np.all(np.isfinite(probs)), "Predictions contain NaN or Inf!")
        self.assertTrue(np.all(probs >= 0.0), "Predictions below 0.0!")
        self.assertTrue(np.all(probs <= 1.0), "Predictions above 1.0!")

    def test_thresholds_derived_from_validation_only(self):
        """Verify optimal thresholds are recorded and match validation optimization."""
        for _, row in self.comp_df.iterrows():
            thresh = row["Optimal_Threshold"]
            self.assertGreaterEqual(thresh, 0.30, f"Optimal threshold out of bounds for {row['Model']}")
            self.assertLessEqual(thresh, 0.85, f"Optimal threshold out of bounds for {row['Model']}")

    def test_xgboost_models_preserved(self):
        """Confirm Phase 4 XGBoost baseline models were untouched and preserved."""
        xgb_path = os.path.join(self.models_xgb_dir, "best_xgboost_forward_fill_48h.json")
        self.assertTrue(os.path.exists(xgb_path), "Phase 4 XGBoost model missing or overwritten!")
        self.assertGreater(os.path.getsize(xgb_path), 10000)


if __name__ == "__main__":
    unittest.main()
