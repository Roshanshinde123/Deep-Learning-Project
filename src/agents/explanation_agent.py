"""
Explanation Agent (Phase 6.3)
Agentic AI-Assisted ICU Mortality Risk Prediction from the First 48 Hours.

Responsibilities:
1. Receive validated patient data, Data Quality Agent results, and Prediction Agent outputs.
2. Generate transparent local feature explanations for XGBoost using SHAP TreeExplainer:
   - Rank top positive contributors (increasing predicted mortality probability).
   - Rank top negative contributors (decreasing predicted mortality probability / protective).
   - Report actual patient feature values alongside exact SHAP attribution scores.
   - Optionally render local SHAP waterfall plots saved to reports/figures/.
3. Generate principled LSTM temporal and feature attributions via perturbation analysis:
   - Feature occlusion (Delta P = P_original - P_occluded) to identify physiological variable impacts.
   - Temporal segment occlusion (Quarters Q1-Q4) to identify critical time windows in the stay.
4. Perform physiological temporal trend analysis across verified clinical variables:
   - First value, last value, net delta, trajectory trend (INCREASING, DECREASING, STABLE).
   - Observation counts and missingness percentage across the prediction horizon.
   - Report strictly on variables actually present in the patient record.
5. Provide in-depth Model Discordance analysis when XGBoost and LSTM disagree:
   - Compare probabilities, thresholds, and architectural perspectives (tabular vs sequential).
   - Strictly prohibit probability averaging or artificial consensus fabrication.
6. Connect with Data Quality findings:
   - Propagate DQI, quality grades, missing descriptors, and pathological alerts.
   - Flag top-contributing features that were heavily imputed or sparsely measured as limitations.
7. Synthesize research-oriented, human-readable explanations:
   - Enforce strict clinical safety language (model associations, not medical causation; zero treatment advice).

RESEARCH DISCLAIMER:
Attributions and explanations represent mathematical associations derived from retrospective models.
They do NOT represent clinical causality, etiology, or diagnostic directives, and must NOT be used
for clinical treatment or medication decisions.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

# Set environment variables for PyTorch / Keras backend before imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import keras

# Project internal imports
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import DYNAMIC_PARAMS, STATIC_PARAMS, parse_time_to_minutes
from src.agents.data_quality_agent import DataQualityAgent
from src.agents.prediction_agent import PredictionAgent

logger = logging.getLogger("ExplanationAgent")


class ExplanationAgent:
    """
    Autonomous Explanation Agent for Interpreting ICU Mortality Predictions.
    Provides dual-model interpretability: SHAP TreeExplainer for XGBoost and
    perturbation-based temporal/feature attribution for LSTM.
    """

    # Core clinical physiological variables monitored for longitudinal trajectory trends
    KEY_TREND_VARIABLES = [
        "HR", "Temp", "GCS", "BUN", "Urine", "PaCO2", "RespRate",
        "Na", "pH", "Lactate", "MAP", "Creatinine", "Glucose", "WBC",
        "Platelets", "SysABP", "DiasABP", "SaO2", "FiO2", "K"
    ]

    def __init__(
        self,
        base_dir: Optional[str] = None,
        prediction_agent: Optional[PredictionAgent] = None,
        data_quality_agent: Optional[DataQualityAgent] = None,
    ):
        """
        Initializes the Explanation Agent.

        Parameters:
            base_dir: Root directory of the project.
            prediction_agent: Optional pre-instantiated PredictionAgent instance.
            data_quality_agent: Optional pre-instantiated DataQualityAgent instance.
        """
        self.base_dir = str(base_dir or ROOT_DIR)
        self.dq_agent = data_quality_agent or DataQualityAgent()
        self.prediction_agent = prediction_agent or PredictionAgent(
            base_dir=self.base_dir, data_quality_agent=self.dq_agent
        )

        # Cache for SHAP TreeExplainer instances per horizon
        self._shap_explainers: Dict[int, shap.TreeExplainer] = {}

    # =========================================================================
    # 1. SHAP TREE-EXPLAINER MANAGEMENT
    # =========================================================================

    def _get_tree_explainer(self, horizon_hours: int) -> shap.TreeExplainer:
        """Retrieves or instantiates cached SHAP TreeExplainer for XGBoost model."""
        if horizon_hours in self._shap_explainers:
            return self._shap_explainers[horizon_hours]

        xgb_model = self.prediction_agent._load_model("xgboost", horizon_hours)
        explainer = shap.TreeExplainer(xgb_model)
        self._shap_explainers[horizon_hours] = explainer
        return explainer

    # =========================================================================
    # 2. XGBOOST SHAP LOCAL EXPLANATION
    # =========================================================================

    def explain_xgboost(
        self,
        X_tabular: np.ndarray,
        feature_names: List[str],
        patient_id: str,
        horizon_hours: int = 48,
        top_n: int = 10,
        generate_plots: bool = False,
        output_plot_dir: str = "reports/figures",
    ) -> Dict[str, Any]:
        """
        Generates comprehensive local SHAP explanations for XGBoost:
        1. Computes feature-level SHAP values via TreeExplainer.
        2. Separates top positive and top negative contributors with actual feature values.
        3. Optionally generates a SHAP waterfall plot.

        Returns:
            Dict with top_positive_contributors, top_negative_contributors, shap_values, base_value.
        """
        explainer = self._get_tree_explainer(horizon_hours)
        shap_exp = explainer(X_tabular)
        shap_exp.feature_names = feature_names

        shap_values_arr = shap_exp.values[0]  # Shape: (334,)
        base_val = float(shap_exp.base_values[0]) if hasattr(shap_exp.base_values, "__getitem__") else float(shap_exp.base_values)

        all_attributions = []
        shap_dict = {}

        for i, f_name in enumerate(feature_names):
            act_val = float(X_tabular[0, i])
            shap_val = float(shap_values_arr[i])
            shap_dict[f_name] = round(shap_val, 4)

            all_attributions.append({
                "feature": f_name,
                "actual_value": round(act_val, 4),
                "shap_value": round(shap_val, 4),
                "abs_shap": abs(shap_val),
            })

        # Separate positive (increasing risk) and negative (reducing risk / protective)
        pos_features = [f for f in all_attributions if f["shap_value"] > 0]
        neg_features = [f for f in all_attributions if f["shap_value"] < 0]

        top_pos = sorted(pos_features, key=lambda x: x["shap_value"], reverse=True)[:top_n]
        top_neg = sorted(neg_features, key=lambda x: x["shap_value"])[:top_n]  # Most negative first

        # Optional Waterfall Plot Generation
        plot_path = None
        if generate_plots:
            full_plot_dir = os.path.join(self.base_dir, output_plot_dir)
            os.makedirs(full_plot_dir, exist_ok=True)
            plot_path = os.path.join(full_plot_dir, f"shap_{patient_id}_{horizon_hours}h.png")

            try:
                fig = plt.figure(figsize=(10, 6.5))
                shap.plots.waterfall(shap_exp[0], max_display=min(12, top_n), show=False)
                plt.title(
                    f"SHAP Waterfall Attribution: Patient #{patient_id} ({horizon_hours}h Horizon)",
                    fontsize=11, fontweight="bold", pad=15
                )
                plt.tight_layout()
                plt.savefig(plot_path, dpi=200, bbox_inches="tight")
                plt.close()
            except Exception as e:
                logger.warning(f"Failed to generate SHAP waterfall plot: {e}")
                plot_path = None

        return {
            "top_positive_contributors": top_pos,
            "top_negative_contributors": top_neg,
            "base_value": round(base_val, 4),
            "shap_values": shap_dict,
            "waterfall_plot_path": plot_path,
        }

    # =========================================================================
    # 3. LSTM PERTURBATION-BASED ATTRIBUTION
    # =========================================================================

    def explain_lstm(
        self,
        temporal_tensor: np.ndarray,
        static_tensor: np.ndarray,
        baseline_prob: float,
        horizon_hours: int = 48,
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """
        Generates principled feature-level and temporal-window attributions for LSTM:
        Method: Occlusion Perturbation Analysis (Delta P = P_baseline - P_occluded).

        1. Feature Occlusion: Each dynamic variable's values and mask are zeroed out (neutral baseline).
        2. Temporal Window Occlusion: Temporal quarters (Q1-Q4) are occluded to isolate time-window importance.

        Returns:
            Dict with important_features, important_time_periods, attribution_method.
        """
        lstm_model = self.prediction_agent._load_model("lstm", horizon_hours)
        num_dynamic = len(DYNAMIC_PARAMS)

        # -------------------------------------------------------------
        # A. DYNAMIC FEATURE PERTURBATION (37 FEATURES)
        # -------------------------------------------------------------
        # Construct batched perturbations for fast single-pass evaluation
        batch_temp = []
        batch_stat = []
        feature_labels = []

        for f in range(num_dynamic):
            temp_copy = temporal_tensor.copy()
            # Occlude feature values (channel f) and observation mask (channel 37 + f)
            temp_copy[0, :, f] = 0.0
            temp_copy[0, :, num_dynamic + f] = 0.0

            batch_temp.append(temp_copy[0])
            batch_stat.append(static_tensor[0])
            feature_labels.append(DYNAMIC_PARAMS[f])

        # -------------------------------------------------------------
        # B. STATIC DESCRIPTOR PERTURBATION (STATIC FEATURES)
        # -------------------------------------------------------------
        static_names = [
            "Age_scaled", "Height_scaled", "Weight_admission_scaled",
            "Gender", "MechVent_ever_recorded", "ICUType"
        ]
        for s_idx, s_name in enumerate(static_names[:5]):
            stat_copy = static_tensor.copy()
            stat_copy[0, s_idx] = 0.0  # Continuous standard mean or reference level

            batch_temp.append(temporal_tensor[0])
            batch_stat.append(stat_copy[0])
            feature_labels.append(s_name)

        # -------------------------------------------------------------
        # C. TEMPORAL WINDOW PERTURBATION (4 QUARTERS)
        # -------------------------------------------------------------
        quarters = [
            (0, horizon_hours // 4, f"Q1 (Hours 00-{horizon_hours // 4:02d})"),
            (horizon_hours // 4, horizon_hours // 2, f"Q2 (Hours {horizon_hours // 4:02d}-{horizon_hours // 2:02d})"),
            (horizon_hours // 2, 3 * horizon_hours // 4, f"Q3 (Hours {horizon_hours // 2:02d}-{3 * horizon_hours // 4:02d})"),
            (3 * horizon_hours // 4, horizon_hours, f"Q4 (Hours {3 * horizon_hours // 4:02d}-{horizon_hours:02d})"),
        ]
        window_labels = []

        for start_h, end_h, w_name in quarters:
            temp_w = temporal_tensor.copy()
            # Occlude all dynamic measurements and masks in this temporal window
            temp_w[0, start_h:end_h, :] = 0.0

            batch_temp.append(temp_w[0])
            batch_stat.append(static_tensor[0])
            window_labels.append(w_name)

        # Run fast batched forward pass
        batch_temp_arr = np.array(batch_temp, dtype=np.float32)
        batch_stat_arr = np.array(batch_stat, dtype=np.float32)

        try:
            preds_arr = lstm_model.predict([batch_temp_arr, batch_stat_arr], verbose=0).flatten()
        except Exception as e:
            logger.warning(f"LSTM batched perturbation failed: {e}")
            return {
                "important_features": [],
                "important_time_periods": [],
                "attribution_method": "Occlusion perturbation failed.",
            }

        # Separate feature attributions from temporal window attributions
        n_features = len(feature_labels)
        feature_preds = preds_arr[:n_features]
        window_preds = preds_arr[n_features:]

        feature_attributions = []
        for i, f_name in enumerate(feature_labels):
            p_perturbed = float(feature_preds[i])
            delta_p = float(baseline_prob - p_perturbed)
            feature_attributions.append({
                "feature": f_name,
                "delta_probability": round(delta_p, 4),
                "perturbed_probability": round(p_perturbed, 4),
                "impact_direction": "RISK_INCREASING" if delta_p > 0.0005 else "RISK_DECREASING" if delta_p < -0.0005 else "NEUTRAL",
                "abs_delta": abs(delta_p),
            })

        ranked_features = sorted(feature_attributions, key=lambda x: x["abs_delta"], reverse=True)[:top_n]

        temporal_attributions = []
        for j, w_name in enumerate(window_labels):
            p_w = float(window_preds[j])
            delta_w = float(baseline_prob - p_w)
            temporal_attributions.append({
                "time_period": w_name,
                "delta_probability": round(delta_w, 4),
                "perturbed_probability": round(p_w, 4),
                "impact_direction": "RISK_INCREASING" if delta_w > 0.0005 else "RISK_DECREASING" if delta_w < -0.0005 else "NEUTRAL",
                "abs_delta": abs(delta_w),
            })

        ranked_time_periods = sorted(temporal_attributions, key=lambda x: x["abs_delta"], reverse=True)

        return {
            "important_features": ranked_features,
            "important_time_periods": ranked_time_periods,
            "attribution_method": "Temporal and feature occlusion perturbation (Delta P = P_baseline - P_occluded)",
        }

    # =========================================================================
    # 4. TEMPORAL TREND ANALYSIS (CLINICAL MEASUREMENTS)
    # =========================================================================

    def analyze_temporal_trends(
        self, df_patient: pd.DataFrame, horizon_hours: int = 48
    ) -> List[Dict[str, Any]]:
        """
        Extracts temporal trajectory summaries for verified clinical measurements:
        First observed value, last observed value, net delta, trajectory direction,
        observation frequency, and missingness percentage.
        Strictly reports variables actually present in the patient record.
        """
        max_minutes = horizon_hours * 60
        trend_records = []

        # Filter strictly within prediction horizon
        df_valid = df_patient.copy()
        df_valid["minutes"] = df_valid["Time"].apply(
            lambda t: parse_time_to_minutes(str(t).strip()) if ":" in str(t) else -1
        )
        df_valid = df_valid[(df_valid["minutes"] >= 0) & (df_valid["minutes"] <= max_minutes)]

        available_params = set(df_valid["Parameter"].unique())
        params_to_inspect = [p for p in self.KEY_TREND_VARIABLES if p in available_params]

        for param in params_to_inspect:
            p_data = df_valid[df_valid["Parameter"] == param].dropna(subset=["Value"])
            if p_data.empty:
                continue

            p_data = p_data.sort_values("minutes")
            vals = p_data["Value"].values
            times = p_data["Time"].values

            obs_count = len(vals)
            first_val = float(vals[0])
            last_val = float(vals[-1])
            first_time = str(times[0])
            last_time = str(times[-1])
            delta = float(last_val - first_val)

            # Determine direction (using 5% relative change or 0.5 absolute threshold)
            threshold = max(0.5, abs(first_val) * 0.05) if first_val != 0 else 0.5
            if delta > threshold:
                trend_dir = "INCREASING"
            elif delta < -threshold:
                trend_dir = "DECREASING"
            else:
                trend_dir = "STABLE"

            # Missingness across hourly grid
            hours_observed = len(set(p_data["minutes"] // 60))
            missing_pct = max(0.0, 100.0 * (1.0 - (hours_observed / float(horizon_hours))))

            trend_records.append({
                "parameter": param,
                "first_value": round(first_val, 2),
                "first_time": first_time,
                "last_value": round(last_val, 2),
                "last_time": last_time,
                "delta": round(delta, 2),
                "trend": trend_dir,
                "observation_count": obs_count,
                "missingness_percentage": round(missing_pct, 1),
            })

        return trend_records

    # =========================================================================
    # 5. MODEL DISCORDANCE ANALYSIS
    # =========================================================================

    def analyze_model_discordance(
        self,
        pred_res: Dict[str, Any],
        top_xgb_features: List[Dict[str, Any]],
        top_lstm_features: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Analyzes and explains divergence between XGBoost and LSTM predictions:
        Compares probability levels, decision thresholds, and distinct model perspectives.
        Strictly prohibits probability averaging or consensus fabrication.
        """
        xgb_info = pred_res.get("predictions", {}).get("xgboost", {})
        lstm_info = pred_res.get("predictions", {}).get("lstm", {})

        p_xgb = float(xgb_info.get("predicted_probability", 0.0))
        th_xgb = float(xgb_info.get("threshold", 0.58))
        dec_xgb = xgb_info.get("threshold_decision", "UNKNOWN")

        p_lstm = float(lstm_info.get("predicted_probability", 0.0))
        th_lstm = float(lstm_info.get("threshold", 0.58))
        dec_lstm = lstm_info.get("threshold_decision", "UNKNOWN")

        is_discordant = dec_xgb != dec_lstm
        prob_diff = abs(p_xgb - p_lstm)

        top_xgb_names = [f["feature"] for f in top_xgb_features[:3]]
        top_lstm_names = [f["feature"] for f in top_lstm_features[:3]]

        if is_discordant:
            explanation = (
                f"MODEL DISCORDANCE DETECTED: Primary XGBoost evaluated the patient as {dec_xgb} "
                f"(p={p_xgb:.4f}, threshold={th_xgb}), whereas Secondary LSTM evaluated the patient as "
                f"{dec_lstm} (p={p_lstm:.4f}, threshold={th_lstm}). The absolute probability discrepancy "
                f"is {prob_diff:.4f}. "
                f"This divergence occurs because XGBoost is a tree-based tabular model heavily influenced "
                f"by aggregate summary statistics and extreme static/lab values ({', '.join(top_xgb_names)}), "
                f"whereas LSTM processes sequential longitudinal trajectories ({', '.join(top_lstm_names)}) "
                f"and may capture temporal stability or late decompensation differently. "
                "Clinicians must review individual vital trajectories rather than relying on a heuristic consensus."
            )
        else:
            explanation = (
                f"MODEL CONCORDANCE: Both XGBoost ({dec_xgb}, p={p_xgb:.4f}) and LSTM ({dec_lstm}, p={p_lstm:.4f}) "
                f"reached concordant risk classifications relative to their validation-selected thresholds. "
                f"The models exhibit an absolute probability difference of {prob_diff:.4f}."
            )

        return {
            "model_discordance": is_discordant,
            "xgboost_probability": p_xgb,
            "xgboost_threshold": th_xgb,
            "xgboost_decision": dec_xgb,
            "lstm_probability": p_lstm,
            "lstm_threshold": th_lstm,
            "lstm_decision": dec_lstm,
            "probability_difference": round(prob_diff, 4),
            "explanation": explanation,
        }

    # =========================================================================
    # 6. HUMAN-READABLE SYNTHESIS & RESEARCH NARRATIVE
    # =========================================================================

    def _synthesize_human_readable_summary(
        self,
        patient_id: str,
        horizon_hours: int,
        pred_res: Dict[str, Any],
        top_pos_xgb: List[Dict[str, Any]],
        top_neg_xgb: List[Dict[str, Any]],
        top_lstm: List[Dict[str, Any]],
        time_periods: List[Dict[str, Any]],
        trends: List[Dict[str, Any]],
        discordance_info: Dict[str, Any],
        dq_info: Dict[str, Any],
    ) -> str:
        """
        Synthesizes a research-oriented narrative summary adhering to clinical safety rules:
        - Never states 'patient will die' or 'this variable caused death'.
        - Emphasizes model associations and statistical contributions.
        - Strictly provides zero clinical treatment or intervention advice.
        """
        p_xgb = pred_res.get("predictions", {}).get("xgboost", {}).get("predicted_probability", 0.0)
        dec_xgb = pred_res.get("predictions", {}).get("xgboost", {}).get("threshold_decision", "UNKNOWN")
        th_xgb = pred_res.get("predictions", {}).get("xgboost", {}).get("threshold", 0.58)

        p_lstm = pred_res.get("predictions", {}).get("lstm", {}).get("predicted_probability", 0.0)
        dec_lstm = pred_res.get("predictions", {}).get("lstm", {}).get("threshold_decision", "UNKNOWN")
        th_lstm = pred_res.get("predictions", {}).get("lstm", {}).get("threshold", 0.58)

        # Top XGBoost contributors
        pos_names = [f"{f['feature']} (val: {f['actual_value']}, SHAP: +{f['shap_value']:.3f})" for f in top_pos_xgb[:3]]
        neg_names = [f"{f['feature']} (val: {f['actual_value']}, SHAP: {f['shap_value']:.3f})" for f in top_neg_xgb[:3]]

        pos_str = f"Strongest model contributors increasing predicted risk: {', '.join(pos_names)}." if pos_names else "No strong risk-increasing tabular features detected."
        neg_str = f"Strongest protective associations reducing predicted risk: {', '.join(neg_names)}." if neg_names else "No strong protective tabular features detected."

        # Key trends
        trend_sentences = []
        for tr in trends[:4]:
            trend_sentences.append(
                f"{tr['parameter']} transitioned from {tr['first_value']} ({tr['first_time']}) to {tr['last_value']} ({tr['last_time']}) [{tr['trend']}]"
            )
        trend_str = f"Longitudinal physiological trends observed: {'; '.join(trend_sentences)}." if trend_sentences else "Sparse longitudinal measurements recorded."

        # Top LSTM temporal period
        top_w = time_periods[0] if time_periods else {"time_period": "Full window", "delta_probability": 0.0}
        lstm_str = (
            f"Sequential LSTM attribution indicated that {top_w['time_period']} exerted the highest dynamic impact "
            f"(Delta P = {top_w['delta_probability']:+.4f}) on temporal risk encoding."
        )

        discordance_str = discordance_info["explanation"]

        narrative = (
            f"Prediction Summary for Patient #{patient_id} ({horizon_hours}-Hour ICU Horizon):\n"
            f"Primary XGBoost assigned a model-estimated mortality probability of {p_xgb:.4f} "
            f"({dec_xgb} validation threshold {th_xgb:.2f}). "
            f"Secondary LSTM assigned a model-estimated mortality probability of {p_lstm:.4f} "
            f"({dec_lstm} validation threshold {th_lstm:.2f}).\n\n"
            f"Feature Attribution (XGBoost SHAP):\n"
            f"{pos_str}\n"
            f"{neg_str}\n\n"
            f"Temporal Dynamics & Trajectory Analysis:\n"
            f"{lstm_str}\n"
            f"{trend_str}\n\n"
            f"Model Concordance Assessment:\n"
            f"{discordance_str}\n\n"
            "Clinical Safety & Research Boundary:\n"
            "The attributions above reflect statistical model associations derived from retrospective EHR data "
            "and must NOT be interpreted as causal medical conclusions, physiological certainties, or clinical treatment recommendations."
        )

        return narrative

    # =========================================================================
    # 7. MAIN EXPLANATION ENTRY POINT
    # =========================================================================

    def explain_patient(
        self,
        patient_input: Union[int, str, pd.DataFrame, List[str]],
        prediction_horizon_hours: int = 48,
        generate_plots: bool = False,
        output_plot_dir: str = "reports/figures",
        top_n: int = 10,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end model explanation for an individual patient:
        1. Ingests patient telemetry and executes Data Quality & Prediction Agents.
        2. Blocks explanation if Data Quality Agent blocked prediction (DQI < 50).
        3. Computes local XGBoost SHAP values (top positive and negative contributors).
        4. Computes LSTM feature and temporal window attributions via perturbation.
        5. Computes temporal trajectory trends across verified physiological variables.
        6. Assesses model discordance between XGBoost and LSTM.
        7. Cross-references top contributors with data quality missingness.
        8. Synthesizes structured output and research-oriented clinical narrative.

        Returns:
            Structured dictionary matching Phase 6.3 schema.
        """
        # Step 1: Execute Prediction Agent
        pred_res = self.prediction_agent.predict_patient(
            patient_input, prediction_horizon_hours=prediction_horizon_hours
        )
        patient_id = pred_res.get("patient_id", "UNKNOWN")
        dq_info = pred_res.get("data_quality", {})
        status = pred_res.get("status", "UNKNOWN")

        # Step 2: Quality Gatekeeper Check
        if status == "BLOCKED_BY_DATA_QUALITY":
            return {
                "patient_id": patient_id,
                "prediction_horizon": prediction_horizon_hours,
                "status": "BLOCKED_BY_DATA_QUALITY",
                "model_outputs": {},
                "xgboost_explanation": {
                    "top_positive_contributors": [],
                    "top_negative_contributors": [],
                    "shap_values": {},
                },
                "lstm_explanation": {
                    "important_features": [],
                    "important_time_periods": [],
                    "attribution_method": "Inhibited due to critical telemetry quality failure.",
                },
                "temporal_changes": [],
                "model_discordance": False,
                "discordance_explanation": "Model explanation blocked due to critical data quality failure (DQI < 50.0).",
                "data_quality": dq_info,
                "human_readable_summary": (
                    f"Automated explanation is blocked for Patient #{patient_id} because input telemetry failed "
                    f"minimum data quality standards (DQI={dq_info.get('dqi', 0.0)}/100, {dq_info.get('category')}). "
                    "No model explanations or synthetic feature attributions are generated for unreliable inputs."
                ),
                "limitations": [
                    "Explanations are inhibited when input telemetry fails data quality thresholds.",
                    *pred_res.get("limitations", []),
                ],
                "audit_recommendations": pred_res.get("audit_recommendations", []),
            }

        if status.startswith("ERROR"):
            return {
                "patient_id": patient_id,
                "prediction_horizon": prediction_horizon_hours,
                "status": status,
                "warnings": pred_res.get("warnings", []),
                "limitations": pred_res.get("limitations", []),
            }

        # Step 3: Run Preprocessing to get exact arrays
        _, df_patient = self.prediction_agent.parse_patient_input(patient_input)
        prep = self.prediction_agent.preprocess_patient(
            df_patient, horizon_hours=prediction_horizon_hours
        )

        X_tabular = prep["tabular_features"]
        feature_names = prep["feature_names"]
        temporal_tensor = prep["temporal_tensor"]
        static_tensor = prep["static_tensor"]

        # Step 4: XGBoost SHAP Explanation
        xgb_expl = self.explain_xgboost(
            X_tabular=X_tabular,
            feature_names=feature_names,
            patient_id=patient_id,
            horizon_hours=prediction_horizon_hours,
            top_n=top_n,
            generate_plots=generate_plots,
            output_plot_dir=output_plot_dir,
        )

        # Step 5: LSTM Attribution
        p_lstm_baseline = float(pred_res["predictions"]["lstm"]["predicted_probability"])
        lstm_expl = self.explain_lstm(
            temporal_tensor=temporal_tensor,
            static_tensor=static_tensor,
            baseline_prob=p_lstm_baseline,
            horizon_hours=prediction_horizon_hours,
            top_n=top_n,
        )

        # Step 6: Temporal Trend Analysis
        temporal_trends = self.analyze_temporal_trends(
            df_patient=df_patient, horizon_hours=prediction_horizon_hours
        )

        # Step 7: Model Discordance Analysis
        discordance_info = self.analyze_model_discordance(
            pred_res=pred_res,
            top_xgb_features=xgb_expl["top_positive_contributors"] + xgb_expl["top_negative_contributors"],
            top_lstm_features=lstm_expl["important_features"],
        )

        # Step 8: Cross-Reference Top Contributors with Data Quality & Sparsity
        sparsity_limitations = []
        trend_dict = {t["parameter"]: t for t in temporal_trends}

        for item in xgb_expl["top_positive_contributors"][:5] + xgb_expl["top_negative_contributors"][:5]:
            f_name = item["feature"]
            base_param = f_name.split("_")[0]
            if base_param in trend_dict:
                t_info = trend_dict[base_param]
                if t_info["missingness_percentage"] > 80.0:
                    sparsity_limitations.append(
                        f"HIGH_IMPUTATION_WARNING: Tabular feature '{f_name}' (SHAP: {item['shap_value']:+.4f}) "
                        f"was derived from '{base_param}', which had {t_info['missingness_percentage']:.1f}% missingness "
                        f"({t_info['observation_count']} recorded observations). Interpret attribution with caution."
                    )

        # Step 9: Synthesize Human-Readable Research Summary
        narrative = self._synthesize_human_readable_summary(
            patient_id=patient_id,
            horizon_hours=prediction_horizon_hours,
            pred_res=pred_res,
            top_pos_xgb=xgb_expl["top_positive_contributors"],
            top_neg_xgb=xgb_expl["top_negative_contributors"],
            top_lstm=lstm_expl["important_features"],
            time_periods=lstm_expl["important_time_periods"],
            trends=temporal_trends,
            discordance_info=discordance_info,
            dq_info=dq_info,
        )

        # Step 10: Assemble Structured Response
        response = {
            "patient_id": str(patient_id),
            "prediction_horizon": prediction_horizon_hours,
            "status": "SUCCESS",
            "model_outputs": pred_res.get("predictions", {}),
            "xgboost_explanation": {
                "top_positive_contributors": xgb_expl["top_positive_contributors"],
                "top_negative_contributors": xgb_expl["top_negative_contributors"],
                "base_value": xgb_expl["base_value"],
                "shap_values": xgb_expl["shap_values"],
                "waterfall_plot_path": xgb_expl["waterfall_plot_path"],
            },
            "lstm_explanation": {
                "important_features": lstm_expl["important_features"],
                "important_time_periods": lstm_expl["important_time_periods"],
                "attribution_method": lstm_expl["attribution_method"],
            },
            "temporal_changes": temporal_trends,
            "model_discordance": discordance_info["model_discordance"],
            "discordance_explanation": discordance_info["explanation"],
            "data_quality": dq_info,
            "human_readable_summary": narrative,
            "limitations": [
                *pred_res.get("limitations", []),
                *sparsity_limitations,
                "SHAP values and occlusion perturbations reflect model-specific statistical sensitivities, not clinical etiology.",
                "Feature attributions are derived from retrospective MIMIC-II observational data with prevalent clinical missingness.",
            ],
            "warnings": [
                *pred_res.get("warnings", []),
                *sparsity_limitations,
            ],
        }

        return response
