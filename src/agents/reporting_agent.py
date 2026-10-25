"""
Reporting / Priority Agent (Phase 6.4)
Agentic AI-Assisted ICU Mortality Risk Prediction from the First 48 Hours.

Responsibilities:
1. Orchestrates and synthesizes outputs from the previous three clinical agents:
   - Data Quality Agent (Phase 6.1)
   - Prediction Agent (Phase 6.2)
   - Explanation Agent (Phase 6.3)
2. Enforces non-redundant computation:
   - Reuses precomputed prediction and explanation artifacts when provided.
   - Orchestrates missing upstream steps seamlessly when only raw patient data is supplied.
3. Categorizes Model Agreement into explicit clinical research statuses:
   - MODEL_CONCORDANT_LOW (Both models below validation threshold)
   - MODEL_CONCORDANT_HIGH (Both models above validation threshold)
   - MODEL_DISCORDANT (Models diverge across thresholds)
   - PREDICTION_BLOCKED (Data Quality failure, DQI < 50.0)
4. Evaluates Deterministic Research Review Priority:
   - LOW: Concordant low risk with sound data quality.
   - MODERATE: Model discordance, suboptimal data quality (DQI 50-70), or borderline intermediate risk.
   - HIGH: Concordant high risk crossing validation thresholds with sound data quality.
   - CRITICAL_DATA_QUALITY: DQI < 50.0, automated predictions inhibited.
5. Implements Calibrated Confidence Language:
   - Strictly uses 'model-estimated mortality probability' (never 'clinical confidence').
   - Explicitly flags increased uncertainty when telemetry is sparse or suboptimal.
6. Generates Dual Deliverables:
   - Machine-readable structured JSON matching project schema.
   - Clinician-oriented structured ASCII/Markdown review report.

RESEARCH DISCLAIMER:
This system is an academic research prototype evaluated on retrospective EHR data.
Review priority levels and model estimates represent research workflow triage labels,
NOT clinical treatment, medication, or intervention directives.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

# Project internal imports
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.data_quality_agent import DataQualityAgent
from src.agents.prediction_agent import PredictionAgent
from src.agents.explanation_agent import ExplanationAgent

logger = logging.getLogger("ReportingAgent")


class ReportingAgent:
    """
    Autonomous Reporting and Research Prioritization Agent.
    Orchestrates the multi-agent decision support workflow, synthesizing data quality,
    dual-model predictions, and SHAP/perturbation explanations into actionable summaries.
    """

    # Deterministic priority levels
    PRIORITY_LEVELS = ["LOW", "MODERATE", "HIGH", "CRITICAL_DATA_QUALITY"]

    # Model agreement statuses
    AGREEMENT_STATUSES = [
        "MODEL_CONCORDANT_LOW",
        "MODEL_CONCORDANT_HIGH",
        "MODEL_DISCORDANT",
        "PREDICTION_BLOCKED",
    ]

    def __init__(
        self,
        base_dir: Optional[str] = None,
        data_quality_agent: Optional[DataQualityAgent] = None,
        prediction_agent: Optional[PredictionAgent] = None,
        explanation_agent: Optional[ExplanationAgent] = None,
    ):
        """
        Initializes the Reporting Agent with upstream agent dependencies.
        """
        self.base_dir = str(base_dir or ROOT_DIR)
        self.dq_agent = data_quality_agent or DataQualityAgent()
        self.prediction_agent = prediction_agent or PredictionAgent(
            base_dir=self.base_dir, data_quality_agent=self.dq_agent
        )
        self.explanation_agent = explanation_agent or ExplanationAgent(
            base_dir=self.base_dir,
            prediction_agent=self.prediction_agent,
            data_quality_agent=self.dq_agent,
        )

    # =========================================================================
    # 1. UPSTREAM ORCHESTRATION & CACHING
    # =========================================================================

    def _resolve_upstream_artifacts(
        self,
        patient_input: Any,
        prediction_horizon_hours: int,
        dq_result: Optional[Dict[str, Any]] = None,
        pred_result: Optional[Dict[str, Any]] = None,
        expl_result: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Orchestrates upstream agent executions without redundant computations:
        - If expl_result is provided, extracts precomputed dq_result and pred_result.
        - If pred_result is provided but expl_result is missing, runs explanation only.
        - If no precomputed results are provided, executes end-to-end multi-agent pipeline.
        """
        # Scenario A: Full explanation result already provided
        if expl_result is not None:
            resolved_expl = expl_result
            resolved_dq = dq_result or resolved_expl.get("data_quality", {})
            resolved_pred = pred_result or {
                "patient_id": resolved_expl.get("patient_id"),
                "status": resolved_expl.get("status"),
                "prediction_horizon_hours": prediction_horizon_hours,
                "data_quality": resolved_dq,
                "predictions": resolved_expl.get("model_outputs", {}),
                "warnings": resolved_expl.get("warnings", []),
                "limitations": resolved_expl.get("limitations", []),
                "audit_recommendations": resolved_expl.get("audit_recommendations", []),
            }
            return resolved_dq, resolved_pred, resolved_expl

        # Scenario B: Prediction result provided, Explanation missing
        if pred_result is not None:
            resolved_pred = pred_result
            resolved_dq = dq_result or resolved_pred.get("data_quality", {})

            # Check if prediction was blocked by data quality
            if resolved_pred.get("status") == "BLOCKED_BY_DATA_QUALITY":
                resolved_expl = self.explanation_agent.explain_patient(
                    patient_input, prediction_horizon_hours=prediction_horizon_hours
                )
            else:
                resolved_expl = self.explanation_agent.explain_patient(
                    patient_input, prediction_horizon_hours=prediction_horizon_hours, generate_plots=False
                )
            return resolved_dq, resolved_pred, resolved_expl

        # Scenario C: No precomputed artifacts; execute complete workflow
        resolved_expl = self.explanation_agent.explain_patient(
            patient_input, prediction_horizon_hours=prediction_horizon_hours, generate_plots=False
        )
        resolved_dq = resolved_expl.get("data_quality", {})
        resolved_pred = {
            "patient_id": resolved_expl.get("patient_id"),
            "status": resolved_expl.get("status"),
            "prediction_horizon_hours": prediction_horizon_hours,
            "data_quality": resolved_dq,
            "predictions": resolved_expl.get("model_outputs", {}),
            "warnings": resolved_expl.get("warnings", []),
            "limitations": resolved_expl.get("limitations", []),
            "audit_recommendations": resolved_expl.get("audit_recommendations", []),
        }

        return resolved_dq, resolved_pred, resolved_expl

    # =========================================================================
    # 2. MODEL AGREEMENT EVALUATION
    # =========================================================================

    def evaluate_model_agreement(
        self, pred_result: Dict[str, Any]
    ) -> Tuple[str, str]:
        """
        Evaluates the consensus status between XGBoost and LSTM:
        - PREDICTION_BLOCKED: Telemetry blocked due to quality failure.
        - MODEL_CONCORDANT_LOW: Both models below validation thresholds.
        - MODEL_CONCORDANT_HIGH: Both models above validation thresholds.
        - MODEL_DISCORDANT: Models produce diverging threshold decisions.

        Returns:
            agreement_status: String constant.
            agreement_meaning: Human-readable clinical interpretation.
        """
        status = pred_result.get("status", "UNKNOWN")
        if status == "BLOCKED_BY_DATA_QUALITY" or status.startswith("ERROR"):
            return (
                "PREDICTION_BLOCKED",
                "Automated risk predictions are blocked due to telemetry quality failure. No consensus evaluated.",
            )

        preds = pred_result.get("predictions", {})
        xgb_info = preds.get("xgboost", {})
        lstm_info = preds.get("lstm", {})

        dec_xgb = xgb_info.get("threshold_decision", "UNKNOWN")
        dec_lstm = lstm_info.get("threshold_decision", "UNKNOWN")

        p_xgb = float(xgb_info.get("predicted_probability", 0.0))
        p_lstm = float(lstm_info.get("predicted_probability", 0.0))
        th_xgb = float(xgb_info.get("threshold", 0.58))
        th_lstm = float(lstm_info.get("threshold", 0.58))

        if dec_xgb == "BELOW_THRESHOLD" and dec_lstm == "BELOW_THRESHOLD":
            return (
                "MODEL_CONCORDANT_LOW",
                f"Both independent models estimate low mortality risk below validation cutoffs (XGBoost: {p_xgb:.4f} < {th_xgb:.2f}, LSTM: {p_lstm:.4f} < {th_lstm:.2f}).",
            )
        elif dec_xgb == "ABOVE_THRESHOLD" and dec_lstm == "ABOVE_THRESHOLD":
            return (
                "MODEL_CONCORDANT_HIGH",
                f"Both independent models estimate elevated mortality risk above validation cutoffs (XGBoost: {p_xgb:.4f} >= {th_xgb:.2f}, LSTM: {p_lstm:.4f} >= {th_lstm:.2f}).",
            )
        else:
            return (
                "MODEL_DISCORDANT",
                f"Models diverge across decision thresholds: XGBoost={dec_xgb} (p={p_xgb:.4f}, th={th_xgb:.2f}), LSTM={dec_lstm} (p={p_lstm:.4f}, th={th_lstm:.2f}). Multidisciplinary clinical trajectory review recommended.",
            )

    # =========================================================================
    # 3. DETERMINISTIC RESEARCH REVIEW PRIORITY LOGIC
    # =========================================================================

    def assign_review_priority(
        self,
        dqi: float,
        agreement_status: str,
        p_xgb: float,
        p_lstm: float,
        th_xgb: float,
        th_lstm: float,
        warnings: Optional[List[str]] = None,
    ) -> Tuple[str, str]:
        """
        Determines the research workflow review priority according to explicit,
        deterministic priority precedence:

        1. DQI < 50 (or PREDICTION_BLOCKED):
           CRITICAL_DATA_QUALITY
        2. DQI from 50 to below 70:
           MODERATE (suboptimal telemetry uncertainty overrides model predictions)
        3. Model discordance (MODEL_DISCORDANT):
           MODERATE (models diverge across thresholds)
        4. Both models above thresholds with DQI >= 70 (MODEL_CONCORDANT_HIGH):
           HIGH
        5. Both models below thresholds with DQI >= 70 (MODEL_CONCORDANT_LOW):
           LOW, unless another documented warning (e.g. borderline probability within 0.10)
           requires MODERATE.

        Returns:
            priority: One of ['LOW', 'MODERATE', 'HIGH', 'CRITICAL_DATA_QUALITY'].
            rationale: Explanation of deterministic assignment criteria.
        """
        # Precedence Rule 1: Critical Telemetry Failure
        if dqi < 50.0 or agreement_status == "PREDICTION_BLOCKED":
            return (
                "CRITICAL_DATA_QUALITY",
                f"CRITICAL_DATA_QUALITY: Automated risk scoring is blocked because the Data Quality Index (DQI={dqi:.1f}/100) is below the critical threshold (50.0). Telemetry requires audit and potential re-acquisition before research interpretation.",
            )

        # Precedence Rule 2: Suboptimal Telemetry Quality (50 <= DQI < 70)
        if 50.0 <= dqi < 70.0:
            return (
                "MODERATE",
                f"MODERATE: Assigned MODERATE review priority due to suboptimal telemetry quality (DQI={dqi:.1f}/100, SUBOPTIMAL). Predictions are subject to increased uncertainty because of sparse or degraded physiological observations.",
            )

        # Precedence Rule 3: Model Discordance
        if agreement_status == "MODEL_DISCORDANT":
            return (
                "MODERATE",
                f"MODERATE: Assigned MODERATE review priority due to model discordance: XGBoost (p={p_xgb:.4f}) and LSTM (p={p_lstm:.4f}) diverge across decision thresholds. Tabular aggregations and sequential trajectories warrant individual inspection.",
            )

        # Precedence Rule 4: Concordant High Risk with DQI >= 70
        if agreement_status == "MODEL_CONCORDANT_HIGH" and dqi >= 70.0:
            return (
                "HIGH",
                f"HIGH: Both primary tabular XGBoost (p={p_xgb:.4f} >= {th_xgb:.2f}) and secondary sequential LSTM (p={p_lstm:.4f} >= {th_lstm:.2f}) independently cross validation-derived decision thresholds under adequate telemetry quality (DQI={dqi:.1f}/100).",
            )

        # Precedence Rule 5: Concordant Low Risk with DQI >= 70
        if agreement_status == "MODEL_CONCORDANT_LOW" and dqi >= 70.0:
            borderline_band = 0.10
            is_borderline = (th_xgb - borderline_band <= p_xgb < th_xgb) or (th_lstm - borderline_band <= p_lstm < th_lstm)
            if is_borderline:
                return (
                    "MODERATE",
                    f"MODERATE: Assigned MODERATE review priority due to borderline intermediate probability (XGBoost: {p_xgb:.4f}, LSTM: {p_lstm:.4f}) approaching the decision cutoff ({th_xgb:.2f}) despite low-risk categorization.",
                )
            if warnings and any("MODERATE_PRIORITY_REQUIRED" in w for w in warnings):
                return (
                    "MODERATE",
                    f"MODERATE: Assigned MODERATE review priority due to documented clinical telemetry warnings despite concordant low predictions.",
                )
            return (
                "LOW",
                f"LOW: Both independent models estimate low mortality risk below validation cutoffs (XGBoost: {p_xgb:.4f} < {th_xgb:.2f}, LSTM: {p_lstm:.4f} < {th_lstm:.2f}) supported by reliable telemetry (DQI={dqi:.1f}/100, {agreement_status}).",
            )

        # Fallback for any unclassified state
        return (
            "MODERATE",
            f"MODERATE: Intermediate research review status assigned based on model values (XGBoost: {p_xgb:.4f}, LSTM: {p_lstm:.4f}, DQI: {dqi:.1f}/100).",
        )

    # =========================================================================
    # 4. HUMAN-READABLE REPORT SYNTHESIS
    # =========================================================================

    def _generate_human_readable_report(
        self,
        patient_id: str,
        horizon_hours: int,
        dq_info: Dict[str, Any],
        pred_info: Dict[str, Any],
        agreement_status: str,
        agreement_meaning: str,
        expl_info: Dict[str, Any],
        priority: str,
        rationale: str,
        data_quality_warnings: List[str],
        limitations: List[str],
    ) -> str:
        """
        Formats a clean, standardized ASCII clinical research review summary.
        """
        dqi = dq_info.get("dqi", 0.0)
        category = dq_info.get("category", "UNKNOWN")
        status = dq_info.get("status", "UNKNOWN")

        xgb = pred_info.get("xgboost", {})
        lstm = pred_info.get("lstm", {})

        p_xgb = xgb.get("predicted_probability", "N/A")
        th_xgb = xgb.get("threshold", "N/A")
        dec_xgb = xgb.get("threshold_decision", "N/A")

        p_lstm = lstm.get("predicted_probability", "N/A")
        th_lstm = lstm.get("threshold", "N/A")
        dec_lstm = lstm.get("threshold_decision", "N/A")

        p_xgb_str = f"{p_xgb:.4f}" if isinstance(p_xgb, (int, float)) else str(p_xgb)
        p_lstm_str = f"{p_lstm:.4f}" if isinstance(p_lstm, (int, float)) else str(p_lstm)

        # Extract top XGBoost contributions
        top_pos = expl_info.get("xgboost_explanation", {}).get("top_positive_contributors", [])
        top_neg = expl_info.get("xgboost_explanation", {}).get("top_negative_contributors", [])

        # Extract top LSTM temporal patterns
        lstm_periods = expl_info.get("lstm_explanation", {}).get("important_time_periods", [])
        lstm_feats = expl_info.get("lstm_explanation", {}).get("important_features", [])

        lines = [
            "=" * 79,
            "                   ICU MORTALITY RISK REVIEW SUMMARY",
            "               Academic Research Prototype — Not for Clinical Use",
            "=" * 79,
            f"Patient ID:             #{patient_id}",
            f"Prediction Horizon:     {horizon_hours} Hours from ICU Admission",
            f"Data Source:            PhysioNet Challenge 2012 / MIMIC-II Dataset",
            f"Data Quality Index:     {dqi:.1f}/100 [{category}] (Status: {status})",
            "-" * 79,
            "1. MODEL PREDICTION SUMMARY",
            "-" * 79,
            f"Primary Model (XGBoost):  Probability = {p_xgb_str}  |  Threshold = {th_xgb}  |  Decision = {dec_xgb}",
            f"Secondary Model (LSTM):   Probability = {p_lstm_str}  |  Threshold = {th_lstm}  |  Decision = {dec_lstm}",
            "",
            "Note: Model probabilities are independent and are NOT averaged or blended.",
            "-" * 79,
            f"2. MODEL AGREEMENT STATUS: {agreement_status}",
            "-" * 79,
            f"{agreement_meaning}",
            "-" * 79,
            "3. TOP MODEL FEATURE CONTRIBUTIONS (XGBoost Tree SHAP)",
            "-" * 79,
        ]

        if top_pos:
            lines.append("Risk-Increasing Contributions (Positive SHAP):")
            for idx, f in enumerate(top_pos[:3], 1):
                lines.append(f"  {idx}. {f['feature']} (Patient Value: {f['actual_value']}, SHAP: +{f['shap_value']:.4f})")
        else:
            lines.append("  No strong positive contributors above baseline.")

        if top_neg:
            lines.append("Protective Associations (Negative SHAP):")
            for idx, f in enumerate(top_neg[:3], 1):
                lines.append(f"  {idx}. {f['feature']} (Patient Value: {f['actual_value']}, SHAP: {f['shap_value']:.4f})")
        else:
            lines.append("  No strong protective contributors below baseline.")

        lines.extend([
            "-" * 79,
            "4. IMPORTANT TEMPORAL PATTERNS (LSTM Perturbation & Trajectories)",
            "-" * 79,
        ])

        if lstm_periods:
            lines.append("Sequential Time Period Sensitivity:")
            for idx, tp in enumerate(lstm_periods[:2], 1):
                lines.append(f"  {idx}. {tp['time_period']} (Impact Delta P: {tp['delta_probability']:+.4f}, Sensitivity: {tp['impact_direction']})")

        if lstm_feats:
            lines.append("Key Dynamic Trajectory Influences:")
            for idx, lf in enumerate(lstm_feats[:3], 1):
                lines.append(f"  {idx}. {lf['feature']} (Occlusion Delta P: {lf['delta_probability']:+.4f}, Direction: {lf['impact_direction']})")

        lines.extend([
            "-" * 79,
            "5. DATA QUALITY AUDIT & SENSOR WARNINGS",
            "-" * 79,
        ])

        if data_quality_warnings:
            for idx, w in enumerate(data_quality_warnings, 1):
                lines.append(f"  {idx}. {w}")
        else:
            lines.append("  No critical data quality or sensor warnings flagged.")

        lines.extend([
            "-" * 79,
            f"6. RESEARCH REVIEW PRIORITY:  [ {priority} ]",
            "-" * 79,
            f"RATIONALE: {rationale}",
            "-" * 79,
            "7. CLINICAL SAFETY LIMITATIONS & RESEARCH BOUNDARIES",
            "-" * 79,
        ])

        for idx, lim in enumerate(limitations[:4], 1):
            lines.append(f"  {idx}. {lim}")

        lines.extend([
            "",
            "\"These outputs are research-model estimates and must not be used to make clinical treatment decisions.\"",
            "=" * 79,
        ])

        return "\n".join(lines)

    # =========================================================================
    # 5. MAIN REPORT GENERATION ENTRY POINT
    # =========================================================================

    def generate_report(
        self,
        patient_input: Any = None,
        prediction_horizon_hours: int = 48,
        dq_result: Optional[Dict[str, Any]] = None,
        pred_result: Optional[Dict[str, Any]] = None,
        expl_result: Optional[Dict[str, Any]] = None,
        prediction_horizon: Optional[int] = None,
        patient_id: Optional[Union[str, int]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Orchestrates and generates the comprehensive ICU Mortality Risk Review Report.
        Non-redundantly reuses precomputed artifacts if supplied.

        Returns:
            Structured JSON dictionary matching Phase 6.4 machine-readable schema.
        """
        if prediction_horizon is not None:
            prediction_horizon_hours = prediction_horizon

        # Step 1: Upstream artifact resolution (zero redundant compute)
        dq_info, pred_info, expl_info = self._resolve_upstream_artifacts(
            patient_input=patient_input,
            prediction_horizon_hours=prediction_horizon_hours,
            dq_result=dq_result,
            pred_result=pred_result,
            expl_result=expl_result,
        )

        resolved_pid = str(patient_id or pred_info.get("patient_id") or expl_info.get("patient_id") or "UNKNOWN")
        patient_id = resolved_pid
        dqi = float(dq_info.get("dqi", 0.0))
        dq_category = dq_info.get("category", "UNKNOWN")
        dq_status = dq_info.get("status", "UNKNOWN")

        # Step 2: Quality Confidence Classification
        if dqi >= 85.0:
            confidence_label = "EXCELLENT_DATA_QUALITY"
        elif 70.0 <= dqi < 85.0:
            confidence_label = "ACCEPTABLE_WITH_WARNING"
        elif 50.0 <= dqi < 70.0:
            confidence_label = "SUBOPTIMAL_DATA_QUALITY"
        else:
            confidence_label = "PREDICTION_BLOCKED"

        # Step 3: Evaluate Model Agreement Status
        agreement_status, agreement_meaning = self.evaluate_model_agreement(pred_info)

        # Step 4: Extract Numerical Predictions
        preds = pred_info.get("predictions", {})
        xgb_info = preds.get("xgboost", {})
        lstm_info = preds.get("lstm", {})

        p_xgb = float(xgb_info.get("predicted_probability", 0.0))
        p_lstm = float(lstm_info.get("predicted_probability", 0.0))
        th_xgb = float(xgb_info.get("threshold", 0.58))
        th_lstm = float(lstm_info.get("threshold", 0.58))

        # Step 5: Assign Deterministic Research Review Priority
        priority, rationale = self.assign_review_priority(
            dqi=dqi,
            agreement_status=agreement_status,
            p_xgb=p_xgb,
            p_lstm=p_lstm,
            th_xgb=th_xgb,
            th_lstm=th_lstm,
            warnings=pred_info.get("warnings", []),
        )

        # Step 6: Compile Warnings & Limitations
        data_quality_warnings = []
        if confidence_label == "SUBOPTIMAL_DATA_QUALITY":
            data_quality_warnings.append(
                "Prediction is subject to increased uncertainty because of limited or unreliable observations (DQI 50-70)."
            )
        elif confidence_label == "ACCEPTABLE_WITH_WARNING":
            data_quality_warnings.append(
                "Acceptable telemetry with warnings: some descriptors or sparse laboratory measurements were imputed."
            )
        elif confidence_label == "PREDICTION_BLOCKED":
            data_quality_warnings.append(
                "Automated prediction blocked due to critical telemetry quality failure (DQI < 50.0)."
            )

        if dq_info.get("missing_descriptors"):
            data_quality_warnings.append(f"Missing static descriptors: {', '.join(dq_info['missing_descriptors'])}")

        for w in pred_info.get("warnings", []):
            if w not in data_quality_warnings and not any(w in dw for dw in data_quality_warnings):
                data_quality_warnings.append(w)

        all_limitations = [
            "Research prototype only; not intended for autonomous clinical diagnosis, triage, or treatment guidance.",
            "Decision thresholds (0.58 / 0.58) are validation-selected cutoffs optimizing F1-score, not clinical severity limits.",
            "Priority levels reflect internal research workflow prioritization, NOT clinical urgency scores.",
            "Predictions and attributions are derived from retrospective MIMIC-II ICU data with prevalent clinical missingness.",
            *pred_info.get("limitations", []),
        ]
        # Deduplicate limitations while preserving order
        unique_limitations = []
        for lim in all_limitations:
            if lim not in unique_limitations:
                unique_limitations.append(lim)

        # Step 7: Synthesize Human-Readable Report
        human_readable_report = self._generate_human_readable_report(
            patient_id=patient_id,
            horizon_hours=prediction_horizon_hours,
            dq_info=dq_info,
            pred_info=preds,
            agreement_status=agreement_status,
            agreement_meaning=agreement_meaning,
            expl_info=expl_info,
            priority=priority,
            rationale=rationale,
            data_quality_warnings=data_quality_warnings,
            limitations=unique_limitations,
        )

        # Step 8: Assemble Structured JSON Response
        response = {
            "patient_id": str(patient_id),
            "horizon_hours": prediction_horizon_hours,
            "data_quality": {
                "dqi": round(dqi, 2),
                "category": dq_category,
                "status": dq_status,
                "confidence_label": confidence_label,
                "missing_descriptors": dq_info.get("missing_descriptors", []),
                "sensor_artifacts": dq_info.get("sensor_artifacts", 0),
                "observation_density": dq_info.get("observation_density", 0),
            },
            "prediction": {
                "status": pred_info.get("status", "SUCCESS"),
                "xgboost": {
                    "predicted_probability": round(p_xgb, 4) if agreement_status != "PREDICTION_BLOCKED" else None,
                    "threshold": th_xgb,
                    "decision": xgb_info.get("threshold_decision"),
                },
                "lstm": {
                    "predicted_probability": round(p_lstm, 4) if agreement_status != "PREDICTION_BLOCKED" else None,
                    "threshold": th_lstm,
                    "decision": lstm_info.get("threshold_decision"),
                },
            },
            "agreement_status": agreement_status,
            "agreement_meaning": agreement_meaning,
            "explanation": {
                "top_xgboost_risk_contributors": expl_info.get("xgboost_explanation", {}).get("top_positive_contributors", []),
                "top_xgboost_protective_contributors": expl_info.get("xgboost_explanation", {}).get("top_negative_contributors", []),
                "top_lstm_temporal_features": expl_info.get("lstm_explanation", {}).get("important_features", []),
                "important_time_periods": expl_info.get("lstm_explanation", {}).get("important_time_periods", []),
                "physiological_trends": expl_info.get("temporal_changes", []),
            },
            "review_priority": priority,
            "rationale": rationale,
            "human_readable_report": human_readable_report,
            "limitations": unique_limitations,
            "research_only": True,
        }

        return response
