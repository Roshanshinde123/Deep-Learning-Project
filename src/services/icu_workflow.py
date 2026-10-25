"""
ICU Workflow Service (Phase 7 Integration Service)
Orchestrates the 4-tier Agentic AI system:
1. Data Quality Agent
2. Prediction Agent
3. Explanation Agent
4. Reporting / Priority Agent

Ensures patient isolation, cached model resources, non-redundant computation,
and structured progress tracking for the Streamlit frontend.
"""

import os
import sys
import io
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.data_quality_agent import DataQualityAgent
from src.agents.prediction_agent import PredictionAgent
from src.agents.explanation_agent import ExplanationAgent
from src.agents.reporting_agent import ReportingAgent


class ICUWorkflowService:
    """
    Central orchestration service connecting the 4 clinical research agents.
    Provides safe resource management, progress callbacks, and format adapters
    for the Streamlit application.
    """

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        self.base_dir = Path(base_dir or ROOT_DIR)
        
        # Instantiate agents (agents manage their own model checkpoints)
        self.dq_agent = DataQualityAgent()
        self.pred_agent = PredictionAgent(base_dir=str(self.base_dir))
        self.expl_agent = ExplanationAgent(base_dir=str(self.base_dir))
        self.reporting_agent = ReportingAgent(base_dir=str(self.base_dir))

    # =========================================================================
    # 1. SAMPLE PATIENT DISCOVERY & UPLOAD PARSING
    # =========================================================================

    def get_permitted_sample_patients(self) -> List[Dict[str, Any]]:
        """
        Returns permitted sample patient records from Dataset/set-a for demonstration.
        Does NOT expose ground truth outcomes to live prediction pages.
        """
        dataset_dir = self.base_dir / "Dataset" / "set-a"
        samples = [
            {
                "patient_id": "132539",
                "label": "Patient #132539 (54F Cardiac Surgery Recovery)",
                "description": "54-year-old female post-cardiac surgery. High observation density with stable vital signs.",
                "file_path": str(dataset_dir / "132539.txt"),
            },
            {
                "patient_id": "132540",
                "label": "Patient #132540 (76M Medical ICU)",
                "description": "76-year-old male admitted with respiratory and metabolic abnormalities.",
                "file_path": str(dataset_dir / "132540.txt"),
            },
            {
                "patient_id": "132543",
                "label": "Patient #132543 (29M Surgical / Trauma ICU)",
                "description": "29-year-old male with postoperative trauma monitoring.",
                "file_path": str(dataset_dir / "132543.txt"),
            },
            {
                "patient_id": "132551",
                "label": "Patient #132551 (77M Septic Shock & Vasopressors)",
                "description": "77-year-old male presenting with severe sepsis, hypotensive episodes, and vasopressor requirement.",
                "file_path": str(dataset_dir / "132551.txt"),
            },
            {
                "patient_id": "132588",
                "label": "Patient #132588 (85F Acute Organ Failure)",
                "description": "85-year-old female with acute multiorgan dysfunction and high tabular risk extremity.",
                "file_path": str(dataset_dir / "132588.txt"),
            },
        ]
        return samples

    def parse_uploaded_patient_file(self, file_content: Union[str, bytes, io.BytesIO, io.StringIO]) -> Tuple[pd.DataFrame, str]:
        """
        Parses an uploaded PhysioNet patient .txt file into a validated DataFrame.
        Validates column structure (Time, Parameter, Value) and extracts RecordID.
        """
        if isinstance(file_content, bytes):
            text = file_content.decode("utf-8", errors="replace")
        elif isinstance(file_content, (io.BytesIO, io.StringIO)):
            text = file_content.getvalue()
            if isinstance(text, bytes):
                text = text.decode("utf-8", errors="replace")
        elif isinstance(file_content, str):
            text = file_content
        else:
            raise ValueError(f"Unsupported file content type: {type(file_content)}")

        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        if not lines:
            raise ValueError("The uploaded patient file is completely empty.")

        # Check header
        first_line = lines[0].split(",")
        has_header = "Time" in first_line and "Parameter" in first_line

        data = []
        patient_id = "UNKNOWN"

        start_idx = 1 if has_header else 0
        for line in lines[start_idx:]:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                t, param, val = parts[0], parts[1], parts[2]
                try:
                    num_val = float(val)
                except ValueError:
                    num_val = np.nan
                
                if param == "RecordID":
                    try:
                        patient_id = str(int(float(val)))
                    except (ValueError, TypeError):
                        patient_id = str(val)

                data.append({"Time": t, "Parameter": param, "Value": num_val})

        if not data:
            raise ValueError("No valid clinical telemetry records could be parsed from the file.")

        df = pd.DataFrame(data)
        return df, patient_id

    # =========================================================================
    # 2. END-TO-END MULTI-AGENT EXECUTION
    # =========================================================================

    def run_workflow(
        self,
        patient_input: Union[str, Path, pd.DataFrame],
        prediction_horizon_hours: int = 48,
        generate_plots: bool = True,
        progress_callback: Optional[Callable[[str, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the complete multi-agent workflow in sequence:
        1. Data Quality Agent -> Audits telemetry, checks DQI Gate (50.0).
        2. Prediction Agent -> XGBoost tabular + LSTM sequential inference.
        3. Explanation Agent -> Tree SHAP attributions + LSTM temporal sensitivity.
        4. Reporting Agent -> Synthesizes structured JSON and human-readable report.
        """
        if prediction_horizon_hours not in (24, 48):
            raise ValueError(f"Invalid prediction horizon: {prediction_horizon_hours}. Must be 24 or 48.")

        # Step 1: Data Quality Audit
        if progress_callback:
            progress_callback("data_quality", "Agent 1/4: Auditing physiological telemetry, density & sensor plausibility...")

        if isinstance(patient_input, pd.DataFrame):
            dq_result = self.dq_agent.audit_dataframe(patient_input, prediction_horizon_hours=prediction_horizon_hours)
        elif isinstance(patient_input, (str, Path)) and os.path.isfile(str(patient_input)):
            dq_result = self.dq_agent.audit_patient_file(str(patient_input), prediction_horizon_hours=prediction_horizon_hours)
        else:
            # Assume it's a patient ID string
            pid_file = self.base_dir / "Dataset" / "set-a" / f"{patient_input}.txt"
            if pid_file.exists():
                dq_result = self.dq_agent.audit_patient_file(str(pid_file), prediction_horizon_hours=prediction_horizon_hours)
            else:
                raise FileNotFoundError(f"Patient record '{patient_input}' not found on filesystem.")

        raw_dqi = dq_result.get("data_quality_index") if "data_quality_index" in dq_result else dq_result.get("dqi", 0.0)
        dqi = float(raw_dqi)
        dq_result["dqi"] = round(dqi, 2)
        dq_result["category"] = dq_result.get("quality_grade", dq_result.get("category", "UNKNOWN"))

        # Resolve patient ID accurately
        if isinstance(patient_input, pd.DataFrame):
            pid_cand = dq_result.get("record_id") or dq_result.get("patient_id")
            if not pid_cand or str(pid_cand) == "UNKNOWN":
                rec_rows = patient_input[patient_input["Parameter"] == "RecordID"]
                if not rec_rows.empty:
                    try:
                        pid_cand = str(int(float(rec_rows.iloc[0]["Value"])))
                    except Exception:
                        pid_cand = str(rec_rows.iloc[0]["Value"])
            patient_id = str(pid_cand or "UNKNOWN")
        elif isinstance(patient_input, (str, int)):
            # If path, take stem, else string itself
            p_str = str(patient_input)
            patient_id = Path(p_str).stem if "/" in p_str or "\\" in p_str or p_str.endswith(".txt") else p_str
        elif isinstance(patient_input, Path):
            patient_id = patient_input.stem
        else:
            patient_id = str(dq_result.get("record_id") or dq_result.get("patient_id") or "UNKNOWN")

        dq_result["patient_id"] = patient_id

        # Step 2: Quality Gate Evaluation
        if dqi < 50.0:
            if progress_callback:
                progress_callback("gate_block", "Quality Gate: DQI < 50.0 — Inhibiting automated prediction to prevent clinical misinformation.")

            pred_result = {
                "patient_id": patient_id,
                "status": "BLOCKED_BY_DATA_QUALITY",
                "prediction_horizon_hours": prediction_horizon_hours,
                "data_quality": dq_result,
                "predictions": {},
                "warnings": [f"Automated risk scoring blocked: DQI={dqi:.1f}/100 fails minimum quality threshold (50.0)."],
                "limitations": ["Prediction inhibited due to critical telemetry quality failure."],
                "audit_recommendations": dq_result.get("audit_recommendations", []),
            }
            expl_result = {
                "patient_id": patient_id,
                "prediction_horizon": prediction_horizon_hours,
                "status": "BLOCKED_BY_DATA_QUALITY",
                "data_quality": dq_result,
                "model_outputs": {},
                "xgboost_explanation": {"top_positive_contributors": [], "top_negative_contributors": []},
                "lstm_explanation": {"important_features": [], "important_time_periods": []},
                "temporal_changes": [],
                "warnings": pred_result["warnings"],
                "audit_recommendations": dq_result.get("audit_recommendations", []),
            }
        else:
            # Step 3: Prediction Agent
            if progress_callback:
                progress_callback("prediction", "Agent 2/4: Computing primary XGBoost baseline & secondary sequential LSTM inference...")

            pred_result = self.pred_agent.predict_patient(
                patient_input=patient_input,
                prediction_horizon_hours=prediction_horizon_hours,
            )

            # Step 4: Explanation Agent
            if progress_callback:
                progress_callback("explanation", "Agent 3/4: Calculating Tree SHAP feature attributions & LSTM temporal perturbations...")

            expl_result = self.expl_agent.explain_patient(
                patient_input=patient_input,
                prediction_horizon_hours=prediction_horizon_hours,
                generate_plots=generate_plots,
            )

        # Step 5: Reporting / Priority Agent
        if progress_callback:
            progress_callback("reporting", "Agent 4/4: Evaluating model agreement & synthesizing research report...")

        report_result = self.reporting_agent.generate_report(
            patient_input=patient_input,
            prediction_horizon_hours=prediction_horizon_hours,
            dq_result=dq_result,
            pred_result=pred_result,
            expl_result=expl_result,
            patient_id=patient_id,
        )

        if progress_callback:
            progress_callback("completed", "Workflow Completed: Structured research risk report ready.")

        return {
            "patient_id": patient_id,
            "prediction_horizon_hours": prediction_horizon_hours,
            "data_quality": dq_result,
            "predictions": pred_result,
            "explanation": expl_result,
            "report": report_result,
            "human_readable_report": report_result.get("human_readable_report", ""),
            "waterfall_plot_path": expl_result.get("xgboost_explanation", {}).get("waterfall_plot_path"),
        }

    # =========================================================================
    # 3. BENCHMARK & PERFORMANCE DATA ACCESS
    # =========================================================================

    def get_stored_model_benchmarks(self) -> pd.DataFrame:
        """
        Loads the precomputed, officially frozen benchmark metrics across all
        models (XGBoost, 1D CNN, Simple RNN, LSTM) from reports/model_comparison.csv.
        """
        comp_csv = self.base_dir / "reports" / "model_comparison.csv"
        if comp_csv.exists():
            return pd.read_csv(comp_csv)
        return pd.DataFrame()

    def get_stored_imputation_comparison(self) -> pd.DataFrame:
        """
        Loads the precomputed imputation comparison from reports/imputation_comparison.csv.
        """
        imp_csv = self.base_dir / "reports" / "imputation_comparison.csv"
        if imp_csv.exists():
            return pd.read_csv(imp_csv)
        return pd.DataFrame()
