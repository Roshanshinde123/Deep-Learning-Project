"""
Data Quality Agent (Phase 6.1 Refined)
Autonomous Clinical Data Auditing, Sensor Artifact Detection, and Input Validation.

Responsibilities:
1. Validates static admission descriptors, explicitly recognizing -1 sentinels as missing values.
2. Audits temporal measurement density and comprehensive Blood Pressure monitoring across both
   invasive (MAP, SysABP, DiasABP) and non-invasive (NIMAP, NISysABP, NIDiasABP) modalities.
3. Separates physiologically impossible sensor artifacts from genuine extreme pathological values.
4. Computes a transparent, reproducible, research-defined Data Quality Index (DQI: 0-100).
5. Generates structured clinician briefings with clear data-quality grades and audit recommendations.

DISCLAIMER:
The Data Quality Index (DQI) is a research-defined operational measure developed specifically
for the PhysioNet 2012 / MIMIC-II study to quantify EHR observational completeness and sensor
plausibility. It is NOT a clinically validated diagnostic score (such as SOFA, APACHE, or SAPS).
"""

import os
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class DataQualityAgent:
    """
    Autonomous clinical data quality audit agent.
    Acts as the first line of defense in the ICU Agentic Decision Support System,
    ensuring that only verified, biologically plausible, and structurally sound
    clinical data is forwarded to the Prediction and Explanation Agents.
    """

    # 1. Biological Impossibility Limits (Sensor disconnection, lead detachment, transcription error)
    # Values outside these bounds represent physical or biological impossibilities.
    IMPOSSIBLE_BOUNDS = {
        "HR": {"min": 15.0, "max": 300.0, "unit": "bpm"},
        "SysABP": {"min": 20.0, "max": 320.0, "unit": "mmHg"},
        "DiasABP": {"min": 10.0, "max": 200.0, "unit": "mmHg"},
        "MAP": {"min": 15.0, "max": 250.0, "unit": "mmHg"},
        "NISysABP": {"min": 20.0, "max": 320.0, "unit": "mmHg"},
        "NIDiasABP": {"min": 10.0, "max": 200.0, "unit": "mmHg"},
        "NIMAP": {"min": 15.0, "max": 250.0, "unit": "mmHg"},
        "RespRate": {"min": 2.0, "max": 80.0, "unit": "breaths/min"},
        "Temp": {"min": 24.0, "max": 44.0, "unit": "°C"},
        "GCS": {"min": 3.0, "max": 15.0, "unit": "score"},
        "SaO2": {"min": 20.0, "max": 100.0, "unit": "%"},
        "FiO2": {"min": 0.20, "max": 1.0, "unit": "fraction"},
        "PaO2": {"min": 15.0, "max": 750.0, "unit": "mmHg"},
        "PaCO2": {"min": 8.0, "max": 160.0, "unit": "mmHg"},
        "pH": {"min": 6.60, "max": 7.90, "unit": "pH"},
        "Glucose": {"min": 5.0, "max": 2000.0, "unit": "mg/dL"},
        "BUN": {"min": 1.0, "max": 300.0, "unit": "mg/dL"},
        "Creatinine": {"min": 0.1, "max": 35.0, "unit": "mg/dL"},
        "K": {"min": 0.8, "max": 12.0, "unit": "mEq/L"},
        "Na": {"min": 85.0, "max": 200.0, "unit": "mEq/L"},
        "Lactate": {"min": 0.1, "max": 40.0, "unit": "mmol/L"},
        "Urine": {"min": 0.0, "max": 5000.0, "unit": "mL"},
        "WBC": {"min": 0.05, "max": 250.0, "unit": "k/uL"},
        "Platelets": {"min": 1.0, "max": 2000.0, "unit": "k/uL"},
        "Weight": {"min": 15.0, "max": 400.0, "unit": "kg"},
        "Height": {"min": 40.0, "max": 250.0, "unit": "cm"},
    }

    # 2. Extreme Pathological Alerts (Genuine severe clinical emergencies; NOT deleted as artifacts)
    PATHOLOGICAL_EXTREMES = {
        "HR": {"low": 40.0, "high": 150.0, "desc": "Severe Bradycardia / Extreme Tachycardia"},
        "MAP": {"low": 55.0, "high": 135.0, "desc": "Profound Hypoperfusion (Shock) / Malignant Hypertension"},
        "NIMAP": {"low": 55.0, "high": 135.0, "desc": "Profound Hypoperfusion / Severe Hypertension"},
        "SysABP": {"low": 75.0, "high": 190.0, "desc": "Cardiogenic / Septic Shock / Hypertensive Crisis"},
        "NISysABP": {"low": 75.0, "high": 190.0, "desc": "Hypotensive Shock / Hypertensive Crisis"},
        "RespRate": {"low": 8.0, "high": 40.0, "desc": "Central Respiratory Depression / Severe Tachypnea"},
        "Temp": {"low": 34.5, "high": 39.5, "desc": "Severe Hypothermia / Extreme Pyrexia"},
        "GCS": {"low": 6.0, "high": 15.0, "desc": "Deep Comatose State (Severe Encephalopathy)"},
        "pH": {"low": 7.20, "high": 7.55, "desc": "Severe Acidemia / Severe Alkalemia"},
        "Lactate": {"low": 0.0, "high": 4.0, "desc": "Severe Tissue Hypoperfusion (Lactate >= 4.0 mmol/L)"},
        "K": {"low": 2.8, "high": 6.0, "desc": "Severe Hypokalemia / Cardiac Arrest Risk Hyperkalemia"},
        "Creatinine": {"low": 0.0, "high": 4.0, "desc": "Acute Kidney Injury (Stage 3 AKI)"},
        "Urine": {"low": 15.0, "high": 3000.0, "desc": "Severe Oliguria (< 15 mL/hr)"},
    }

    # Blood Pressure measurement modalities
    INVASIVE_BP_PARAMS = ["MAP", "SysABP", "DiasABP"]
    NONINVASIVE_BP_PARAMS = ["NIMAP", "NISysABP", "NIDiasABP"]
    ALL_BP_PARAMS = INVASIVE_BP_PARAMS + NONINVASIVE_BP_PARAMS

    # Critical vital domains required for safe ICU monitoring
    CRITICAL_VITAL_DOMAINS = ["HR", "BloodPressure", "GCS", "RespRate", "Temp"]

    def __init__(self, agent_name: str = "DataQualityAgent_v1.1"):
        self.agent_name = agent_name
        self.version = "1.1.0"

    def audit_patient_file(self, file_path: str, prediction_horizon_hours: int = 48) -> Dict[str, Any]:
        """
        Parses and audits an official raw PhysioNet format ICU patient record text file.
        Returns a comprehensive diagnostic dictionary.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Patient file not found at: {file_path}")

        records = []
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) == 3:
                time_str, param, val_str = parts[0].strip(), parts[1].strip(), parts[2].strip()
                if not param:
                    continue
                try:
                    val = float(val_str)
                    records.append({"Time": time_str, "Parameter": param, "Value": val})
                except ValueError:
                    continue

        df = pd.DataFrame(records)
        return self.audit_dataframe(df, file_name=os.path.basename(file_path), prediction_horizon_hours=prediction_horizon_hours)

    def audit_dataframe(
        self,
        df: pd.DataFrame,
        file_name: str = "Unknown_Record",
        prediction_horizon_hours: int = 48,
    ) -> Dict[str, Any]:
        """
        Audits a DataFrame containing ['Time', 'Parameter', 'Value'] for an individual ICU patient.
        """
        if df.empty:
            return {
                "agent": self.agent_name,
                "file_name": file_name,
                "status": "FAIL",
                "quality_grade": "UNRELIABLE_CRITICAL",
                "data_quality_index": 0.0,
                "summary": "Patient record is completely empty. No descriptors or temporal signals available.",
                "missing_descriptors": ["RecordID", "Age", "Gender", "Height", "Weight", "ICUType"],
                "descriptor_issues": ["EMPTY_RECORD"],
                "recommendations": ["Do not pass to prediction models. Re-fetch patient telemetry."],
            }

        # ---------------------------------------------------------
        # 1. PARSE & VALIDATE STATIC ADMISSION DESCRIPTORS
        # ---------------------------------------------------------
        # Sentinels of -1 represent standard PhysioNet missing values and must NOT be classified as valid!
        raw_descriptors = {}
        cleaned_descriptors = {}
        missing_descriptors = []
        descriptor_issues = []

        static_params = ["RecordID", "Age", "Gender", "Height", "Weight", "ICUType"]
        for p in static_params:
            match = df[df["Parameter"] == p]
            if not match.empty:
                val = match.iloc[0]["Value"]
                raw_descriptors[p] = val
                # Check for recognized missing sentinel (-1 or -1.0)
                if val == -1 or val == -1.0 or val is None or np.isnan(val):
                    cleaned_descriptors[p] = None
                    missing_descriptors.append(f"{p} (sentinel -1)")
                else:
                    if p in ["RecordID", "Gender", "ICUType"]:
                        cleaned_descriptors[p] = int(val)
                    else:
                        cleaned_descriptors[p] = float(val)
            else:
                raw_descriptors[p] = None
                cleaned_descriptors[p] = None
                missing_descriptors.append(f"{p} (unrecorded)")

        record_id = cleaned_descriptors.get("RecordID") or raw_descriptors.get("RecordID") or file_name.replace(".txt", "")

        # Descriptor Validity Checks
        if cleaned_descriptors.get("Age") is None:
            descriptor_issues.append("MISSING_AGE")
        elif cleaned_descriptors["Age"] <= 0 or cleaned_descriptors["Age"] > 115:
            descriptor_issues.append("INVALID_AGE_VALUE")

        if cleaned_descriptors.get("Gender") is None:
            descriptor_issues.append("MISSING_GENDER")
        elif cleaned_descriptors["Gender"] not in [0, 1]:
            descriptor_issues.append("INVALID_GENDER_CODE")

        if cleaned_descriptors.get("ICUType") is None:
            descriptor_issues.append("MISSING_ICU_TYPE")
        elif cleaned_descriptors["ICUType"] not in [1, 2, 3, 4]:
            descriptor_issues.append("INVALID_ICU_TYPE_CODE")

        if cleaned_descriptors.get("Height") is None:
            # Non-fatal missing descriptor (92.5% missing in MIMIC-II)
            pass
        elif cleaned_descriptors["Height"] < 40 or cleaned_descriptors["Height"] > 250:
            descriptor_issues.append("PHYSIOLOGICALLY_IMPOSSIBLE_HEIGHT")

        if cleaned_descriptors.get("Weight") is None:
            # Missing baseline admission weight
            pass
        elif cleaned_descriptors["Weight"] < 20 or cleaned_descriptors["Weight"] > 350:
            descriptor_issues.append("PHYSIOLOGICALLY_IMPOSSIBLE_ADMISSION_WEIGHT")

        # ---------------------------------------------------------
        # 2. FILTER TEMPORAL MEASUREMENTS TO PREDICTION WINDOW
        # ---------------------------------------------------------
        temporal_rows = []
        for idx, row in df.iterrows():
            p = row["Parameter"]
            if p in static_params and row["Time"] == "00:00":
                continue  # static admission descriptor
            t_str = str(row["Time"])
            try:
                hh, mm = t_str.split(":")
                t_float = float(hh) + float(mm) / 60.0
            except Exception:
                t_float = 0.0

            if t_float <= prediction_horizon_hours:
                temporal_rows.append(row)

        temporal_df = pd.DataFrame(temporal_rows) if temporal_rows else pd.DataFrame(columns=["Time", "Parameter", "Value"])
        obs_count = len(temporal_df)
        unique_params = list(temporal_df["Parameter"].unique()) if not temporal_df.empty else []

        # ---------------------------------------------------------
        # 3. MULTIMODAL BLOOD PRESSURE & VITAL DOMAINS AUDITING
        # ---------------------------------------------------------
        # Check both invasive (MAP, SysABP, DiasABP) and non-invasive (NIMAP, NISysABP, NIDiasABP)
        invasive_bp_count = 0
        noninvasive_bp_count = 0
        invasive_details = {}
        noninvasive_details = {}

        if not temporal_df.empty:
            for bp_param in self.INVASIVE_BP_PARAMS:
                c = int((temporal_df["Parameter"] == bp_param).sum())
                invasive_details[bp_param] = c
                invasive_bp_count += c

            for bp_param in self.NONINVASIVE_BP_PARAMS:
                c = int((temporal_df["Parameter"] == bp_param).sum())
                noninvasive_details[bp_param] = c
                noninvasive_bp_count += c

        total_bp_observations = invasive_bp_count + noninvasive_bp_count
        bp_monitoring_type = "NONE"
        if invasive_bp_count >= 4 and noninvasive_bp_count >= 4:
            bp_monitoring_type = "DUAL_INVASIVE_AND_NONINVASIVE"
        elif invasive_bp_count >= 4:
            bp_monitoring_type = "CONTINUOUS_INVASIVE_ARTERIAL_LINE"
        elif noninvasive_bp_count >= 4:
            bp_monitoring_type = "INTERMITTENT_NONINVASIVE_CUFF"
        elif total_bp_observations > 0:
            bp_monitoring_type = "SPARSE_BLOOD_PRESSURE"

        # Comprehensive Critical Vitals Assessment
        critical_vitals_summary = {}

        # 1. Heart Rate
        hr_cnt = int((temporal_df["Parameter"] == "HR").sum()) if not temporal_df.empty else 0
        critical_vitals_summary["HR"] = {
            "observed_count": hr_cnt,
            "status": "SUFFICIENT" if hr_cnt >= 4 else ("SPARSE" if hr_cnt > 0 else "ABSENT"),
            "details": f"{hr_cnt} measurements",
        }

        # 2. Blood Pressure (Recognizes invasive and non-invasive modalities appropriately)
        critical_vitals_summary["BloodPressure"] = {
            "observed_count": total_bp_observations,
            "status": "SUFFICIENT" if total_bp_observations >= 4 else ("SPARSE" if total_bp_observations > 0 else "ABSENT"),
            "monitoring_modality": bp_monitoring_type,
            "invasive_count": invasive_bp_count,
            "noninvasive_count": noninvasive_bp_count,
            "details": f"Invasive={invasive_bp_count} (MAP={invasive_details.get('MAP', 0)}, Sys={invasive_details.get('SysABP', 0)}), Non-Invasive={noninvasive_bp_count} (NIMAP={noninvasive_details.get('NIMAP', 0)}, NISys={noninvasive_details.get('NISysABP', 0)})",
        }

        # 3. Glasgow Coma Scale
        gcs_cnt = int((temporal_df["Parameter"] == "GCS").sum()) if not temporal_df.empty else 0
        critical_vitals_summary["GCS"] = {
            "observed_count": gcs_cnt,
            "status": "SUFFICIENT" if gcs_cnt >= 4 else ("SPARSE" if gcs_cnt > 0 else "ABSENT"),
            "details": f"{gcs_cnt} assessments",
        }

        # 4. Respiratory Rate
        rr_cnt = int((temporal_df["Parameter"] == "RespRate").sum()) if not temporal_df.empty else 0
        critical_vitals_summary["RespRate"] = {
            "observed_count": rr_cnt,
            "status": "SUFFICIENT" if rr_cnt >= 4 else ("SPARSE" if rr_cnt > 0 else "ABSENT"),
            "details": f"{rr_cnt} measurements",
        }

        # 5. Body Temperature
        temp_cnt = int((temporal_df["Parameter"] == "Temp").sum()) if not temporal_df.empty else 0
        critical_vitals_summary["Temp"] = {
            "observed_count": temp_cnt,
            "status": "SUFFICIENT" if temp_cnt >= 4 else ("SPARSE" if temp_cnt > 0 else "ABSENT"),
            "details": f"{temp_cnt} measurements",
        }

        # ---------------------------------------------------------
        # 4. SENSOR ARTIFACTS VS GENUINE PATHOLOGICAL EXTREMES
        # ---------------------------------------------------------
        # Strict separation: Sensor artifacts are physiological impossibilities (e.g. HR=450 or pH=3.0).
        # Pathological extremes are real life-threatening values (e.g. Lactate=9.0, GCS=3, pH=7.12)
        # and must NEVER be deleted or classified as artifacts.
        sensor_artifacts = []
        pathological_alerts = []

        if not temporal_df.empty:
            for idx, row in temporal_df.iterrows():
                p = row["Parameter"]
                val = row["Value"]
                t = row["Time"]

                # A. Check for impossible sensor artifacts
                if p in self.IMPOSSIBLE_BOUNDS:
                    bounds = self.IMPOSSIBLE_BOUNDS[p]
                    if val < bounds["min"] or val > bounds["max"]:
                        sensor_artifacts.append({
                            "time": t,
                            "parameter": p,
                            "value": val,
                            "plausible_bounds": f"[{bounds['min']}, {bounds['max']}] {bounds['unit']}",
                            "classification": "SENSOR_ARTIFACT_PHYSIOLOGICAL_IMPOSSIBILITY",
                        })
                        continue  # Do not double-count as pathological

                # B. Check for genuine pathological extremes
                if p in self.PATHOLOGICAL_EXTREMES:
                    crit = self.PATHOLOGICAL_EXTREMES[p]
                    if val < crit["low"] or val > crit["high"]:
                        pathological_alerts.append({
                            "time": t,
                            "parameter": p,
                            "value": val,
                            "clinical_significance": crit["desc"],
                            "classification": "EXTREME_PATHOLOGICAL_VALUE",
                        })

        # Check Mechanical Ventilation
        mechvent_obs = temporal_df[temporal_df["Parameter"] == "MechVent"] if not temporal_df.empty else pd.DataFrame()
        has_mechvent = len(mechvent_obs) > 0

        # ---------------------------------------------------------
        # 5. REPRODUCIBLE DATA QUALITY INDEX (DQI) CALCULATION
        # ---------------------------------------------------------
        # FORMULATION (Research-defined metric: 0 - 100):
        # Base: 100.0
        # Deductions:
        # - Missing Essential Descriptors:
        #     * Age missing / invalid: -15.0 pts
        #     * Gender missing / invalid: -5.0 pts
        #     * ICUType missing / invalid: -5.0 pts
        #     * Admission Weight missing (sentinel -1): -5.0 pts
        #     * Height missing (sentinel -1): -2.0 pts
        # - Critical Vital Domain Deficiencies (HR, BP, GCS, RespRate, Temp):
        #     * Absent domain (0 observations): -10.0 pts per domain
        #     * Sparse domain (1 to 3 observations): -4.0 pts per domain
        # - Sensor Artifacts (Impossible values):
        #     * -5.0 pts per artifact (capped at -20.0 pts maximum)
        # - General Observation Density:
        #     * Total measurements < 20: -15.0 pts
        #     * Total measurements < 50: -8.0 pts
        # Bound: [0.0, 100.0]

        dqi_score = 100.0
        penalties = []

        # Descriptor Deductions
        if cleaned_descriptors.get("Age") is None or "INVALID_AGE_VALUE" in descriptor_issues:
            dqi_score -= 15.0
            penalties.append("Missing or invalid Age (-15 pts)")
        if cleaned_descriptors.get("Gender") is None or "INVALID_GENDER_CODE" in descriptor_issues:
            dqi_score -= 5.0
            penalties.append("Missing or invalid Gender (-5 pts)")
        if cleaned_descriptors.get("ICUType") is None or "INVALID_ICU_TYPE_CODE" in descriptor_issues:
            dqi_score -= 5.0
            penalties.append("Missing or invalid ICUType (-5 pts)")
        if cleaned_descriptors.get("Weight") is None:
            dqi_score -= 5.0
            penalties.append("Missing admission Weight sentinel (-5 pts)")
        if cleaned_descriptors.get("Height") is None:
            dqi_score -= 2.0
            penalties.append("Missing Height sentinel (-2 pts)")

        # Vital Domain Deductions
        for domain, info in critical_vitals_summary.items():
            if info["status"] == "ABSENT":
                dqi_score -= 10.0
                penalties.append(f"Zero monitoring for critical domain {domain} (-10 pts)")
            elif info["status"] == "SPARSE":
                dqi_score -= 4.0
                penalties.append(f"Sparse monitoring for critical domain {domain} (< 4 obs: -4 pts)")

        # Sensor Artifact Deductions
        if sensor_artifacts:
            art_penalty = min(20.0, len(sensor_artifacts) * 5.0)
            dqi_score -= art_penalty
            penalties.append(f"Physiological sensor artifacts detected ({len(sensor_artifacts)} events: -{art_penalty:.0f} pts)")

        # General Density Deductions
        if obs_count < 20:
            dqi_score -= 15.0
            penalties.append(f"Critically sparse overall telemetry ({obs_count} measurements: -15 pts)")
        elif obs_count < 50:
            dqi_score -= 8.0
            penalties.append(f"Moderately sparse overall telemetry ({obs_count} measurements: -8 pts)")

        dqi_score = round(max(0.0, min(100.0, dqi_score)), 1)

        # Assign DQI Grade
        if dqi_score >= 85.0:
            quality_grade = "EXCELLENT"
            status = "PASS"
        elif dqi_score >= 70.0:
            quality_grade = "ACCEPTABLE"
            status = "PASS"
        elif dqi_score >= 50.0:
            quality_grade = "SUBOPTIMAL_WARNING"
            status = "WARNING"
        else:
            quality_grade = "UNRELIABLE_CRITICAL"
            status = "FAIL"

        # Actionable Agent Recommendations
        recommendations = []
        if status == "PASS":
            recommendations.append("Data quality verified. Patient is suitable for automated AI risk inference.")
            if pathological_alerts:
                recommendations.append(f"Clinical Alert: {len(pathological_alerts)} acute pathological measurement(s) identified. Forward alerts to Downstream Prediction & SBAR Agents.")
        elif status == "WARNING":
            recommendations.append("Data quality is suboptimal. Downstream predictions should be interpreted with high-uncertainty flags.")
            absent_domains = [d for d, s in critical_vitals_summary.items() if s["status"] == "ABSENT"]
            if absent_domains:
                recommendations.append(f"Request immediate telemetry update for unmonitored domains: {', '.join(absent_domains)}.")
        else:
            recommendations.append("Data quality audit FAILED. Severe missingness or sensor corruption prevents reliable AI inference.")

        return {
            "agent": self.agent_name,
            "version": self.version,
            "record_id": record_id,
            "status": status,
            "quality_grade": quality_grade,
            "data_quality_index": dqi_score,
            "dqi_measure_disclaimer": "Research-defined observational completeness score (0-100); not a clinically validated diagnostic index.",
            "prediction_horizon_hours": prediction_horizon_hours,
            "descriptors_raw": raw_descriptors,
            "descriptors_cleaned": cleaned_descriptors,
            "missing_descriptors": missing_descriptors,
            "descriptor_issues": descriptor_issues,
            "total_observations_in_window": obs_count,
            "unique_parameters_measured": len(unique_params),
            "critical_vitals_summary": critical_vitals_summary,
            "blood_pressure_summary": {
                "total_bp_observations": total_bp_observations,
                "modality": bp_monitoring_type,
                "invasive_count": invasive_bp_count,
                "noninvasive_count": noninvasive_bp_count,
            },
            "sensor_artifacts_detected": len(sensor_artifacts),
            "sensor_artifact_details": sensor_artifacts[:5],
            "extreme_pathological_events": len(pathological_alerts),
            "extreme_pathological_details": pathological_alerts[:5],
            "mechanical_ventilation_recorded": has_mechvent,
            "mechvent_observation_count": len(mechvent_obs),
            "scoring_penalties": penalties,
            "recommendations": recommendations,
        }

    def generate_markdown_report(self, audit: Dict[str, Any]) -> str:
        """Formats the audit result into a clean, clinician-friendly markdown briefing."""
        status_badge = "[PASS]" if audit["status"] == "PASS" else ("[WARNING]" if audit["status"] == "WARNING" else "[FAIL]")

        report = f"""### Data Quality Audit Report: Patient #{audit.get('record_id', 'Unknown')}
**Agent:** `{audit['agent']} (v{audit.get('version', '1.1.0')})` | **Status:** **{status_badge}** | **Data Quality Index (DQI):** **{audit['data_quality_index']}/100 ({audit['quality_grade']})**  
*Notice: DQI is a research-defined operational measure of EHR completeness, not a clinically validated score.*

#### 1. Admission Descriptors Audit
- **Record ID:** {audit.get('record_id', 'N/A')}
- **Age:** {audit['descriptors_cleaned'].get('Age', 'Missing')} | **Gender:** {'Male (1)' if audit['descriptors_cleaned'].get('Gender') == 1 else ('Female (0)' if audit['descriptors_cleaned'].get('Gender') == 0 else 'Missing')} | **ICU Unit:** Type {audit['descriptors_cleaned'].get('ICUType', 'Missing')}
- **Height:** {str(audit['descriptors_cleaned'].get('Height')) + ' cm' if audit['descriptors_cleaned'].get('Height') is not None else 'Missing (sentinel -1)'}
- **Admission Weight:** {str(audit['descriptors_cleaned'].get('Weight')) + ' kg' if audit['descriptors_cleaned'].get('Weight') is not None else 'Missing (sentinel -1)'}
- **Missing Descriptors Identified:** {', '.join(audit['missing_descriptors']) if audit['missing_descriptors'] else 'None (Complete admission profile)'}
- **Descriptor Violations:** {', '.join(audit['descriptor_issues']) if audit['descriptor_issues'] else 'None (All recorded fields valid)'}

#### 2. Temporal Monitoring Coverage (First {audit['prediction_horizon_hours']} Hours)
- **Total Measurements in Window:** {audit['total_observations_in_window']} observations across {audit['unique_parameters_measured']} clinical parameters.
- **Blood Pressure Monitoring Modality:** **{audit['blood_pressure_summary']['modality']}** ({audit['blood_pressure_summary']['total_bp_observations']} total BP checks)
  * Invasive (A-Line): {audit['blood_pressure_summary']['invasive_count']} checks (MAP, SysABP, DiasABP)
  * Non-Invasive (NIBP Cuff): {audit['blood_pressure_summary']['noninvasive_count']} checks (NIMAP, NISysABP, NIDiasABP)
- **Critical Vital Sign Domains:**
"""
        for cv, info in audit["critical_vitals_summary"].items():
            report += f"  * **{cv}:** {info['details']} | Status: **[{info['status']}]**\n"

        report += f"""
#### 3. Sensor Artifacts vs. Extreme Pathological Alerts
- **Sensor Artifacts (Biological Impossibilities):** **{audit['sensor_artifacts_detected']} events**.
"""
        if audit["sensor_artifact_details"]:
            for art in audit["sensor_artifact_details"]:
                report += f"  * Hour {art['time']}: {art['parameter']} = {art['value']} (Expected bounds: {art['plausible_bounds']})\n"
        else:
            report += "  * Zero impossible sensor artifacts detected. Observational integrity confirmed.\n"

        report += f"- **Extreme Pathological Alerts (Genuine Critical Physiology):** **{audit['extreme_pathological_events']} events flagged**.\n"
        if audit["extreme_pathological_details"]:
            for alert in audit["extreme_pathological_details"]:
                report += f"  * Hour {alert['time']}: {alert['parameter']} = {alert['value']} ({alert['clinical_significance']})\n"

        report += f"- **Mechanical Ventilation:** {'Ventilated (' + str(audit['mechvent_observation_count']) + ' checks recorded)' if audit['mechanical_ventilation_recorded'] else 'No ventilation events explicitly recorded'}\n"

        report += "\n#### 4. DQI Quality Scoring Deductions\n"
        if audit["scoring_penalties"]:
            for p in audit["scoring_penalties"]:
                report += f"- {p}\n"
        else:
            report += "- Zero deductions. Pristine observational fidelity (100.0/100).\n"

        report += "\n#### 5. Autonomous Agent Recommendations\n"
        for rec in audit["recommendations"]:
            report += f"- {rec}\n"

        return report
