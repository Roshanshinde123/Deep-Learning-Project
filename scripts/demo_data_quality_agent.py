"""
Demonstration and Verification Script for Data Quality Agent (Phase 6.1)
Runs live autonomous auditing on selected patient files across diverse ICU cohorts.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.data_quality_agent import DataQualityAgent


def run_demo():
    agent = DataQualityAgent()
    base_dir = r"D:\ICU_Prediction"
    demo_files = [
        os.path.join(base_dir, "Dataset", "set-a", "132539.txt"),
        os.path.join(base_dir, "Dataset", "set-a", "132540.txt"),
        os.path.join(base_dir, "Dataset", "set-a", "132543.txt"),
    ]

    print("=" * 75)
    print("DEMONSTRATING PHASE 6.1: DATA QUALITY AGENT ON REAL ICU PATIENT RECORDS")
    print("=" * 75)

    markdown_reports = ["# Phase 6.1: Data Quality Agent Demonstration Audit\n\n"]

    for fpath in demo_files:
        if os.path.exists(fpath):
            print(f"\nAuditing Patient File: {os.path.basename(fpath)}...")
            audit = agent.audit_patient_file(fpath, prediction_horizon_hours=48)
            print(f"  Record ID:             {audit['record_id']}")
            print(f"  Audit Status:          {audit['status']}")
            print(f"  Data Quality Index:    {audit['data_quality_index']}/100 ({audit['quality_grade']})")
            print(f"  Measurements in 48h:   {audit['total_observations_in_window']}")
            print(f"  Sensor Artifacts:      {audit['sensor_artifacts_detected']}")
            print(f"  Extreme Alert Events:  {audit['extreme_pathological_events']}")
            print(f"  Mechanical Vent:       {audit['mechanical_ventilation_recorded']} ({audit['mechvent_observation_count']} events)")

            md = agent.generate_markdown_report(audit)
            markdown_reports.append(md + "\n---\n\n")

    report_path = os.path.join(base_dir, "reports", "data_quality_agent_demo.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(markdown_reports)
    print(f"\nSaved demonstration clinical reports to: {report_path}")


if __name__ == "__main__":
    run_demo()
