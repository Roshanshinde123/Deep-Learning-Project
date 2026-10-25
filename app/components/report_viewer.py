"""
Report Viewer UI Component for ICU Risk Intelligence.
Renders formatted human-readable ASCII reports and provides instant
downloads for JSON and Markdown deliverables.
"""

import json
import streamlit as st
from typing import Dict, Any


def render_report_view(report_dict: Dict[str, Any], human_report: str):
    """
    Renders structured report sections and download buttons.
    """
    if not report_dict:
        st.info("No report generated yet.")
        return

    pid = report_dict.get("patient_id", "UNKNOWN")
    horizon = report_dict.get("horizon_hours", 48)

    # Top Action Bar with Download Buttons
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.subheader(f"📄 Clinical Risk Review Report — Patient #{pid}")

    with col2:
        # JSON Download
        clean_json = {k: v for k, v in report_dict.items() if k != "human_readable_report"}
        json_str = json.dumps(clean_json, indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"icu_mortality_report_{pid}_{horizon}h.json",
            mime="application/json",
            use_container_width=True,
        )

    with col3:
        # Markdown Download
        md_text = f"# ICU Mortality Risk Review — Patient #{pid}\n\n```text\n{human_report}\n```"
        st.download_button(
            label="📥 Download Markdown",
            data=md_text,
            file_name=f"icu_mortality_report_{pid}_{horizon}h.md",
            mime="text/markdown",
            use_container_width=True,
        )

    # Tabs for formatted vs raw view
    tab1, tab2 = st.tabs(["Formatted Research Review", "Machine-Readable JSON"])

    with tab1:
        st.code(human_report, language="text")

    with tab2:
        st.json(clean_json)
