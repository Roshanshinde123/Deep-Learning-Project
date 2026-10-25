"""
Sidebar component for ICU Risk Intelligence.
Renders project identity, dataset specifications, active patient status,
and persistent clinical research safety disclaimers.
"""

import streamlit as st


def render_sidebar():
    """Renders the standard sidebar across all dashboard pages."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 10px 0px 5px 0px; text-align: left;">
                <h2 style="color: #1E3A8A; margin: 0; font-size: 1.5rem; font-weight: 700;">
                    🏥 ICU Risk Intelligence
                </h2>
                <p style="color: #4B5563; font-size: 0.85rem; margin-top: 4px; line-height: 1.3;">
                    Agentic AI-Assisted ICU Mortality Risk Prediction from First 48 Hours
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # Active Session Information Card
        st.markdown("#### 📋 Active Session Context")
        if "patient_result" in st.session_state and st.session_state["patient_result"] is not None:
            res = st.session_state["patient_result"]
            pid = res["patient_id"]
            horizon = res["prediction_horizon_hours"]
            dqi = res["data_quality"]["dqi"]
            priority = res["report"]["review_priority"]

            priority_color = {
                "LOW": "#10B981",
                "MODERATE": "#F59E0B",
                "HIGH": "#EF4444",
                "CRITICAL_DATA_QUALITY": "#7C3AED",
            }.get(priority, "#6B7280")

            st.markdown(
                f"""
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                    <div style="font-size: 0.85rem; color: #64748B;">Analyzed Patient:</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #0F172A;">#{pid}</div>
                    <div style="display: flex; justify-content: space-between; margin-top: 6px; font-size: 0.8rem;">
                        <span>Horizon: <b>{horizon}h</b></span>
                        <span>DQI: <b>{dqi:.1f}</b></span>
                    </div>
                    <div style="margin-top: 8px;">
                        <span style="background-color: {priority_color}; color: white; padding: 3px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;">
                            {priority}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("No active patient analyzed. Select or upload a record in **Patient Analysis**.")

        st.divider()

        # Dataset & Model Architecture Specs
        st.markdown("#### 🔬 System Specifications")
        st.markdown(
            """
            - **Cohort:** PhysioNet Challenge 2012 / MIMIC-II
            - **Primary Model:** XGBoost Tabular (334 Features)
            - **Secondary Model:** Multimodal LSTM (74 Channels)
            - **Decision Threshold:** τ = 0.58 (F1-Optimal)
            - **Multi-Agent Stack:**
              - *Tier 1:* Data Quality Agent (DQI)
              - *Tier 2:* Prediction Agent (Dual-Model)
              - *Tier 3:* Explanation Agent (Tree SHAP)
              - *Tier 4:* Reporting / Priority Agent
            """
        )

        st.divider()

        # Mandatory Safety Disclaimer
        st.markdown(
            """
            <div style="background-color: #FEF2F2; border-left: 4px solid #EF4444; padding: 10px; border-radius: 4px;">
                <p style="color: #991B1B; font-size: 0.75rem; margin: 0; font-weight: 600;">
                    ⚠️ RESEARCH & EDUCATIONAL PROTOTYPE ONLY
                </p>
                <p style="color: #B91C1C; font-size: 0.72rem; margin-top: 4px; line-height: 1.3;">
                    Not approved for clinical diagnosis, triage, medication dosing, or treatment intervention decisions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
