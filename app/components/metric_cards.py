"""
Card and Metric UI Components for ICU Risk Intelligence.
Renders responsive, styled cards for DQI, model predictions, agreement, and priority.
"""

import streamlit as st
from typing import Dict, Any, Optional


def render_dqi_card(dq: Dict[str, Any]):
    """Renders the comprehensive Data Quality Index (DQI) audit card."""
    dqi = dq.get("dqi", 0.0)
    category = dq.get("category", "UNKNOWN")
    status = dq.get("status", "UNKNOWN")
    density = dq.get("observation_density", 0.0)
    artifacts = dq.get("sensor_artifacts", 0)
    missing_desc = dq.get("missing_descriptors", [])

    cat_color = {
        "EXCELLENT": "#10B981",
        "ACCEPTABLE": "#3B82F6",
        "SUBOPTIMAL": "#F59E0B",
        "UNRELIABLE_CRITICAL": "#EF4444",
        "FAIL": "#EF4444",
    }.get(category, "#6B7280")

    status_badge = "#10B981" if status == "PASS" else "#F59E0B" if status == "WARN" else "#EF4444"

    st.markdown(
        f"""
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-size: 0.9rem; font-weight: 600; color: #475569; text-transform: uppercase; letter-spacing: 0.5px;">Data Quality Index (DQI)</span>
                <span style="background-color: {status_badge}; color: white; padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 700;">{status}</span>
            </div>
            <div style="display: flex; align-items: baseline; gap: 8px;">
                <span style="font-size: 2.4rem; font-weight: 800; color: {cat_color}; line-height: 1;">{dqi:.1f}</span>
                <span style="font-size: 1.1rem; color: #94A3B8; font-weight: 500;">/ 100</span>
                <span style="background-color: {cat_color}20; color: {cat_color}; border: 1px solid {cat_color}40; padding: 3px 8px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; margin-left: 8px;">
                    {category}
                </span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 16px; padding-top: 14px; border-top: 1px solid #F1F5F9; font-size: 0.85rem;">
                <div>
                    <span style="color: #64748B;">Observation Density:</span>
                    <b style="color: #1E293B; margin-left: 4px;">{density * 100:.1f}%</b>
                </div>
                <div>
                    <span style="color: #64748B;">Sensor Artifacts:</span>
                    <b style="color: {'#EF4444' if artifacts > 0 else '#10B981'}; margin-left: 4px;">{artifacts} detected</b>
                </div>
                <div style="grid-column: span 2;">
                    <span style="color: #64748B;">Missing Descriptors:</span>
                    <b style="color: #1E293B; margin-left: 4px;">{', '.join(missing_desc) if missing_desc else 'None (Complete)'}</b>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_cards(pred_info: Dict[str, Any], horizon_hours: int = 48):
    """Renders dual model prediction cards with strict probability independence."""
    status = pred_info.get("status", "UNKNOWN")
    preds = pred_info.get("predictions", {})

    if status == "BLOCKED_BY_DATA_QUALITY":
        st.error("Automated predictions blocked: telemetry failed minimum Data Quality standards (DQI < 50.0). No speculative probabilities were generated.")
        return

    xgb = preds.get("xgboost", {})
    lstm = preds.get("lstm", {})

    p_xgb = xgb.get("predicted_probability")
    th_xgb = xgb.get("threshold", 0.58)
    dec_xgb = xgb.get("threshold_decision", "N/A")

    p_lstm = lstm.get("predicted_probability")
    th_lstm = lstm.get("threshold", 0.58)
    dec_lstm = lstm.get("threshold_decision", "N/A")

    col1, col2 = st.columns(2)

    with col1:
        color_xgb = "#EF4444" if dec_xgb == "ABOVE_THRESHOLD" else "#10B981"
        st.markdown(
            f"""
            <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); height: 100%;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 0.85rem; font-weight: 700; color: #1E3A8A; text-transform: uppercase;">Primary Model: XGBoost Tabular</span>
                    <span style="background-color: #EFF6FF; color: #1D4ED8; font-size: 0.75rem; font-weight: 600; padding: 2px 8px; border-radius: 4px;">334 Features</span>
                </div>
                <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 10px;">Model-Estimated Mortality Probability:</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: {color_xgb}; line-height: 1;">
                    {f"{p_xgb:.4f}" if p_xgb is not None else "N/A"}
                </div>
                <div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid #F1F5F9; font-size: 0.82rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="color: #64748B;">Decision Threshold:</span>
                        <b style="color: #1E293B;">τ = {th_xgb:.2f}</b>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #64748B;">Threshold Classification:</span>
                        <span style="color: {color_xgb}; font-weight: 700;">{dec_xgb}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        color_lstm = "#EF4444" if dec_lstm == "ABOVE_THRESHOLD" else "#10B981"
        st.markdown(
            f"""
            <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); height: 100%;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 0.85rem; font-weight: 700; color: #4338CA; text-transform: uppercase;">Secondary Model: Multimodal LSTM</span>
                    <span style="background-color: #EEF2FF; color: #4338CA; font-size: 0.75rem; font-weight: 600; padding: 2px 8px; border-radius: 4px;">74 Channels</span>
                </div>
                <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 10px;">Model-Estimated Mortality Probability:</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: {color_lstm}; line-height: 1;">
                    {f"{p_lstm:.4f}" if p_lstm is not None else "N/A"}
                </div>
                <div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid #F1F5F9; font-size: 0.82rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="color: #64748B;">Decision Threshold:</span>
                        <b style="color: #1E293B;">τ = {th_lstm:.2f}</b>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #64748B;">Threshold Classification:</span>
                        <span style="color: {color_lstm}; font-weight: 700;">{dec_lstm}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_agreement_card(agreement_status: str, agreement_meaning: str):
    """Renders the model agreement / discordance status card."""
    color_map = {
        "MODEL_CONCORDANT_LOW": ("#10B981", "#ECFDF5", "Concordant Low Risk"),
        "MODEL_CONCORDANT_HIGH": ("#EF4444", "#FEF2F2", "Concordant High Risk"),
        "MODEL_DISCORDANT": ("#F59E0B", "#FFFBEB", "Model Discordance"),
        "PREDICTION_BLOCKED": ("#7C3AED", "#F5F3FF", "Prediction Inhibited"),
    }
    border_color, bg_color, title = color_map.get(agreement_status, ("#6B7280", "#F8FAFC", agreement_status))

    st.markdown(
        f"""
        <div style="background-color: {bg_color}; border-left: 5px solid {border_color}; border-radius: 8px; padding: 16px; margin: 12px 0px;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <h4 style="margin: 0; color: {border_color}; font-size: 1.05rem;">{title} ({agreement_status})</h4>
            </div>
            <p style="margin: 6px 0 0 0; color: #334155; font-size: 0.9rem; line-height: 1.4;">
                {agreement_meaning}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_priority_card(priority: str, rationale: str):
    """Renders the deterministic research review priority card."""
    prio_color = {
        "LOW": "#10B981",
        "MODERATE": "#F59E0B",
        "HIGH": "#EF4444",
        "CRITICAL_DATA_QUALITY": "#7C3AED",
    }.get(priority, "#6B7280")

    st.markdown(
        f"""
        <div style="background-color: #FFFFFF; border: 2px solid {prio_color}; border-radius: 10px; padding: 18px; margin: 12px 0px; box-shadow: 0 2px 4px rgba(0,0,0,0.06);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.85rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Research Review Priority</span>
                <span style="background-color: {prio_color}; color: white; padding: 4px 12px; border-radius: 16px; font-size: 0.9rem; font-weight: 800;">
                    {priority}
                </span>
            </div>
            <p style="color: #1E293B; font-size: 0.92rem; margin: 10px 0 0 0; line-height: 1.45;">
                <b>Rationale:</b> {rationale}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
