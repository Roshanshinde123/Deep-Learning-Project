"""
Page 2: Patient Analysis
Primary live execution tier for the 4-agent ICU Intelligence pipeline.
Allows selecting permitted benchmark records or uploading an individual PhysioNet .txt file.
Executes real Multi-Agent workflow with live Streamlit status containers.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.components.sidebar import render_sidebar
from app.components.metric_cards import render_dqi_card
from app.components.charts import plot_vitals_trajectories, plot_parameter_observation_frequency
from src.services.icu_workflow import ICUWorkflowService


# Safe immutable resource caching: workflow service initialized once per session
@st.cache_resource
def get_workflow_service() -> ICUWorkflowService:
    return ICUWorkflowService(base_dir=ROOT_DIR)


st.set_page_config(page_title="Patient Analysis | ICU Risk Intelligence", page_icon="🔬", layout="wide")
render_sidebar()

st.title("🔬 Patient Intake & Autonomous Agent Execution")
st.markdown("Supply patient telemetry, select observation horizon, and trigger the integrated 4-tier agent workflow.")

service = get_workflow_service()

# -----------------------------------------------------------------------------
# 1. INPUT SELECTION: OPTION A (UPLOAD) OR OPTION B (SAMPLE RECORD)
# -----------------------------------------------------------------------------
st.subheader("1. Patient Telemetry Input")

input_mode = st.radio(
    "Choose input modality:",
    ["Select Permitted Sample Record (Set A)", "Upload PhysioNet Patient File (.txt)"],
    horizontal=True,
)

selected_patient_id = None
patient_df = None
patient_input_target = None

if input_mode == "Select Permitted Sample Record (Set A)":
    samples = service.get_permitted_sample_patients()
    sample_options = {s["label"]: s for s in samples}
    chosen_label = st.selectbox("Select patient cohort case:", list(sample_options.keys()))
    chosen_meta = sample_options[chosen_label]

    selected_patient_id = chosen_meta["patient_id"]
    patient_input_target = chosen_meta["patient_id"]
    st.info(f"**Case Details:** {chosen_meta['description']}")

else:
    uploaded_file = st.file_uploader(
        "Upload an individual PhysioNet patient record (.txt file):",
        type=["txt"],
        help="Expected format: comma-separated lines of Time, Parameter, Value.",
    )
    if uploaded_file is not None:
        try:
            patient_df, selected_patient_id = service.parse_uploaded_patient_file(uploaded_file)
            patient_input_target = patient_df
            st.success(f"Successfully parsed patient file (Record ID: **#{selected_patient_id}**, {len(patient_df)} measurements).")
        except Exception as e:
            st.error(f"Error parsing uploaded file: {str(e)}")
            st.stop()

# -----------------------------------------------------------------------------
# 2. PREDICTION HORIZON SELECTION
# -----------------------------------------------------------------------------
st.subheader("2. Observation Horizon")
horizon = st.radio(
    "Select ICU observation window:",
    [48, 24],
    format_func=lambda h: f"First {h} Hours ({'Standard ICU Evaluation' if h == 48 else 'Early Triage Window'})",
    horizontal=True,
)

st.divider()

# -----------------------------------------------------------------------------
# 3. TRIGGER MULTI-AGENT EXECUTION
# -----------------------------------------------------------------------------
btn_run = st.button("🚀 Execute Multi-Agent Workflow", type="primary", use_container_width=True)

if btn_run:
    if patient_input_target is None:
        st.warning("Please select or upload a valid patient record before executing.")
        st.stop()

    # Live Streamlit Status Container for real agent execution
    with st.status("Executing 4-Tier Agentic AI Intelligence Workflow...", expanded=True) as status_box:
        
        def update_progress(agent_key: str, message: str):
            status_box.write(f"🔄 **{message}**")

        try:
            workflow_result = service.run_workflow(
                patient_input=patient_input_target,
                prediction_horizon_hours=horizon,
                generate_plots=True,
                progress_callback=update_progress,
            )

            # Store in session state for cross-page isolation
            st.session_state["patient_result"] = workflow_result
            st.session_state["selected_patient_id"] = workflow_result["patient_id"]
            st.session_state["prediction_horizon_hours"] = horizon

            status_box.update(label="✅ Multi-Agent Workflow Completed Successfully!", state="complete", expanded=False)
            st.success(f"Analysis complete for Patient #{workflow_result['patient_id']} ({horizon}h horizon). Navigate to Predictions or Explainability for details.")

        except Exception as ex:
            status_box.update(label="❌ Multi-Agent Execution Encountered an Error", state="error", expanded=True)
            st.error(f"Execution Error: {str(ex)}")
            st.stop()

# -----------------------------------------------------------------------------
# 4. DATA QUALITY AUDIT VIEW (PART 7)
# -----------------------------------------------------------------------------
if "patient_result" in st.session_state and st.session_state["patient_result"] is not None:
    res = st.session_state["patient_result"]
    dq = res["data_quality"]

    st.divider()
    st.subheader(f"3. Data Quality Audit View — Patient #{res['patient_id']}")

    # Render DQI Card
    render_dqi_card(dq)

    # Data Quality Warnings
    warnings = dq.get("warnings", [])
    if warnings:
        with st.expander(f"⚠️ Telemetry & Imputation Warnings ({len(warnings)})", expanded=True):
            for w in warnings:
                st.markdown(f"- {w}")

    # Visualizations
    st.markdown("#### 📈 Physiological Trajectories & Observation Density")
    
    # Load patient telemetry for plotting if available
    plot_df = None
    if isinstance(patient_input_target, pd.DataFrame):
        plot_df = patient_input_target
    else:
        # Load sample file for plotting
        sample_path = ROOT_DIR / "Dataset" / "set-a" / f"{res['patient_id']}.txt"
        if sample_path.exists():
            try:
                plot_df, _ = service.parse_uploaded_patient_file(sample_path.read_text(encoding="utf-8"))
            except Exception:
                pass

    if plot_df is not None and not plot_df.empty:
        col_plot1, col_plot2 = st.columns([3, 2])
        with col_plot1:
            st.plotly_chart(plot_vitals_trajectories(plot_df), use_container_width=True)
        with col_plot2:
            st.plotly_chart(plot_parameter_observation_frequency(plot_df), use_container_width=True)
    else:
        st.info("Continuous telemetry points not directly available for trajectory plotting.")
