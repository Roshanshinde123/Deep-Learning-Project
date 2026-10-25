"""
Interactive and Statistical Chart Components for ICU Risk Intelligence.
Uses Plotly and Matplotlib for rich, responsive clinical visualizations.
"""

import os
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def plot_vitals_trajectories(df: pd.DataFrame, selected_params: Optional[List[str]] = None) -> go.Figure:
    """
    Renders an interactive multi-parameter physiological trajectory timeline.
    Distinguishes actual measured points from interpolated trajectories.
    """
    if df.empty or "Time" not in df.columns or "Parameter" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No telemetry data available for plotting.", showarrow=False)
        return fig

    # Default key parameters
    if not selected_params:
        default_candidates = ["HR", "SysABP", "DiasABP", "MAP", "Temp", "RespRate", "GCS", "Urine"]
        present_params = df["Parameter"].unique()
        selected_params = [p for p in default_candidates if p in present_params][:4]
        if not selected_params and len(present_params) > 0:
            selected_params = list(present_params[:4])

    # Convert Time (HH:MM) to decimal hours
    def parse_time_to_hours(t_str):
        try:
            parts = str(t_str).split(":")
            return float(parts[0]) + float(parts[1]) / 60.0
        except Exception:
            return np.nan

    plot_df = df[df["Parameter"].isin(selected_params)].copy()
    plot_df["Hour"] = plot_df["Time"].apply(parse_time_to_hours)
    plot_df = plot_df.dropna(subset=["Hour", "Value"]).sort_values("Hour")

    fig = go.Figure()

    palette = ["#2563EB", "#DC2626", "#059669", "#D97706", "#7C3AED", "#DB2777", "#0891B2"]

    for idx, param in enumerate(selected_params):
        sub = plot_df[plot_df["Parameter"] == param]
        if sub.empty:
            continue
        color = palette[idx % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=sub["Hour"],
                y=sub["Value"],
                mode="lines+markers",
                name=param,
                line=dict(color=color, width=2),
                marker=dict(size=6, color=color),
                hovertemplate=f"<b>{param}</b><br>Time: %{{x:.1f}}h<br>Value: %{{y:.2f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Physiological Trajectories Over Observation Horizon",
        xaxis_title="Time from ICU Admission (Hours)",
        yaxis_title="Measured Clinical Value",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=50, b=40),
        height=380,
    )
    return fig


def plot_parameter_observation_frequency(df: pd.DataFrame) -> go.Figure:
    """
    Renders horizontal bar chart of observation counts per parameter.
    """
    if df.empty or "Parameter" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No observation data available.", showarrow=False)
        return fig

    # Filter out static admission descriptors
    static_params = {"RecordID", "Age", "Gender", "Height", "Weight", "ICUType"}
    dyn_df = df[~df["Parameter"].isin(static_params)]

    counts = dyn_df["Parameter"].value_counts().reset_index()
    counts.columns = ["Parameter", "Count"]
    counts = counts.sort_values("Count", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=counts["Count"],
            y=counts["Parameter"],
            orientation="h",
            marker=dict(color="#3B82F6"),
            hovertemplate="Parameter: %{y}<br>Observations: %{x}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Measurement Density by Physiological Parameter",
        xaxis_title="Number of Discrete Observations",
        yaxis_title="Parameter",
        template="plotly_white",
        margin=dict(l=80, r=40, t=50, b=40),
        height=max(350, len(counts) * 20),
    )
    return fig


def plot_benchmark_comparison(benchmarks_df: pd.DataFrame, metric_set: str = "TestB") -> go.Figure:
    """
    Renders grouped bar chart comparing models on Test B or Test C benchmark metrics.
    """
    if benchmarks_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No benchmark metrics found.", showarrow=False)
        return fig

    auroc_col = f"{metric_set}_AUROC"
    auprc_col = f"{metric_set}_AUPRC"
    f1_col = f"{metric_set}_F1"

    if auroc_col not in benchmarks_df.columns:
        fig = go.Figure()
        fig.add_annotation(text=f"Metric set {metric_set} not present in benchmarks.", showarrow=False)
        return fig

    models = benchmarks_df["Model"].tolist()
    aurocs = benchmarks_df[auroc_col].tolist()
    auprcs = benchmarks_df[auprc_col].tolist()
    f1s = benchmarks_df[f1_col].tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=models, y=aurocs, name="AUROC", marker_color="#2563EB"))
    fig.add_trace(go.Bar(x=models, y=auprcs, name="AUPRC", marker_color="#10B981"))
    fig.add_trace(go.Bar(x=models, y=f1s, name="F1-Score", marker_color="#F59E0B"))

    fig.update_layout(
        title=f"Independent Model Performance Benchmarks ({metric_set} Evaluation)",
        barmode="group",
        yaxis=dict(title="Score", range=[0.0, 1.0]),
        xaxis=dict(title="Model Architecture", tickangle=-20),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=50, b=80),
        height=400,
    )
    return fig
