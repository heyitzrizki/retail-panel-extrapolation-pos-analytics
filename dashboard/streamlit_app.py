from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# 0.1 Project Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"


# 0.2 Data Loading
@st.cache_data
def load_csv(filename: str) -> pd.DataFrame:
    path = OUTPUT_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def show_table(title: str, df: pd.DataFrame) -> None:
    st.subheader(title)
    if df.empty:
        st.info(f"{title} is not available yet. Run the notebooks to generate the CSV output.")
        return
    st.dataframe(df, use_container_width=True)


# 0.3 Dashboard
st.set_page_config(page_title="Retail Panel Extrapolation", layout="wide")
st.title("Retail Panel Extrapolation and Panel Health Monitoring")

universe_summary = load_csv("universe_summary.csv")
sample_panel_comparison = load_csv("sample_panel_comparison.csv")
extrapolation_method_comparison = load_csv("extrapolation_method_comparison.csv")
weekly_error = load_csv("weekly_extrapolation_error.csv")
category_error = load_csv("category_extrapolation_error.csv")
missing_retailer = load_csv("missing_retailer_impact_summary.csv")
panel_health = load_csv("panel_health_kpi_summary.csv")
dashboard_summary = load_csv("dashboard_summary.csv")

show_table("Dashboard Summary", dashboard_summary)
show_table("Universe Summary", universe_summary)
show_table("Sample Panel Comparison", sample_panel_comparison)
show_table("Extrapolation Method Comparison", extrapolation_method_comparison)

st.subheader("Weekly Extrapolation Error")
if weekly_error.empty:
    st.info("Weekly extrapolation error is not available yet.")
else:
    fig = px.line(
        weekly_error,
        x="period",
        y="error_pct",
        color="method",
        markers=True,
        title="Weekly Error Percentage by Method",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(weekly_error, use_container_width=True)

st.subheader("Category-Level Extrapolation Error")
if category_error.empty:
    st.info("Category-level extrapolation error is not available yet.")
else:
    category_view = (
        category_error.groupby("category", as_index=False)
        .agg(error_pct=("error_pct", "mean"), actual_sales=("actual_sales", "sum"), estimated_sales=("estimated_sales", "sum"))
        .sort_values("error_pct", key=lambda col: col.abs(), ascending=False)
    )
    st.dataframe(category_view, use_container_width=True)

show_table("Missing Retailer Impact Scenarios", missing_retailer)
show_table("Panel Health KPI Table", panel_health)

st.subheader("Recommended Actions")
if panel_health.empty or "recommended_action" not in panel_health.columns:
    st.info("Recommended actions are not available yet.")
else:
    st.dataframe(panel_health[["panel_name", "risk_level", "recommended_action"]], use_container_width=True)
