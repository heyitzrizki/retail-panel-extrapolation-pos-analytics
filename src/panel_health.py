from __future__ import annotations

import numpy as np
import pandas as pd


# 0.1 KPI Components
def active_store_rate(panel: pd.DataFrame) -> float:
    if not len(panel):
        return np.nan
    if "is_active" not in panel.columns:
        return 1.0
    return panel["is_active"].mean()


def missing_store_rate(expected_stores: pd.Series, observed_stores: pd.Series) -> float:
    expected = set(expected_stores)
    observed = set(observed_stores)
    return len(expected - observed) / len(expected) if expected else np.nan


def sample_coverage_score(panel: pd.DataFrame, universe: pd.DataFrame) -> float:
    active_count = universe[universe["is_active"]]["store_nbr"].nunique()
    return panel["store_nbr"].nunique() / active_count if active_count else np.nan


def category_coverage_score(panel_store_list: pd.DataFrame, category_sales: pd.DataFrame) -> float:
    all_categories = category_sales["family"].nunique()
    panel_categories = category_sales[category_sales["store_nbr"].isin(panel_store_list["store_nbr"])]["family"].nunique()
    return panel_categories / all_categories if all_categories else np.nan


def top_store_concentration_score(panel: pd.DataFrame, top_n: int = 5) -> float:
    total = panel["total_sales"].sum()
    return panel["total_sales"].nlargest(min(top_n, len(panel))).sum() / total if total else np.nan


def outlier_rate(values: pd.Series, z_threshold: float = 3.0) -> float:
    std = values.std()
    if std == 0 or np.isnan(std):
        return 0.0
    z_scores = (values - values.mean()).abs() / std
    return (z_scores > z_threshold).mean()


def extrapolation_reliability_score(error_pct: float) -> float:
    return max(0.0, 1 - abs(error_pct)) if pd.notna(error_pct) else np.nan


def panel_stability_score(current_panel: pd.DataFrame, prior_panel: pd.DataFrame | None = None) -> float:
    if prior_panel is None or prior_panel.empty:
        return 1.0
    current = set(current_panel["store_nbr"])
    prior = set(prior_panel["store_nbr"])
    return len(current & prior) / len(current | prior) if current or prior else np.nan


def overall_panel_health_score(
    sample_score: float,
    category_score: float,
    concentration: float,
    reliability_score: float,
    stability_score: float,
) -> float:
    concentration_component = 1 - concentration if pd.notna(concentration) else np.nan
    values = np.array([sample_score, category_score, concentration_component, reliability_score, stability_score], dtype=float)
    return float(np.nanmean(values))


# 0.2 Risk and Actions
def assign_risk_level(score: float) -> str:
    if pd.isna(score):
        return "unknown"
    if score >= 0.8:
        return "low"
    if score >= 0.6:
        return "medium"
    return "high"


def recommended_action(row: pd.Series) -> str:
    if row["risk_level"] == "low":
        return "Maintain current panel and monitor weekly."
    if row["top_store_concentration"] > 0.5:
        return "Reduce dependence on top stores or add comparable stores."
    if row["sample_coverage_score"] < 0.3:
        return "Increase store coverage in underrepresented strata."
    if row["category_coverage_score"] < 0.8:
        return "Add stores that improve missing category coverage."
    return "Review panel composition and extrapolation error drivers."


def build_panel_health_summary(
    panels: pd.DataFrame,
    universe: pd.DataFrame,
    category_sales: pd.DataFrame,
    extrapolation_comparison: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for panel_name, panel in panels.groupby("panel_name"):
        error_pct = _panel_extrapolation_error(panel_name, extrapolation_comparison)
        sample_score = sample_coverage_score(panel, universe)
        category_score = category_coverage_score(panel, category_sales)
        concentration = top_store_concentration_score(panel)
        reliability = extrapolation_reliability_score(error_pct)
        stability = panel_stability_score(panel)
        overall = overall_panel_health_score(sample_score, category_score, concentration, reliability, stability)
        rows.append(
            {
                "panel_name": panel_name,
                "active_store_rate": active_store_rate(panel),
                "sample_coverage_score": sample_score,
                "category_coverage_score": category_score,
                "top_store_concentration": concentration,
                "extrapolation_error_pct": error_pct,
                "extrapolation_reliability_score": reliability,
                "panel_stability_score": stability,
                "overall_panel_health_score": overall,
                "risk_level": assign_risk_level(overall),
            }
        )
    output = pd.DataFrame(rows)
    output["recommended_action"] = output.apply(recommended_action, axis=1)
    return output


def _panel_extrapolation_error(panel_name: str, extrapolation_comparison: pd.DataFrame) -> float:
    if extrapolation_comparison.empty:
        return np.nan
    comparison = extrapolation_comparison.copy()
    if "panel_name" in comparison.columns:
        comparison = comparison[comparison["panel_name"] == panel_name]
    if comparison.empty:
        return np.nan
    best_method_error = comparison.sort_values("wape").head(1)
    return float(best_method_error["bias_pct"].iloc[0]) if not best_method_error.empty else np.nan
