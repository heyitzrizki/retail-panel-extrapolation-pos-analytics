from __future__ import annotations

import numpy as np
import pandas as pd


# 0.1 Holdout Scenarios
def holdout_random_stores(panel: pd.DataFrame, holdout_count: int, random_state: int = 42) -> pd.DataFrame:
    return panel.sample(n=min(holdout_count, len(panel)), random_state=random_state)


def holdout_top_contributing_stores(panel: pd.DataFrame, holdout_count: int) -> pd.DataFrame:
    return panel.sort_values("total_sales", ascending=False).head(holdout_count)


def holdout_stores_by_group(panel: pd.DataFrame, group_col: str, group_value: str | int) -> pd.DataFrame:
    return panel[panel[group_col] == group_value].copy()


# 0.2 Imputation
def estimate_missing_sales_similar_store_average(
    sales: pd.DataFrame,
    panel: pd.DataFrame,
    heldout: pd.DataFrame,
    period_col: str = "week",
    strata_col: str = "type",
) -> pd.DataFrame:
    observed_panel = panel[~panel["store_nbr"].isin(heldout["store_nbr"])]
    sales_with_type = sales.merge(panel[["store_nbr", strata_col]], on="store_nbr", how="left")
    observed_sales = sales_with_type[sales_with_type["store_nbr"].isin(observed_panel["store_nbr"])]
    avg_sales = (
        observed_sales.groupby([period_col, strata_col], as_index=False)
        .agg(avg_store_sales=("unit_sales", "mean"))
    )
    heldout_counts = heldout.groupby(strata_col, as_index=False).agg(heldout_store_count=("store_nbr", "nunique"))
    estimates = avg_sales.merge(heldout_counts, on=strata_col, how="inner")
    estimates["imputed_sales"] = estimates["avg_store_sales"] * estimates["heldout_store_count"]
    return estimates.groupby(period_col, as_index=False).agg(imputed_sales=("imputed_sales", "sum"))


def estimate_missing_sales_historical_trend(
    sales: pd.DataFrame,
    heldout: pd.DataFrame,
    period_col: str = "week",
) -> pd.DataFrame:
    heldout_sales = sales[sales["store_nbr"].isin(heldout["store_nbr"])]
    trend = heldout_sales.groupby(period_col, as_index=False).agg(imputed_sales=("unit_sales", "sum"))
    if trend.empty:
        return pd.DataFrame(columns=[period_col, "imputed_sales"])
    trend["imputed_sales"] = trend["imputed_sales"].rolling(4, min_periods=1).mean()
    return trend


def estimate_missing_sales_category_trend(
    category_sales: pd.DataFrame,
    panel: pd.DataFrame,
    heldout: pd.DataFrame,
    period_col: str = "week",
) -> pd.DataFrame:
    heldout_categories = category_sales[category_sales["store_nbr"].isin(heldout["store_nbr"])]
    observed_categories = category_sales[
        category_sales["store_nbr"].isin(set(panel["store_nbr"]) - set(heldout["store_nbr"]))
    ]
    category_ratio = (
        heldout_categories.groupby("family")["unit_sales"].sum()
        / observed_categories.groupby("family")["unit_sales"].sum().replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    observed_period_category = (
        observed_categories.groupby([period_col, "family"], as_index=False)
        .agg(observed_sales=("unit_sales", "sum"))
    )
    observed_period_category["ratio"] = observed_period_category["family"].map(category_ratio).fillna(0)
    observed_period_category["imputed_sales"] = observed_period_category["observed_sales"] * observed_period_category["ratio"]
    return observed_period_category.groupby(period_col, as_index=False).agg(imputed_sales=("imputed_sales", "sum"))


# 0.3 Impact Comparison
def _risk_level(error_pct: float) -> str:
    if pd.isna(error_pct):
        return "unknown"
    return "high" if abs(error_pct) >= 0.15 else "medium" if abs(error_pct) >= 0.05 else "low"


def compare_panel_recovery_with_imputation(
    scenario_name: str,
    panel_sales: pd.DataFrame,
    heldout: pd.DataFrame,
    imputed_sales: pd.DataFrame | None,
    method: str,
) -> dict[str, float | str | int]:
    actual_panel_total = panel_sales["unit_sales"].sum()
    observed_total = panel_sales[~panel_sales["store_nbr"].isin(heldout["store_nbr"])]["unit_sales"].sum()
    imputed_total = 0 if imputed_sales is None or imputed_sales.empty else imputed_sales["imputed_sales"].sum()
    recovered_total = observed_total + imputed_total
    heldout_actual = panel_sales[panel_sales["store_nbr"].isin(heldout["store_nbr"])]["unit_sales"].sum()
    error = recovered_total - actual_panel_total
    error_pct = error / actual_panel_total if actual_panel_total else np.nan
    return {
        "scenario_name": scenario_name,
        "evaluation_level": "panel_recovery",
        "heldout_store_count": heldout["store_nbr"].nunique(),
        "heldout_sales_contribution": heldout_actual / actual_panel_total if actual_panel_total else np.nan,
        "method": method,
        "actual_panel_sales": actual_panel_total,
        "recovered_panel_sales": recovered_total,
        "actual_universe_sales": np.nan,
        "estimated_market_sales": np.nan,
        "market_extrapolation_factor": np.nan,
        "error": error,
        "error_pct": error_pct,
        "business_risk_level": _risk_level(error_pct),
    }


def compare_market_estimate_with_imputation(
    scenario_name: str,
    universe_sales: pd.DataFrame,
    panel_sales: pd.DataFrame,
    heldout: pd.DataFrame,
    imputed_sales: pd.DataFrame | None,
    method: str,
    period_col: str = "week",
    market_factor: float | None = None,
) -> dict[str, float | str | int]:
    actual_total = universe_sales["unit_sales"].sum()
    observed_total = panel_sales[~panel_sales["store_nbr"].isin(heldout["store_nbr"])]["unit_sales"].sum()
    imputed_total = 0 if imputed_sales is None or imputed_sales.empty else imputed_sales["imputed_sales"].sum()
    panel_recovered_total = observed_total + imputed_total
    if market_factor is None:
        universe_store_count = universe_sales["store_nbr"].nunique()
        panel_store_count = panel_sales["store_nbr"].nunique()
        market_factor = universe_store_count / panel_store_count if panel_store_count else np.nan
    estimated_total = panel_recovered_total * market_factor
    heldout_actual = panel_sales[panel_sales["store_nbr"].isin(heldout["store_nbr"])]["unit_sales"].sum()
    error = estimated_total - actual_total
    error_pct = error / actual_total if actual_total else np.nan
    return {
        "scenario_name": scenario_name,
        "evaluation_level": "market_impact",
        "heldout_store_count": heldout["store_nbr"].nunique(),
        "heldout_sales_contribution": heldout_actual / actual_total if actual_total else np.nan,
        "method": method,
        "actual_panel_sales": panel_sales["unit_sales"].sum(),
        "recovered_panel_sales": panel_recovered_total,
        "actual_universe_sales": actual_total,
        "estimated_market_sales": estimated_total,
        "market_extrapolation_factor": market_factor,
        "error": error,
        "error_pct": error_pct,
        "business_risk_level": _risk_level(error_pct),
    }
