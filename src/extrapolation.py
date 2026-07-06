from __future__ import annotations

import numpy as np
import pandas as pd


# 0.1 Helpers
def _panel_store_set(panel: pd.DataFrame) -> set[int]:
    return set(panel["store_nbr"].unique())


def _actual_by_period(sales: pd.DataFrame, period_col: str) -> pd.DataFrame:
    return sales.groupby(period_col, as_index=False).agg(actual_sales=("unit_sales", "sum"))


def _sample_by_period(sales: pd.DataFrame, panel: pd.DataFrame, period_col: str) -> pd.DataFrame:
    return (
        sales[sales["store_nbr"].isin(_panel_store_set(panel))]
        .groupby(period_col, as_index=False)
        .agg(sample_sales=("unit_sales", "sum"))
    )


# 0.2 Extrapolation Methods
def naive_extrapolation(
    sales: pd.DataFrame,
    universe: pd.DataFrame,
    panel: pd.DataFrame,
    period_col: str = "week",
    method_name: str = "naive",
) -> pd.DataFrame:
    active_count = universe[universe["is_active"]]["store_nbr"].nunique()
    panel_count = panel["store_nbr"].nunique()
    factor = active_count / panel_count if panel_count else np.nan
    actual = _actual_by_period(sales, period_col)
    sample = _sample_by_period(sales, panel, period_col)
    estimates = actual.merge(sample, on=period_col, how="left").fillna({"sample_sales": 0})
    estimates["estimated_sales"] = estimates["sample_sales"] * factor
    estimates["method"] = method_name
    return estimates.rename(columns={period_col: "period"})


def stratified_extrapolation(
    sales: pd.DataFrame,
    universe: pd.DataFrame,
    panel: pd.DataFrame,
    strata_col: str = "type",
    period_col: str = "week",
    method_name: str = "stratified",
) -> pd.DataFrame:
    store_strata = universe[["store_nbr", strata_col, "is_active"]].copy()
    active_counts = store_strata[store_strata["is_active"]].groupby(strata_col)["store_nbr"].nunique()
    panel_counts = panel.groupby(strata_col)["store_nbr"].nunique()
    factors = (active_counts / panel_counts).replace([np.inf, -np.inf], np.nan).fillna(0).rename("factor")
    panel_sales = sales[sales["store_nbr"].isin(_panel_store_set(panel))].merge(
        store_strata[["store_nbr", strata_col]], on="store_nbr", how="left"
    )
    estimates = (
        panel_sales.groupby([period_col, strata_col], as_index=False)
        .agg(sample_sales=("unit_sales", "sum"))
        .merge(factors, on=strata_col, how="left")
    )
    estimates["estimated_sales"] = estimates["sample_sales"] * estimates["factor"]
    estimated_total = estimates.groupby(period_col, as_index=False).agg(estimated_sales=("estimated_sales", "sum"))
    actual = _actual_by_period(sales, period_col)
    output = actual.merge(estimated_total, on=period_col, how="left").fillna({"estimated_sales": 0})
    output["method"] = method_name
    return output.rename(columns={period_col: "period"})


def contribution_weighted_extrapolation(
    sales: pd.DataFrame,
    universe: pd.DataFrame,
    panel: pd.DataFrame,
    period_col: str = "week",
    method_name: str = "contribution_weighted",
) -> pd.DataFrame:
    universe_sales = universe.loc[universe["is_active"], "total_sales"].sum()
    panel_contribution = panel["total_sales"].sum()
    factor = universe_sales / panel_contribution if panel_contribution else np.nan
    actual = _actual_by_period(sales, period_col)
    sample = _sample_by_period(sales, panel, period_col)
    estimates = actual.merge(sample, on=period_col, how="left").fillna({"sample_sales": 0})
    estimates["estimated_sales"] = estimates["sample_sales"] * factor
    estimates["method"] = method_name
    return estimates.rename(columns={period_col: "period"})


def category_level_extrapolation(
    category_sales: pd.DataFrame,
    universe: pd.DataFrame,
    panel: pd.DataFrame,
    period_col: str = "week",
    category_col: str = "family",
    method_name: str = "category_level",
) -> pd.DataFrame:
    panel_stores = _panel_store_set(panel)
    actual = (
        category_sales.groupby([period_col, category_col], as_index=False)
        .agg(actual_sales=("unit_sales", "sum"))
    )
    sample = (
        category_sales[category_sales["store_nbr"].isin(panel_stores)]
        .groupby([period_col, category_col], as_index=False)
        .agg(sample_sales=("unit_sales", "sum"))
    )
    universe_category = category_sales.groupby(category_col, as_index=False).agg(universe_category_sales=("unit_sales", "sum"))
    panel_category = (
        category_sales[category_sales["store_nbr"].isin(panel_stores)]
        .groupby(category_col, as_index=False)
        .agg(panel_category_sales=("unit_sales", "sum"))
    )
    factors = universe_category.merge(panel_category, on=category_col, how="left")
    factors["factor"] = np.where(
        factors["panel_category_sales"] > 0,
        factors["universe_category_sales"] / factors["panel_category_sales"],
        0,
    )
    estimates = actual.merge(sample, on=[period_col, category_col], how="left").merge(
        factors[[category_col, "factor"]], on=category_col, how="left"
    )
    estimates["sample_sales"] = estimates["sample_sales"].fillna(0)
    estimates["estimated_sales"] = estimates["sample_sales"] * estimates["factor"].fillna(0)
    estimates["method"] = method_name
    return estimates.rename(columns={period_col: "period", category_col: "category"})


# 0.3 Comparisons
def generate_weekly_market_estimates(
    weekly_sales: pd.DataFrame,
    universe: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    methods = [
        naive_extrapolation(weekly_sales, universe, panel),
        stratified_extrapolation(weekly_sales, universe, panel),
        contribution_weighted_extrapolation(weekly_sales, universe, panel),
    ]
    return pd.concat(methods, ignore_index=True)


def compare_market_estimates(estimates: pd.DataFrame) -> pd.DataFrame:
    output = estimates.copy()
    output["error"] = output["estimated_sales"] - output["actual_sales"]
    output["error_pct"] = np.where(output["actual_sales"] != 0, output["error"] / output["actual_sales"], np.nan)
    return output
