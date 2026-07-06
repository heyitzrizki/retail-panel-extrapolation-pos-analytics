from __future__ import annotations

import numpy as np
import pandas as pd


# 0.1 Error Metrics
def absolute_error(actual: float, estimated: float) -> float:
    return estimated - actual


def percentage_error(actual: float, estimated: float) -> float:
    return (estimated - actual) / actual if actual else np.nan


def mape(actual: pd.Series, estimated: pd.Series) -> float:
    valid = actual != 0
    return ((estimated[valid] - actual[valid]).abs() / actual[valid].abs()).mean()


def wape(actual: pd.Series, estimated: pd.Series) -> float:
    denominator = actual.abs().sum()
    return (estimated - actual).abs().sum() / denominator if denominator else np.nan


def bias_percentage(actual: pd.Series, estimated: pd.Series) -> float:
    denominator = actual.sum()
    return (estimated.sum() - actual.sum()) / denominator if denominator else np.nan


# 0.2 Summaries
def weekly_error_summary(estimates: pd.DataFrame) -> pd.DataFrame:
    df = estimates.copy()
    df["error"] = df["estimated_sales"] - df["actual_sales"]
    df["error_pct"] = np.where(df["actual_sales"] != 0, df["error"] / df["actual_sales"], np.nan)
    return df


def category_level_error_summary(estimates: pd.DataFrame) -> pd.DataFrame:
    df = estimates.copy()
    df["error"] = df["estimated_sales"] - df["actual_sales"]
    df["error_pct"] = np.where(df["actual_sales"] != 0, df["error"] / df["actual_sales"], np.nan)
    return df


def method_comparison_table(estimates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in estimates.groupby("method"):
        rows.append(
            {
                "method": method,
                "actual_sales": group["actual_sales"].sum(),
                "estimated_sales": group["estimated_sales"].sum(),
                "absolute_error": group["estimated_sales"].sum() - group["actual_sales"].sum(),
                "mape": mape(group["actual_sales"], group["estimated_sales"]),
                "wape": wape(group["actual_sales"], group["estimated_sales"]),
                "bias_pct": bias_percentage(group["actual_sales"], group["estimated_sales"]),
            }
        )
    return pd.DataFrame(rows).sort_values("wape")
