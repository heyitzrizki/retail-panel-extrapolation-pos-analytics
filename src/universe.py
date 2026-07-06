from __future__ import annotations

import numpy as np
import pandas as pd


# 0.1 Universe Scope
def define_store_universe(stores: pd.DataFrame) -> pd.DataFrame:
    universe = stores.copy()
    universe["in_universe"] = True
    return universe


def identify_active_stores(
    store_sales: pd.DataFrame,
    store_transactions: pd.DataFrame | None = None,
    min_sales_days: int = 30,
    min_total_sales: float = 1.0,
    min_transaction_days: int | None = None,
) -> pd.DataFrame:
    active = store_sales.copy()
    active["sales_active"] = (active["sales_days"] >= min_sales_days) & (active["total_sales"] > min_total_sales)
    if store_transactions is not None and min_transaction_days is not None:
        active = active.merge(store_transactions[["store_nbr", "transaction_days"]], on="store_nbr", how="left")
        active["transaction_days"] = active["transaction_days"].fillna(0)
        active["transaction_active"] = active["transaction_days"] >= min_transaction_days
        active["is_active"] = active["sales_active"] | active["transaction_active"]
    else:
        active["is_active"] = active["sales_active"]
    return active


# 0.2 Strata
def add_sales_size_group(universe: pd.DataFrame, sales_col: str = "total_sales") -> pd.DataFrame:
    df = universe.copy()
    positive_sales = df.loc[df[sales_col] > 0, sales_col]
    if positive_sales.nunique() < 3:
        df["sales_size_group"] = np.where(df[sales_col] > 0, "active_sales", "no_sales")
        return df
    labels = ["small", "medium", "large"]
    df["sales_size_group"] = "no_sales"
    df.loc[df[sales_col] > 0, "sales_size_group"] = pd.qcut(
        positive_sales.rank(method="first"), q=3, labels=labels
    ).astype(str)
    return df


def create_store_strata(universe: pd.DataFrame) -> pd.DataFrame:
    df = universe.copy()
    required = ["city", "state", "type", "cluster", "sales_size_group"]
    for column in required:
        if column not in df.columns:
            df[column] = "unknown"
    df["cluster"] = df["cluster"].astype(str)
    df["store_stratum"] = (
        df["state"].astype(str)
        + "|"
        + df["city"].astype(str)
        + "|"
        + df["type"].astype(str)
        + "|"
        + df["cluster"].astype(str)
        + "|"
        + df["sales_size_group"].astype(str)
    )
    return df


# 0.3 Summaries
def summarize_universe(universe: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return (
        universe.groupby(group_col, dropna=False, as_index=False)
        .agg(
            store_count=("store_nbr", "nunique"),
            active_store_count=("is_active", "sum"),
            total_sales=("total_sales", "sum"),
        )
        .assign(active_store_rate=lambda df: df["active_store_count"] / df["store_count"].replace(0, np.nan))
        .sort_values("total_sales", ascending=False)
    )


def build_universe_summaries(universe: pd.DataFrame) -> pd.DataFrame:
    summary_frames = []
    for column in ["type", "cluster", "city", "state", "sales_size_group"]:
        summary = summarize_universe(universe, column)
        summary.insert(0, "summary_dimension", column)
        summary = summary.rename(columns={column: "summary_value"})
        summary_frames.append(summary)
    return pd.concat(summary_frames, ignore_index=True)
