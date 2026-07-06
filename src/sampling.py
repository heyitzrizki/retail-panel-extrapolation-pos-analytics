from __future__ import annotations

import numpy as np
import pandas as pd


# 0.1 Helpers
def _sample_n(universe: pd.DataFrame, sample_size: int | float) -> int:
    active_count = len(universe)
    if isinstance(sample_size, float):
        return max(1, int(round(active_count * sample_size)))
    return min(sample_size, active_count)


def _active_universe(universe: pd.DataFrame) -> pd.DataFrame:
    return universe[universe["is_active"]].copy()


def _panel_frame(panel: pd.DataFrame, panel_name: str) -> pd.DataFrame:
    output = panel.copy()
    output["panel_name"] = panel_name
    return output


# 0.2 Panel Creation
def create_random_sample_panel(
    universe: pd.DataFrame,
    sample_size: int | float = 0.35,
    random_state: int = 42,
    panel_name: str = "random_panel",
) -> pd.DataFrame:
    active = _active_universe(universe)
    panel = active.sample(n=_sample_n(active, sample_size), random_state=random_state)
    return _panel_frame(panel, panel_name)


def create_stratified_sample_panel(
    universe: pd.DataFrame,
    strata_col: str = "type",
    sample_size: int | float = 0.35,
    random_state: int = 42,
    panel_name: str = "stratified_panel",
) -> pd.DataFrame:
    active = _active_universe(universe)
    frac = sample_size if isinstance(sample_size, float) else sample_size / len(active)
    panel = (
        active.groupby(strata_col, group_keys=False, dropna=False)
        .apply(lambda group: group.sample(n=max(1, int(round(len(group) * frac))), random_state=random_state))
        .drop_duplicates("store_nbr")
    )
    return _panel_frame(panel, panel_name)


def create_biased_large_store_panel(
    universe: pd.DataFrame,
    sample_size: int | float = 0.35,
    random_state: int = 42,
    panel_name: str = "large_store_biased_panel",
) -> pd.DataFrame:
    active = _active_universe(universe)
    n = _sample_n(active, sample_size)
    weights = active["total_sales"].clip(lower=0) + 1
    panel = active.sample(n=n, weights=weights, random_state=random_state)
    return _panel_frame(panel, panel_name)


def create_reduced_panel(
    panel: pd.DataFrame,
    excluded_store_nbrs: list[int],
    panel_name: str = "reduced_panel",
) -> pd.DataFrame:
    reduced = panel[~panel["store_nbr"].isin(excluded_store_nbrs)].copy()
    reduced["panel_name"] = panel_name
    return reduced


def create_optimized_panel(
    universe: pd.DataFrame,
    sample_size: int | float = 0.35,
    min_type_coverage: bool = True,
    min_cluster_coverage: bool = True,
    panel_name: str = "optimized_panel",
) -> pd.DataFrame:
    active = _active_universe(universe).sort_values("total_sales", ascending=False).copy()
    n = _sample_n(active, sample_size)
    selected = []
    if min_type_coverage and "type" in active.columns:
        selected.extend(active.groupby("type").head(1)["store_nbr"].tolist())
    if min_cluster_coverage and "cluster" in active.columns:
        selected.extend(active.groupby("cluster").head(1)["store_nbr"].tolist())
    selected = list(dict.fromkeys(selected))
    remaining = active[~active["store_nbr"].isin(selected)]
    selected.extend(remaining.head(max(0, n - len(selected)))["store_nbr"].tolist())
    panel = active[active["store_nbr"].isin(selected[:n])].copy()
    return _panel_frame(panel, panel_name)


# 0.3 Panel Comparison
def compare_panel_coverage(
    panels: list[pd.DataFrame],
    universe: pd.DataFrame,
    category_sales: pd.DataFrame | None = None,
) -> pd.DataFrame:
    active = _active_universe(universe)
    universe_sales = active["total_sales"].sum()
    category_count = category_sales["family"].nunique() if category_sales is not None and not category_sales.empty else np.nan
    rows = []
    for panel in panels:
        name = panel["panel_name"].iloc[0]
        panel_stores = set(panel["store_nbr"])
        panel_sales = panel["total_sales"].sum()
        top_sales = panel["total_sales"].nlargest(min(5, len(panel))).sum()
        if category_sales is not None and not category_sales.empty:
            panel_categories = category_sales[category_sales["store_nbr"].isin(panel_stores)]["family"].nunique()
            category_coverage = panel_categories / category_count if category_count else np.nan
        else:
            category_coverage = np.nan
        rows.append(
            {
                "panel_name": name,
                "store_count": len(panel),
                "store_type_coverage": panel["type"].nunique() / active["type"].nunique(),
                "cluster_coverage": panel["cluster"].nunique() / active["cluster"].nunique(),
                "city_coverage": panel["city"].nunique() / active["city"].nunique(),
                "state_coverage": panel["state"].nunique() / active["state"].nunique(),
                "sales_contribution_coverage": panel_sales / universe_sales if universe_sales else np.nan,
                "category_coverage": category_coverage,
                "top_store_concentration": top_sales / panel_sales if panel_sales else np.nan,
            }
        )
    return pd.DataFrame(rows)


def combine_panel_store_lists(panels: list[pd.DataFrame]) -> pd.DataFrame:
    base_columns = ["panel_name", "store_nbr", "city", "state", "type", "cluster", "sales_size_group", "total_sales"]
    optional_columns = ["is_active"]
    return pd.concat(
        [panel[[column for column in base_columns + optional_columns if column in panel.columns]] for panel in panels],
        ignore_index=True,
    )
