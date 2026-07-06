from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Iterable

import pandas as pd


# 0.1 File Discovery
def find_data_file(raw_dir: str | Path, stem: str, csv_only: bool = False) -> Path:
    raw_path = Path(raw_dir)
    candidates = [raw_path / f"{stem}.csv"]
    if not csv_only:
        candidates.extend([raw_path / f"{stem}.csv.7z", raw_path / f"{stem}.csv.zip"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing {stem}.csv, {stem}.csv.7z, or {stem}.csv.zip in {raw_path}")


def _validate_supported_file(path: str | Path) -> Path:
    file_path = Path(path)
    if file_path.name.endswith((".csv", ".csv.7z", ".csv.zip")):
        return file_path
    raise ValueError(f"Unsupported file format: {file_path.name}")


# 0.2 Raw Loading
def load_csv_or_7z(
    path: str | Path,
    parse_dates: list[str] | None = None,
    usecols: list[str] | None = None,
    dtype: dict[str, str] | None = None,
    chunksize: int | None = None,
) -> pd.DataFrame | Iterable[pd.DataFrame]:
    file_path = _validate_supported_file(path)
    kwargs = {
        "parse_dates": parse_dates,
        "usecols": usecols,
        "dtype": dtype,
        "low_memory": False,
    }
    if chunksize:
        kwargs["chunksize"] = chunksize
    if file_path.name.endswith(".csv.7z"):
        return _read_7z_csv(file_path, **kwargs)
    return pd.read_csv(file_path, **kwargs)


def _read_7z_csv(path: Path, **read_csv_kwargs) -> pd.DataFrame | Iterable[pd.DataFrame]:
    chunksize = read_csv_kwargs.get("chunksize")
    if chunksize:
        return _iter_7z_csv_chunks(path, read_csv_kwargs)

    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = _extract_single_csv(path, Path(temp_dir))
        return pd.read_csv(csv_path, **read_csv_kwargs)


def _iter_7z_csv_chunks(path: Path, read_csv_kwargs: dict) -> Iterable[pd.DataFrame]:
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = _extract_single_csv(path, Path(temp_dir))
        for chunk in pd.read_csv(csv_path, **read_csv_kwargs):
            yield chunk


def _extract_single_csv(path: Path, temp_dir: Path) -> Path:
    try:
        import py7zr
    except ImportError as exc:
        raise ImportError("Install py7zr to read .csv.7z files, or extract the CSV into data/raw/.") from exc
    with py7zr.SevenZipFile(path, mode="r") as archive:
        archive.extractall(path=temp_dir)
    csv_files = sorted(temp_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found inside {path.name}")
    return csv_files[0]


def load_stores(raw_dir: str | Path, csv_only: bool = False) -> pd.DataFrame:
    return load_csv_or_7z(find_data_file(raw_dir, "stores", csv_only=csv_only))


def load_items(raw_dir: str | Path, csv_only: bool = False) -> pd.DataFrame:
    return load_csv_or_7z(find_data_file(raw_dir, "items", csv_only=csv_only))


def load_transactions(raw_dir: str | Path, csv_only: bool = False) -> pd.DataFrame:
    path = find_data_file(raw_dir, "transactions", csv_only=csv_only)
    return load_csv_or_7z(path, parse_dates=["date"])


def load_train_sales(
    raw_dir: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
    families: list[str] | None = None,
    chunksize: int = 500_000,
    csv_only: bool = False,
) -> pd.DataFrame:
    path = find_data_file(raw_dir, "train", csv_only=csv_only)
    frames: list[pd.DataFrame] = []
    columns = ["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"]
    items = load_items(raw_dir, csv_only=csv_only)[["item_nbr", "family"]] if families else None
    for chunk in load_csv_or_7z(path, parse_dates=["date"], usecols=columns, chunksize=chunksize):
        chunk = _filter_date_range(chunk, start_date, end_date)
        if families and items is not None:
            chunk = chunk.merge(items, on="item_nbr", how="left")
            chunk = chunk[chunk["family"].isin(families)]
        frames.append(chunk)
    sales = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    return sales


def _filter_date_range(df: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]
    return df


# 0.3 Aggregations
def attach_item_family(sales: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    if "family" in sales.columns:
        return sales.copy()
    return sales.merge(items[["item_nbr", "family"]], on="item_nbr", how="left")


def create_daily_store_sales(sales: pd.DataFrame) -> pd.DataFrame:
    return (
        sales.groupby(["date", "store_nbr"], as_index=False)
        .agg(unit_sales=("unit_sales", "sum"), observed_rows=("unit_sales", "size"))
        .sort_values(["date", "store_nbr"])
    )


def create_weekly_store_sales(daily_store_sales: pd.DataFrame) -> pd.DataFrame:
    weekly = daily_store_sales.copy()
    weekly["week"] = weekly["date"].dt.to_period("W").apply(lambda period: period.start_time)
    return (
        weekly.groupby(["week", "store_nbr"], as_index=False)
        .agg(unit_sales=("unit_sales", "sum"), observed_rows=("observed_rows", "sum"))
        .sort_values(["week", "store_nbr"])
    )


def create_daily_store_category_sales(sales: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    sales_with_family = attach_item_family(sales, items)
    return (
        sales_with_family.groupby(["date", "store_nbr", "family"], as_index=False)
        .agg(unit_sales=("unit_sales", "sum"), observed_rows=("unit_sales", "size"))
        .sort_values(["date", "store_nbr", "family"])
    )


def create_weekly_store_category_sales(daily_store_category_sales: pd.DataFrame) -> pd.DataFrame:
    weekly = daily_store_category_sales.copy()
    weekly["week"] = weekly["date"].dt.to_period("W").apply(lambda period: period.start_time)
    return (
        weekly.groupby(["week", "store_nbr", "family"], as_index=False)
        .agg(unit_sales=("unit_sales", "sum"), observed_rows=("observed_rows", "sum"))
        .sort_values(["week", "store_nbr", "family"])
    )


def create_item_family_sales(sales: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    sales_with_family = attach_item_family(sales, items)
    return (
        sales_with_family.groupby("family", as_index=False)
        .agg(unit_sales=("unit_sales", "sum"), observed_rows=("unit_sales", "size"))
        .sort_values("unit_sales", ascending=False)
    )


def create_store_transaction_counts(transactions: pd.DataFrame) -> pd.DataFrame:
    return (
        transactions.groupby("store_nbr", as_index=False)
        .agg(transaction_days=("date", "nunique"), transactions=("transactions", "sum"))
    )


def create_store_metadata_sales(stores: pd.DataFrame, daily_store_sales: pd.DataFrame) -> pd.DataFrame:
    store_sales = (
        daily_store_sales.groupby("store_nbr", as_index=False)
        .agg(total_sales=("unit_sales", "sum"), sales_days=("date", "nunique"))
    )
    return stores.merge(store_sales, on="store_nbr", how="left").fillna({"total_sales": 0, "sales_days": 0})


# 0.4 Memory-Efficient Processing
def aggregate_train_sales_chunks(
    raw_dir: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
    families: list[str] | None = None,
    chunksize: int = 500_000,
    csv_only: bool = False,
) -> dict[str, pd.DataFrame]:
    path = find_data_file(raw_dir, "train", csv_only=csv_only)
    items = load_items(raw_dir, csv_only=csv_only)[["item_nbr", "family"]]
    if families:
        items = items[items["family"].isin(families)].copy()

    daily_store_parts: list[pd.DataFrame] = []
    daily_category_parts: list[pd.DataFrame] = []
    family_parts: list[pd.DataFrame] = []
    columns = ["date", "store_nbr", "item_nbr", "unit_sales"]
    dtype = {"store_nbr": "int16", "item_nbr": "int32", "unit_sales": "float32"}

    for chunk in load_csv_or_7z(
        path,
        parse_dates=["date"],
        usecols=columns,
        dtype=dtype,
        chunksize=chunksize,
    ):
        chunk = _filter_date_range(chunk, start_date, end_date)
        if chunk.empty:
            continue

        chunk = chunk.merge(items, on="item_nbr", how="inner" if families else "left")
        daily_store_parts.append(
            chunk.groupby(["date", "store_nbr"], as_index=False)
            .agg(unit_sales=("unit_sales", "sum"), observed_rows=("unit_sales", "size"))
        )
        daily_category_parts.append(
            chunk.groupby(["date", "store_nbr", "family"], as_index=False)
            .agg(unit_sales=("unit_sales", "sum"), observed_rows=("unit_sales", "size"))
        )
        family_parts.append(
            chunk.groupby("family", as_index=False)
            .agg(unit_sales=("unit_sales", "sum"), observed_rows=("unit_sales", "size"))
        )

    daily_store_sales = _combine_aggregated_parts(daily_store_parts, ["date", "store_nbr"])
    daily_store_category_sales = _combine_aggregated_parts(daily_category_parts, ["date", "store_nbr", "family"])
    item_family_sales = _combine_aggregated_parts(family_parts, ["family"])
    return {
        "daily_store_sales": daily_store_sales.sort_values(["date", "store_nbr"]),
        "daily_store_category_sales": daily_store_category_sales.sort_values(["date", "store_nbr", "family"]),
        "item_family_sales": item_family_sales.sort_values("unit_sales", ascending=False),
    }


def _combine_aggregated_parts(parts: list[pd.DataFrame], group_cols: list[str]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=group_cols + ["unit_sales", "observed_rows"])
    return (
        pd.concat(parts, ignore_index=True)
        .groupby(group_cols, as_index=False)
        .agg(unit_sales=("unit_sales", "sum"), observed_rows=("observed_rows", "sum"))
    )


def build_processed_tables(
    raw_dir: str | Path,
    processed_dir: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
    families: list[str] | None = None,
    chunksize: int = 500_000,
    csv_only: bool = False,
) -> dict[str, pd.DataFrame]:
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    stores = load_stores(raw_dir, csv_only=csv_only)
    sales_tables = aggregate_train_sales_chunks(
        raw_dir,
        start_date=start_date,
        end_date=end_date,
        families=families,
        chunksize=chunksize,
        csv_only=csv_only,
    )
    transactions = load_transactions(raw_dir, csv_only=csv_only)
    transactions = _filter_date_range(transactions, start_date, end_date)

    daily_store_sales = sales_tables["daily_store_sales"]
    weekly_store_sales = create_weekly_store_sales(daily_store_sales)
    daily_store_category_sales = sales_tables["daily_store_category_sales"]
    weekly_store_category_sales = create_weekly_store_category_sales(daily_store_category_sales)
    item_family_sales = sales_tables["item_family_sales"]
    store_transaction_counts = create_store_transaction_counts(transactions)
    store_metadata_sales = create_store_metadata_sales(stores, daily_store_sales)

    tables = {
        "daily_store_sales": daily_store_sales,
        "weekly_store_sales": weekly_store_sales,
        "daily_store_category_sales": daily_store_category_sales,
        "weekly_store_category_sales": weekly_store_category_sales,
        "item_family_sales": item_family_sales,
        "store_transaction_counts": store_transaction_counts,
        "store_metadata_sales": store_metadata_sales,
    }
    for name, table in tables.items():
        table.to_csv(processed_path / f"{name}.csv", index=False)
    return tables


def load_processed_table(processed_dir: str | Path, table_name: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    path = Path(processed_dir) / f"{table_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed table: {path}")
    return pd.read_csv(path, parse_dates=parse_dates)
