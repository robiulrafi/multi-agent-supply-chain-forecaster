"""
Data loading and preparation for the Rossmann Store Sales dataset.

The forecasting agent works on a per-store daily sales time series. This module
handles loading the raw CSVs, cleaning them, and producing a clean time series
for any given store.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_raw(data_dir: Path | str = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw train.csv and store.csv."""
    data_dir = Path(data_dir)
    train = pd.read_csv(
        data_dir / "train.csv",
        dtype={"StateHoliday": str},   # column has mixed types ('0' and 0)
        low_memory=False,
    )
    store = pd.read_csv(data_dir / "store.csv")
    train["Date"] = pd.to_datetime(train["Date"])
    return train, store


def get_store_series(
    store_id: int,
    data_dir: Path | str = DATA_DIR,
    open_only: bool = True,
) -> pd.DataFrame:
    """
    Return a clean daily sales time series for a single store.

    Returns a DataFrame indexed by Date with columns: Sales, Customers, Promo,
    plus day-of-week. Optionally drops closed days (Sales == 0, Open == 0),
    which is standard for demand forecasting since a closed day is not demand.
    """
    train, _ = load_raw(data_dir)
    s = train[train["Store"] == store_id].copy()
    if s.empty:
        raise ValueError(f"Store {store_id} not found in data.")
    s = s.sort_values("Date").set_index("Date")
    if open_only:
        s = s[s["Open"] == 1]
    return s[["Sales", "Customers", "Promo", "DayOfWeek"]]


def list_stores(data_dir: Path | str = DATA_DIR) -> list[int]:
    """Return the list of available store IDs."""
    train, _ = load_raw(data_dir)
    return sorted(train["Store"].unique().tolist())


if __name__ == "__main__":
    # quick smoke test
    stores = list_stores()
    print(f"Stores available: {len(stores)} (first few: {stores[:5]})")
    series = get_store_series(stores[0])
    print(f"\nStore {stores[0]} series shape: {series.shape}")
    print(f"Date range: {series.index.min().date()} to {series.index.max().date()}")
    print(series.head())
