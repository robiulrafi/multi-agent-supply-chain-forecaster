"""
Time-series forecasting for per-store daily sales.

Uses Holt-Winters exponential smoothing (statsmodels), which captures:
  - trend (sales rising/falling over time)
  - weekly seasonality (retail has strong day-of-week patterns)

We deliberately avoid Prophet here to keep the dependency stack simple and
Windows-friendly. Holt-Winters is a solid, interpretable baseline for demand
forecasting and is easy to explain in an interview.
"""

from __future__ import annotations
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")  # statsmodels is chatty about convergence


@dataclass
class ForecastResult:
    store_id: int
    horizon_days: int
    forecast: pd.Series          # predicted sales, indexed by future date
    mean_forecast: float         # average predicted daily sales over horizon
    total_forecast: float        # sum over horizon
    train_end: pd.Timestamp
    mape: float | None = None    # backtest error, if computed
    lower: pd.Series | None = None   # lower prediction interval (optional)
    upper: pd.Series | None = None   # upper prediction interval (optional)
    interval_pct: int | None = None  # e.g. 80 for an 80% interval

    def summary(self) -> str:
        lines = [
            f"Store {self.store_id} \u2014 {self.horizon_days}-day sales forecast",
            f"  Avg daily sales (predicted): {self.mean_forecast:,.0f}",
            f"  Total predicted sales:       {self.total_forecast:,.0f}",
            f"  Forecast starts after:       {self.train_end.date()}",
        ]
        if self.lower is not None and self.upper is not None:
            lines.append(
                f"  {self.interval_pct or 80}% interval (daily): "
                f"{self.lower.mean():,.0f} \u2013 {self.upper.mean():,.0f}"
            )
        if self.mape is not None:
            lines.append(f"  Backtest MAPE:               {self.mape:.1f}%")
        return "\n".join(lines)


def _resample_daily(series: pd.Series) -> pd.Series:
    """Fill the calendar so Holt-Winters sees a regular daily frequency.
    Closed days were dropped upstream; reindex to continuous days and
    interpolate so the seasonal model has an even grid."""
    full_idx = pd.date_range(series.index.min(), series.index.max(), freq="D")
    return series.reindex(full_idx).interpolate(method="linear").ffill().bfill()


def forecast_store_sales(
    store_series: pd.DataFrame,
    store_id: int,
    horizon_days: int = 30,
    backtest: bool = True,
) -> ForecastResult:
    """
    Fit Holt-Winters on a store's historical sales and forecast `horizon_days`
    into the future.

    store_series: DataFrame with a 'Sales' column, indexed by Date
                  (as produced by loader.get_store_series).
    """
    sales = _resample_daily(store_series["Sales"].astype(float))

    # Optional backtest: hold out the last `horizon_days`, forecast them,
    # measure MAPE, so the agent can report how trustworthy the forecast is.
    mape = None
    if backtest and len(sales) > horizon_days * 3:
        train = sales.iloc[:-horizon_days]
        test = sales.iloc[-horizon_days:]
        model_bt = ExponentialSmoothing(
            train, trend="add", seasonal="add", seasonal_periods=7
        ).fit()
        pred_bt = model_bt.forecast(horizon_days)
        # MAPE, guarding against divide-by-zero
        denom = test.replace(0, np.nan)
        mape = float((np.abs((test - pred_bt) / denom)).dropna().mean() * 100)

    # Fit on ALL data for the real forward forecast
    model = ExponentialSmoothing(
        sales, trend="add", seasonal="add", seasonal_periods=7
    ).fit()
    fc = model.forecast(horizon_days)
    fc = fc.clip(lower=0)  # sales can't be negative

    return ForecastResult(
        store_id=store_id,
        horizon_days=horizon_days,
        forecast=fc,
        mean_forecast=float(fc.mean()),
        total_forecast=float(fc.sum()),
        train_end=sales.index.max(),
        mape=mape,
    )


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.data.loader import get_store_series, list_stores

    store_id = list_stores()[0]
    series = get_store_series(store_id)
    result = forecast_store_sales(series, store_id, horizon_days=30)
    print(result.summary())
    print("\nFirst 7 forecast days:")
    print(result.forecast.head(7).round(0))
