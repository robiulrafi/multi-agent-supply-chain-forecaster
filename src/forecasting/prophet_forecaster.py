"""
Prophet-based forecaster.

Prophet's advantages over Holt-Winters for retail data:
  - native holiday effects (we feed it the promo days as a regressor and
    holidays as holiday events)
  - handles weekly AND yearly seasonality simultaneously
  - robust to missing days

We return the same ForecastResult shape as the Holt-Winters forecaster so the
two are interchangeable and directly comparable.
"""

from __future__ import annotations
import warnings
import logging

import numpy as np
import pandas as pd
from prophet import Prophet

from src.forecasting.forecaster import ForecastResult

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


def _to_prophet_frame(store_series: pd.DataFrame) -> pd.DataFrame:
    """Prophet wants columns ds (date) and y (value); Promo becomes a regressor."""
    df = store_series.reset_index().rename(
        columns={"index": "ds", "Date": "ds", "Sales": "y"}
    )
    # the date column may be named 'Date' or the index name
    if "ds" not in df.columns:
        df = df.rename(columns={df.columns[0]: "ds"})
    df["ds"] = pd.to_datetime(df["ds"])
    keep = ["ds", "y"]
    if "Promo" in df.columns:
        keep.append("Promo")
    return df[keep]


def _fit_prophet(train_df: pd.DataFrame) -> Prophet:
    m = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
    )
    if "Promo" in train_df.columns:
        m.add_regressor("Promo")
    m.fit(train_df)
    return m


def forecast_store_sales_prophet(
    store_series: pd.DataFrame,
    store_id: int,
    horizon_days: int = 30,
    backtest: bool = True,
) -> ForecastResult:
    df = _to_prophet_frame(store_series)

    mape = None
    if backtest and len(df) > horizon_days * 3:
        train_df = df.iloc[:-horizon_days]
        test_df = df.iloc[-horizon_days:]
        m_bt = _fit_prophet(train_df)
        future_bt = test_df[["ds"] + (["Promo"] if "Promo" in df.columns else [])].copy()
        pred_bt = m_bt.predict(future_bt)["yhat"].clip(lower=0).values
        actual = test_df["y"].values
        denom = np.where(actual == 0, np.nan, actual)
        mape = float(np.nanmean(np.abs((actual - pred_bt) / denom)) * 100)

    # fit on all data, forecast forward
    m = _fit_prophet(df)
    future = m.make_future_dataframe(periods=horizon_days)
    if "Promo" in df.columns:
        # assume promo pattern continues at its historical rate for future days
        promo_rate = df["Promo"].mean()
        future["Promo"] = future["ds"].map(
            dict(zip(df["ds"], df["Promo"]))
        )
        future["Promo"] = future["Promo"].fillna((promo_rate > 0.5) * 1)
    fc_df = m.predict(future).tail(horizon_days)
    fc = pd.Series(
        fc_df["yhat"].clip(lower=0).values,
        index=pd.to_datetime(fc_df["ds"].values),
    )

    return ForecastResult(
        store_id=store_id,
        horizon_days=horizon_days,
        forecast=fc,
        mean_forecast=float(fc.mean()),
        total_forecast=float(fc.sum()),
        train_end=pd.to_datetime(df["ds"].max()),
        mape=mape,
    )


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.data.loader import get_store_series, list_stores

    sid = list_stores()[0]
    series = get_store_series(sid)
    res = forecast_store_sales_prophet(series, sid, horizon_days=30)
    print(res.summary())
