"""
Model selection: run multiple forecasters, compare by backtest error (MAPE),
and return the best one.

This is the heart of a credible forecasting system: instead of trusting a single
model, we benchmark several on held-out data and select empirically. The agent
can then report which model won and why.
"""

from __future__ import annotations
from dataclasses import dataclass

import pandas as pd

from src.forecasting.forecaster import forecast_store_sales, ForecastResult
from src.forecasting.prophet_forecaster import forecast_store_sales_prophet


@dataclass
class ModelComparison:
    best_model: str
    best_result: ForecastResult
    all_mape: dict[str, float | None]

    def summary(self) -> str:
        lines = ["Model comparison (backtest MAPE, lower is better):"]
        for name, mape in self.all_mape.items():
            marker = "  <-- selected" if name == self.best_model else ""
            mape_str = f"{mape:.1f}%" if mape is not None else "n/a"
            lines.append(f"  {name:16s} {mape_str}{marker}")
        lines.append("")
        lines.append(self.best_result.summary())
        return "\n".join(lines)


# registry of available models: name -> forecaster fn
MODELS = {
    "holt_winters": forecast_store_sales,
    "prophet": forecast_store_sales_prophet,
}


def compare_and_forecast(
    store_series: pd.DataFrame,
    store_id: int,
    horizon_days: int = 30,
) -> ModelComparison:
    """Fit every model, compare backtest MAPE, return the best forecast."""
    results: dict[str, ForecastResult] = {}
    mapes: dict[str, float | None] = {}

    for name, fn in MODELS.items():
        try:
            res = fn(store_series, store_id, horizon_days=horizon_days, backtest=True)
            results[name] = res
            mapes[name] = res.mape
        except Exception as e:  # a model failing shouldn't kill the whole comparison
            mapes[name] = None
            results[name] = None  # type: ignore

    # pick the model with the lowest valid MAPE; fall back to any that ran
    valid = {n: m for n, m in mapes.items() if m is not None and results[n] is not None}
    if valid:
        best = min(valid, key=valid.get)  # type: ignore
    else:
        best = next((n for n, r in results.items() if r is not None), None)
        if best is None:
            raise RuntimeError("All forecasting models failed for this store.")

    return ModelComparison(best_model=best, best_result=results[best], all_mape=mapes)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.data.loader import get_store_series, list_stores

    sid = list_stores()[0]
    series = get_store_series(sid)
    comp = compare_and_forecast(series, sid, horizon_days=30)
    print(comp.summary())
