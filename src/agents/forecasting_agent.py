"""
Forecasting Agent
-----------------
The first specialized agent in the multi-agent supply-chain system.

Its job: given a store (and a horizon), produce a demand forecast plus a
plain-language explanation and a trust signal (backtest error). It exposes a
single `run` method that the supervisor agent will call, and a `TOOL_SPEC`
describing it so an LLM supervisor can decide when to route here.

This agent is intentionally self-contained and testable WITHOUT an LLM, so the
forecasting core can be validated on its own before orchestration is added.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any

from src.data.loader import get_store_series, list_stores
from src.forecasting.model_selector import compare_and_forecast


TOOL_SPEC = {
    "name": "forecast_demand",
    "description": (
        "Forecast future daily sales/demand for a specific store over a given "
        "horizon. Use when the user asks about future sales, demand, or whether "
        "a store will hit certain sales levels."
    ),
    "parameters": {
        "store_id": "int \u2014 the store to forecast",
        "horizon_days": "int \u2014 how many days ahead to forecast (default 30)",
    },
}


@dataclass
class AgentResponse:
    agent: str
    ok: bool
    data: dict[str, Any]
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


class ForecastingAgent:
    name = "forecasting_agent"

    def run(self, store_id: int, horizon_days: int = 30) -> AgentResponse:
        # validate the store exists
        available = list_stores()
        if store_id not in available:
            return AgentResponse(
                agent=self.name,
                ok=False,
                data={"available_stores_sample": available[:10]},
                message=(
                    f"Store {store_id} not found. "
                    f"Available stores include: {available[:10]}..."
                ),
            )

        series = get_store_series(store_id)
        comparison = compare_and_forecast(series, store_id, horizon_days=horizon_days)
        result = comparison.best_result

        # a trust signal derived from backtest error
        if result.mape is None:
            trust = "unknown (not enough history to backtest)"
        elif result.mape < 12:
            trust = "high"
        elif result.mape < 25:
            trust = "moderate"
        else:
            trust = "low"

        data = {
            "store_id": store_id,
            "horizon_days": horizon_days,
            "model_selected": comparison.best_model,
            "model_comparison_mape": {
                k: (round(v, 1) if v is not None else None)
                for k, v in comparison.all_mape.items()
            },
            "mean_daily_forecast": round(result.mean_forecast),
            "total_forecast": round(result.total_forecast),
            "prediction_interval": (
                {
                    "pct": result.interval_pct,
                    "lower_daily": round(float(result.lower.mean())),
                    "upper_daily": round(float(result.upper.mean())),
                }
                if result.lower is not None and result.upper is not None
                else None
            ),
            "backtest_mape_pct": round(result.mape, 1) if result.mape else None,
            "trust": trust,
            "forecast_start": str((result.train_end).date()),
            "forecast_first_7_days": {
                str(d.date()): round(v)
                for d, v in result.forecast.head(7).items()
            },
        }

        message = (
            f"Store {store_id}: predicted ~{data['mean_daily_forecast']:,} sales/day "
            f"over the next {horizon_days} days "
            f"(~{data['total_forecast']:,} total). "
            f"Best model: {comparison.best_model}. "
            f"Forecast confidence: {trust}"
            + (f" (backtest error {data['backtest_mape_pct']}%)." if result.mape else ".")
        )

        return AgentResponse(agent=self.name, ok=True, data=data, message=message)


if __name__ == "__main__":
    agent = ForecastingAgent()
    stores = list_stores()
    resp = agent.run(store_id=stores[0], horizon_days=30)
    print("OK:", resp.ok)
    print("MESSAGE:", resp.message)
    print("\nSTRUCTURED DATA:")
    for k, v in resp.data.items():
        print(f"  {k}: {v}")

    print("\n--- error handling test (bad store) ---")
    bad = agent.run(store_id=99999)
    print("OK:", bad.ok, "|", bad.message)
