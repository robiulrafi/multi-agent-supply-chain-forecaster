"""
Anomaly Detection Agent
-----------------------
The second specialized agent. Given a store, it scans historical sales and
reports unusual demand days (spikes/drops). Different capability from the
forecasting agent — this explains the past rather than predicting the future.

Same AgentResponse contract + TOOL_SPEC as the forecasting agent, so the
supervisor can route to either.
"""

from __future__ import annotations

from src.agents.forecasting_agent import AgentResponse  # reuse the response type
from src.data.loader import get_store_series, list_stores
from src.forecasting.anomaly import detect_anomalies


TOOL_SPEC = {
    "name": "detect_anomalies",
    "description": (
        "Find unusual sales days (spikes or drops) for a specific store in its "
        "historical data. Use when the user asks about anomalies, unusual sales, "
        "outliers, strange days, or what went wrong/right on certain days."
    ),
    "parameters": {
        "store_id": "int — the store to analyze",
        "z_threshold": "float — sensitivity; higher = fewer, stronger anomalies (default 3.0)",
    },
}


class AnomalyAgent:
    name = "anomaly_agent"

    def run(self, store_id: int, z_threshold: float = 3.0) -> AgentResponse:
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
        report = detect_anomalies(series, store_id, z_threshold=z_threshold)

        data = {
            "store_id": store_id,
            "days_analyzed": report.n_days_analyzed,
            "anomalies_found": report.n_anomalies,
            "top_anomalies": [
                {
                    "date": a.date,
                    "kind": a.kind,
                    "sales": round(a.sales),
                    "expected": round(a.expected),
                    "z_score": round(a.z_score, 1),
                }
                for a in report.anomalies[:5]
            ],
        }

        if report.n_anomalies == 0:
            message = (
                f"Store {store_id}: no significant anomalies across "
                f"{report.n_days_analyzed} sales days."
            )
        else:
            top = report.anomalies[0]
            message = (
                f"Store {store_id}: found {report.n_anomalies} anomalous days. "
                f"Most extreme: {top.date} ({top.kind}, {top.sales:,.0f} sales "
                f"vs expected ~{top.expected:,.0f})."
            )

        return AgentResponse(agent=self.name, ok=True, data=data, message=message)


if __name__ == "__main__":
    agent = AnomalyAgent()
    sid = list_stores()[0]
    resp = agent.run(store_id=sid)
    print("OK:", resp.ok)
    print("MESSAGE:", resp.message)
    print("\nSTRUCTURED DATA:")
    for k, v in resp.data.items():
        print(f"  {k}: {v}")
