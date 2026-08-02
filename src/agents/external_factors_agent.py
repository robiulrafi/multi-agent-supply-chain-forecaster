"""
External-Factors Agent
----------------------
Explains the CONTEXT around a store's demand: how much do promotions, holidays,
and day-of-week drive its sales? This gives the reporting agent the "why" behind
forecasts and anomalies.

It computes, from the store's own history:
  - promo lift  (avg sales on promo days vs non-promo days)
  - holiday effect (state/school holiday vs normal)
  - the strongest / weakest weekdays
This is data-driven context, not a forecast — a different capability again.
"""

from __future__ import annotations

import pandas as pd

from src.agents.forecasting_agent import AgentResponse
from src.data.loader import load_raw, list_stores


TOOL_SPEC = {
    "name": "external_factors",
    "description": (
        "Explain what drives a store's sales: promotion lift, holiday effects, "
        "and day-of-week patterns. Use when the user asks WHY sales move, what "
        "drives demand, or about the impact of promos/holidays."
    ),
    "parameters": {"store_id": "int — the store to analyze"},
}


class ExternalFactorsAgent:
    name = "external_factors_agent"

    def run(self, store_id: int) -> AgentResponse:
        available = list_stores()
        if store_id not in available:
            return AgentResponse(
                agent=self.name, ok=False,
                data={"available_stores_sample": available[:10]},
                message=f"Store {store_id} not found. Available: {available[:10]}...",
            )

        train, _ = load_raw()
        s = train[(train["Store"] == store_id) & (train["Open"] == 1)].copy()
        open_sales = s["Sales"].astype(float)
        base = open_sales.mean()

        # promo lift
        promo_on = s.loc[s["Promo"] == 1, "Sales"].mean()
        promo_off = s.loc[s["Promo"] == 0, "Sales"].mean()
        promo_lift_pct = ((promo_on - promo_off) / promo_off * 100) if promo_off else None

        # holiday effect (StateHoliday != '0' means some holiday)
        is_hol = s["StateHoliday"].astype(str) != "0"
        hol_sales = s.loc[is_hol, "Sales"].mean() if is_hol.any() else None
        hol_effect_pct = (
            ((hol_sales - base) / base * 100) if (hol_sales is not None and base) else None
        )

        # day-of-week pattern
        dow_avg = s.groupby("DayOfWeek")["Sales"].mean()
        strongest_dow = int(dow_avg.idxmax())
        weakest_dow = int(dow_avg.idxmin())
        dow_names = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}

        data = {
            "store_id": store_id,
            "avg_daily_sales": round(base),
            "promo_lift_pct": round(promo_lift_pct, 1) if promo_lift_pct is not None else None,
            "holiday_effect_pct": round(hol_effect_pct, 1) if hol_effect_pct is not None else None,
            "strongest_weekday": dow_names[strongest_dow],
            "weakest_weekday": dow_names[weakest_dow],
        }

        parts = [f"Store {store_id} drivers:"]
        if promo_lift_pct is not None:
            parts.append(f"promotions lift sales ~{promo_lift_pct:.0f}%")
        parts.append(f"strongest day {dow_names[strongest_dow]}, weakest {dow_names[weakest_dow]}")
        if hol_effect_pct is not None:
            parts.append(f"holidays shift sales ~{hol_effect_pct:+.0f}%")
        message = "; ".join(parts) + "."

        return AgentResponse(agent=self.name, ok=True, data=data, message=message)


if __name__ == "__main__":
    agent = ExternalFactorsAgent()
    sid = list_stores()[0]
    resp = agent.run(store_id=sid)
    print("OK:", resp.ok)
    print("MESSAGE:", resp.message)
    for k, v in resp.data.items():
        print(f"  {k}: {v}")
