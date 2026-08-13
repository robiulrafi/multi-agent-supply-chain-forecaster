"""
Reporting Agent
---------------
The synthesizer. Instead of routing to ONE specialist, it calls several agents
for a store and combines their outputs into a single narrative answer:
forecast + anomalies + drivers. This is the "give me the full picture" path.

If a Groq LLM is configured it writes a polished narrative; otherwise it falls
back to a clean template, so it always works.
"""

from __future__ import annotations
import os

from src.agents.forecasting_agent import AgentResponse, ForecastingAgent
from src.agents.anomaly_agent import AnomalyAgent
from src.agents.external_factors_agent import ExternalFactorsAgent
from src.data.loader import list_stores


TOOL_SPEC = {
    "name": "full_report",
    "description": (
        "Produce a complete supply-chain briefing for a store: demand forecast, "
        "notable anomalies, and the factors driving sales. Use when the user asks "
        "for an overview, a summary, a report, or 'the full picture'."
    ),
    "parameters": {"store_id": "int", "horizon_days": "int (default 30)"},
}


class ReportingAgent:
    name = "reporting_agent"

    def __init__(self):
        self.forecasting = ForecastingAgent()
        self.anomaly = AnomalyAgent()
        self.factors = ExternalFactorsAgent()

    def run(self, store_id: int, horizon_days: int = 30) -> AgentResponse:
        if store_id not in list_stores():
            return AgentResponse(
                agent=self.name, ok=False, data={},
                message=f"Store {store_id} not found.",
            )

        # gather from the specialists
        fc = self.forecasting.run(store_id, horizon_days=horizon_days)
        an = self.anomaly.run(store_id)
        fx = self.factors.run(store_id)

        bundle = {
            "forecast": fc.data,
            "anomalies": an.data,
            "drivers": fx.data,
        }

        narrative = self._narrate(store_id, fc, an, fx)
        return AgentResponse(agent=self.name, ok=True, data=bundle, message=narrative)

    def _narrate(self, store_id, fc, an, fx) -> str:
        # Try LLM synthesis if available; else template.
        template = (
            f"Supply-chain briefing — Store {store_id}\n"
            f"  Forecast: {fc.message}\n"
            f"  Anomalies: {an.message}\n"
            f"  Drivers: {fx.message}"
        )
        if not os.getenv("GROQ_API_KEY"):
            return template
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)
            prompt = (
                "Write a concise 3-4 sentence supply-chain briefing for a store "
                "manager, using ONLY the facts below. Be specific and practical.\n\n"
                f"Forecast: {fc.message}\n"
                f"Anomalies: {an.message}\n"
                f"Drivers: {fx.message}\n\nBriefing:"
            )
            return llm.invoke(prompt).content.strip()
        except Exception:
            return template


if __name__ == "__main__":
    agent = ReportingAgent()
    sid = list_stores()[0]
    resp = agent.run(store_id=sid)
    print("OK:", resp.ok)
    print("\n" + resp.message)
