"""
Supervisor Agent (LangGraph)
----------------------------
The orchestrator. Takes a natural-language query about a store, decides which
specialist agent should handle it (forecasting vs. anomaly), invokes that agent,
and returns the result.

Routing is done with a small LLM classifier (Groq/Ollama). To keep the system
testable WITHOUT an LLM, there is also a deterministic keyword router used as a
fallback and for offline tests — so the graph runs even with no model configured.

Graph shape:
    START -> route -> (forecasting_agent | anomaly_agent) -> END
"""

from __future__ import annotations
import os
import re
from typing import TypedDict, Literal, Optional

from langgraph.graph import StateGraph, START, END

from src.agents.forecasting_agent import ForecastingAgent
from src.agents.anomaly_agent import AnomalyAgent
from src.agents.reporting_agent import ReportingAgent
from src.ops.cost_tracker import tracker as _cost_tracker
import time as _time


# ---- shared state passed between graph nodes ----
class SupervisorState(TypedDict, total=False):
    query: str
    store_id: int
    horizon_days: int
    route: str                # which agent was chosen
    result: dict              # the agent's AgentResponse as a dict
    message: str              # human-readable answer


# ---- agents (instantiated once) ----
_forecasting = ForecastingAgent()
_anomaly = AnomalyAgent()
_reporting = ReportingAgent()


# ---- deterministic keyword router (fallback + offline tests) ----
_FORECAST_KWS = ("forecast", "predict", "future", "next month", "next week",
                 "demand", "will we", "run low", "run out", "expect", "projection")
_ANOMALY_KWS = ("anomaly", "anomalies", "unusual", "outlier", "spike", "drop",
                "strange", "weird", "went wrong", "abnormal")
_REPORT_KWS = ("report", "full picture", "overview", "summary", "summarize",
               "briefing", "brief", "everything", "complete", "full report",
               "tell me about", "whole picture")


def _keyword_route(query: str) -> Literal["forecasting_agent", "anomaly_agent", "reporting_agent"]:
    q = query.lower()
    report_hits = sum(kw in q for kw in _REPORT_KWS)
    anomaly_hits = sum(kw in q for kw in _ANOMALY_KWS)
    forecast_hits = sum(kw in q for kw in _FORECAST_KWS)
    # a report request ("full picture / overview / everything") wins if present,
    # since it composes the other agents anyway
    if report_hits > 0 and report_hits >= anomaly_hits and report_hits >= forecast_hits:
        return "reporting_agent"
    return "anomaly_agent" if anomaly_hits > forecast_hits else "forecasting_agent"


def _llm_route(query: str) -> Optional[str]:
    """Use an LLM to classify the query, if a model is configured.
    Returns 'forecasting_agent' / 'anomaly_agent', or None if no LLM available."""
    if not os.getenv("GROQ_API_KEY"):
        return None
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
        prompt = (
            "You route supply-chain questions to one of three tools.\n"
            "Reply with EXACTLY one word: 'forecasting', 'anomaly', or 'report'.\n"
            "- 'forecasting' = predicting future sales/demand.\n"
            "- 'anomaly' = finding unusual/outlier sales days in history.\n"
            "- 'report' = a full overview/summary/briefing combining everything.\n\n"
            f"Question: {query}\nAnswer:"
        )
        ans = llm.invoke(prompt).content.strip().lower()
        if "report" in ans:
            return "reporting_agent"
        if "anomaly" in ans:
            return "anomaly_agent"
        if "forecast" in ans:
            return "forecasting_agent"
    except Exception:
        return None
    return None


def _extract_store_id(query: str, default: int = 1) -> int:
    m = re.search(r"store\s*#?\s*(\d+)", query.lower())
    return int(m.group(1)) if m else default


# ---- graph nodes ----
def route_node(state: SupervisorState) -> SupervisorState:
    query = state["query"]
    route = _llm_route(query) or _keyword_route(query)
    store_id = state.get("store_id") or _extract_store_id(query)
    return {**state, "route": route, "store_id": store_id}


def forecasting_node(state: SupervisorState) -> SupervisorState:
    _t0 = _time.perf_counter()
    resp = _forecasting.run(
        store_id=state["store_id"],
        horizon_days=state.get("horizon_days", 30),
    )
    _cost_tracker.record("forecasting_agent", _time.perf_counter() - _t0, error=not resp.ok)
    return {**state, "result": resp.to_dict(), "message": resp.message}


def anomaly_node(state: SupervisorState) -> SupervisorState:
    _t0 = _time.perf_counter()
    resp = _anomaly.run(store_id=state["store_id"])
    _cost_tracker.record("anomaly_agent", _time.perf_counter() - _t0, error=not resp.ok)
    return {**state, "result": resp.to_dict(), "message": resp.message}


def reporting_node(state: SupervisorState) -> SupervisorState:
    _t0 = _time.perf_counter()
    resp = _reporting.run(
        store_id=state["store_id"],
        horizon_days=state.get("horizon_days", 30),
    )
    _cost_tracker.record("reporting_agent", _time.perf_counter() - _t0, error=not resp.ok)
    return {**state, "result": resp.to_dict(), "message": resp.message}


def _route_selector(state: SupervisorState) -> str:
    return state["route"]


# ---- build the graph ----
def build_supervisor():
    g = StateGraph(SupervisorState)
    g.add_node("route", route_node)
    g.add_node("forecasting_agent", forecasting_node)
    g.add_node("anomaly_agent", anomaly_node)
    g.add_node("reporting_agent", reporting_node)

    g.add_edge(START, "route")
    g.add_conditional_edges(
        "route",
        _route_selector,
        {
            "forecasting_agent": "forecasting_agent",
            "anomaly_agent": "anomaly_agent",
            "reporting_agent": "reporting_agent",
        },
    )
    g.add_edge("forecasting_agent", END)
    g.add_edge("anomaly_agent", END)
    g.add_edge("reporting_agent", END)
    return g.compile()


# convenience wrapper
def ask(query: str, store_id: int | None = None) -> dict:
    app = build_supervisor()
    state: SupervisorState = {"query": query}
    if store_id is not None:
        state["store_id"] = store_id
    return app.invoke(state)


if __name__ == "__main__":
    tests = [
        "Will store 1 run low on sales next month?",
        "Were there any unusual sales days at store 1?",
        "Forecast demand for store 1",
        "Show me anomalies for store 1",
        "Give me a full report on store 1",
        "Complete overview of store 1",
    ]
    app = build_supervisor()
    for q in tests:
        out = app.invoke({"query": q})
        print(f"\nQ: {q}")
        print(f"   routed -> {out['route']}")
        print(f"   answer  : {out['message']}")
