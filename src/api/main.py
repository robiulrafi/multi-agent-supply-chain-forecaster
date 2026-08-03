"""
FastAPI application exposing the multi-agent supply-chain system.

Endpoints:
  GET  /health   — lightweight liveness probe (does NOT invoke any model/agent)
  POST /ask      — route a natural-language query to the right agent
  POST /report   — full multi-agent briefing for a store (forecast+anomaly+drivers)
  GET  /stores   — list available store IDs

Design mirrors the RAG project: Pydantic validation at the boundary, a health
check that never touches the model, and clear error handling.
"""

from __future__ import annotations
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agents.supervisor import build_supervisor
from src.agents.reporting_agent import ReportingAgent
from src.data.loader import list_stores
from src.ops.cost_tracker import tracker as cost_tracker

app = FastAPI(
    title="Multi-Agent Supply Chain Forecaster",
    description="Routes supply-chain questions across forecasting, anomaly, "
                "external-factors, and reporting agents over real retail data.",
    version="0.1.0",
)

# build once at startup (compiling the graph / instantiating agents is not free)
_supervisor = build_supervisor()
_reporter = ReportingAgent()


# ---------- request/response models ----------
class AskRequest(BaseModel):
    query: str = Field(..., examples=["Will store 5 run low next month?"])
    store_id: Optional[int] = Field(None, description="Override the store id parsed from the query")
    horizon_days: int = Field(30, ge=1, le=365)


class AskResponse(BaseModel):
    route: str
    message: str
    data: dict


class ReportRequest(BaseModel):
    store_id: int = Field(..., ge=1)
    horizon_days: int = Field(30, ge=1, le=365)


# ---------- endpoints ----------
@app.get("/health")
def health():
    """Liveness probe. Deliberately does NOT invoke any agent or model —
    it just confirms the service is up, so orchestrators can poll it cheaply."""
    return {"status": "ok", "service": "supply-chain-multi-agent"}


@app.get("/metrics")
def metrics():
    """AI Ops: per-agent usage, latency, and estimated cost since startup."""
    return cost_tracker.snapshot()


@app.get("/stores")
def stores():
    """List available store IDs (first 50)."""
    try:
        ids = list_stores()
        return {"count": len(ids), "stores": ids[:50]}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Data unavailable: {e}")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Route a natural-language query to the right specialist agent."""
    try:
        state = {"query": req.query, "horizon_days": req.horizon_days}
        if req.store_id is not None:
            state["store_id"] = req.store_id
        out = _supervisor.invoke(state)
        return AskResponse(
            route=out.get("route", "unknown"),
            message=out.get("message", ""),
            data=out.get("result", {}),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")


@app.post("/report")
def report(req: ReportRequest):
    """Full multi-agent briefing: forecast + anomalies + drivers, synthesized."""
    try:
        resp = _reporter.run(store_id=req.store_id, horizon_days=req.horizon_days)
        if not resp.ok:
            raise HTTPException(status_code=404, detail=resp.message)
        return {"message": resp.message, "data": resp.data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reporting error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
