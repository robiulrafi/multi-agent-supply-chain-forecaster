"""
MCP Server
----------
Exposes the supply-chain agents as Model Context Protocol (MCP) tools, so any
MCP-compatible client (Claude Desktop, other agents, IDEs) can call them.

Tools exposed:
  - forecast_demand(store_id, horizon_days)  -> forecasting agent
  - detect_anomalies(store_id)               -> anomaly agent
  - generate_report(store_id, horizon_days)  -> reporting agent (full briefing)
  - list_stores()                            -> available store IDs

Run (stdio transport, the standard for local MCP clients):
    python -m src.mcp_server

To use from Claude Desktop, add to claude_desktop_config.json:
    {
      "mcpServers": {
        "supply-chain": {
          "command": "python",
          "args": ["-m", "src.mcp_server"],
          "cwd": "/absolute/path/to/multi-agent-supply-chain"
        }
      }
    }
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from src.agents.forecasting_agent import ForecastingAgent
from src.agents.anomaly_agent import AnomalyAgent
from src.agents.reporting_agent import ReportingAgent
from src.data.loader import list_stores as _list_stores

mcp = FastMCP("supply-chain-forecaster")

_forecasting = ForecastingAgent()
_anomaly = AnomalyAgent()
_reporting = ReportingAgent()


@mcp.tool()
def forecast_demand(store_id: int, horizon_days: int = 30) -> dict:
    """Forecast future daily sales/demand for a store over a horizon.

    Benchmarks Holt-Winters vs Prophet on held-out data and returns the better
    model with a confidence signal (backtest MAPE).

    Args:
        store_id: the store to forecast.
        horizon_days: how many days ahead to forecast (default 30).
    """
    resp = _forecasting.run(store_id=store_id, horizon_days=horizon_days)
    return {"ok": resp.ok, "message": resp.message, "data": resp.data}


@mcp.tool()
def detect_anomalies(store_id: int) -> dict:
    """Find unusual sales days (spikes/drops) for a store in its history.

    Uses a weekday-aware robust z-score so it respects retail's weekly pattern.

    Args:
        store_id: the store to analyze.
    """
    resp = _anomaly.run(store_id=store_id)
    return {"ok": resp.ok, "message": resp.message, "data": resp.data}


@mcp.tool()
def generate_report(store_id: int, horizon_days: int = 30) -> dict:
    """Produce a full supply-chain briefing for a store: forecast + anomalies +
    demand drivers, synthesized into one actionable summary.

    Args:
        store_id: the store to report on.
        horizon_days: forecast horizon for the report (default 30).
    """
    resp = _reporting.run(store_id=store_id, horizon_days=horizon_days)
    return {"ok": resp.ok, "message": resp.message, "data": resp.data}


@mcp.tool()
def list_available_stores() -> dict:
    """List the store IDs available in the dataset."""
    ids = _list_stores()
    return {"count": len(ids), "stores": ids[:50]}


if __name__ == "__main__":
    # stdio transport — the standard for local MCP clients like Claude Desktop
    mcp.run(transport="stdio")
