# Multi-Agent Supply Chain Forecaster

A multi-agent system for supply-chain intelligence. A supervisor agent routes
questions across specialized agents — forecasting, anomaly detection,
external-factors, and reporting — over **real retail sales data** (Rossmann
Store Sales: 1,115 stores, daily sales 2013–2015, with promotions and holidays).

The goal: given a natural-language question about a store ("Will store 5 run low
next month?" or "Give me a full report on store 5"), route it to the right
specialist agent (or compose several) and return a grounded, explained answer
with a confidence signal.

## Demo

The system runs as a FastAPI service with interactive docs, and is also exposed
as an MCP server so the agents are callable from any MCP client.

**Interactive API (`/docs`)** — auto-generated OpenAPI docs for every endpoint:

![API docs](assets/screenshot_docs.png)

**`POST /ask`** — a query is routed to the forecasting agent, which benchmarks
Holt-Winters vs Prophet and returns the better model with a confidence signal:

![Forecast response](assets/screenshot_forecast.png)

**`POST /report`** — the reporting agent calls the forecasting, anomaly, and
external-factors agents, then synthesizes an actionable briefing:

![Report response](assets/screenshot_report.png)

> *"Store 1 is expected to experience a high volume of sales over the next 30
> days... Prophet has high confidence (7.6% backtest error). However, we
> identified 15 anomalous days, including a spike on December 20, 2014 (+74%).
> We recommend closely monitoring sales on Mondays (strongest) and Thursdays
> (weakest)."*

## Status

| Component | Status |
|---|---|
| Data loader (Rossmann per-store daily series) | ✅ Done |
| **Forecasting agent** — multi-model (Holt-Winters + Prophet), self-backtesting, best model by MAPE, **prediction intervals** | ✅ Done |
| **Anomaly-detection agent** — weekday-aware robust z-score (median + MAD), flags spikes/drops | ✅ Done |
| **External-factors agent** — promo lift, holiday effect, weekday patterns from history | ✅ Done |
| **Reporting agent** — calls all specialists and synthesizes an LLM briefing | ✅ Done |
| **Supervisor (LangGraph)** — 3-way routing to forecasting / anomaly / reporting (LLM classifier + keyword fallback) | ✅ Done |
| **Evaluation harness** — 50 routing cases; keyword router 84% vs LLM router 100% | ✅ Done |
| **FastAPI + Docker + CI/CD** — REST API, containerized, GitHub Actions (12 tests) | ✅ Done |
| **MCP server** — agents exposed as MCP tools for any MCP client | ✅ Done |
| **AI Ops** — per-agent cost/latency tracking via `/metrics` | ✅ Done |
| **Security guardrails** — input validation + prompt-injection defense | ✅ Done |

## Agents

**Forecasting agent** — predicts future daily sales for a store. Benchmarks
multiple models on held-out data (backtest MAPE) and selects the best one
automatically, and returns a **prediction interval** alongside the point forecast:
- **Holt-Winters** (statsmodels) — exponential smoothing with weekly seasonality; robust baseline
- **Prophet** — additive model with weekly + yearly seasonality and promotions as a regressor

On real data, Prophet typically wins (e.g. ~7.6% MAPE vs ~25% for Holt-Winters on
Store 1). A forecast reads, for example: *3,712 sales/day, 80% interval
2,923–4,507* — a range for inventory planning, not just a point estimate.

**Anomaly-detection agent** — explains the past by finding unusual sales days.
Uses a **weekday-aware robust z-score** (median + MAD within each day-of-week
group), so it accounts for retail's strong weekly pattern and flags days unusual
*for that weekday*. On real data it automatically surfaces meaningful events —
e.g. the pre-Christmas demand spikes (Dec 18–21, ~+80% vs expected).

**External-factors agent** — explains what drives a store's sales: promotion lift,
holiday effects, and day-of-week patterns computed from its own history.

**Reporting agent** — the synthesizer. Calls the forecasting, anomaly, and
external-factors agents, then combines their outputs into a single actionable
briefing (LLM narrative when a Groq key is configured, clean template otherwise).

All agents share a common `AgentResponse` contract and a `TOOL_SPEC`, so the
supervisor can route to any of them.

## Data

[Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales):
1,115 drug stores, daily sales Jan 2013 – Jul 2015, with promotions, state/school
holidays, and store metadata. Real data is not committed; place `train.csv` and
`store.csv` in `./data`.

## Quickstart

```bash
python -m venv venv312 && source venv312/Scripts/activate   # Python 3.12
pip install -r requirements.txt
# place Rossmann train.csv and store.csv in ./data

# --- run the API (interactive docs at http://localhost:8000/docs) ---
uvicorn src.api.main:app --reload --port 8000

# --- or run individual components directly ---
python -m src.agents.forecasting_agent    # forecasting agent
python -m src.agents.anomaly_agent         # anomaly-detection agent
python -m src.agents.supervisor            # supervisor (routes a query)
python -m src.agents.reporting_agent       # full multi-agent briefing
python -m src.eval.eval_routing            # evaluation harness (50 cases)

# --- run the tests ---
pytest tests/ -q
```

Set `GROQ_API_KEY` to enable LLM-based routing and LLM-written briefings;
without it, the system falls back to keyword routing and template reports so it
still runs fully offline.

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness probe (does not invoke any model) |
| GET | `/metrics` | Per-agent cost, latency, and usage (AI Ops) |
| GET | `/stores` | List available store IDs |
| POST | `/ask` | Route a natural-language query to the right agent (guardrail-protected) |
| POST | `/report` | Full multi-agent briefing for a store |

## MCP server

The agents are also exposed as **Model Context Protocol** tools, so any MCP
client (e.g. Claude Desktop) can call them:

```bash
python -m src.mcp_server        # stdio transport
```

Tools: `forecast_demand`, `detect_anomalies`, `generate_report`,
`list_available_stores`. To connect from Claude Desktop, add to
`claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "supply-chain": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "/absolute/path/to/multi-agent-supply-chain"
    }
  }
}
```

## Example — supervisor routing

```
Q: Will store 1 run low on sales next month?
   routed -> forecasting_agent
   Store 1: predicted ~3,712 sales/day (80% interval 2,923–4,507) over 30 days.
   Best model: prophet. Confidence: high (backtest error 7.6%).

Q: Were there any unusual sales days at store 1?
   routed -> anomaly_agent
   Store 1: found 15 anomalous days. Most extreme: 2014-12-20 (spike).

Q: Give me a full report on store 1
   routed -> reporting_agent
   Supply-chain briefing — Store 1: <forecast + anomalies + drivers>
```

## Evaluation

Multi-agent systems have a failure mode single-agent systems don't: the
supervisor can route to the *wrong* agent. The eval harness measures **routing
accuracy** over 50 labeled cases (25 forecasting + 25 anomaly):

| Router | Accuracy |
|---|---|
| Keyword (fast, offline) | 84% |
| LLM (Groq, intent-aware) | 100% |

The eval didn't just score — it **diagnosed the failure**: the keyword router
misroutes anomaly queries containing "demand" (e.g. "detect abnormal demand"),
because "demand" collides with the forecasting keywords. This is exactly why the
LLM router exists — it routes on intent, not keyword overlap — and the harness
quantifies the improvement (84% → 100%).

## AI Ops (observability)

The `/metrics` endpoint reports per-agent call counts, average latency, token
usage, and estimated cost. This makes a real tradeoff visible: the reporting
agent (~26s) is far slower than forecasting (~9s) or anomaly detection, because
it composes all specialists sequentially plus an LLM call — exactly the signal
that would drive an optimization like caching forecasts or running the
sub-agents in parallel.

## Security

User text reaches an LLM (router + reporting synthesis), so `/ask` is protected
by guardrails that run **before** any model call: length/type validation, bounds
checks, and prompt-injection detection (e.g. "ignore previous instructions",
role-tag spoofing). Suspicious queries are rejected with a 400.

## Architecture

```
User query
    │
    ▼
Security guardrails  — validate + sanitize (block injection)
    │
    ▼
Supervisor (LangGraph)  — routes to the right specialist
    │
    ├── Forecasting agent   → predict future demand (multi-model, intervals)
    ├── Anomaly agent       → find unusual sales days (weekday-aware robust z-score)
    ├── External-factors    → promo lift / holiday / weekday drivers
    └── Reporting agent     → calls all specialists, synthesizes a briefing
    │
    ▼
Grounded, explained answer + confidence signal

Also exposed via: FastAPI REST API  +  MCP server  +  /metrics (AI Ops)
```

## Design notes

- **Empirical model selection** — the forecasting agent backtests and picks by measured error, not by assumption.
- **Uncertainty quantified** — forecasts carry prediction intervals, not just point estimates.
- **Weekday-aware anomaly detection** — respects retail's weekly structure; robust statistics (median/MAD) resist outliers.
- **Two orchestration patterns** — the supervisor *routes* to one agent; the reporting agent *composes* several.
- **Graceful failure** — agents validate input (e.g. unknown store) and return a clean response instead of crashing.
- **Swappable LLM** — routing/synthesis use Groq when configured and fall back to offline logic otherwise.
- **Observability + security built in** — per-agent cost/latency via `/metrics`; input validation + prompt-injection defense on `/ask`.
- **CI-tested** — 12 tests cover routing, the eval harness, API endpoints, and security guardrails, running on synthetic data so CI needs no real dataset, GPU, or API key.

## Tech stack

Python 3.12 · pandas · statsmodels · Prophet · scikit-learn · LangGraph (agent
orchestration) · Groq (LLM routing & synthesis) · FastAPI · Docker · GitHub
Actions CI · MCP (Model Context Protocol)

## Limitations & future work

Honest scope notes — this is a portfolio project, not a production system:

- **Forecasting models** — Holt-Winters + Prophet only; modern alternatives
  (LightGBM-TS, N-BEATS, NeuralProphet) are not included.
- **Store metadata** — `store.csv` (store type, competition distance, assortment)
  is loaded but not yet used as forecast features.
- **Answer-quality evaluation** — routing is evaluated (84%→100%); the reporting
  narrative is not yet scored with an LLM-as-judge. *(future work)*
- **Cost optimization** — `/metrics` surfaces the bottleneck, but caching and
  model cascading are not yet implemented. *(future work)*
- **Static data** — no automated retraining, drift detection, or A/B pipeline;
  Rossmann data ends July 2015 (a standard benchmark, not recent dynamics).

## Next steps

1. **Extend evaluation** — LLM-as-judge for answer quality; harder/ambiguous routing cases.
2. **Model cascading / caching** — use `/metrics` signals to cache forecasts and route simple tasks to cheaper models.
