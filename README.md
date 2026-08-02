# Multi-Agent Supply Chain Forecaster

A multi-agent system for supply-chain intelligence. A supervisor agent routes
questions across specialized agents — forecasting, anomaly detection,
external-factors, and reporting — over **real retail sales data** (Rossmann
Store Sales: 1,115 stores, daily sales 2013–2015, with promotions and holidays).

The goal: given a natural-language question about a store ("Will store 5 run low
next month?" or "Were there unusual sales days at store 5?"), route it to the
right specialist agent and return a grounded, explained answer with a
confidence signal.

## Status

| Component | Status |
|---|---|
| Data loader (Rossmann per-store daily series) | ✅ Done |
| **Forecasting agent** — multi-model (Holt-Winters + Prophet), self-backtesting, selects best model by MAPE, reports confidence | ✅ Done |
| **Anomaly-detection agent** — weekday-aware robust z-score (median + MAD), flags demand spikes/drops | ✅ Done |
| Supervisor / LangGraph orchestration (routes query → agent) | 🚧 Next |
| External-factors agent (holidays, seasonality context) | ⬜ Planned |
| Reporting agent (synthesize multi-agent output) | ⬜ Planned |
| Evaluation harness (50+ routing + output cases) | ⬜ Planned |
| FastAPI + Docker + CI/CD | ⬜ Planned |
| Cost tracking, security guardrails, MCP server | ⬜ Planned |

## Agents

**Forecasting agent** — predicts future daily sales for a store. Benchmarks
multiple models on held-out data (backtest MAPE) and selects the best one
automatically, so every forecast comes with an expected-error / confidence
signal:
- **Holt-Winters** (statsmodels) — exponential smoothing with weekly seasonality; robust baseline
- **Prophet** — additive model with weekly + yearly seasonality and promotions as a regressor

On real data, Prophet typically wins (e.g. ~7.6% MAPE vs ~25% for Holt-Winters
on Store 1), and the agent selects it automatically.

**Anomaly-detection agent** — explains the past by finding unusual sales days.
Uses a **weekday-aware robust z-score** (median + MAD within each day-of-week
group), so it accounts for retail's strong weekly pattern and flags days that
are unusual *for that weekday*. On real data it automatically surfaces
meaningful events — e.g. the pre-Christmas demand spikes (Dec 18–21, ~+80% vs
expected).

Both agents share a common `AgentResponse` contract and a `TOOL_SPEC`, so the
forthcoming supervisor can route to either.

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

# run the forecasting agent
python -m src.agents.forecasting_agent

# run the anomaly-detection agent
python -m src.agents.anomaly_agent
```

## Architecture

```
User query
    │
    ▼
Supervisor agent  (routes to the right specialist)
    │
    ├── Forecasting agent   → predict future demand (multi-model, self-selecting)
    ├── Anomaly agent       → find unusual sales days (weekday-aware robust z-score)
    ├── External-factors    → holidays / seasonality context   [planned]
    └── Reporting agent     → synthesize a clear answer         [planned]
    │
    ▼
Grounded, explained answer + confidence signal
```

## Design notes

- **Empirical model selection** over trusting one model — the forecasting agent
  backtests and picks by measured error.
- **Weekday-aware anomaly detection** — respects retail's weekly structure
  instead of a naive global threshold; robust statistics (median/MAD) resist
  outliers.
- **Graceful failure** — agents validate input (e.g. unknown store) and return a
  clean response instead of crashing.
- **Common agent contract** — every agent returns the same `AgentResponse`
  shape, so orchestration and evaluation are uniform.

## Tech stack

Python 3.12 · pandas · statsmodels · Prophet · scikit-learn · LangGraph
(orchestration, next) · FastAPI + Docker + CI (planned)
EOF
echo "written"
head -5 /home/claude/msc/README_new.md