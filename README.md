# Multi-Agent Supply Chain Forecaster

A multi-agent system for supply-chain intelligence. A supervisor agent routes
questions across specialized agents — forecasting, anomaly detection,
external-factors, and reporting — over real retail sales data (Rossmann Store
Sales: 1,115 stores, daily sales 2013–2015, with promotions and holidays).

## Status
- [x] Data loader (Rossmann per-store daily series)
- [x] Forecasting agent — **multi-model** (Holt-Winters + Prophet), self-backtesting,
      selects the best model by MAPE, reports a confidence signal
- [ ] Anomaly-detection agent
- [ ] Supervisor / LangGraph orchestration
- [ ] Reporting agent
- [ ] Evaluation harness (50+ cases)
- [ ] FastAPI + Docker + CI
- [ ] Cost tracking, security guardrails, MCP

## Quickstart
```bash
python -m venv venv312 && source venv312/Scripts/activate   # Python 3.12
pip install -r requirements.txt
# place Rossmann train.csv and store.csv in ./data
python -m src.agents.forecasting_agent
```

## Forecasting approach
The forecasting agent benchmarks multiple models on held-out data (backtest MAPE)
and selects the best one automatically:
- **Holt-Winters** (statsmodels) — exponential smoothing with weekly seasonality; robust baseline
- **Prophet** — additive model with weekly + yearly seasonality and promo as a regressor

Instead of trusting one model, it compares them empirically and reports which won
and the expected error — so every forecast comes with a confidence signal.

## Architecture
Supervisor agent → {forecasting, anomaly, external-factors, reporting} agents →
grounded, explained answer with a forecast-confidence (backtest MAPE) signal.
