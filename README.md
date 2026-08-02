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
| **Supervisor (LangGraph)** — routes a query to the right agent (LLM classifier + keyword fallback) | ✅ Done |
| **External-factors agent** — promo lift, holiday effect, weekday patterns from history | ✅ Done |
| **Reporting agent** — calls all specialists and synthesizes an LLM briefing | ✅ Done |
| **Evaluation harness** — 50 routing cases; keyword router 84% vs LLM router 100% | ✅ Done |
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
python -m venv venv312 && source venv312/Scripts/activate   # Python 3.12 · pandas · statsmodels · Prophet · scikit-learn · LangGraph (agent orchestration) · Groq (LLM routing & synthesis) · FastAPI + Docker + CI (planned)
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

*Next: expand to more ambiguous/edge cases and add answer-quality evaluation
(LLM-as-judge) beyond routing.*

## Next steps

1. **Wire the reporting agent into the supervisor** — add a "full report" route so
   the supervisor hands off to composition when the user wants the whole picture.
2. **Extend evaluation** — add answer-quality cases (LLM-as-judge) beyond routing,
   and harder/ambiguous routing cases to stress-test.
3. **FastAPI + Docker + CI/CD** — expose `/ask`, `/health`; containerize; GitHub Actions.
4. **AI Ops** — cost dashboard ($/request per agent), model cascading, caching.
5. **Security & guardrails** — input validation, prompt-injection defense, access control.
6. **MCP server** — expose the agents as MCP tools.