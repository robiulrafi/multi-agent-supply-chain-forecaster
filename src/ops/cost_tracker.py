"""
AI Ops: cost & usage tracking
-----------------------------
Lightweight, dependency-free instrumentation for the multi-agent system. Wraps
agent/LLM calls to record per-agent latency, call counts, LLM token usage, and
an estimated dollar cost, so the system can answer "what does each agent cost?"

This addresses a real production concern: multi-agent systems fan out to several
LLM calls, and cost/latency can balloon silently. Here we make it observable.

The tracker is a process-global singleton (fine for a single-service demo). In a
real deployment you'd export these to Prometheus/StatsD; the shape is the same.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from threading import Lock
import time
import functools


# Rough Groq pricing for llama-3.1-8b-instant (USD per 1M tokens).
# Adjust to your model; the point is per-agent $ visibility, not exact billing.
_PRICE_PER_1M_INPUT = 0.075
_PRICE_PER_1M_OUTPUT = 0.30


@dataclass
class AgentStats:
    calls: int = 0
    errors: int = 0
    total_latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def avg_latency_s(self) -> float:
        return self.total_latency_s / self.calls if self.calls else 0.0

    @property
    def est_cost_usd(self) -> float:
        return (
            self.input_tokens / 1_000_000 * _PRICE_PER_1M_INPUT
            + self.output_tokens / 1_000_000 * _PRICE_PER_1M_OUTPUT
        )


class CostTracker:
    def __init__(self):
        self._stats: dict[str, AgentStats] = {}
        self._lock = Lock()

    def _get(self, agent: str) -> AgentStats:
        if agent not in self._stats:
            self._stats[agent] = AgentStats()
        return self._stats[agent]

    def record(self, agent: str, latency_s: float, *, error: bool = False,
               input_tokens: int = 0, output_tokens: int = 0) -> None:
        with self._lock:
            s = self._get(agent)
            s.calls += 1
            if error:
                s.errors += 1
            s.total_latency_s += latency_s
            s.input_tokens += input_tokens
            s.output_tokens += output_tokens

    def snapshot(self) -> dict:
        with self._lock:
            per_agent = {}
            total_cost = 0.0
            total_calls = 0
            for name, s in self._stats.items():
                per_agent[name] = {
                    "calls": s.calls,
                    "errors": s.errors,
                    "avg_latency_s": round(s.avg_latency_s, 3),
                    "input_tokens": s.input_tokens,
                    "output_tokens": s.output_tokens,
                    "est_cost_usd": round(s.est_cost_usd, 6),
                }
                total_cost += s.est_cost_usd
                total_calls += s.calls
            return {
                "total_calls": total_calls,
                "total_est_cost_usd": round(total_cost, 6),
                "per_agent": per_agent,
            }

    def reset(self) -> None:
        with self._lock:
            self._stats.clear()


# process-global tracker
tracker = CostTracker()


def tracked(agent_name: str):
    """Decorator: time a call and record it against `agent_name`.
    If the wrapped function returns a dict with token counts under
    'usage' -> {'input_tokens', 'output_tokens'}, those are recorded too."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            error = False
            try:
                return fn(*args, **kwargs)
            except Exception:
                error = True
                raise
            finally:
                latency = time.perf_counter() - start
                tracker.record(agent_name, latency, error=error)
        return wrapper
    return deco


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) for logging LLM usage without a
    tokenizer dependency."""
    return max(1, len(text) // 4)


if __name__ == "__main__":
    # demo
    import random
    for _ in range(3):
        t0 = time.perf_counter()
        time.sleep(random.uniform(0.01, 0.05))
        tracker.record("forecasting_agent", time.perf_counter() - t0)
    tracker.record("reporting_agent", 0.9, input_tokens=350, output_tokens=120)
    import json
    print(json.dumps(tracker.snapshot(), indent=2))
