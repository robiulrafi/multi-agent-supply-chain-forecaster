"""
Evaluation Harness
------------------
Multi-agent systems have a failure mode single-agent RAG doesn't: the SUPERVISOR
can route to the wrong agent. So the first thing we evaluate is ROUTING ACCURACY
— given a query, did the supervisor pick the intended agent?

This mirrors the eval philosophy from the RAG project: decompose into checkable
cases, score each with a clear pass/fail (not a fuzzy number), and report an
aggregate. Here each case is (query -> expected_agent); we run the router and
check the match.

Run:  python -m src.eval.eval_routing
"""

from __future__ import annotations
from dataclasses import dataclass

from src.agents.supervisor import _keyword_route, _llm_route


@dataclass
class RoutingCase:
    query: str
    expected: str          # 'forecasting_agent' | 'anomaly_agent'


# ---- 50+ labeled routing cases ----
FORECAST_CASES = [
    "Will store 5 run low on sales next month?",
    "Forecast demand for store 12",
    "Predict next week's sales for store 3",
    "What will sales look like in 30 days for store 7?",
    "How much product will store 9 sell next month?",
    "Give me a demand projection for store 2",
    "Are we going to run out of stock at store 4?",
    "Expected sales for store 6 over the next quarter",
    "What's the sales forecast for store 100?",
    "Project future revenue for store 55",
    "How many units will store 8 need next month?",
    "Estimate upcoming demand for store 20",
    "Sales prediction for store 33 next week",
    "Will store 15 hit its targets next month?",
    "What demand should I plan for at store 44?",
    "Forecast the next 60 days for store 11",
    "Upcoming sales outlook for store 77",
    "How much will store 88 sell in August?",
    "Predict store 1 sales trend going forward",
    "Give me the demand forecast for store 66",
    "What are expected sales next month for store 22?",
    "Store 30 future demand estimate please",
    "How will store 5 perform next quarter?",
    "Projected sales volume for store 99",
    "Anticipated demand at store 40 next week",
]

ANOMALY_CASES = [
    "Were there any unusual sales days at store 5?",
    "Show me anomalies for store 12",
    "Find outliers in store 3's sales",
    "Any strange sales days at store 7?",
    "Detect abnormal demand for store 9",
    "What days had unusual spikes at store 2?",
    "Were there sales drops at store 4?",
    "Show weird sales patterns for store 6",
    "Which days were anomalous for store 100?",
    "Find sales spikes for store 55",
    "Any outlier days at store 8?",
    "Detect unusual activity at store 20",
    "What went wrong with sales at store 33?",
    "Highlight abnormal sales days for store 15",
    "Were there demand anomalies at store 44?",
    "Find the strange days at store 11",
    "Show me the sales outliers for store 77",
    "Any unexpected drops at store 88?",
    "Identify unusual sales for store 1",
    "What days deviated from normal at store 66?",
    "Spot the anomalies in store 22 sales",
    "Were there odd sales days at store 30?",
    "Find irregular demand at store 5",
    "Show abnormal spikes for store 99",
    "Any unusual demand events at store 40?",
]

CASES = (
    [RoutingCase(q, "forecasting_agent") for q in FORECAST_CASES]
    + [RoutingCase(q, "anomaly_agent") for q in ANOMALY_CASES]
)


def evaluate_routing(use_llm: bool = False) -> dict:
    """Run all cases through the router and report accuracy."""
    correct = 0
    misses = []
    for case in CASES:
        route = None
        if use_llm:
            route = _llm_route(case.query)
        if route is None:
            route = _keyword_route(case.query)
        if route == case.expected:
            correct += 1
        else:
            misses.append((case.query, case.expected, route))

    total = len(CASES)
    accuracy = correct / total * 100
    return {
        "total_cases": total,
        "correct": correct,
        "accuracy_pct": round(accuracy, 1),
        "misses": misses,
        "router": "llm" if use_llm else "keyword",
    }


if __name__ == "__main__":
    report = evaluate_routing(use_llm=False)
    print(f"Routing evaluation ({report['router']} router)")
    print(f"  Cases:    {report['total_cases']}")
    print(f"  Correct:  {report['correct']}")
    print(f"  Accuracy: {report['accuracy_pct']}%")
    if report["misses"]:
        print(f"\n  Misroutes ({len(report['misses'])}):")
        for q, exp, got in report["misses"]:
            print(f"    '{q}'\n       expected {exp}, got {got}")
