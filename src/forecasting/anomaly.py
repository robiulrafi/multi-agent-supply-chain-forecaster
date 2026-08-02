"""
Anomaly detection for per-store daily sales.

Flags unusual demand days — spikes and drops — that deviate from the store's
normal pattern. This is a DIFFERENT capability from forecasting: forecasting
predicts the future; anomaly detection explains the past/present by finding
days that don't fit the expected pattern.

Method: we account for the strong day-of-week effect in retail (Mondays differ
from Saturdays), so we compute a robust z-score *within each weekday group*
using median and MAD (median absolute deviation), which is robust to outliers.
A day is anomalous if its sales are far from the typical sales for that weekday.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# 1.4826 makes MAD a consistent estimator of std for normal data
_MAD_SCALE = 1.4826


@dataclass
class Anomaly:
    date: str
    sales: float
    expected: float          # typical sales for that weekday
    z_score: float           # how many robust-std's away
    kind: str                # "spike" or "drop"


@dataclass
class AnomalyReport:
    store_id: int
    n_days_analyzed: int
    n_anomalies: int
    anomalies: list[Anomaly] = field(default_factory=list)

    def summary(self) -> str:
        if self.n_anomalies == 0:
            return (
                f"Store {self.store_id}: no significant anomalies in "
                f"{self.n_days_analyzed} days of sales."
            )
        lines = [
            f"Store {self.store_id}: {self.n_anomalies} anomalies found "
            f"in {self.n_days_analyzed} days.",
        ]
        for a in self.anomalies[:10]:
            lines.append(
                f"  {a.date}: {a.kind.upper()} — sales {a.sales:,.0f} "
                f"vs expected ~{a.expected:,.0f} (z={a.z_score:+.1f})"
            )
        if self.n_anomalies > 10:
            lines.append(f"  ... and {self.n_anomalies - 10} more")
        return "\n".join(lines)


def detect_anomalies(
    store_series: pd.DataFrame,
    store_id: int,
    z_threshold: float = 3.0,
) -> AnomalyReport:
    """
    Detect anomalous sales days using a weekday-aware robust z-score.

    store_series: DataFrame with 'Sales' and 'DayOfWeek', indexed by Date.
    z_threshold: how many robust-std's from the weekday median counts as anomalous.
    """
    df = store_series.copy()
    df = df[df["Sales"] > 0]  # ignore closed days
    sales = df["Sales"].astype(float)
    dow = df["DayOfWeek"]

    anomalies: list[Anomaly] = []

    # compute robust stats per weekday (retail has strong day-of-week effects)
    for day in sorted(dow.unique()):
        grp = sales[dow == day]
        if len(grp) < 5:
            continue
        median = grp.median()
        mad = np.median(np.abs(grp - median)) * _MAD_SCALE
        if mad == 0:
            continue
        z = (grp - median) / mad
        flagged = grp[np.abs(z) >= z_threshold]
        for date, val in flagged.items():
            zval = float((val - median) / mad)
            anomalies.append(
                Anomaly(
                    date=str(pd.Timestamp(date).date()),
                    sales=float(val),
                    expected=float(median),
                    z_score=zval,
                    kind="spike" if zval > 0 else "drop",
                )
            )

    anomalies.sort(key=lambda a: abs(a.z_score), reverse=True)

    return AnomalyReport(
        store_id=store_id,
        n_days_analyzed=int(len(sales)),
        n_anomalies=len(anomalies),
        anomalies=anomalies,
    )


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.data.loader import get_store_series, list_stores

    sid = list_stores()[0]
    series = get_store_series(sid)
    report = detect_anomalies(series, sid)
    print(report.summary())
