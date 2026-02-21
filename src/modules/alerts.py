"""
Alerting and anomaly detection utilities.
"""

from typing import Dict

import numpy as np
import pandas as pd


def evaluate_alerts(metrics: Dict[str, float], thresholds: Dict[str, Dict]) -> pd.DataFrame:
    """
    Evaluate metric thresholds to generate alerts.

    thresholds format:
        {
            "metric_name": {"type": ">" or "<", "value": float, "severity": str}
        }
    """
    rows = []
    for name, rule in thresholds.items():
        value = metrics.get(name, np.nan)
        if value is None or np.isnan(value):
            continue

        t_type = rule.get("type", ">")
        t_val = rule.get("value", 0)
        severity = rule.get("severity", "Medium")

        triggered = (value > t_val) if t_type == ">" else (value < t_val)
        if triggered:
            rows.append(
                {
                    "Alert": name,
                    "Severity": severity,
                    "Value": value,
                    "Threshold": f"{t_type} {t_val}",
                }
            )

    return pd.DataFrame(rows)


def detect_zscore_anomalies(
    returns: pd.Series, window: int = 60, z_threshold: float = 3.0
) -> pd.DataFrame:
    """
    Detect anomalies in returns using rolling z-scores.
    """
    if returns.empty:
        return pd.DataFrame()

    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    zscore = (returns - rolling_mean) / rolling_std

    flagged = zscore.abs() >= z_threshold
    out = pd.DataFrame({"Return": returns, "ZScore": zscore})
    out = out[flagged].dropna()
    return out
