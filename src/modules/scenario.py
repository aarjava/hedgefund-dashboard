"""
Scenario shock utilities.
"""

from typing import Dict

import pandas as pd


def run_scenario_shocks(betas: pd.Series, shocks: Dict[str, float]) -> pd.Series:
    """
    Compute scenario PnL impact from factor betas and shocks.

    Args:
        betas: Series of betas indexed by factor names.
        shocks: Dict of factor -> shock (decimal, e.g., 0.01 for +1%).

    Returns:
        Series with factor impacts and total.
    """
    if betas is None or betas.empty:
        return pd.Series(dtype=float)

    impacts = {}
    total = 0.0
    for factor, beta in betas.items():
        shock = shocks.get(factor, 0.0)
        impact = beta * shock
        impacts[factor] = impact
        total += impact
    impacts["Total"] = total
    return pd.Series(impacts)
