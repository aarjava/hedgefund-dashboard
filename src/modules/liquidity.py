"""
Liquidity analytics utilities.
"""

import numpy as np
import pandas as pd


def compute_liquidity_metrics(
    price_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    weights: pd.Series,
    portfolio_value: float,
    adv_pct: float = 0.10,
    window: int = 20,
) -> pd.DataFrame:
    """
    Compute ADV and days-to-liquidate metrics for each asset.

    Args:
        price_df: DataFrame of prices.
        volume_df: DataFrame of volumes.
        weights: Series of portfolio weights.
        portfolio_value: Total portfolio value in USD.
        adv_pct: Fraction of ADV used for liquidation.
        window: Rolling window for ADV.

    Returns:
        DataFrame with ADV, dollar ADV, and days to liquidate.
    """
    if price_df.empty or volume_df.empty:
        return pd.DataFrame()

    latest_prices = price_df.iloc[-1]
    aligned_weights = weights.reindex(price_df.columns).fillna(0.0)

    dollar_volume = volume_df * price_df
    adv = dollar_volume.rolling(window).mean()

    latest_adv = adv.iloc[-1]
    position_value = aligned_weights * portfolio_value

    # Avoid division by zero
    dttl = position_value / (latest_adv * adv_pct)

    out = pd.DataFrame(
        {
            "Price": latest_prices,
            "Weight": aligned_weights,
            "PositionValue": position_value,
            "ADV$": latest_adv,
            "DaysToLiquidate": dttl,
        }
    )

    return out.replace([np.inf, -np.inf], np.nan)
