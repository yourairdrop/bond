"""
Alpha Suite — Risk Management Utilities.

Provides drawdown heat scaling, max-drawdown checks,
Kelly criterion sizing, and expected value calculations.
"""

import math
from typing import List


# ══════════════════════════════════════════════════════════════════════
# Drawdown Heat — Dynamic Position Sizing
# ══════════════════════════════════════════════════════════════════════

class DrawdownHeat:
    """Reduces position sizing as cumulative losses grow.

    Thresholds (cumulative loss):
        $0  - $10  -> 100% sizing (full)
        $10 - $25  -> 70% sizing
        $25 - $50  -> 40% sizing
        >= $50     -> 0% sizing (stop trading)
    """

    LEVELS = [
        (50.0, 0.00),  # >= $50 loss -> stop trading
        (25.0, 0.40),  # >= $25 loss -> 40% sizing
        (10.0, 0.70),  # >= $10 loss -> 70% sizing
    ]

    @staticmethod
    def multiplier(cumulative_loss: float) -> float:
        """Return sizing multiplier based on cumulative loss.

        Args:
            cumulative_loss: Total P&L (negative = loss). Uses abs() internally.

        Returns:
            Float between 0.0 and 1.0 — multiply bet size by this.
        """
        loss = abs(cumulative_loss)
        for threshold, mult in DrawdownHeat.LEVELS:
            if loss >= threshold:
                return mult
        return 1.0


# ══════════════════════════════════════════════════════════════════════
# Max Drawdown Check
# ══════════════════════════════════════════════════════════════════════

def check_mdd(equity_curve: List[float], threshold: float = 0.15) -> bool:
    """Check if maximum drawdown exceeds threshold.

    Args:
        equity_curve: List of portfolio values over time.
        threshold: MDD fraction threshold (e.g. 0.15 = 15%).

    Returns:
        True if MDD exceeds threshold (should stop trading).
        False if within limits or insufficient data.
    """
    if not equity_curve or len(equity_curve) < 2:
        return False

    peak = equity_curve[0]
    max_dd = 0.0

    for val in equity_curve:
        if val > peak:
            peak = val
        if peak > 0:
            dd = (peak - val) / peak
            if dd > max_dd:
                max_dd = dd

    return max_dd >= threshold


# ══════════════════════════════════════════════════════════════════════
# Kelly Criterion Sizing
# ══════════════════════════════════════════════════════════════════════

def kelly_size(
    p_fair: float,
    price: float,
    bankroll: float,
    exec_prob: float = 0.80,
    fraction: float = 0.25,
) -> float:
    """Compute Kelly-optimal bet size with execution probability discount.

    Formula: f = (bp - q) / b * sqrt(exec_prob) * fraction
    where b = 1/price - 1 (net payout per $1 bet), p = p_fair, q = 1-p.

    Args:
        p_fair: Estimated true probability of the outcome.
        price: Market price (probability implied by the market).
        bankroll: Available capital in USD.
        exec_prob: Probability the order fills (discount factor).
        fraction: Kelly fraction (0.25 = quarter-Kelly for safety).

    Returns:
        Recommended bet size in USD, or 0.0 if no edge.
    """
    if price <= 0 or price >= 1 or bankroll <= 0:
        return 0.0

    b = 1.0 / price - 1.0
    if b <= 0:
        return 0.0

    edge = p_fair * b - (1 - p_fair)
    if edge <= 0:
        return 0.0

    f = (edge / b) * math.sqrt(exec_prob) * fraction
    bet = f * bankroll
    return max(0.0, round(bet, 2))


# ══════════════════════════════════════════════════════════════════════
# Expected Value
# ══════════════════════════════════════════════════════════════════════

def ev_calc(p_fair: float, price: float, fee: float = 0.02) -> float:
    """Calculate expected value per dollar bet.

    EV = p_fair * (1/price - 1) - (1 - p_fair) - fee

    Args:
        p_fair: Estimated true probability.
        price: Market price to buy at.
        fee: Trading fee as fraction (default 2%).

    Returns:
        EV per dollar. Positive = profitable in expectation.
        Returns -999 for degenerate prices.
    """
    if price <= 0 or price >= 1:
        return -999.0

    win_per_dollar = 1.0 / price - 1.0
    return p_fair * win_per_dollar - (1 - p_fair) - fee


# ══════════════════════════════════════════════════════════════════════
# Statistical Helpers
# ══════════════════════════════════════════════════════════════════════

def norm_cdf(x: float, mu: float = 0.0, s: float = 1.0) -> float:
    """Normal cumulative distribution function."""
    if s <= 0:
        return 0.5
    return 0.5 * (1 + math.erf((x - mu) / (s * math.sqrt(2))))


def bs_fair(S: float, K: float, sigma: float, T_sec: float) -> tuple:
    """Black-Scholes binary option fair prices for crypto 15-min windows.

    Args:
        S: Current spot price.
        K: Strike (window start price).
        sigma: Annualized volatility.
        T_sec: Seconds remaining in the window.

    Returns:
        (fair_up, fair_down) probabilities.
    """
    if S <= 0 or K <= 0 or sigma <= 0 or T_sec <= 0:
        return 0.5, 0.5
    T = T_sec / (365.25 * 24 * 3600)
    sqrt_T = math.sqrt(T)
    d2 = (math.log(S / K) - 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    fair_up = norm_cdf(d2)
    return fair_up, 1.0 - fair_up
