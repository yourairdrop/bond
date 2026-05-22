"""PnL and fee calculations for the Alpha Suite backtest."""

from __future__ import annotations


def binary_trade_cost(notional: float, fee_rate: float) -> float:
    """Total cash spent to open a binary position."""
    return round(notional * (1.0 + fee_rate), 2)


def binary_trade_pnl(notional: float, entry_price: float, won: bool, fee_rate: float) -> float:
    """PnL for a binary buy-and-hold trade."""
    if notional <= 0 or entry_price <= 0 or entry_price >= 1:
        return 0.0

    fee = notional * fee_rate
    if won:
        shares = notional / entry_price
        return round(shares - notional - fee, 2)
    return round(-(notional + fee), 2)


def complete_set_edge(price_sum: float, fee_rate: float) -> float | None:
    """Expected edge per notional dollar for a complete-set buy."""
    if price_sum <= 0:
        return None
    return (1.0 / price_sum) - 1.0 - fee_rate


def complete_set_trade_cost(notional: float, fee_rate: float) -> float:
    """Total cash spent to open a complete-set position."""
    return round(notional * (1.0 + fee_rate), 2)


def complete_set_trade_pnl(notional: float, price_sum: float, fee_rate: float) -> float:
    """PnL for buying a complete set and holding to resolution."""
    edge = complete_set_edge(price_sum, fee_rate)
    if edge is None or notional <= 0:
        return 0.0
    return round(notional * edge, 2)


def edge_to_bps(edge: float | None) -> int:
    """Convert a per-dollar edge to basis points."""
    if edge is None:
        return 0
    return round(edge * 10000)
