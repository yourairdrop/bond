"""Summary statistics for the Alpha Suite backtest."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable

from alpha_suite.backtesting.models import BacktestTrade


LogFn = Callable[[str], None]


def build_daily_pnl(trades: list[BacktestTrade]) -> list[tuple[str, float]]:
    """Aggregate realized PnL by UTC resolution date in chronological order."""
    daily = defaultdict(float)
    for trade in sorted(trades, key=lambda item: (item.event_ts, item.trade_ts, item.label)):
        day = datetime.fromtimestamp(trade.event_ts, tz=timezone.utc).strftime("%Y-%m-%d")
        daily[day] += trade.pnl
    return [(day, daily[day]) for day in sorted(daily)]


def compute_trade_stats(trades: list[BacktestTrade]) -> dict:
    """Compute stable summary metrics for a set of trades."""
    if not trades:
        return {
            "n_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "total_cost": 0.0,
            "roi": 0.0,
            "avg_pnl": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
        }

    total_pnl = round(sum(trade.pnl for trade in trades), 2)
    total_cost = round(sum(trade.cost for trade in trades), 2)
    wins = sum(1 for trade in trades if trade.won)
    losses = len(trades) - wins
    win_rate = round(wins / len(trades) * 100, 1)
    avg_pnl = round(total_pnl / len(trades), 2)

    daily_pnls = [pnl for _, pnl in build_daily_pnl(trades)]
    sharpe = 0.0
    if len(daily_pnls) > 1:
        mean = sum(daily_pnls) / len(daily_pnls)
        variance = sum((value - mean) ** 2 for value in daily_pnls) / (len(daily_pnls) - 1)
        std = math.sqrt(variance) if variance > 0 else 0.0
        sharpe = mean / std * math.sqrt(252) if std > 0 else 0.0

    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for pnl in daily_pnls:
        cumulative += pnl
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)

    gross_win = sum(trade.pnl for trade in trades if trade.pnl > 0)
    gross_loss = abs(sum(trade.pnl for trade in trades if trade.pnl <= 0))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

    return {
        "n_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "total_cost": total_cost,
        "roi": round(total_pnl / total_cost * 100, 1) if total_cost > 0 else 0.0,
        "avg_pnl": avg_pnl,
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_drawdown, 2),
        "profit_factor": round(profit_factor, 2) if math.isfinite(profit_factor) else float("inf"),
    }


def log_strategy_results(log: LogFn, name: str, trades: list[BacktestTrade]) -> None:
    """Pretty-print a strategy summary."""
    if not trades:
        log(f"\n  {name}: No trades")
        return

    stats = compute_trade_stats(trades)
    log(
        f"\n  {name}: {stats['n_trades']} trades, "
        f"{stats['wins']}W/{stats['losses']}L ({stats['win_rate']:.1f}%)"
    )
    log(
        f"    PnL: ${stats['total_pnl']:+.2f} | Cost: ${stats['total_cost']:.2f} "
        f"| ROI: {stats['roi']:+.1f}%"
    )
    if stats["n_trades"] > 1:
        log(
            f"    Sharpe: {stats['sharpe']:.2f} | MDD: ${stats['max_drawdown']:.2f} "
            f"| PF: {stats['profit_factor']:.2f} | Avg: ${stats['avg_pnl']:.2f}"
        )
