"""Typed models for the Alpha Suite backtest."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PricePoint:
    """Historical price observation for a token."""

    ts: int
    price: float


@dataclass(frozen=True)
class BinarySnapshot:
    """Historical YES/NO snapshot for a binary market."""

    yes: PricePoint
    no: PricePoint
    target_ts: int
    end_ts: int


@dataclass(frozen=True)
class OutcomeLeg:
    """Historical YES snapshot for one outcome in a complete-set event."""

    question: str
    token_id: str
    snapshot: PricePoint
    yes_winner: bool


@dataclass(frozen=True)
class CompleteSetSnapshot:
    """Synchronized historical YES-side snapshot for a complete-set event."""

    legs: tuple[OutcomeLeg, ...]
    target_ts: int
    end_ts: int


@dataclass
class BacktestTrade:
    """One simulated trade or complete-set position."""

    strategy: str
    label: str
    cost: float
    pnl: float
    won: bool
    trade_ts: int
    event_ts: int
    side: str = ""
    entry_price: float = 0.0
    edge: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
