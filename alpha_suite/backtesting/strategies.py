"""Strategy-specific backtest implementations."""

from __future__ import annotations

import os
import random
import re
import time
from typing import Callable

from alpha_suite.backtesting.calculations import (
    binary_trade_cost,
    binary_trade_pnl,
    complete_set_edge,
    complete_set_trade_cost,
    complete_set_trade_pnl,
    edge_to_bps,
)
from alpha_suite.backtesting.data import HistoricalDataClient, parse_json_list, winner_side
from alpha_suite.backtesting.models import BacktestTrade


LogFn = Callable[[str], None]

DEFAULT_API_KEY = ""  # Set via OPENAI_API_KEY env
DEFAULT_BASE_URL = "https://api.openai.com"


class BacktestLLMClient:
    """Thin wrapper around the historical LLM prompt used by the backtest."""

    def __init__(self) -> None:
        try:
            from openai import OpenAI
        except Exception:
            self.client = None
            return

        api_key = os.environ.get("OPENAI_API_KEY", DEFAULT_API_KEY)
        base_url = os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)
        try:
            self.client = OpenAI(api_key=api_key, base_url=base_url + "/v1")
        except Exception:
            self.client = None

    @property
    def is_available(self) -> bool:
        return self.client is not None

    def estimate_probability(self, question: str) -> float | None:
        """Return the LLM-estimated YES probability for a market question."""
        if not self.client:
            return None

        prompt = (
            "Estimate probability of YES for this prediction market. Think carefully.\n\n"
            f"Question: {question}\n\n"
            'Respond ONLY with JSON: {"probability": 0.XX, "confidence": "low/medium/high"}'
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-5.4",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )
            text = (response.choices[0].message.content or "").strip() if response.choices else ""
            match = re.search(r'"probability"\s*:\s*([\d.]+)', text)
            if not match:
                return None
            return max(0.01, min(0.99, float(match.group(1))))
        except Exception:
            return None


def run_bond_backtest(
    markets: list[dict],
    client: HistoricalDataClient,
    *,
    fee_rate: float,
    history_workers: int,
    log: LogFn,
) -> list[BacktestTrade]:
    """Backtest the bond buyer strategy."""
    log("\n" + "=" * 60)
    log("  BOND BUYER - Buy YES/NO at 90-95%, hold to settlement")
    log("=" * 60)

    candidate_markets: list[dict] = []
    token_ids: list[str] = []
    for market in markets:
        try:
            volume = float(market.get("volumeNum", market.get("volume", 0)) or 0)
        except (TypeError, ValueError):
            volume = 0.0
        if volume < 100:
            continue

        if not winner_side(market):
            continue

        tokens = [str(token) for token in parse_json_list(market.get("clobTokenIds", "[]"))]
        if len(tokens) < 2:
            continue
        candidate_markets.append(market)
        token_ids.extend(tokens[:2])

    client.prefetch_price_histories(token_ids, max_workers=history_workers)

    trades: list[BacktestTrade] = []
    for market in candidate_markets:
        settled = winner_side(market)
        if not settled:
            continue

        snapshot = client.binary_snapshot(market, hours_before_end=24, min_points=5)
        if snapshot is None:
            continue

        event_ts = snapshot.end_ts
        question = market.get("question", "?")[:60]

        for side, point in (("YES", snapshot.yes), ("NO", snapshot.no)):
            price = point.price
            if not (0.90 <= price <= 0.95):
                continue

            if price <= 0.91:
                p_fair = 0.97
            elif price <= 0.93:
                p_fair = 0.96
            else:
                p_fair = 0.95

            edge = p_fair - price
            if edge < 0.01:
                continue

            notional = 15.0
            won = side == settled
            trades.append(
                BacktestTrade(
                    strategy="bond",
                    label=question,
                    side=side,
                    entry_price=round(price, 3),
                    edge=round(edge, 4),
                    cost=binary_trade_cost(notional, fee_rate),
                    pnl=binary_trade_pnl(notional, price, won, fee_rate),
                    won=won,
                    trade_ts=point.ts,
                    event_ts=event_ts,
                    metadata={"notional": notional, "vol": volume},
                )
            )

    return trades


def run_multi_arb_backtest(
    events: list[dict],
    client: HistoricalDataClient,
    *,
    fee_rate: float,
    max_events: int,
    history_workers: int,
    log: LogFn,
) -> list[BacktestTrade]:
    """Backtest multi-outcome complete-set arbitrage."""
    log("\n" + "=" * 60)
    log("  ARB SCANNER - Multi-outcome complete-set arbitrage")
    log("=" * 60)

    trades: list[BacktestTrade] = []
    checked = 0
    eligible_events = [event for event in events if len(event.get("markets", [])) >= 3]
    prefetch_ids: list[str] = []
    for event in eligible_events[:max_events]:
        for market in event.get("markets", []):
            tokens = [str(token) for token in parse_json_list(market.get("clobTokenIds", "[]"))]
            if tokens:
                prefetch_ids.append(tokens[0])
    client.prefetch_price_histories(prefetch_ids, max_workers=history_workers)

    for event in eligible_events:
        markets = event.get("markets", [])
        if len(markets) < 3:
            continue

        checked += 1
        complete_set = client.complete_set(markets, hours_before_end=6, min_points=3)
        if not complete_set:
            if checked >= max_events:
                break
            continue

        price_sum = sum(leg.snapshot.price for leg in complete_set.legs)
        edge = complete_set_edge(price_sum, fee_rate)
        if edge is None or edge <= 0.01:
            if checked >= max_events:
                break
            continue

        notional = 10.0
        trade_ts = complete_set.target_ts
        event_ts = complete_set.end_ts
        trades.append(
            BacktestTrade(
                strategy="multi_arb",
                label=event.get("title", "?")[:60],
                entry_price=round(price_sum, 4),
                edge=round(edge, 4),
                cost=complete_set_trade_cost(notional, fee_rate),
                pnl=complete_set_trade_pnl(notional, price_sum, fee_rate),
                won=True,
                trade_ts=trade_ts,
                event_ts=event_ts,
                metadata={
                    "notional": notional,
                    "n_outcomes": len(complete_set.legs),
                    "sum": round(price_sum, 4),
                    "profit_bps": edge_to_bps(edge),
                },
            )
        )

        if checked % 20 == 0:
            log(f"  [{checked} events checked, {len(trades)} arbs found]")
        if checked >= max_events:
            break

    log(f"  Checked {checked} multi-outcome events")
    return trades


def run_llm_signal_backtest(
    markets: list[dict],
    client: HistoricalDataClient,
    *,
    fee_rate: float,
    llm_sample_size: int,
    include_llm: bool,
    history_workers: int,
    log: LogFn,
) -> list[BacktestTrade]:
    """Backtest the historical LLM signal strategy."""
    log("\n" + "=" * 60)
    log("  LLM SIGNAL - GPT-5.4 probability estimation")
    log("=" * 60)

    if not include_llm:
        log("  LLM disabled via flag")
        return []

    llm = BacktestLLMClient()
    if not llm.is_available:
        log("  LLM unavailable")
        return []

    skip_keywords = {"temperature", "°f", "°c", "bitcoin", "ethereum", "btc", "eth", "highest temp"}
    token_ids: list[str] = []
    prefiltered: list[dict] = []
    for market in markets:
        question = (market.get("question", "") or "").lower()
        if any(keyword in question for keyword in skip_keywords):
            continue

        try:
            volume = float(market.get("volumeNum", market.get("volume", 0)) or 0)
        except (TypeError, ValueError):
            volume = 0.0
        if volume < 5000:
            continue

        if not winner_side(market):
            continue

        tokens = [str(token) for token in parse_json_list(market.get("clobTokenIds", "[]"))]
        if len(tokens) < 2:
            continue
        prefiltered.append(market)
        token_ids.extend(tokens[:2])

    client.prefetch_price_histories(token_ids, max_workers=history_workers)

    candidates: list[tuple[dict, str, object]] = []
    for market in prefiltered:
        settled = winner_side(market)
        if not settled:
            continue

        snapshot = client.binary_snapshot(market, hours_before_end=48, min_points=5)
        if snapshot is None:
            continue
        if snapshot.yes.price < 0.10 or snapshot.yes.price > 0.90:
            continue

        candidates.append((market, settled, snapshot))

    log(f"  {len(candidates)} eligible markets")
    random.seed(42)
    sample = random.sample(candidates, min(llm_sample_size, len(candidates)))

    trades: list[BacktestTrade] = []
    for market, settled, snapshot in sample:
        question = market.get("question", "?")
        probability = llm.estimate_probability(question)
        if probability is None:
            log("    LLM error: no probability returned")
            continue

        yes_edge = probability - snapshot.yes.price
        no_edge = (1.0 - probability) - snapshot.no.price
        if max(yes_edge, no_edge) < 0.05:
            continue

        if yes_edge >= no_edge:
            side = "YES"
            point = snapshot.yes
            won = settled == "YES"
            edge = yes_edge
        else:
            side = "NO"
            point = snapshot.no
            won = settled == "NO"
            edge = no_edge

        notional = 10.0
        event_ts = snapshot.end_ts
        trades.append(
            BacktestTrade(
                strategy="llm_signal",
                label=question[:60],
                side=side,
                entry_price=round(point.price, 3),
                edge=round(edge, 3),
                cost=binary_trade_cost(notional, fee_rate),
                pnl=binary_trade_pnl(notional, point.price, won, fee_rate),
                won=won,
                trade_ts=point.ts,
                event_ts=event_ts,
                metadata={
                    "notional": notional,
                    "p_llm": round(probability, 3),
                    "yes_price": round(snapshot.yes.price, 3),
                    "no_price": round(snapshot.no.price, 3),
                },
            )
        )

        mark = "✓" if won else "✗"
        log(
            f"    {mark} {question[:45]} | "
            f"LLM={probability:.2f} yes={snapshot.yes.price:.2f} no={snapshot.no.price:.2f} side={side}"
        )
        time.sleep(0.3)

    return trades


def run_coverage_backtest(
    events: list[dict],
    client: HistoricalDataClient,
    *,
    fee_rate: float,
    max_weather_events: int,
    history_workers: int,
    log: LogFn,
) -> list[BacktestTrade]:
    """Backtest weather complete-set coverage arbitrage."""
    log("\n" + "=" * 60)
    log("  COVERAGE ARB - Weather full-bucket coverage")
    log("=" * 60)

    weather_events = [
        event
        for event in events
        if "temperature" in (event.get("title", "") or "").lower()
        or "highest-temperature" in (event.get("slug", "") or "").lower()
    ]
    log(f"  {len(weather_events)} weather events")
    prefetch_ids: list[str] = []
    for event in weather_events[:max_weather_events]:
        for market in event.get("markets", []):
            tokens = [str(token) for token in parse_json_list(market.get("clobTokenIds", "[]"))]
            if tokens:
                prefetch_ids.append(tokens[0])
    client.prefetch_price_histories(prefetch_ids, max_workers=history_workers)

    trades: list[BacktestTrade] = []
    for event in weather_events[:max_weather_events]:
        markets = event.get("markets", [])
        if len(markets) < 5:
            continue

        complete_set = client.complete_set(markets, hours_before_end=4, min_points=3)
        if not complete_set or len(complete_set.legs) < 5:
            continue

        price_sum = sum(leg.snapshot.price for leg in complete_set.legs)
        edge = complete_set_edge(price_sum, fee_rate)
        if edge is None or edge <= 0.01:
            continue

        notional = 10.0
        trade_ts = complete_set.target_ts
        event_ts = complete_set.end_ts
        trades.append(
            BacktestTrade(
                strategy="coverage",
                label=event.get("title", "?")[:60],
                entry_price=round(price_sum, 4),
                edge=round(edge, 4),
                cost=complete_set_trade_cost(notional, fee_rate),
                pnl=complete_set_trade_pnl(notional, price_sum, fee_rate),
                won=True,
                trade_ts=trade_ts,
                event_ts=event_ts,
                metadata={
                    "notional": notional,
                    "n_buckets": len(complete_set.legs),
                    "sum": round(price_sum, 4),
                    "profit_bps": edge_to_bps(edge),
                },
            )
        )

    return trades
