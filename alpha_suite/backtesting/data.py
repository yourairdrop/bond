"""Historical data access for the Alpha Suite backtest."""

from __future__ import annotations

import json
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Callable

from alpha_suite.utils.api import CLOB_API, GAMMA_API, http_get

from alpha_suite.backtesting.models import BinarySnapshot, CompleteSetSnapshot, OutcomeLeg, PricePoint


LogFn = Callable[[str], None]


def parse_json_list(raw) -> list:
    """Parse a Gamma list field that may already be a list or a JSON string."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def parse_price_list(raw) -> list[float]:
    """Parse a price array into floats."""
    prices = []
    for value in parse_json_list(raw):
        try:
            prices.append(float(value))
        except (TypeError, ValueError):
            continue
    return prices


def parse_iso_ts(value: str | None) -> int | None:
    """Parse an ISO-8601 timestamp into a UTC unix timestamp."""
    if not value:
        return None
    normalized = value.strip().replace("Z", "+00:00")
    if len(normalized) >= 3 and normalized[-3:] in {"+00", "-00"}:
        normalized = normalized + ":00"
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def choose_item_anchor_ts(item: dict, now: datetime | None = None) -> int | None:
    """Pick a stable resolution timestamp for window filtering."""
    if now is None:
        now = datetime.now(timezone.utc)

    upper_bound = int((now + timedelta(days=1)).timestamp())
    markets = item.get("markets")
    if isinstance(markets, list) and markets:
        anchors = [choose_item_anchor_ts(market, now=now) for market in markets]
        anchors = [anchor for anchor in anchors if anchor is not None]
        if anchors:
            return max(anchors)

    closed_ts = parse_iso_ts(item.get("closedTime"))
    if closed_ts is not None and closed_ts <= upper_bound:
        return closed_ts

    end_ts = parse_iso_ts(item.get("endDate"))
    if end_ts is not None and end_ts <= upper_bound:
        return end_ts
    return None


def item_in_window(item: dict, cutoff: datetime, now: datetime | None = None) -> bool:
    """Return True when a market/event anchor falls inside the backtest window."""
    if now is None:
        now = datetime.now(timezone.utc)

    anchor_ts = choose_item_anchor_ts(item, now=now)
    if anchor_ts is None:
        return False

    lower = int(cutoff.timestamp())
    upper = int((now + timedelta(days=1)).timestamp())
    return lower <= anchor_ts <= upper


def recommended_max_deviation_seconds(hours_before_end: int) -> int:
    """Return the maximum allowed distance from the intended lookback target."""
    return max(30 * 60, min(4 * 3600, hours_before_end * 3600 // 12))


def recommended_max_skew_seconds(hours_before_end: int) -> int:
    """Return the maximum allowed timestamp skew across synchronized legs."""
    deviation = recommended_max_deviation_seconds(hours_before_end)
    return max(10 * 60, min(30 * 60, deviation // 4))


def winner_side(market: dict) -> str | None:
    """Return the resolved side for a binary market."""
    prices = parse_price_list(market.get("outcomePrices", "[]"))
    if len(prices) >= 2:
        if prices[0] > 0.5 and prices[1] < 0.5:
            return "YES"
        if prices[1] > 0.5 and prices[0] < 0.5:
            return "NO"
    return None


def yes_winner(market: dict) -> bool | None:
    """Return whether the YES outcome won for a binary market."""
    prices = parse_price_list(market.get("outcomePrices", "[]"))
    if len(prices) >= 2:
        if prices[0] > 0.5 and prices[1] < 0.5:
            return True
        if prices[1] > 0.5 and prices[0] < 0.5:
            return False
    return None


class HistoricalDataClient:
    """HTTP-backed historical data client with simple in-memory caching."""

    def __init__(
        self,
        log: LogFn,
        market_sleep_seconds: float = 0.2,
        history_sleep_seconds: float = 0.05,
    ) -> None:
        self.log = log
        self.market_sleep_seconds = market_sleep_seconds
        self.history_sleep_seconds = history_sleep_seconds
        self._price_history_cache: dict[str, list[PricePoint]] = {}
        self._cache_lock = threading.Lock()

    def fetch_closed_markets(self, days_back: int, max_pages: int) -> list[dict]:
        """Fetch closed markets inside the backtest window."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)

        markets: list[dict] = []
        for page in range(max_pages):
            params = {
                "limit": 100,
                "offset": page * 100,
                "closed": "true",
                "order": "closedTime",
                "ascending": "false",
            }
            url = f"{GAMMA_API}/markets?{urllib.parse.urlencode(params)}"
            data = http_get(url)
            if not data or not isinstance(data, list):
                break

            filtered = [item for item in data if item_in_window(item, cutoff)]
            markets.extend(filtered)

            if page % 10 == 0:
                self.log(f"  Markets: {len(markets)} kept after filtering...")

            if len(data) < 100:
                break
            time.sleep(self.market_sleep_seconds)

        return markets

    def fetch_closed_events(self, days_back: int, max_pages: int) -> list[dict]:
        """Fetch closed events inside the backtest window."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)

        events: list[dict] = []
        for page in range(max_pages):
            params = {
                "limit": 100,
                "offset": page * 100,
                "closed": "true",
                "order": "closedTime",
                "ascending": "false",
            }
            url = f"{GAMMA_API}/events?{urllib.parse.urlencode(params)}"
            data = http_get(url)
            if not data or not isinstance(data, list):
                break

            filtered = [item for item in data if item_in_window(item, cutoff)]
            events.extend(filtered)

            if page % 10 == 0:
                self.log(f"  Events: {len(events)} kept after filtering...")

            if len(data) < 100:
                break
            time.sleep(self.market_sleep_seconds)

        return events

    def fetch_price_history(self, token_id: str) -> list[PricePoint]:
        """Fetch and cache minute-level CLOB history for a token."""
        with self._cache_lock:
            cached = self._price_history_cache.get(token_id)
        if cached is not None:
            return cached

        data = http_get(f"{CLOB_API}/prices-history?market={token_id}&interval=max&fidelity=60")
        history: list[PricePoint] = []
        if isinstance(data, dict):
            for item in data.get("history", []):
                try:
                    ts = int(item.get("t"))
                    price = float(item.get("p"))
                except (TypeError, ValueError):
                    continue
                if 0.0 <= price <= 1.0:
                    history.append(PricePoint(ts=ts, price=price))

        history.sort(key=lambda point: point.ts)
        with self._cache_lock:
            self._price_history_cache[token_id] = history
        if history:
            time.sleep(self.history_sleep_seconds)
        return history

    def prefetch_price_histories(
        self,
        token_ids: list[str],
        max_workers: int = 8,
        batch_size: int = 512,
    ) -> None:
        """Warm the in-memory history cache with bounded parallelism."""
        unique_ids = []
        seen: set[str] = set()
        with self._cache_lock:
            cached_ids = set(self._price_history_cache)
        for token_id in token_ids:
            if not token_id or token_id in seen or token_id in cached_ids:
                continue
            seen.add(token_id)
            unique_ids.append(token_id)

        if not unique_ids:
            return

        workers = max(1, min(max_workers, len(unique_ids)))
        self.log(f"  Prefetching {len(unique_ids)} histories with {workers} workers...")
        completed = 0
        for start in range(0, len(unique_ids), batch_size):
            batch = unique_ids[start:start + batch_size]
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(self.fetch_price_history, token_id) for token_id in batch]
                for future in as_completed(futures):
                    future.result()
                    completed += 1
            self.log(f"  Prefetch progress: {completed}/{len(unique_ids)} histories")

    @staticmethod
    def select_price_point(
        history: list[PricePoint],
        *,
        target_ts: int,
        min_points: int,
        max_deviation_seconds: int,
    ) -> PricePoint | None:
        """Pick the historical point closest to the target lookback."""
        if len(history) < min_points:
            return None

        point = min(history, key=lambda candidate: abs(candidate.ts - target_ts))
        if abs(point.ts - target_ts) > max_deviation_seconds:
            return None
        return point

    def binary_snapshot(
        self,
        market: dict,
        *,
        hours_before_end: int,
        min_points: int,
        max_deviation_seconds: int | None = None,
        max_skew_seconds: int | None = None,
    ) -> BinarySnapshot | None:
        """Return a historical YES/NO snapshot for a binary market."""
        tokens = [str(token) for token in parse_json_list(market.get("clobTokenIds", "[]"))]
        if len(tokens) < 2:
            return None

        yes_history = self.fetch_price_history(tokens[0])
        no_history = self.fetch_price_history(tokens[1])
        if not yes_history or not no_history:
            return None

        max_deviation = max_deviation_seconds or recommended_max_deviation_seconds(hours_before_end)
        max_skew = max_skew_seconds or recommended_max_skew_seconds(hours_before_end)
        end_ts = choose_item_anchor_ts(market) or min(yes_history[-1].ts, no_history[-1].ts)
        target_ts = end_ts - hours_before_end * 3600
        yes_point = self.select_price_point(
            yes_history,
            target_ts=target_ts,
            min_points=min_points,
            max_deviation_seconds=max_deviation,
        )
        no_point = self.select_price_point(
            no_history,
            target_ts=target_ts,
            min_points=min_points,
            max_deviation_seconds=max_deviation,
        )
        if yes_point is None or no_point is None:
            return None
        if abs(yes_point.ts - no_point.ts) > max_skew:
            return None

        return BinarySnapshot(yes=yes_point, no=no_point, target_ts=target_ts, end_ts=end_ts)

    def complete_set(
        self,
        markets: list[dict],
        *,
        hours_before_end: int,
        min_points: int,
        max_deviation_seconds: int | None = None,
        max_skew_seconds: int | None = None,
    ) -> CompleteSetSnapshot | None:
        """Return the full YES-side outcome set for an event.

        Incomplete sets are rejected outright to avoid false arbitrage signals.
        """
        if not markets:
            return None

        max_deviation = max_deviation_seconds or recommended_max_deviation_seconds(hours_before_end)
        max_skew = max_skew_seconds or recommended_max_skew_seconds(hours_before_end)
        histories: list[tuple[dict, list[PricePoint], bool, int]] = []

        for market in markets:
            tokens = [str(token) for token in parse_json_list(market.get("clobTokenIds", "[]"))]
            winner = yes_winner(market)
            if not tokens or winner is None:
                return None

            history = self.fetch_price_history(tokens[0])
            if not history:
                return None

            anchor_ts = choose_item_anchor_ts(market) or history[-1].ts
            histories.append((market, history, winner, anchor_ts))

        end_ts = min(anchor_ts for _, _, _, anchor_ts in histories)
        target_ts = end_ts - hours_before_end * 3600
        legs: list[OutcomeLeg] = []
        timestamps: list[int] = []
        for market, history, winner, _anchor_ts in histories:
            tokens = [str(token) for token in parse_json_list(market.get("clobTokenIds", "[]"))]
            point = self.select_price_point(
                history,
                target_ts=target_ts,
                min_points=min_points,
                max_deviation_seconds=max_deviation,
            )
            if point is None:
                return None
            timestamps.append(point.ts)

            legs.append(
                OutcomeLeg(
                    question=market.get("question", "?"),
                    token_id=tokens[0],
                    snapshot=point,
                    yes_winner=winner,
                )
            )

        if not legs or len(legs) != len(markets):
            return None
        if sum(1 for leg in legs if leg.yes_winner) != 1:
            return None
        if max(timestamps) - min(timestamps) > max_skew:
            return None
        return CompleteSetSnapshot(legs=tuple(legs), target_ts=target_ts, end_ts=end_ts)
