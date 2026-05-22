"""
Alpha Suite — Synchronous HTTP helpers for Polymarket & Binance APIs.

Uses urllib only (no requests dependency). Handles retries, pagination,
SSL context, and graceful timeouts.
"""

import json
import math
import os
import ssl
import time
import http.client
import urllib.parse
import urllib.error
import urllib.request
from typing import Optional


# ── API Endpoints ──
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
BINANCE_API = "https://api.binance.com/api/v3"

# ── SSL / HTTP defaults ──
_SSL_CTX = ssl.create_default_context()
_USER_AGENT = "AlphaSuite/2.0"
_DEFAULT_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "application/json",
}


# ══════════════════════════════════════════════════════════════════════
# Core HTTP
# ══════════════════════════════════════════════════════════════════════

def http_get(url: str, timeout: int = 15, retries: int = 2) -> Optional[dict]:
    """Synchronous HTTP GET returning parsed JSON, or None on failure.

    Args:
        url: Full URL to fetch.
        timeout: Socket timeout in seconds.
        retries: Number of retry attempts after the first failure.

    Returns:
        Parsed JSON dict/list, or None if all attempts fail.
    """
    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=_DEFAULT_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
                raw = resp.read()
                return json.loads(raw)
        except (urllib.error.URLError, urllib.error.HTTPError,
                http.client.IncompleteRead, http.client.RemoteDisconnected,
                json.JSONDecodeError, OSError, TimeoutError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
    return None


# ══════════════════════════════════════════════════════════════════════
# Polymarket — Gamma API
# ══════════════════════════════════════════════════════════════════════

def fetch_gamma_markets(
    params: dict,
    max_pages: int = 20,
    start_offset: int = 0,
) -> list:
    """Paginated fetch of markets from the Gamma API.

    Args:
        params: Query parameters dict (e.g. {"active": "true", "closed": "false"}).
        max_pages: Maximum number of pages to fetch (100 markets per page).
        start_offset: Gamma offset to start from. Use this to continue a
            prior paginated scan without re-fetching earlier pages.

    Returns:
        List of raw market dicts from the API.
    """
    all_markets = []
    offset = max(0, int(start_offset or 0))
    limit = 100
    timeout = int(float(os.environ.get("GAMMA_HTTP_TIMEOUT", "15") or "15"))
    retries = int(os.environ.get("GAMMA_HTTP_RETRIES", "2") or "2")
    page_sleep = float(os.environ.get("GAMMA_PAGE_SLEEP_SEC", "0.3") or "0.3")

    for _ in range(max_pages):
        merged = {**params, "limit": str(limit), "offset": str(offset)}
        qs = urllib.parse.urlencode(merged)
        url = f"{GAMMA_API}/markets?{qs}"

        page = http_get(url, timeout=timeout, retries=retries)
        if not page or not isinstance(page, list):
            break

        all_markets.extend(page)

        if len(page) < limit:
            break

        offset += limit
        if page_sleep > 0:
            time.sleep(page_sleep)  # Rate-limit politeness

    return all_markets


# ══════════════════════════════════════════════════════════════════════
# Polymarket — CLOB Orderbook
# ══════════════════════════════════════════════════════════════════════

def fetch_market_resolution(condition_id: str) -> Optional[dict]:
    """Fetch a single market by conditionId via CLOB API for resolution checks.

    Returns a normalized dict:
        {
            "closed": bool,
            "yes_winner": bool,        # True if YES token resolved to 1
            "no_winner": bool,         # True if NO token resolved to 1
            "yes_price": float,        # 0 or 1 if closed, else last price
            "no_price": float,
            "yes_token": str,
            "no_token": str,
        }
    Or None if the market couldn't be fetched.

    Why CLOB and not Gamma: Gamma's `markets?condition_ids=X` returns an
    empty list for every cid (filter is broken), and `?conditionId=X` is
    silently ignored. CLOB's `/markets/{cid}` is the only endpoint that
    actually filters by conditionId — and it directly exposes each
    token's `winner: true/false` field, perfect for resolution.
    """
    url = f"{CLOB_API}/markets/{condition_id}"
    data = http_get(url, timeout=12, retries=2)
    if not isinstance(data, dict):
        return None
    tokens = data.get("tokens") or []
    # Polymarket condition_ids are always binary (2 tokens). Multi-outcome
    # events are modeled as multiple condition_ids grouped under an event
    # id — each condition has exactly 2 tokens. Refuse anything else so a
    # schema surprise can't silently slip the positional fallback below.
    if len(tokens) != 2:
        return None
    yes_tok = next(
        (t for t in tokens if str(t.get("outcome", "")).lower() == "yes"),
        None,
    )
    no_tok = next(
        (t for t in tokens if str(t.get("outcome", "")).lower() == "no"),
        None,
    )
    # Position-based fallback for non-YES/NO binary markets:
    #   - Crypto markets: Up / Down
    #   - Esports: <team1_name> / <team2_name>
    #   - Sports: <home> / <away>, Over / Under
    #   - Over/Under spread markets, Option markets, etc.
    # By Polymarket convention tokens[0] is the "affirmative"/YES-equivalent
    # side, tokens[1] is the "negative"/NO-equivalent side.
    # Our trades' side_label is written at place-time ("YES" if we bought
    # tokens[0], "NO" if tokens[1]), so this mapping stays correct.
    if not yes_tok or not no_tok:
        yes_tok = yes_tok or tokens[0]
        no_tok = no_tok or tokens[1]
    if not yes_tok or not no_tok:
        return None
    return {
        "closed": bool(data.get("closed")),
        "yes_winner": bool(yes_tok.get("winner", False)),
        "no_winner": bool(no_tok.get("winner", False)),
        "yes_price": float(yes_tok.get("price", 0) or 0),
        "no_price": float(no_tok.get("price", 0) or 0),
        "yes_token": str(yes_tok.get("token_id", "")),
        "no_token": str(no_tok.get("token_id", "")),
    }


def fetch_orderbook(token_id: str) -> dict:
    """Fetch CLOB orderbook for a token, normalized to canonical ordering.

    Polymarket's CLOB `/book` endpoint returns:
      - `asks` sorted **descending** by price (asks[0] = highest ask)
      - `bids` sorted **ascending** by price (bids[0] = lowest bid)

    These orderings are the OPPOSITE of what every walking algorithm expects
    ("walk asks from cheapest = best ask, walk bids from highest = best bid").
    This caused `vwap_ask` to systematically buy at the most expensive level
    in the book, making VWAP ≈ market max ask, and breaking every strategy
    that uses a `vwap > price * X` sanity check (especially longshots where
    quote=0.05-0.15 vs VWAP=0.99 → all signals rejected).

    Fix: normalize on the way out. Callers receive:
      - asks sorted **ascending** by price (asks[0] = best/lowest ask)
      - bids sorted **descending** by price (bids[0] = best/highest bid)

    Args:
        token_id: Polymarket CLOB token ID.

    Returns:
        Dict with normalized 'bids' and 'asks' lists. Empty lists on failure.
    """
    url = f"{CLOB_API}/book?token_id={token_id}"
    data = http_get(url)
    if not data or not isinstance(data, dict):
        return {"bids": [], "asks": []}

    raw_asks = data.get("asks") or []
    raw_bids = data.get("bids") or []

    def _safe_price(level):
        try:
            return float(level.get("price", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    return {
        "asks": sorted(raw_asks, key=_safe_price),                 # cheapest first
        "bids": sorted(raw_bids, key=_safe_price, reverse=True),   # highest first
    }


# ══════════════════════════════════════════════════════════════════════
# Orderbook Math (copied from math_engine.py for self-containment)
# ══════════════════════════════════════════════════════════════════════

def vwap_ask(asks: list, target_usd: float) -> tuple:
    """Compute VWAP from orderbook asks for a given target USD spend.

    Walks through ask levels, accumulating cost until target_usd is reached.

    Args:
        asks: List of dicts with 'price' and 'size' keys.
        target_usd: Total USD to spend.

    Returns:
        Tuple of (vwap_price, total_shares_available).
        Returns (0.0, 0.0) if orderbook is empty or invalid.
    """
    if not asks or target_usd <= 0:
        return 0.0, 0.0

    total_cost = 0.0
    total_shares = 0.0

    for a in asks:
        px = float(a.get("price", 0))
        sz = float(a.get("size", 0))
        if px <= 0 or sz <= 0:
            continue

        cost_this_level = px * sz
        if total_cost + cost_this_level >= target_usd:
            remaining = target_usd - total_cost
            shares_here = remaining / px
            total_shares += shares_here
            total_cost += remaining
            break

        total_cost += cost_this_level
        total_shares += sz

    if total_shares <= 0:
        return 0.0, 0.0

    return round(total_cost / total_shares, 4), round(total_shares, 2)


# ══════════════════════════════════════════════════════════════════════
# Binance
# ══════════════════════════════════════════════════════════════════════

def fetch_binance_price(symbol: str) -> float:
    """Fetch current price for a Binance trading pair.

    Args:
        symbol: Binance symbol (e.g. 'BTCUSDT', 'ETHUSDT').

    Returns:
        Current price as float, or 0.0 on failure.
    """
    url = f"{BINANCE_API}/ticker/price?symbol={symbol}"
    data = http_get(url, timeout=10, retries=1)
    if not data:
        return 0.0
    try:
        return float(data.get("price", 0))
    except (ValueError, TypeError):
        return 0.0


def fetch_binance_klines(
    symbol: str,
    interval: str = "1m",
    limit: int = 60,
    start_time_ms: int | None = None,
    end_time_ms: int | None = None,
) -> list:
    """Fetch Binance kline/candlestick data.

    Args:
        symbol: Binance symbol (e.g. 'BTCUSDT').
        interval: Kline interval (e.g. '1m', '5m', '1h').
        limit: Number of klines to fetch (max 1000).
        start_time_ms: Optional window start (Unix ms). When set, Binance
            returns klines whose open_time >= start_time_ms, which is the
            only way to anchor on a specific window's opening candle rather
            than the latest one. Caller should pass `window_start_sec * 1000`.
        end_time_ms: Optional window end (Unix ms). Paired with start_time_ms
            to clip the range — otherwise Binance returns up to `limit` bars
            from the start.

    Returns:
        List of kline arrays. Each kline:
        [open_time, open, high, low, close, volume, close_time, ...]
        Returns empty list on failure.
    """
    params = [f"symbol={symbol}", f"interval={interval}", f"limit={limit}"]
    if start_time_ms is not None:
        params.append(f"startTime={int(start_time_ms)}")
    if end_time_ms is not None:
        params.append(f"endTime={int(end_time_ms)}")
    url = f"{BINANCE_API}/klines?" + "&".join(params)
    data = http_get(url, timeout=10, retries=1)
    if not data or not isinstance(data, list):
        return []
    return data
