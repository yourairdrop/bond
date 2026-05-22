"""
Alpha Suite -- BondBuyer Strategy.

Scans for high-probability YES tokens (90-95%) expiring within 14 days.
Uses price-graded Kelly sizing and relaxed thresholds for more opportunities.

Ported from alpha_bot.py bond scanner with relaxed thresholds.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone, timedelta
from typing import List

from alpha_suite.base import Strategy, Signal, Trade, Result
from alpha_suite.position_manager import SellHighConfig, CancelStaleConfig
from alpha_suite.state import TERMINAL_STATUSES
from alpha_suite.utils.api import (
    http_get,
    fetch_gamma_markets,
    fetch_orderbook,
    vwap_ask,
)
from alpha_suite.utils.risk import kelly_size, ev_calc

# ── Configuration (Relaxed thresholds) ──
BOND_EXPIRY_DAYS = float(os.environ.get("BOND_EXPIRY_DAYS", "14"))  # Markets expiring within N days
BOND_GAMMA_MAX_PAGES = int(os.environ.get("BOND_GAMMA_MAX_PAGES", "40"))
BOND_DEEP_SCAN_MAX_PAGES = int(os.environ.get("BOND_DEEP_SCAN_MAX_PAGES", str(BOND_GAMMA_MAX_PAGES)))
BOND_MIN_PRICE = 0.90           # Minimum YES price (90%)
BOND_MAX_PRICE = 0.95           # Maximum YES price (95%)
WEATHER_BOND_MIN_HOURS = 6.0    # weather5000: 3h loses, 6-12h wins after friction
WEATHER_BOND_MAX_HOURS = 12.0
WEATHER_BOND_MAX_PRICE = 0.95   # 60d city replay: 0.90-0.95 profitable at 6-12h
WEATHER_NO_MIN_EDGE = 0.02      # NO legs cause most weather losses; require extra edge
WEATHER_GAMMA_LOOKBACK_HOURS = 8
WEATHER_LOCAL_DAY_MIN_HOURS = float(os.environ.get("WEATHER_LOCAL_DAY_MIN_HOURS", "6.0"))
WEATHER_LOCAL_DAY_MAX_HOURS = float(os.environ.get("WEATHER_LOCAL_DAY_MAX_HOURS", "30.0"))
# Entry timing window (Bug-25 fix, 2026-04-28). Empirical data + backtest:
#   live data:  0-6h bin   354 trades 92.8% WR +$X
#               6-12h bin  146 trades 76.9% WR -$X  ← leak
#   backtest:   weather 12h before  +8.3% ROI 100% WR
#               weather 24h before  -2.4% ROI  90% WR
# Capping at 12h excludes only future entries (0 historical trades at
# >12h since natural bond price 0.90-0.95 only happens close to expiry).
# Acts as safety net + formalizes the empirical sweet spot. Bug-24
# (catastrophe_mid 0.30→0.10) should recover the 6-12h band; if after
# 1 week of monitoring it's still negative, tighten to 6h.
BOND_MAX_HOURS_TO_END = 12
MIN_VOLUME = float(os.environ.get("BOND_MIN_VOLUME", "100"))  # Relaxed: was 500
MIN_EDGE = float(os.environ.get("BOND_MIN_EDGE", "0.01"))      # Relaxed: was 3%
# After-fee EV floor per $X risked. Edge alone is not enough in the 90c+ bond
# zone: a 1-2c misestimate can turn a small-looking edge into negative EV.
MIN_EV = float(os.environ.get("BOND_MIN_EV", "0.01"))
MIN_LIQUIDITY = 5               # Relaxed: 5 shares minimum liquidity
POLY_FEE = 0.02
KELLY_FRACTION = 0.25
VWAP_SIZE = 5.0                 # VWAP verification size

# Polymarket minimum order = 5 shares. At bond zone max 0.95, that's $X notional.
# Round up to $X by default so we always clear the floor with buffer for
# rounding. Live can override this via env when position sizing is scaled.
MIN_NOTIONAL = float(os.environ.get("BOND_MIN_NOTIONAL", "6.0"))
FIXED_BET_SIZE = float(os.environ.get("BOND_FIXED_BET_SIZE", "0.0"))
FIXED_BET_SHARES = float(os.environ.get("BOND_FIXED_BET_SHARES", "0.0"))

# Category blacklist (2026-04-24): independent audit vs 04-19 snapshot
# confirmed esports + sports-prop (spread/exact-score/O-U/handicap) have
# ALWAYS been net-negative for bond, in both profitable-era ($X bet,
# esports -$X / sports-prop -$57) and current-era ($X bet, esports
# -$X / sports-prop -$X). Filtering both categories counterfactually
# adds +$X to the 04-19 sample and +$X to the current sample. Sole fix
# that improves BOTH samples; all time-/price-based filters fail on one.
# Keeping the constants here (not in scan()) so they're trivial to audit
# and toggle without editing scan logic.
BOND_BLACKLIST_KEYWORDS = (
    # TODO: Populate based on your own per-category audit. These keywords
    # are matched against market question text to skip systematically
    # net-negative categories before any other filter. See
    # METHODOLOGY.md "Category Audit" section.
    # Examples:
    #   "spread:", "exact score:", "handicap",          # sports-prop
    #   "counter-strike", " dota", "league of legends", # esports
)
    "spread:", "exact score:", "handicap", "o/u", "over/under",
    # Esports tournaments — short-horizon high-variance
    "counter-strike", " cs2", " dota", "league of legends", " lol ",
    " lol:", "valorant", "honor of kings", "mobile legends",
)

BOND_CATEGORY_KEYWORDS = {
    "weather": (
        "temperature", "highest temperature", "low temperature", "rain",
        "snow", "weather", "heat", "cold", "degree", "°c", "°f",
    ),
    "esports": (
        "counter-strike", " cs2", " dota", "league of legends", " lol ",
        " lol:", "valorant", "honor of kings", "mobile legends",
        "starcraft", "overwatch", "rocket league", "esports",
    ),
    "sports": (
        "soccer", "champions league", "premier league",
        "will club ", " fc win", " sk win", " atlético win",
    ),
}

SPORTS_CATEGORY_RE = re.compile(
    r"\b(nba|nfl|nhl|mlb|ufc|wnba|ncaa|fifa|uefa|afc|mls)\b|"
    r"\bwill\s+.+\s+(?:fc|cf|sc|club|united|city|rangers|athletic|"
    r"atl[eé]tico|bk|sk)\s+win\s+on\s+\d{4}-\d{2}-\d{2}\b|"
    r"\b(?:vs\.?|versus)\b.*\b(?:win|draw|o/u|over/under)\b|"
    r"\bend\s+in\s+a\s+draw\b",
    re.IGNORECASE,
)

STOCK_PRICE_RE = re.compile(
    r"\([A-Z]{1,6}\).*(?:up or down|close above|close below)|"
    r"(?:up or down|close above|close below).*\([A-Z]{1,6}\)",
    re.IGNORECASE,
)

CRYPTO_SINGLE_RE = re.compile(
    r"\b(?:bitcoin|ethereum|solana|xrp|dogecoin|litecoin|btc|eth|sol)\b"
    r".*(?:up or down|dip to|\babove\b|\bbelow\b)",
    re.IGNORECASE,
)

CRYPTO_RANGE_RE = re.compile(
    r"\b(?:bitcoin|ethereum|solana|xrp|dogecoin|litecoin|btc|eth|sol)\b.*"
    r"(?:between|reach|range)",
    re.IGNORECASE,
)

COUNT_TEXT_BUCKET_RE = re.compile(
    r"\b(?:subscribers?|users?|posts?|tweets?|followers?|downloads?|views?|"
    r"mentions?|dissent|say\s+[\"']|publicly insult|monthly active users?)\b",
    re.IGNORECASE,
)

CATEGORY_BASE_P_FAIR = {
    # TODO: Calibrate from your own historical analysis. These are admission-
    # control priors used to compute edge = p_fair - market_price. Higher
    # values mean you trust the category more (will buy at thinner edges).
    # See METHODOLOGY.md "Per-Category p_fair Calibration" section.
    # Conservative starting values:
    "stocks_price": 0.90,
    "crypto_single": 0.85,
    "sports": 0.85,
    "weather": 0.90,
    "esports": 0.50,
    "crypto_range": 0.50,
    "count_or_text_bucket": 0.50,
    "other": 0.50,
}

# Conservative city shrinkage from the same 60d replay. These are negative
# adjustments only; good-looking cities get no boost until the live sample is
# large enough. Small-sample weak cities are softened instead of hard-blocked.
WEATHER_CITY_P_FAIR_ADJUSTMENTS = {
    # TODO: Populate from your own city-level analysis.
    # Each entry: city_name_lowercased: p_fair_adjustment (negative shrinks).
    # See METHODOLOGY.md "Per-Category p_fair Calibration" section.
    # Example: "denver": -0.03,  # if Denver weather buckets historically over-predict
}

WEATHER_TEMP_RE = re.compile(
    r"\bhighest\s+temperature\s+in\s+(.+?)\s+be\b",
    re.IGNORECASE,
)

# Diagnostic clock only. A 2026-05-01 live check showed `closedTime` can be
# per-bucket auto-resolution time while sibling buckets remain active and
# accepting orders, so these medians must not be used as the primary live
# trade gate. Live entry uses Gamma `gameStartTime` below.
WEATHER_CITY_CLOSE_CST = {
    # TODO: City-specific market close times in your local timezone.
    # Polymarket weather markets have inconsistent endDate vs actual
    # closedTime per city; use the median of historical closedTime per
    # city as the strategy clock. Format: "city": "HH:MM".
    # Example: "amsterdam": "20:21",
}

MONTH_NAME_TO_NUM = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

LOCAL_TZ = timezone(timedelta(hours=8))


def _excluded_categories() -> set[str]:
    raw = os.environ.get("BOND_EXCLUDE_CATEGORIES", "")
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


def _log_excluded_markets() -> bool:
    return os.environ.get("BOND_LOG_EXCLUDED", "").strip().lower() in {"1", "true", "yes"}


def classify_bond_question(question: str) -> str:
    """Coarse bond category used for risk filtering and p_fair priors."""
    q = f" {(question or '').lower()} "
    for category in ("weather", "esports"):
        keywords = BOND_CATEGORY_KEYWORDS[category]
        if any(kw in q for kw in keywords):
            return category
    if COUNT_TEXT_BUCKET_RE.search(question or ""):
        return "count_or_text_bucket"
    if CRYPTO_RANGE_RE.search(question or ""):
        return "crypto_range"
    if CRYPTO_SINGLE_RE.search(question or ""):
        return "crypto_single"
    if STOCK_PRICE_RE.search(question or ""):
        return "stocks_price"
    if SPORTS_CATEGORY_RE.search(question or ""):
        return "sports"
    return "other"


def weather_city_adjustment(question: str) -> float:
    """Return conservative p_fair city adjustment for temperature markets."""
    city = weather_city_key(question)
    return WEATHER_CITY_P_FAIR_ADJUSTMENTS.get(city, 0.0)


def is_weather_temperature_market(question: str) -> bool:
    """True for the city temperature markets covered by the 60d replay."""
    return bool(WEATHER_TEMP_RE.search(question or ""))


def is_weather_range_bucket(question: str) -> bool:
    """True for temperature range buckets like 'between 76-77°F'.

    The 60d weather replay that justified `CATEGORY_BASE_P_FAIR["weather"]`
    was for city temperature markets, but the worst recent live loss came from
    a narrow range bucket. Those buckets behave like sibling strikes on a
    continuous variable and should not inherit the same 0.95-0.96 prior.
    """
    q = question or ""
    return bool(re.search(
        r"\bbetween\s+\d+(?:[\.,]\d+)?\s*[-–to]+\s*\d+(?:[\.,]\d+)?\s*°?[cf]\b",
        q,
        re.IGNORECASE,
    ))


def weather_city_key(question: str) -> str:
    """Normalize city from a temperature-market question."""
    match = WEATHER_TEMP_RE.search(question or "")
    if not match:
        return ""
    return re.sub(r"[^a-z0-9 ]+", "", match.group(1).lower()).strip()


def _parse_question_date_cst(question: str, fallback: datetime | None = None) -> datetime.date | None:
    """Parse the target date in a weather question as a CST date."""
    match = re.search(
        r"\bon\s+([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?",
        question or "",
        re.IGNORECASE,
    )
    if not match:
        return fallback.astimezone(LOCAL_TZ).date() if fallback else None
    month = MONTH_NAME_TO_NUM.get(match.group(1).lower())
    if not month:
        return fallback.astimezone(LOCAL_TZ).date() if fallback else None
    day = int(match.group(2))
    if match.group(3):
        year = int(match.group(3))
    elif fallback:
        year = fallback.astimezone(LOCAL_TZ).year
    else:
        year = datetime.now(LOCAL_TZ).year
    try:
        return datetime(year, month, day, tzinfo=LOCAL_TZ).date()
    except ValueError:
        return fallback.astimezone(LOCAL_TZ).date() if fallback else None


def expected_weather_close_dt(question: str, fallback_end_dt: datetime | None = None) -> datetime | None:
    """Expected temperature-market close time in UTC.

    Gamma's endDate is currently a uniform daily cutoff, not city-specific.
    For known cities, combine the question's target date with the observed
    city median closedTime in CST. Unknown cities fall back to Gamma endDate.
    """
    city = weather_city_key(question)
    close_hhmm = WEATHER_CITY_CLOSE_CST.get(city)
    if not close_hhmm:
        return fallback_end_dt.astimezone(timezone.utc) if fallback_end_dt else None
    target_date = _parse_question_date_cst(question, fallback_end_dt)
    if not target_date:
        return fallback_end_dt.astimezone(timezone.utc) if fallback_end_dt else None
    hour, minute = (int(x) for x in close_hhmm.split(":", 1))
    return datetime(
        target_date.year, target_date.month, target_date.day,
        hour, minute, tzinfo=LOCAL_TZ,
    ).astimezone(timezone.utc)


def _parse_dt(value: object) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    return default


def weather_entry_clock(
    question: str,
    now: datetime,
    fallback_end_dt: datetime | None = None,
    game_start_dt: datetime | None = None,
) -> tuple[float | None, datetime | None, str]:
    """Return the live weather timing metric.

    For live markets, Gamma's `gameStartTime` is the start of the city's local
    weather date in UTC. That is a better live anchor than historical
    `closedTime` medians, which can reflect partial bucket auto-resolution.
    The returned value is hours since the local weather day started.
    """
    if game_start_dt is not None:
        return (now - game_start_dt).total_seconds() / 3600, game_start_dt, "game_start_age"
    return weather_hours_to_close(question, now, fallback_end_dt)


def weather_hours_to_close(
    question: str,
    now: datetime,
    fallback_end_dt: datetime | None = None,
) -> tuple[float | None, datetime | None, str]:
    """Return hours until the city-specific weather clock closes."""
    expected = expected_weather_close_dt(question, fallback_end_dt)
    if not expected:
        return None, None, "unknown"
    source = "city_median" if WEATHER_CITY_CLOSE_CST.get(weather_city_key(question)) else "gamma_end"
    return (expected - now).total_seconds() / 3600, expected, source


def compute_bond_p_fair(question: str, market_price: float) -> float:
    """Estimate conservative fair probability from market type.

    Bug-37: the old model assigned 0.95-0.97 purely from price. That made
    weather/KPI/esports/count buckets look like positive-edge bonds even when
    live results showed their realized win rate was far below breakeven.
    """
    category = classify_bond_question(question)
    base = CATEGORY_BASE_P_FAIR.get(category, CATEGORY_BASE_P_FAIR["other"])
    if category == "weather":
        if not is_weather_temperature_market(question):
            return 0.82
        if is_weather_range_bucket(question):
            return 0.82
        return round(max(0.01, min(0.99, base + weather_city_adjustment(question))), 4)
    # A 94c market is somewhat stronger evidence than a 90c market, but the
    # boost is small so weak categories cannot become tradable from price alone.
    anchor_boost = max(0.0, (float(market_price or 0.0) - BOND_MIN_PRICE) * 0.5)
    return round(min(0.99, base + anchor_boost), 4)


def _parse_risk_epoch(value: str) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _trade_open_after_epoch(t: dict, epoch: datetime | None) -> bool:
    if not epoch:
        return True
    raw = t.get("placed_at") or t.get("time") or ""
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc) >= epoch

# Multi-strike range bucket regex (Bug-12 fix, 2026-04-27).
# Markets with patterns like "60-79 posts" / "100-119 tweets" / "47.5 kills"
# are sibling buckets — exactly ONE bucket must resolve YES, so any NO bond
# bought across multiple buckets has guaranteed loss. Same root cause as the
# weather temperature multi-bucket trap. Verified ALL 8 multi-strike-post
# trades in 5 days of data: 1W/6L = 14% WR, net -$X — worst per-attempt
# loss rate of any category. These slip past the keyword blacklist because
# "post" / "tweet" alone are too broad (would also catch legitimate political
# markets). Regex catches the specific X-Y range structure.
_MULTI_STRIKE_RE = re.compile(
    r'\b\d+-\d+\b\s+(?:posts?|tweets?|kills?|goals?|points?|runs?|sets?)',
    re.IGNORECASE,
)


def _is_multi_strike(question: str) -> bool:
    """True if market is a sibling-bucket range bet on a continuous count."""
    return bool(_MULTI_STRIKE_RE.search(question or ""))


def _event_date_key(question: str) -> str:
    """Return a normalized date suffix for event-level weather dedup."""
    q = question or ""
    month_match = re.search(
        r"\bon\s+([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?",
        q,
        re.IGNORECASE,
    )
    if month_match:
        month = MONTH_NAME_TO_NUM.get(month_match.group(1).lower())
        if month:
            day = int(month_match.group(2))
            year = month_match.group(3)
            if year:
                return f"on-{int(year):04d}-{month:02d}-{day:02d}"
            return f"on-{month:02d}-{day:02d}"

    iso_match = re.search(r"\bon\s+(\d{4})-(\d{2})-(\d{2})\b", q, re.IGNORECASE)
    if iso_match:
        return f"on-{iso_match.group(1)}-{iso_match.group(2)}-{iso_match.group(3)}"
    return ""


def derive_event_key(question: str) -> str:
    """Derive a stable per-event key from question text.

    Sibling-bucket markets in the same event collapse to the same key.
    Used by v2's _has_open_on_event dedup. Why synthetic instead of
    Gamma API's events[0].slug?
      - API slug is null for closed markets (so backfill needs a fallback)
      - Backfilled (synthetic) and live (API) keys would have different
        formats → mismatch → dedup fails
      - Synthetic from question is consistent and works across closed/active
      - Same normalization for any source = no mismatch

    Examples:
     "Will the highest temperature in Lagos be 35°C on April 24?" →
       "highest-temperature-in-lagos-be-on-04-24"
     "Will the highest temperature in Lagos be 32°C on April 26?" →
       "highest-temperature-in-lagos-be-on-04-26"
     "Will Zelenskyy post 60-79 posts from April 17 to April 24, 2026?" →
       "zelenskyy-post-posts"
    """
    if not question:
        return ""
    date_key = _event_date_key(question) if is_weather_temperature_market(question) else ""
    s = question.lower().strip()
    s = re.sub(r'^will\s+(the\s+)?', '', s)
    s = s.rstrip('?').strip()
    # Strip temp ranges + units
    s = re.sub(r'\bbe\s+(?:between\s+)?\d+(?:[\.,]\d+)?\s*[-–to]+\s*\d+(?:[\.,]\d+)?\s*°?[cf]?\b',
               'be', s)
    s = re.sub(r'\bbe\s+\d+(?:[\.,]\d+)?\s*°?[cf](?:\s+or\s+(?:higher|lower))?\b',
               'be', s)
    s = re.sub(r'\b\d+(?:[\.,]\d+)?\s*°[cf]\b', '', s)
    # Strip multi-strike ranges
    s = re.sub(r'\b\d+-\d+\s+(posts?|tweets?|kills?|goals?)\b', r'\1', s)
    # Strip date suffixes
    s = re.sub(r'\bon\s+\w+\s+\d+(,?\s*\d{4})?', '', s)
    s = re.sub(r'\bfrom\s+\w+\s+\d+\s+to\s+\w+\s+\d+(,?\s*\d{4})?', '', s)
    s = re.sub(r'\s+', '-', s.strip())
    s = re.sub(r'-+', '-', s).strip('-')
    if date_key:
        s = f"{s}-{date_key}"
    return s[:80]

# Stale-scan guard (2026-04-23). scan() reads Gamma's cached outcomePrices
# which can lag the live CLOB book by minutes. If the live VWAP is
# significantly below the bond zone floor, the Gamma-quoted price is
# stale — this market isn't a bond anymore, it's a collapsed longshot.
# Without this guard, evaluate() sees "p_fair - VWAP = huge edge" and
# happily places an order, then fill-sim eats a near-certain loser.
# bond_pro has its own version (VWAP<0.60 rejects); this is bond-buyer's.
STALE_SCAN_FLOOR = 0.85         # Reject if live VWAP < this (5c below BOND_MIN)


class BondBuyer(Strategy):
    """High-probability YES token buyer (the 'bond' strategy)."""

    name = "bond-buyer"
    interval_sec = 300
    bankroll_pct = 0.25
    max_bet = 8.0           # was 30.0 @ $X bankroll; scaled to $X bankroll
    max_daily_loss = 40.0
    dry_run = True

    # Position lifecycle (PositionManager): place-and-hold to redemption.
    #
    # 2026-04-24 revision: the previous time-based ladder
    # (0.99→6h→0.97→12h→0.95→24h→0.93) ate 70% of the per-trade edge:
    # only 3/39 ladder exits hit ≥0.99; 67% exited at ≤0.94, burning
    # avg win from $X down to $X per $X bet. At 95% WR that pushed
    # breakeven from 92% to 97.2%, i.e. the strategy was structurally
    # losing even though WR looked fine.
    #
    # New behavior mirrors the 04-19 profitable baseline:
    #   - Place SELL at 0.99 the moment BUY fills.
    #   - No time-based downgrades — if 0.99 doesn't fill, wait for
    #     redemption (+$X/share = full bond alpha).
    #   - Emergency exit only if mid collapses to ≤0.60. Sports/weather
    #     bonds that fail typically snap from 0.9→0.001 at resolution,
    #     so an emergency threshold higher than 0.60 fires on WR-jitter
    #     noise without actually saving catastrophic losses.
    #   - cancel_stale hard_ttl_min (180m below) still cancels the TP
    #     limit if nothing fills; the held position then auto-redeems.
    #
    # Old polling-based `threshold`/`floor` ignored when resting_tp_enabled.
    sell_high_cfg = SellHighConfig(
        enabled=True, profile="bond",
        threshold=0.99, floor=0.97, offset=0.01,  # legacy (ignored)
        resting_tp_enabled=True,
        resting_tp_initial=0.99,
        resting_tp_ladder=(),                # no time-based downgrades
        resting_tp_emergency_mid=0.60,       # only fire on true collapse
        # Bug-27 fix (2026-04-28): catastrophe_mid 0.10 → 0.0 (DISABLED).
        # Counterfactual on all 11 historical catastrophe-locked trades
        # (queried via CLOB API for actual market resolution):
        #   Catastrophe ON:           -$X realized
        #   Hold-to-resolution:       -$X (would have been)
        #   Disabling saves:          +$X
        # Math: 5 wrong saves cost $X/trade each. 6 correct saves
        # only recovered $X/trade each. Net asymmetric — single wrong
        # save offsets 5 correct saves. At 60% accuracy this still loses.
        # Backtest +4.3% ROI assumes hold-to-resolution; matches reality.
        # Bug-24's 0.10 threshold was already de-facto disabled by
        # min_bid 0.20 dead-zone (mid<0.10 → bid<0.20 → can't sell).
        # Setting 0.0 makes the logic explicit and kills dead code paths.
        resting_tp_catastrophe_mid=0.0,
    )
    cancel_stale_cfg = CancelStaleConfig(
        enabled=True,
        hard_ttl_min=720,          # 12h — matches backtest "+6.2% ROI at TTL=12h"
                                    # (was 180 / 3h — too tight, killed filling window)
        cancel_on_crossed=True,
        drift_bps=300,
        out_of_range_ask=0.95,
        expiring_hours=1.0,
    )

    def __init__(self, state_manager, logger: logging.Logger):
        super().__init__(state_manager, logger)
        self._seen_bonds: set[str] = set()

    # ── Scan ──

    def scan(self) -> List[Signal]:
        """Fetch markets expiring within 14 days, YES price 90-95%."""
        self.log.info("-- [BOND] Scanning high-probability markets --")

        # Bug-20 (2026-04-27): _seen_bonds was meant to dedup WITHIN one
        # scan() invocation but never got cleared, accumulating cids over
        # the container's lifetime. After 5 days of scans, thousands of
        # markets were silently locked out, fill rate decayed, and bonds
        # whose orders cancelled and were re-eligible never came back.
        # Clear at scan() entry — cross-cycle dedup is enforced by the
        # `held_cids` check in evaluate (and by v2's same-cid filter).
        self._seen_bonds.clear()

        if not self.state.check_bankroll(self.name):
            self.log.info("  BOND bankroll exhausted")
            return []

        now = datetime.now(timezone.utc)
        params = {
            "active": "true",
            "closed": "false",
            # Weather temperature markets can stay active after Gamma's
            # uniform endDate while their city-specific closeTime is still
            # in the future. Look back a few hours and let per-market gates
            # below reject stale non-weather markets.
            "end_date_min": (now - timedelta(hours=WEATHER_GAMMA_LOOKBACK_HOURS)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_date_max": (now + timedelta(days=BOND_EXPIRY_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "order": "endDate",
            "ascending": "true",
        }
        markets = fetch_gamma_markets(params, max_pages=BOND_GAMMA_MAX_PAGES)
        deep_pages = max(BOND_GAMMA_MAX_PAGES, BOND_DEEP_SCAN_MAX_PAGES)
        if deep_pages > BOND_GAMMA_MAX_PAGES and len(markets) >= BOND_GAMMA_MAX_PAGES * 100:
            extra_pages = deep_pages - BOND_GAMMA_MAX_PAGES
            extra = fetch_gamma_markets(
                params,
                max_pages=extra_pages,
                start_offset=BOND_GAMMA_MAX_PAGES * 100,
            )
            markets.extend(extra)
            self.log.info(
                f"  Deep scan fetched {len(extra)} additional markets "
                f"(pages={BOND_GAMMA_MAX_PAGES + extra_pages})"
            )
        self.log.info(
            f"  Fetched {len(markets)} markets expiring within "
            f"{BOND_EXPIRY_DAYS}d (pages={deep_pages})"
        )

        signals: list[Signal] = []
        excluded_categories = _excluded_categories()
        excluded_counts: dict[str, int] = {}
        log_excluded_markets = _log_excluded_markets()

        for m in markets:
            parsed = self._parse_market(m)
            if not parsed:
                continue

            cid = parsed["condition_id"]
            if cid in self._seen_bonds:
                continue
            if parsed.get("volume", 0) < MIN_VOLUME:
                continue

            # Category blacklist: skip esports + sports-prop (always net-negative
            # in both 04-19 and current samples — see audit 2026-04-24).
            q_full = parsed.get("question") or ""
            q_lower = q_full.lower()
            if any(kw in q_lower for kw in BOND_BLACKLIST_KEYWORDS):
                continue
            category = classify_bond_question(q_full)
            if category in excluded_categories:
                excluded_counts[category] = excluded_counts.get(category, 0) + 1
                if log_excluded_markets:
                    self.log.info(
                        f"  Skip {q_full[:50]} -- category {category} excluded"
                    )
                continue

            # Multi-strike range buckets (Bug-12): "60-79 posts", "47.5 kills",
            # etc. are sibling-bucket traps where one bucket must resolve YES.
            if _is_multi_strike(q_full):
                continue

            # Expiry check
            end_str = parsed.get("end_date", "")
            if not end_str:
                continue
            try:
                end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                if end_dt.tzinfo is None:
                    end_dt = end_dt.replace(tzinfo=timezone.utc)
                end_dt = end_dt.astimezone(timezone.utc)
            except (ValueError, TypeError):
                continue

            # Weather re-entry gate (2026-04-29): only the city temperature
            # family has been re-tested cleanly. Bug-40 (2026-05-01):
            # do not use historical closedTime medians as live hard-close
            # times. Gamma can auto-close individual bucket markets while
            # sibling buckets still accept orders. For live timing, anchor on
            # `gameStartTime` (start of the city's local weather date) and only
            # consider markets that are currently active/open/accepting orders.
            weather_expected_close = None
            weather_clock_source = ""
            gamma_hours_left = (end_dt - now).total_seconds() / 3600
            if category == "weather":
                if not is_weather_temperature_market(q_full):
                    continue
                if is_weather_range_bucket(q_full):
                    continue
                if parsed.get("closed") or not parsed.get("active", True):
                    continue
                if not parsed.get("accepting_orders", True):
                    continue

                game_start_dt = _parse_dt(parsed.get("game_start_time"))
                weather_clock_value, weather_clock_dt, weather_clock_source = weather_entry_clock(
                    q_full, now, end_dt, game_start_dt
                )
                weather_expected_close = expected_weather_close_dt(q_full, end_dt)
                if weather_clock_value is None:
                    continue
                if weather_clock_source == "game_start_age":
                    if not (
                        WEATHER_LOCAL_DAY_MIN_HOURS
                        <= weather_clock_value
                        <= WEATHER_LOCAL_DAY_MAX_HOURS
                    ):
                        continue
                else:
                    if weather_clock_dt and weather_clock_dt < now + timedelta(hours=1):
                        continue
                    if not (
                        WEATHER_BOND_MIN_HOURS
                        <= weather_clock_value
                        <= WEATHER_BOND_MAX_HOURS
                    ):
                        continue
                hours_left = weather_clock_value
            else:
                if end_dt < now + timedelta(hours=1):
                    continue
                hours_left = gamma_hours_left
                # Bug-25 (2026-04-28): cap at BOND_MAX_HOURS_TO_END (12h).
                # See module-level config comment for rationale.
                if hours_left > BOND_MAX_HOURS_TO_END:
                    continue

            # Find best YES outcome in [90%, 95%]
            best_idx = -1
            best_price = 0.0
            for i, price in enumerate(parsed.get("prices", [])):
                max_price = WEATHER_BOND_MAX_PRICE if category == "weather" else BOND_MAX_PRICE
                if BOND_MIN_PRICE <= price <= max_price and price > best_price:
                    best_price = price
                    best_idx = i

            if best_idx < 0:
                continue

            p_fair = compute_bond_p_fair(q_full, best_price)

            edge = p_fair - best_price
            outcome_name = str(parsed["outcomes"][best_idx] or "").strip().lower()
            min_edge = MIN_EDGE
            if category == "weather" and outcome_name == "no":
                min_edge = max(min_edge, WEATHER_NO_MIN_EDGE)
            if edge < min_edge:
                continue

            ev = ev_calc(p_fair, best_price)
            if ev < MIN_EV:
                self.log.info(
                    f"  Skip {q_full[:50]} -- after-fee EV "
                    f"{ev:.4f} < {MIN_EV:.4f}"
                )
                continue

            token_id = parsed["tokens"][best_idx]
            self._seen_bonds.add(cid)

            signals.append(Signal(
                strategy=self.name,
                market_question=parsed.get("question", "?"),
                condition_id=cid,
                token_id=token_id,
                side="BUY",
                price=best_price,
                p_fair=round(p_fair, 3),
                edge=round(edge, 4),
                ev=round(ev, 4),
                metadata={
                    "outcome": parsed["outcomes"][best_idx],
                    "outcome_index": best_idx,
                    "hours_to_expiry": round(hours_left, 1),
                    "end_date": parsed.get("end_date", ""),
                    "gamma_hours_to_expiry": round(gamma_hours_left, 1),
                    "volume": parsed.get("volume", 0),
                    "neg_risk": parsed.get("neg_risk", False),
                    # Bug-13 fix (2026-04-27): event_slug propagated so v2
                    # can dedup across sibling buckets (same city/day temp,
                    # same event multi-line). Without this, v2's _has_open_on_cid
                    # only catches duplicate condition_ids — not sibling cids
                    # within the same event.
                    "event_slug": parsed.get("event_slug", ""),
                    "slug": parsed.get("slug", ""),
                    # Bug-36 (2026-04-28): per-market tick_size from Gamma.
                    # Polymarket migrated bond markets 0.01→0.001 tick;
                    # evaluate() uses this to size maker-undercut correctly.
                    # Fallback 0.01 for legacy/unmigrated markets.
                    "tick_size": float(parsed.get("orderPriceMinTickSize", 0.01) or 0.01),
                    "category": category,
                    "min_edge_required": min_edge,
                    "weather_city_adjustment": weather_city_adjustment(q_full) if category == "weather" else 0.0,
                    "weather_city": weather_city_key(q_full) if category == "weather" else "",
                    "weather_expected_close": weather_expected_close.isoformat() if weather_expected_close else "",
                    "weather_clock_source": weather_clock_source,
                    "weather_hours_to_close": (
                        round(hours_left, 1)
                        if category == "weather" and weather_clock_source != "game_start_age"
                        else None
                    ),
                    "weather_local_day_age": (
                        round(hours_left, 1)
                        if category == "weather" and weather_clock_source == "game_start_age"
                        else None
                    ),
                    "weather_game_start": parsed.get("game_start_time", "") if category == "weather" else "",
                    "accepting_orders": parsed.get("accepting_orders", True),
                },
            ))
            time_label = (
                "wday"
                if category == "weather" and weather_clock_source == "game_start_age"
                else ("wclose" if category == "weather" else "exp")
            )
            self.log.info(
                f"  [BOND] {parsed['outcomes'][best_idx]}={best_price:.3f} "
                f"ev={ev:.4f} edge={edge:.3f} {time_label}={hours_left:.0f}h "
                f"| {parsed['question'][:50]}"
            )

        if excluded_counts:
            summary = ", ".join(
                f"{category}={count}" for category, count in sorted(excluded_counts.items())
            )
            self.log.info(f"  [BOND] Excluded categories: {summary}")
        self.log.info(f"  [BOND] Done: {len(signals)} signals")
        return signals

    # ── Evaluate ──

    def evaluate(self, signals: List[Signal]) -> List[Trade]:
        """VWAP-verify and Kelly-size each bond opportunity."""
        trades: list[Trade] = []
        bankroll = self.state.get_bankroll(self.name, default=5000.0 * self.bankroll_pct)

        # Per-trade cap enforcement (bond_pro pattern, 2026-04-22).
        # scan() already checks check_bankroll at start, but if 20+ bonds
        # qualify in one cycle, evaluate can still add 20 × $X = $160+ past
        # the existing open_cost — each trade appended without re-checking.
        strat_state = self.state.state.get("strategies", {}).get(self.name, {})
        cap = float(strat_state.get("max_open_cost", bankroll))
        risk_epoch = _parse_risk_epoch(strat_state.get("risk_epoch", ""))
        current_open = sum(
            float(t.get("size") or 0)
            for t in strat_state.get("trades", [])
            if t.get("outcome") is None
            and t.get("status") not in TERMINAL_STATUSES
            and _trade_open_after_epoch(t, risk_epoch)
        )
        if current_open >= cap:
            self.log.info(f"  [BOND] open ${current_open:.2f} ≥ cap ${cap:.0f} — skip")
            return []

        # Bug-32 (2026-04-28): cross-cycle dedup. Bug-20's comment promised
        # held_cids would be checked here, but the check was never written.
        # Without it, plain bond-buyer would scan the same market every 5min
        # and add a duplicate trade record — corrupting dry-run aggregate
        # stats used to validate v2's expected EV. v2 has its own _has_open_on_cid
        # in evaluate(), so this fix only touches plain BondBuyer + bond_pro.
        held_cids: set[str] = set()
        for t in strat_state.get("trades", []):
            if t.get("outcome") is not None:
                continue
            if t.get("status") in TERMINAL_STATUSES:
                continue
            cid = t.get("condition_id", "")
            if cid:
                held_cids.add(cid)

        for signal in signals:
            if signal.condition_id and signal.condition_id in held_cids:
                # Already holding an open trade on this market — skip.
                continue
            token_id = signal.token_id

            # Orderbook verification
            prechecked = signal.metadata or {}
            if (
                "_v2_prechecked_vwap_price" in prechecked
                and "_v2_prechecked_vwap_liq" in prechecked
            ):
                vwap_px = float(prechecked["_v2_prechecked_vwap_price"])
                vwap_liq = float(prechecked["_v2_prechecked_vwap_liq"])
            else:
                book = fetch_orderbook(token_id)
                asks = book.get("asks", [])
                vwap_px, vwap_liq = vwap_ask(asks, VWAP_SIZE)

            if vwap_liq < MIN_LIQUIDITY:
                self.log.info(
                    f"  Skip {signal.market_question[:40]} -- liquidity {vwap_liq:.1f} < {MIN_LIQUIDITY}"
                )
                continue

            actual_price = vwap_px if vwap_px > 0 else signal.price

            # Stale-scan guard — if live VWAP is drastically below the
            # bond zone, Gamma's quoted price was stale. Skip as bad data
            # rather than treating the huge phantom "edge" as real.
            if actual_price < STALE_SCAN_FLOOR:
                self.log.warning(
                    f"  Skip {signal.market_question[:40]} -- live VWAP "
                    f"{actual_price:.3f} < {STALE_SCAN_FLOOR} (Gamma quote "
                    f"{signal.price:.3f} stale — market already collapsed)"
                )
                continue

            # Recheck edge with actual price
            actual_edge = signal.p_fair - actual_price
            min_edge_required = float((signal.metadata or {}).get("min_edge_required", MIN_EDGE))
            if actual_edge < min_edge_required:
                self.log.info(
                    f"  Skip {signal.market_question[:40]} -- VWAP {actual_price:.3f} erodes edge"
                )
                continue

            # Kelly sizing
            # MAKER switch (2026-04-18): place limit 1¢ below VWAP so the
            # order rests on the book as maker (0% fee) instead of crossing
            # as effective taker (~1% fee). Backtest with proper friction
            # model showed +6.2% ROI at TTL=12h (from -2.21% taker +3¢).
            # If market runs away before fill, reconcile_state will cancel
            # after TTL=8h (enforced downstream).
            # Bug-36 (2026-04-28): Polymarket upgraded tick_size 0.01→0.001
            # for active bond markets. Our hardcoded $X maker undercut
            # was placing orders 10 ticks below best_bid → queue-ass-end
            # → most orders never filled (133 cancelled vs 85 filled).
            # Read tick from signal metadata (api.py populates this from
            # market.orderPriceMinTickSize); fallback to 0.01 for legacy
            # markets that haven't been migrated.
            tick = float(signal.metadata.get("tick_size", 0.01)) if signal.metadata else 0.01
            maker_limit = max(tick, round(actual_price - tick, 4))
            maker_ev = ev_calc(signal.p_fair, maker_limit)
            if maker_ev < MIN_EV:
                self.log.info(
                    f"  Skip {signal.market_question[:40]} -- maker EV "
                    f"{maker_ev:.4f} < {MIN_EV:.4f}"
                )
                continue

            raw_size = kelly_size(signal.p_fair, maker_limit, bankroll, fraction=KELLY_FRACTION)
            if raw_size < MIN_NOTIONAL:
                self.log.info(
                    f"  Skip {signal.market_question[:40]} -- Kelly ${raw_size:.2f} "
                    f"< ${MIN_NOTIONAL:.2f} min order"
                )
                continue
            if FIXED_BET_SHARES > 0:
                size = round(FIXED_BET_SHARES * maker_limit, 2)
            else:
                size = FIXED_BET_SIZE if FIXED_BET_SIZE > 0 else min(raw_size, self.max_bet)

            # Cap check: would this push past max_open_cost?
            if current_open + size > cap:
                remaining = max(0.0, cap - current_open)
                # Same min-size guard on the last-trade-in-cap case.
                if remaining < MIN_NOTIONAL:
                    self.log.info(
                        f"  [BOND] remaining cap ${remaining:.2f} < "
                        f"${MIN_NOTIONAL} min — stop this cycle"
                    )
                    break
                size = round(remaining, 2)
            current_open += size

            sig = Signal(
                strategy=self.name,
                market_question=signal.market_question,
                condition_id=signal.condition_id,
                token_id=token_id,
                side="BUY",
                price=maker_limit,
                p_fair=signal.p_fair,
                edge=round(signal.p_fair - maker_limit, 4),
                ev=maker_ev,
                metadata={
                    **signal.metadata,
                    "vwap_price": actual_price,
                    "vwap_liq": vwap_liq,
                    "kelly_size": raw_size,
                    "sized_usd": size,
                    "fixed_shares_target": FIXED_BET_SHARES if FIXED_BET_SHARES > 0 else None,
                    "maker_discount": tick,
                    "tick_size": tick,
                    "ttl_sec": 8 * 3600,
                },
            )
            trades.append(Trade(signal=sig, size_usd=round(size, 2), order_type="GTC"))

        return trades

    # ── Helpers ──

    @staticmethod
    def _parse_market(m: dict) -> dict | None:
        """Parse Gamma API market dict into clean format."""
        try:
            prices_raw = m.get("outcomePrices", "[]")
            prices = [float(x) for x in (json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw)]
            outcomes_raw = m.get("outcomes", "[]")
            outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else list(outcomes_raw)
            tokens_raw = m.get("clobTokenIds", "[]")
            tokens = json.loads(tokens_raw) if isinstance(tokens_raw, str) else list(tokens_raw)
            if len(prices) < 2 or len(tokens) < 2:
                return None
            # Bug-13 v3 (2026-04-27): derive synthetic event key from
            # question text. We tried API events[0].slug first, but it's
            # null for closed markets, breaking backfill. Synthetic key
            # is consistent across closed/active and across new/legacy
            # trades. Strip out the variable parts (temps, dates, range
            # buckets) so siblings collapse to one key.
            event_slug = derive_event_key(m.get("question", ""))
            return {
                "condition_id": m.get("conditionId", ""),
                "question": m.get("question", "?"),
                "slug": m.get("slug", ""),
                "end_date": m.get("endDate", ""),
                "active": _as_bool(m.get("active"), True),
                "closed": _as_bool(m.get("closed"), False),
                "accepting_orders": _as_bool(m.get("acceptingOrders"), True),
                "game_start_time": m.get("gameStartTime", ""),
                "closed_time": m.get("closedTime", ""),
                "neg_risk": m.get("negRisk", False),
                "volume": float(m.get("volume", "0") or "0"),
                "outcomes": outcomes,
                "prices": prices,
                "tokens": tokens,
                "event_slug": event_slug,
                # Bug-36 (2026-04-28): per-market tick from Gamma. Polymarket
                # migrated bond markets 0.01→0.001 tick. Read this here so
                # evaluate() places maker undercut at the right precision.
                "orderPriceMinTickSize": float(m.get("orderPriceMinTickSize", 0.01) or 0.01),
            }
        except Exception:
            return None
