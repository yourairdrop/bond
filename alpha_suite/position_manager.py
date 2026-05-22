"""
Alpha Suite — Position Lifecycle Manager.

Provides four features on top of each Strategy's basic place-order path:

    1. Fill simulation (dry-run)  — promote `dry_open` → `dry_filled`
       when the real CLOB orderbook crosses our limit price.
    2. Cancel-stale                — per-strategy TTL + adverse-selection
       rules. Dry mode: mark `dry_cancelled`. Live: subprocess cancel.
    3. Sell-high (take profit)    — per-strategy profile (bond / llm /
       whale / longshot / none). Dry: mark `dry_sold`. Live: place SELL.
    4. Auto-redeem                 — after resolution, trigger
       `redeem_position.py` subprocess. Dry: mark `dry_redeemed`.

Each strategy declares its config via class attributes. Strategies that
opt out of a feature set `enabled=False` in the corresponding config.

## Trade record fields (added on top of existing schema)

    status: "dry_open" | "dry_filled" | "dry_cancelled" | "dry_sold"
          | "dry_redeemed" | "open" | "filled" | "cancelled" | "sold"
          | "redeemed"
    placed_at:     ISO 8601 when the order was placed
    filled_at:     ISO 8601 when the order filled (set by fill-sim or exchange)
    filled_price:  float, actual fill price (may differ from limit)
    cancelled_at:  ISO 8601 when cancelled
    cancel_reason: str, one of: "hard_ttl" | "crossed" | "drift" | "expiring"
                                | "out_of_range" | ...
    sold_at:       ISO 8601
    sold_price:    float
    sold_reason:   str, e.g. "bond_high" | "llm_target" | "whale_exit"
    redeemed_at:   ISO 8601

`outcome` (existing field) remains the authoritative market-resolution flag
(None=open, "WIN"/"LOSS"=resolved). `status` tracks the *order lifecycle*
independently — a trade can be `dry_sold` with `outcome=None` (we sold
before resolution).

## PnL accounting interaction

  * Fill: no PnL change (position just opens).
  * Cancel: no PnL change (order never filled, capital returned).
  * Sell-high: realized_pnl += (sold_price - price) × shares (locked early).
  * Redeem on WIN: realized_pnl += (1.0 - price) × shares.
  * Redeem on LOSS: realized_pnl += (0.0 - price) × shares = -price × shares.

Only resolved (sold or redeemed) trades move `realized_pnl`; open/filled
trades contribute `unrealized_pnl` via the existing `update_unrealized()`
path (unchanged).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import http.client
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from alpha_suite.utils.api import fetch_market_resolution, fetch_orderbook


# ══════════════════════════════════════════════════════════════════════
# Config dataclasses (per-strategy)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class CancelStaleConfig:
    """Per-strategy cancel-stale config.

    All rules are evaluated; if ANY fires, the order is cancelled. Rules:

      * hard_ttl_min:       age > N minutes → cancel (safety net)
      * cancel_on_crossed:  best_bid > our_limit (for BUY orders) → adverse
                            selection; also best_ask < our_limit for SELL.
      * drift_bps:          |mid - our_limit| > bps → market walked away
      * expiring_hours:     market close in < N hours and still unfilled → cancel
      * out_of_range_ask:   e.g. for bond, cancel if best_ask > 0.95
                            (signals bond range abandoned)

    Strategies that don't use GTC (e.g. longshot-taker uses FOK) should set
    `enabled=False`.
    """
    enabled: bool = True
    hard_ttl_min: int = 180
    cancel_on_crossed: bool = True
    drift_bps: Optional[int] = None                 # e.g. 300 = 3%
    expiring_hours: Optional[float] = None          # None = ignore
    out_of_range_ask: Optional[float] = None        # e.g. 0.95 for bond


@dataclass
class SellHighConfig:
    """Per-strategy take-profit config.

    `profile` controls the *logic*:

      * "none"      — never sell early (arb, coverage — must hold basket)
      * "bond"      — sell when price >= threshold; sell_price = current - offset,
                      floor at `floor`. Classic bond-buyer style.
      * "llm"       — sell when price >= p_fair - buffer; sell at current - offset.
                      Uses the trade's own `p_fair` as the target.
      * "whale"     — dual rule: (a) whale wallet exited the position, or
                      (b) price >= threshold (bond fallback).
      * "longshot"  — sell when price >= min(entry × multiplier, ceiling).
                      Default: entry × 2, ceiling 0.50.

    Trailing peak protection (orthogonal to profile, evaluated FIRST):
      If `trail_enabled=True`, we track the highest bid seen since fill
      (`peak_bid` on the trade dict). Once the peak reaches
      `entry × (1 + trail_activate_gain)`, we arm trailing mode. From
      then on, if current bid falls `trail_drawdown_pct` below the peak,
      we sell. This protects a big run-up from reverting all the way to
      entry (user's 2026-04-23 ask: "low entry, big gain → lock it in").

    Strategies with basket/atomic semantics MUST set profile="none" — selling
    one leg of an arb basket destroys the edge.
    """
    enabled: bool = False
    profile: str = "none"
    threshold: float = 0.92          # bond / whale price trigger
    floor: float = 0.90              # minimum sell price (bond)
    offset: float = 0.01             # sell at current - offset
    llm_buffer: float = 0.02         # llm: sell when price >= p_fair - buffer
    longshot_multiplier: float = 2.0
    longshot_ceiling: float = 0.50
    # Trailing peak protection (checked before profile logic when enabled).
    trail_enabled: bool = False
    trail_activate_gain: float = 0.50   # arm trailing after peak = entry × 1.5
    trail_drawdown_pct: float = 0.25    # sell when bid drops 25% below peak
    # ── Resting TP ladder (place-and-wait, bond style) ────────────────
    # When enabled, the moment a BUY fills we "place" a SELL limit at
    # `resting_tp_initial` (e.g. 0.99). If that doesn't fill within each
    # ladder rung's hour window, we lower the price to the next rung. If
    # mid drops below `resting_tp_emergency_mid`, we abandon the ladder
    # and urgently exit at best_bid - offset (but never below entry+1¢).
    #
    # This REPLACES the polling-based profile logic for trades with this
    # enabled — mid-based `threshold` / `floor` / `offset` are ignored.
    resting_tp_enabled: bool = False
    resting_tp_initial: float = 0.99
    resting_tp_ladder: tuple = (
        (6.0, 0.97),    # 6h without fill → lower to 0.97
        (12.0, 0.95),   # 12h → 0.95
        (24.0, 0.93),   # 24h → 0.93
    )
    resting_tp_emergency_mid: float = 0.88
    # ── Catastrophe lock (Bug-11 fix, 2026-04-26) ────────────────────
    # When mid drops below `resting_tp_catastrophe_mid` (default 0.30),
    # we override the "don't lock loss" guard and sell at best_bid -
    # offset, even if that locks a loss. Reason: at this point the
    # position is virtually guaranteed to settle at $X (full loss).
    # Locking a partial loss now is strictly better than waiting.
    # Above catastrophe but below emergency_mid → hold to resolution
    # (mid might recover before settlement).
    resting_tp_catastrophe_mid: float = 0.30
    # Minimum best_bid to bother selling at. Bug-19 (2026-04-27): raised
    # 0.05→0.16 because Polymarket has a $X USD order minimum (NOT just
    # 5 shares). At bid=$X × 6.67 shares = $X, CLOB silently rejects
    # the SELL — PM marks state="sold" but chain still has the shares
    # (state-vs-chain divergence). With min_bid=$X, $X × 6 = $X
    # is borderline; min_bid=$X gives $X+ comfortably above floor.
    resting_tp_catastrophe_min_bid: float = 0.20


@dataclass
class PositionManagerSettings:
    """Cross-strategy knobs."""
    fill_check_delay_sec: int = 30        # gap after placement before fill-check fires
    fill_check_interval_sec: int = 30     # how often the manager runs check_fills
    redeem_enabled: bool = True           # False during rollout if concerned
    redeem_script_path: str = ""          # absolute path to redeem_position.py helper
    redeem_venv_python: str = ""          # absolute path to venv python
    redeem_losses_onchain: bool = False   # burn losing redeemable tokens instead of leaving UI dust


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def _fetch_data_api_positions(
    proxy: str,
    *,
    size_threshold: Optional[float] = None,
    limit: Optional[int] = None,
    timeout: int = 15,
    retries: int = 3,
) -> list:
    """Fetch Polymarket data-api positions with retries for flaky TLS/API edges."""
    params = {"user": proxy}
    if size_threshold is not None:
        params["sizeThreshold"] = str(size_threshold)
    if limit is not None:
        params["limit"] = str(limit)
    qs = urllib.parse.urlencode(params)
    url = f"https://data-api.polymarket.com/positions?{qs}"

    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "alpha-suite-pm/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
            return data if isinstance(data, list) else []
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
            json.JSONDecodeError,
            OSError,
            TimeoutError,
        ) as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.75 * (attempt + 1))
    raise RuntimeError(str(last_err) if last_err else "unknown data-api error")


def _parse_iso(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _age_hours(trade: dict) -> float:
    """Hours since placed_at. 0.0 if unknown."""
    placed = _parse_iso(trade.get("placed_at", "") or trade.get("time", ""))
    if placed is None:
        return 0.0
    return (_now() - placed).total_seconds() / 3600.0


def _trade_is_open(t: dict) -> bool:
    """True when the trade represents an open GTC-like position awaiting fill.

    Legacy trades (before status was tracked) default to `dry_filled` if
    `outcome is None` (existing behavior was instant-fill). Only trades
    with explicit `status in {dry_open, open}` count as unfilled.
    """
    return t.get("status") in ("dry_open", "open")


def _trade_is_filled_holding(t: dict) -> bool:
    """True when the trade is filled and still holding (not sold, not resolved)."""
    if t.get("outcome") is not None:
        return False  # already resolved
    status = t.get("status")
    if status in ("dry_sold", "sold", "dry_cancelled", "cancelled",
                  "dry_redeemed", "redeemed"):
        return False
    # Explicit filled or legacy trade (no status) with outcome=None → holding
    if status in ("dry_filled", "filled"):
        return True
    if status is None:
        # Legacy: no status field yet; treat as filled if outcome is None
        return True
    return False


def _side_is_buy(t: dict) -> bool:
    """True if the trade represents a long/buy leg.

    Accepts all of: "BUY", "BUY_YES", "BUY_NO", "YES", "" (legacy default).
    W3 strategies mostly use plain "BUY"; longshot and W2 ladder use
    "BUY_YES"/"BUY_NO". Treat any side starting with BUY/YES as buy.
    """
    side = (t.get("side") or "").upper()
    if side == "" or side == "YES":
        return True
    return side.startswith("BUY")


def _cost_basis(t: dict) -> float:
    """Per-share cost basis for PnL computation.

    Prefers actual `filled_price` (set by fill-sim or live fill callback) over
    the order's limit `price`. When a GTC BUY at 0.92 gets crossed by a
    counterparty at 0.90, the true cost basis is 0.90 — using 0.92 would
    systematically understate profit by 2¢/share.
    """
    try:
        fp = t.get("filled_price")
        if fp is not None and float(fp) > 0:
            return float(fp)
    except (TypeError, ValueError):
        pass
    try:
        return float(t.get("price", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _fallback_chain_cost(existing_trade: dict, chain_size: float) -> float:
    """Infer cost basis when data-api reports a held position with zero cost.

    Polymarket's data-api occasionally surfaces `size>0,currentValue>0` but
    `initialValue=0` for a freshly held position. Treat that as missing data,
    not as a free position. Prefer the local requested notional, then the
    live sizing metadata, then an entry-price × shares reconstruction.
    """
    entry = _cost_basis(existing_trade)
    if entry > 0 and chain_size > 0:
        return round(entry * chain_size, 4)

    try:
        prior_size = float(existing_trade.get("size") or 0.0)
    except (TypeError, ValueError):
        prior_size = 0.0
    try:
        prior_shares = float(existing_trade.get("shares") or 0.0)
    except (TypeError, ValueError):
        prior_shares = 0.0
    if prior_size > 0:
        if prior_shares > 0 and chain_size > 0 and chain_size < prior_shares:
            return round(prior_size * (chain_size / prior_shares), 4)
        return round(prior_size, 4)

    meta = existing_trade.get("metadata") or {}
    try:
        sized_usd = float(meta.get("sized_usd") or 0.0)
    except (TypeError, ValueError):
        sized_usd = 0.0
    if sized_usd > 0:
        if prior_shares > 0 and chain_size > 0 and chain_size < prior_shares:
            return round(sized_usd * (chain_size / prior_shares), 4)
        return round(sized_usd, 4)
    return 0.0


# Re-export the shared terminal-status set from state.py so callers that
# import from position_manager don't have to know the origin. state.py is
# the source of truth (check_bankroll and update_resolution live there).
from alpha_suite.state import TERMINAL_STATUSES  # noqa: E402, F401


# ══════════════════════════════════════════════════════════════════════
# Position Manager
# ══════════════════════════════════════════════════════════════════════

class PositionManager:
    """Runs lifecycle checks across all strategies.

    Typical usage (from main loop):

        pm = PositionManager(state, logger, settings)
        while True:
            # ... run each strategy scan/evaluate/execute ...
            pm.check_fills(strategies)           # dry-run only
            pm.check_cancel_stale(strategies)
            pm.check_sell_high(strategies)
            pm.check_and_redeem(strategies)
            time.sleep(30)
    """

    def __init__(self, state, logger: logging.Logger,
                 settings: Optional[PositionManagerSettings] = None):
        self.state = state
        self.log = logger
        self.settings = settings or PositionManagerSettings()
        self._last_fill_check_ts: float = 0.0
        # Book cache per token_id with short TTL to avoid hammering /book
        # when multiple lifecycle checks ask about the same market in the
        # same tick.
        self._book_cache: dict[str, tuple[float, dict]] = {}
        self._book_cache_ttl = 5.0   # seconds

    # ── Internal helpers ────────────────────────────────────────

    def _get_book(self, token_id: str) -> dict:
        """Fetch orderbook with short-TTL cache."""
        now = time.time()
        cached = self._book_cache.get(token_id)
        if cached and now - cached[0] < self._book_cache_ttl:
            return cached[1]
        book = fetch_orderbook(token_id)
        self._book_cache[token_id] = (now, book)
        return book

    def _clear_book_cache(self) -> None:
        self._book_cache.clear()

    def _best_ask(self, book: dict) -> Optional[float]:
        asks = book.get("asks") or []
        if not asks:
            return None
        try:
            return float(asks[0].get("price", 0) or 0)
        except (TypeError, ValueError):
            return None

    def _best_bid(self, book: dict) -> Optional[float]:
        bids = book.get("bids") or []
        if not bids:
            return None
        try:
            return float(bids[0].get("price", 0) or 0)
        except (TypeError, ValueError):
            return None

    def _mid(self, book: dict) -> Optional[float]:
        a = self._best_ask(book)
        b = self._best_bid(book)
        if a is None or b is None:
            return None
        return (a + b) / 2.0

    # ── Feature 1: Fill simulation (dry-run only) ───────────────

    def check_fills(self, strategies: list) -> int:
        """Walk dry_open trades; promote to dry_filled when orderbook crosses.

        Interval-limited by settings.fill_check_interval_sec — calling every
        second is fine (we'll skip if not enough time passed since last call).

        Returns number of trades promoted this tick.
        """
        now_ts = time.time()
        if now_ts - self._last_fill_check_ts < self.settings.fill_check_interval_sec:
            return 0
        self._last_fill_check_ts = now_ts

        self._clear_book_cache()   # fresh data each tick
        filled = 0

        for strat in strategies:
            if not getattr(strat, "dry_run", True):
                continue  # live strategy — real exchange tracks fills
            name = strat.name
            strat_state = self.state.get_strategy_state(name)
            for t in strat_state.get("trades", []):
                if t.get("status") != "dry_open":
                    continue
                placed = _parse_iso(t.get("placed_at", ""))
                if placed is None:
                    continue
                # Skip orders placed in the very-recent window (give scan→
                # evaluate→execute a moment before the fill-sim fires).
                if (_now() - placed).total_seconds() < self.settings.fill_check_delay_sec:
                    continue

                token_id = t.get("token_id", "")
                limit = float(t.get("price", 0) or 0)
                if not token_id or limit <= 0:
                    continue

                book = self._get_book(token_id)
                if _side_is_buy(t):
                    # BUY fills when someone sells at/below our limit.
                    best_ask = self._best_ask(book)
                    if best_ask is None:
                        continue
                    if best_ask <= limit + 1e-6:
                        # IMPORTANT: a maker BUY fills at the MAKER's
                        # limit price, not the taker's ask. If best_ask
                        # is much lower than limit, it means market
                        # crashed; a crossed book snapshot (ask < bid)
                        # can't persist in reality — the engine would
                        # have matched at our 0.91 earlier, we're just
                        # seeing the aftermath where deeper sell orders
                        # sit below. So filled_price = limit.
                        #
                        # Prior bug: recording filled_price = best_ask
                        # caused phantom-windfall shares computations
                        # (2667 shares at 0.003) and distorted MTM.
                        t["status"] = "dry_filled"
                        t["filled_at"] = _now_iso()
                        t["filled_price"] = limit
                        filled += 1
                        self.log.info(
                            f"[PM][fill] {name} {t.get('market','')[:40]} "
                            f"limit={limit:.3f} crossed by ask={best_ask:.3f} "
                            f"→ fill@{limit:.3f} (maker price)"
                        )
                else:
                    # SELL fills when someone buys at/above our limit.
                    # Same maker-price logic as BUY above.
                    best_bid = self._best_bid(book)
                    if best_bid is None:
                        continue
                    if best_bid >= limit - 1e-6:
                        t["status"] = "dry_filled"
                        t["filled_at"] = _now_iso()
                        t["filled_price"] = limit
                        filled += 1
                        self.log.info(
                            f"[PM][fill] {name} {t.get('market','')[:40]} "
                            f"limit={limit:.3f} crossed by bid={best_bid:.3f} "
                            f"→ fill@{limit:.3f} (maker price) (SELL)"
                        )

        if filled:
            self.state.save()
        return filled

    # ── Feature 2: Cancel-stale ─────────────────────────────────

    def check_cancel_stale(self, strategies: list) -> int:
        """Apply per-strategy cancel rules to all open trades.

        Fresh book data each pass — don't reuse the cache from the previous
        lifecycle call (it may be up to 5s stale, enough for a fast-moving
        market to have crossed us).
        """
        self._clear_book_cache()
        cancelled = 0
        for strat in strategies:
            cfg = getattr(strat, "cancel_stale_cfg", None)
            if cfg is None or not cfg.enabled:
                continue
            name = strat.name
            strat_state = self.state.get_strategy_state(name)
            for t in strat_state.get("trades", []):
                if not _trade_is_open(t):
                    continue
                reason = self._should_cancel(t, cfg)
                if reason:
                    self._cancel_trade(strat, t, reason)
                    cancelled += 1
        if cancelled:
            self.state.save()
        return cancelled

    def _should_cancel(self, t: dict, cfg: CancelStaleConfig) -> Optional[str]:
        """Return cancel reason string if any rule fires, else None.

        Rules short-circuit on the cheapest checks first (TTL before /book).
        """
        # Rule: hard TTL
        age_h = _age_hours(t)
        if cfg.hard_ttl_min and age_h * 60 > cfg.hard_ttl_min:
            return f"hard_ttl({age_h:.1f}h)"

        # Rule: expiring market
        if cfg.expiring_hours is not None:
            end_iso = (t.get("metadata") or {}).get("market_end_iso") or \
                      t.get("market_end_iso", "")
            end_dt = _parse_iso(end_iso)
            if end_dt is not None:
                hours_left = (end_dt - _now()).total_seconds() / 3600.0
                if hours_left < cfg.expiring_hours:
                    return f"expiring({hours_left:.1f}h)"

        # Rules requiring orderbook lookup
        token_id = t.get("token_id", "")
        if not token_id:
            return None
        book = self._get_book(token_id)
        limit = float(t.get("price", 0) or 0)
        if limit <= 0:
            return None

        best_ask = self._best_ask(book)
        best_bid = self._best_bid(book)
        mid = self._mid(book)

        # Rule: adverse selection (crossed)
        if cfg.cancel_on_crossed and _side_is_buy(t):
            if best_bid is not None and best_bid > limit + 1e-6:
                # Market's best bid above our limit = we're too cheap;
                # anyone crossing to our price has likely info.
                return f"crossed(bid={best_bid:.3f}>limit={limit:.3f})"

        # Rule: price drift
        if cfg.drift_bps is not None and mid is not None:
            drift_bps = abs(mid - limit) * 10_000
            if drift_bps > cfg.drift_bps:
                return f"drift({drift_bps:.0f}bps)"

        # Rule: out-of-range for bond-style filters
        if cfg.out_of_range_ask is not None and best_ask is not None:
            if best_ask > cfg.out_of_range_ask:
                return f"out_of_range(ask={best_ask:.3f})"

        return None

    # Max times we'll retry a live cancel before giving up and marking
    # the trade as `cancel_failed`. Prevents infinite retry loop on orders
    # the exchange refuses to cancel (e.g., already filled, invalid ID).
    MAX_CANCEL_ATTEMPTS = 3

    def _cancel_trade(self, strat, t: dict, reason: str) -> None:
        """Mark a trade as cancelled. Dry: status=dry_cancelled; live:
        subprocess cancel via place_order.py.

        Live cancel failures are retried up to `MAX_CANCEL_ATTEMPTS` times
        before the trade is moved to terminal `cancel_failed` status so
        further lifecycle checks skip it. This avoids the "infinite cancel
        retry" loop Codex flagged (P0-3, 2026-04-22).
        """
        dry = getattr(strat, "dry_run", True)
        order_id = t.get("order_id", "")
        market_short = t.get("market", "")[:40]

        if dry or not order_id or order_id.startswith("DRY-"):
            t["status"] = "dry_cancelled"
            t["cancelled_at"] = _now_iso()
            t["cancel_reason"] = reason
            # Dry cancel: release the open-exposure slot. Bump open_trades
            # counter down so next scan can reclaim the bankroll.
            strat_state = self.state.get_strategy_state(strat.name)
            strat_state["open_trades"] = max(0, strat_state.get("open_trades", 0) - 1)
            self.log.info(
                f"[PM][cancel-dry] {strat.name} {market_short} reason={reason}"
            )
            return

        # Live cancel: check attempts counter first
        attempts = int(t.get("cancel_attempts", 0))
        if attempts >= self.MAX_CANCEL_ATTEMPTS:
            t["status"] = "cancel_failed"
            t["cancelled_at"] = _now_iso()
            t["cancel_reason"] = f"{reason}|max_retries"
            self.log.error(
                f"[PM][cancel-live] {strat.name} {market_short} GAVE UP after "
                f"{attempts} attempts; order {order_id[:24]} left in "
                f"`cancel_failed` state — manual intervention needed."
            )
            # Release the slot for bankroll accounting (treat as cancelled
            # from our side even though the exchange may still have it).
            strat_state = self.state.get_strategy_state(strat.name)
            strat_state["open_trades"] = max(0, strat_state.get("open_trades", 0) - 1)
            return

        self.log.info(
            f"[PM][cancel-live] {strat.name} {market_short} order={order_id[:24]} "
            f"reason={reason} attempt={attempts+1}/{self.MAX_CANCEL_ATTEMPTS}"
        )
        try:
            ok = strat.cancel_live_order(order_id)
        except AttributeError:
            self.log.error(
                f"[PM][cancel-live] {strat.name} has no cancel_live_order() — "
                f"order {order_id[:24]} is LEAKING. Marking cancel_failed."
            )
            t["status"] = "cancel_failed"
            t["cancelled_at"] = _now_iso()
            t["cancel_reason"] = f"{reason}|no_helper"
            return
        if ok:
            t["status"] = "cancelled"
            t["cancelled_at"] = _now_iso()
            t["cancel_reason"] = reason
            strat_state = self.state.get_strategy_state(strat.name)
            strat_state["open_trades"] = max(0, strat_state.get("open_trades", 0) - 1)
        else:
            t["cancel_attempts"] = attempts + 1
            self.log.warning(
                f"[PM][cancel-live] subprocess failed for {order_id[:24]} "
                f"({attempts+1}/{self.MAX_CANCEL_ATTEMPTS})"
            )

    # ── Feature 3: Sell-high ────────────────────────────────────

    def check_sell_high(self, strategies: list) -> int:
        """Per-strategy take-profit check on filled-and-holding trades.

        Fresh book data each pass (see check_cancel_stale for rationale).

        Side-effect: for every filled-and-holding trade we also refresh
        `peak_bid` on the trade dict so trailing-stop logic sees the true
        running high. Peak is updated even when trailing is disabled — cheap,
        and keeps the data available for later opt-in.
        """
        self._clear_book_cache()
        sold = 0
        peak_bumped = False
        for strat in strategies:
            cfg = getattr(strat, "sell_high_cfg", None)
            if cfg is None or not cfg.enabled or cfg.profile == "none":
                continue
            name = strat.name
            strat_state = self.state.get_strategy_state(name)
            for t in strat_state.get("trades", []):
                if not _trade_is_filled_holding(t):
                    continue
                # Update peak_bid snapshot (always, even if not selling).
                token_id = t.get("token_id", "")
                if token_id:
                    book = self._get_book(token_id)
                    bid = self._best_bid(book)
                    if bid is not None and bid > 0:
                        prev_peak = float(t.get("peak_bid", 0) or 0)
                        if bid > prev_peak:
                            t["peak_bid"] = round(bid, 4)
                            t["peak_at"] = _now_iso()
                            peak_bumped = True
                # Branch: resting TP ladder (bond-style place-and-wait)
                # or classic mid-polling profile logic.
                if cfg.resting_tp_enabled:
                    target = self._process_resting_tp(t, cfg, strat)
                else:
                    target = self._compute_sell_target(t, cfg, strat)
                if target is None:
                    continue
                sold_price, reason = target
                if self._sell_trade(strat, t, sold_price, reason):
                    sold += 1
        if sold or peak_bumped:
            self.state.save()
        return sold

    def _process_resting_tp(self, t: dict, cfg: SellHighConfig, strat) -> \
            Optional[tuple[float, str]]:
        """Manage a resting take-profit limit order on a filled trade.

        State machine:
          (init) No tp_target_price yet
              → set tp_target_price = resting_tp_initial (e.g. 0.99)
          (fill)  best_bid ≥ tp_target_price
              → fill at tp_target_price, return (price, reason)
          (emergency) mid ≤ resting_tp_emergency_mid
              → cancel TP, set tp to best_bid - offset (no lower than entry+1c)
              → re-check fill at new price
          (ladder) age_hours ≥ next_rung_hours
              → lower tp_target_price to ladder rung target
              → re-check fill at new price

        Returns (sell_price, reason) if we fill this tick; None otherwise.

        Trade-dict fields maintained:
          tp_target_price     current resting sell limit
          tp_price_set_at     ISO when this price was last set
          tp_ladder_rung      0=initial, 1..N=ladder rung, -1=emergency
          tp_emergency_logged one-shot log guard for can't-exit case
        """
        token_id = t.get("token_id", "")
        if not token_id:
            return None
        # Bug-33 (2026-04-28): cooldown after Bug-30 phantom-sold revert.
        # If a SELL was placed but didn't fill within 5min (Bug-30 reverted
        # status sold→filled), don't immediately re-fire the same TP — that
        # caused a stacking loop where each cycle placed another resting
        # order, eventually multiple GTC orders for the same shares.
        last_revert = t.get("sell_last_revert_at", "")
        if last_revert:
            try:
                rev_dt = datetime.fromisoformat(last_revert.replace("Z", "+00:00"))
                cooldown_age = (datetime.now(timezone.utc) - rev_dt).total_seconds()
                if cooldown_age < 1800:   # 30min cooldown post-revert
                    return None
            except Exception:
                pass
        book = self._get_book(token_id)
        best_bid = self._best_bid(book)
        best_ask = self._best_ask(book)
        if best_bid is None or best_ask is None:
            return None
        mid = (best_bid + best_ask) / 2.0
        entry = _cost_basis(t)
        if entry <= 0:
            return None

        # (init) — first time this trade is seen after fill
        current_tp = t.get("tp_target_price")
        if current_tp is None:
            current_tp = cfg.resting_tp_initial
            t["tp_target_price"] = current_tp
            t["tp_price_set_at"] = _now_iso()
            t["tp_ladder_rung"] = 0
            self.log.info(
                f"[PM][tp-place] {strat.name} entry={entry:.3f} "
                f"TP@{current_tp:.3f} — {t.get('market','')[:40]}"
            )
        else:
            current_tp = float(current_tp)

        # (fill) check current TP against best_bid
        if best_bid >= current_tp - 1e-6:
            # Resting limit fills at the limit price (maker gets their price).
            return current_tp, (
                f"tp_resting_fill(tp={current_tp:.3f},bid={best_bid:.3f})"
            )

        # (emergency) mid below danger threshold → urgent exit
        if mid <= cfg.resting_tp_emergency_mid:
            can_sell_profit = best_bid >= entry + 0.01
            in_catastrophe = mid <= cfg.resting_tp_catastrophe_mid

            if not can_sell_profit:
                # Can't exit at a profit. Decision tree:
                #   1. Above catastrophe threshold (mid > 0.30) → hold to
                #      resolution (mid might recover before settle).
                #   2. Below catastrophe threshold → lock partial loss
                #      (Bug-11 fix). Position is virtually guaranteed to
                #      settle at $0; selling now at best_bid - 1¢ is
                #      strictly better than waiting for full -100% loss.
                #   3. If best_bid is itself below `catastrophe_min_bid`
                #      ($X default), don't bother — gas + slippage
                #      makes the lock-loss attempt net-negative.
                if not in_catastrophe:
                    if not t.get("tp_emergency_logged"):
                        self.log.warning(
                            f"[PM][tp-emergency] {strat.name} mid={mid:.3f} < "
                            f"{cfg.resting_tp_emergency_mid:.2f} AND best_bid "
                            f"{best_bid:.3f} < entry+1c ({entry+0.01:.3f}). "
                            f"Above catastrophe ({cfg.resting_tp_catastrophe_mid:.2f}) "
                            f"— holding to resolution. "
                            f"{t.get('market','')[:40]}"
                        )
                        t["tp_emergency_logged"] = True
                    return None
                # In catastrophe — lock partial loss
                if best_bid < cfg.resting_tp_catastrophe_min_bid:
                    if not t.get("tp_catastrophe_logged"):
                        self.log.warning(
                            f"[PM][tp-catastrophe] {strat.name} mid={mid:.3f} < "
                            f"{cfg.resting_tp_catastrophe_mid:.2f} BUT best_bid "
                            f"{best_bid:.3f} < min_bid "
                            f"({cfg.resting_tp_catastrophe_min_bid:.2f}). "
                            f"Holding to resolution (gas/slippage > recoverable). "
                            f"{t.get('market','')[:40]}"
                        )
                        t["tp_catastrophe_logged"] = True
                    return None
                # Lock the loss
                lock_px = max(round(best_bid - cfg.offset, 2),
                              cfg.resting_tp_catastrophe_min_bid)
                t["tp_target_price"] = lock_px
                t["tp_price_set_at"] = _now_iso()
                t["tp_ladder_rung"] = -2   # sentinel for catastrophe-lock
                lock_pnl = round((lock_px - entry) * float(t.get("shares", 0) or 0), 2)
                hold_pnl = round((0 - entry) * float(t.get("shares", 0) or 0), 2)
                self.log.warning(
                    f"[PM][tp-catastrophe-lock] {strat.name} mid={mid:.3f} "
                    f"<= {cfg.resting_tp_catastrophe_mid:.2f}. Locking partial "
                    f"loss: sell@{lock_px:.3f} pnl=${lock_pnl:+.2f} (vs hold "
                    f"to settlement = ${hold_pnl:+.2f}). "
                    f"{t.get('market','')[:40]}"
                )
                if best_bid >= lock_px - 1e-6:
                    return lock_px, (
                        f"tp_catastrophe_lock(mid={mid:.3f},"
                        f"lock={lock_px:.3f},save=${lock_pnl-hold_pnl:+.2f})"
                    )
                return None
            # Re-price to best_bid - offset (but ≥ entry+1c for profit)
            urgent_px = max(round(best_bid - cfg.offset, 2),
                            round(entry + 0.01, 2))
            if urgent_px != current_tp:
                t["tp_target_price"] = urgent_px
                t["tp_price_set_at"] = _now_iso()
                t["tp_ladder_rung"] = -1   # sentinel for emergency mode
                self.log.warning(
                    f"[PM][tp-emergency] {strat.name} mid={mid:.3f} dropped "
                    f"below {cfg.resting_tp_emergency_mid:.2f}. "
                    f"TP {current_tp:.3f}→{urgent_px:.3f} "
                    f"(best_bid={best_bid:.3f}). "
                    f"{t.get('market','')[:40]}"
                )
                current_tp = urgent_px
                if best_bid >= current_tp - 1e-6:
                    return current_tp, (
                        f"tp_emergency_fill(tp={current_tp:.3f},"
                        f"bid={best_bid:.3f})"
                    )
            return None

        # (ladder) step-down check (skip if already in emergency mode)
        cur_rung = int(t.get("tp_ladder_rung", 0) or 0)
        if cur_rung < 0:
            return None

        filled_at = _parse_iso(t.get("filled_at", "") or t.get("placed_at", ""))
        if filled_at is None:
            return None
        age_hours = (_now() - filled_at).total_seconds() / 3600.0

        # Find the deepest rung we're eligible for (ladder is pre-sorted
        # by ascending hours trigger in the config).
        eligible = None
        eligible_idx = 0
        for idx, (hours_trigger, target_price) in enumerate(cfg.resting_tp_ladder):
            if age_hours >= hours_trigger:
                eligible = target_price
                eligible_idx = idx + 1   # 1-indexed rung number

        if eligible is not None and eligible < current_tp:
            t["tp_target_price"] = eligible
            t["tp_price_set_at"] = _now_iso()
            t["tp_ladder_rung"] = eligible_idx
            self.log.info(
                f"[PM][tp-ladder] {strat.name} age={age_hours:.1f}h → rung "
                f"{eligible_idx}: TP {current_tp:.3f}→{eligible:.3f} "
                f"{t.get('market','')[:40]}"
            )
            current_tp = eligible
            if best_bid >= current_tp - 1e-6:
                return current_tp, (
                    f"tp_ladder_fill(tp={current_tp:.3f},"
                    f"bid={best_bid:.3f},rung={eligible_idx})"
                )

        # Still waiting
        return None

    def _compute_sell_target(self, t: dict, cfg: SellHighConfig, strat) -> \
            Optional[tuple[float, str]]:
        """Decide if we should sell and at what price.

        Returns (sell_price, reason) or None (don't sell).

        Uses `_cost_basis(t)` (filled_price if known, limit otherwise) for
        the entry reference. A trade's "don't lock a loss" guard applies to
        the actual cost, not the quoted limit — otherwise fill-sim at a
        better-than-limit price would make us refuse profitable early exits.
        """
        token_id = t.get("token_id", "")
        if not token_id:
            return None
        book = self._get_book(token_id)
        best_bid = self._best_bid(book)
        best_ask = self._best_ask(book)
        if best_bid is None or best_ask is None:
            return None
        mid = (best_bid + best_ask) / 2.0
        entry = _cost_basis(t)
        if entry <= 0:
            return None

        def can_fill_sell(limit_price: float) -> bool:
            """Return True only when the visible bid can execute our SELL."""
            return best_bid >= limit_price - 1e-6

        # ── Trailing peak protection (orthogonal to profile) ──────────
        # If enabled and peak has reached activation threshold, a
        # drawdown from peak triggers a sale. Evaluated BEFORE profile
        # logic so we can catch reversions that static thresholds miss
        # (e.g. longshot bought 0.08, peaked 0.40, now 0.26 — profile
        # longshot@mult=2 trigger is 0.16, wouldn't fire now, but we've
        # already given back 35% from peak).
        if cfg.trail_enabled:
            peak = float(t.get("peak_bid", 0) or 0)
            # Peak may not yet be recorded (first tick after fill) — fall
            # back to current bid so we don't prematurely trigger.
            if peak <= 0:
                peak = best_bid
            activate_at = entry * (1.0 + cfg.trail_activate_gain)
            if peak >= activate_at:
                # Trailing is armed. Check drawdown from peak.
                drawdown = (peak - best_bid) / peak if peak > 0 else 0.0
                if drawdown >= cfg.trail_drawdown_pct:
                    sell_px = max(round(best_bid - cfg.offset, 2),
                                  round(entry + 0.01, 2))
                    if sell_px > entry and can_fill_sell(sell_px):
                        return sell_px, (
                            f"trail_stop(entry={entry:.3f},peak={peak:.3f},"
                            f"bid={best_bid:.3f},dd={drawdown*100:.0f}%)"
                        )

        if cfg.profile == "bond":
            # Sell when mid >= threshold. Sell price = max(mid - offset, floor).
            if mid < cfg.threshold:
                return None
            sell_px = max(round(mid - cfg.offset, 2), cfg.floor)
            if sell_px <= entry:
                return None  # don't lock a loss
            if not can_fill_sell(sell_px):
                return None
            return sell_px, f"bond_high(mid={mid:.3f})"

        if cfg.profile == "llm":
            # Sell when mid reaches (p_fair - buffer). p_fair stored on trade.
            p_fair = float(t.get("p_fair", 0) or 0)
            if p_fair <= 0:
                return None
            trigger = p_fair - cfg.llm_buffer
            if mid < trigger:
                return None
            sell_px = max(round(mid - cfg.offset, 2), entry + 0.01)
            if sell_px <= entry:
                return None
            if not can_fill_sell(sell_px):
                return None
            return sell_px, f"llm_target(p_fair={p_fair:.2f},mid={mid:.3f})"

        if cfg.profile == "whale":
            # Dual rule: (a) whale exited; (b) bond fallback.
            # Whale exit check is delegated to strategy's helper if present;
            # fallback is logged ONCE per strategy so ops can diagnose missed
            # exits without spamming logs on every cycle.
            whale_exited = False
            try:
                cid = t.get("condition_id", "")
                token = t.get("token_id", "")
                whale_exited = bool(strat.whale_has_exited(cid, token))
            except AttributeError:
                if not getattr(strat, "_pm_whale_fallback_logged", False):
                    self.log.warning(
                        f"[PM][{strat.name}] whale_has_exited() not implemented — "
                        f"falling back to bond-style sell rule ONLY. Missed-exit "
                        f"diagnosis will need manual wallet tracking."
                    )
                    strat._pm_whale_fallback_logged = True
            except Exception as e:
                # Transient errors (wallet API down) — log but don't spam
                self.log.debug(f"[PM] whale_has_exited transient: {e}")
            if whale_exited:
                sell_px = max(round(mid - cfg.offset, 2), entry + 0.01)
                if sell_px > entry and can_fill_sell(sell_px):
                    return sell_px, "whale_exit"
            # Bond fallback
            if mid >= cfg.threshold:
                sell_px = max(round(mid - cfg.offset, 2), cfg.floor)
                if sell_px > entry and can_fill_sell(sell_px):
                    return sell_px, f"whale_bond_high(mid={mid:.3f})"
            return None

        if cfg.profile == "longshot":
            # Sell when mid >= min(entry * mult, ceiling).
            trigger = min(entry * cfg.longshot_multiplier, cfg.longshot_ceiling)
            if mid < trigger:
                return None
            sell_px = max(round(mid - cfg.offset, 2), entry + 0.01)
            if sell_px <= entry:
                return None
            if not can_fill_sell(sell_px):
                return None
            return sell_px, f"longshot_target(entry={entry:.3f},mid={mid:.3f})"

        # Unknown profile
        return None

    def _sell_trade(self, strat, t: dict, sell_price: float, reason: str) -> bool:
        """Mark trade as sold. Dry: synthetic exit; live: subprocess SELL.

        Cost basis = filled_price when known (fill-sim may have filled us at
        a better price than limit). Uses `_cost_basis(t)` helper.
        """
        dry = getattr(strat, "dry_run", True)
        entry = _cost_basis(t)
        shares = float(t.get("shares", 0) or 0)
        profit = round((sell_price - entry) * shares, 4) if shares > 0 else 0.0
        market_short = t.get("market", "")[:40]

        if dry:
            t["status"] = "dry_sold"
            t["sold_at"] = _now_iso()
            t["sold_price"] = sell_price
            t["sold_reason"] = reason
            # Lock realized_pnl on the sell (instead of waiting for resolution).
            t["realized_pnl"] = profit
            # Zero out the trade's unrealized contribution.
            t["unrealized_pnl"] = 0.0
            # Bump strategy counters (call state helper to keep aggregates in sync).
            self._record_realized(strat.name, profit, win=(profit > 0))
            self.log.info(
                f"[PM][sell-dry] {strat.name} {market_short} "
                f"entry={entry:.3f} sold@{sell_price:.3f} shares={shares:.1f} "
                f"profit={profit:+.2f} reason={reason}"
            )
            return True

        # Live sell: subprocess via strategy helper
        # Polymarket min order = 5 shares. Positions below this can't be
        # sold via CLOB regardless of price; they'll auto-resolve at market
        # close (win → $X/share, loss → $0). Skip the SELL attempt to avoid
        # log noise from "Size lower than minimum" rejects.
        if shares < 5.0:
            if not t.get("sell_skip_min_shares_logged"):
                self.log.info(
                    f"[PM][sell-live] {strat.name} {market_short} skip SELL "
                    f"shares={shares:.2f} < 5 (Polymarket min) — "
                    f"will auto-redeem at resolution"
                )
                t["sell_skip_min_shares_logged"] = True
            t["sell_skip_reason"] = "min_shares"
            return False
        # SELL-side rounding fix (Bug-6 v2, 2026-04-25):
        # place_order.py does ceil(usd/price*100)/100 to convert USD → shares.
        # If we send the EXACT round(shares*price), ceil pushes back up to
        # shares+0.01, exceeding actual chain balance. Solution: floor the
        # USD so the round-trip ceil lands exactly on `shares` (not above).
        # Example: shares=6.39 price=0.99 → 6.3261 → floor→6.32 →
        #          ceil(6.32/0.99*100)/100 = ceil(638.38)/100 = 6.39 ✓
        import math as _math
        sell_shares = _math.floor(shares * 100) / 100   # floor to 2dp shares
        sell_usd = _math.floor(sell_shares * sell_price * 100) / 100   # floor USD too
        # Bug-19 (2026-04-27): Polymarket has a $X USD order minimum
        # IN ADDITION to the 5-share minimum. Catastrophe-lock at very
        # low prices can produce sell_usd < $X → CLOB silently rejects
        # but PM marks state="sold" → state-vs-chain divergence.
        # Skip with INFO log + leave position to settle on chain.
        if sell_usd < 1.0:
            if not t.get("sell_skip_min_usd_logged"):
                self.log.info(
                    f"[PM][sell-live] {strat.name} {market_short} skip SELL "
                    f"sell_usd=${sell_usd:.2f} < $X (Polymarket USD floor) — "
                    f"will hold to resolution"
                )
                t["sell_skip_min_usd_logged"] = True
            t["sell_skip_reason"] = "min_usd"
            return False
        try:
            order_id = strat.place_live_sell(
                token_id=t["token_id"], price=sell_price,
                size=sell_usd,
            )
        except AttributeError:
            self.log.error(
                f"[PM][sell-live] {strat.name} has no place_live_sell() — "
                f"cannot sell. Skipping."
            )
            return False
        if order_id:
            t["status"] = "sold"
            t["sold_at"] = _now_iso()
            t["sold_price"] = sell_price
            t["sold_reason"] = reason
            t["sold_order_id"] = order_id
            # NOTE: realized_pnl updates only when the SELL actually fills —
            # for live, we defer to reconcile_open_positions or a follow-up
            # fill check. For now, we optimistically record assuming fill.
            t["realized_pnl"] = profit
            t["unrealized_pnl"] = 0.0
            self._record_realized(strat.name, profit, win=(profit > 0))
            self.log.info(
                f"[PM][sell-live] {strat.name} {market_short} sold@{sell_price:.3f} "
                f"profit={profit:+.2f} order={order_id[:24]}"
            )
            return True
        else:
            t["sell_last_failed_at"] = _now_iso()
            t["sell_fail_count"] = int(t.get("sell_fail_count", 0) or 0) + 1
            err = str(getattr(strat, "last_live_order_error", "") or "")
            if err:
                t["sell_last_error"] = err[:500]
                if "geoblock" in err.lower() or "trading restricted" in err.lower():
                    t["sell_blocked_by_geoblock"] = True
            self.log.warning(
                f"[PM][sell-live] subprocess failed for {strat.name} {market_short}"
            )
            return False

    def _record_realized(self, strategy_name: str, pnl: float, win: bool) -> None:
        """Update strategy aggregate counters after a sell/redeem.

        Keeps `realized_pnl`, `daily_pnl`, `total_pnl`, `wins`/`losses`,
        `open_trades` in sync. Mirrors the logic in `state.update_resolution()`
        so the pathway is equivalent regardless of whether the trade resolved
        via market settlement or via our own sell-high.
        """
        strat = self.state.get_strategy_state(strategy_name)
        strat["realized_pnl"] = round(strat.get("realized_pnl", 0) + pnl, 4)
        strat["daily_pnl"] = round(strat.get("daily_pnl", 0) + pnl, 4)
        strat["total_pnl"] = round(strat.get("realized_pnl", 0) +
                                   strat.get("unrealized_pnl", 0), 4)
        if win:
            strat["wins"] = strat.get("wins", 0) + 1
        else:
            strat["losses"] = strat.get("losses", 0) + 1
        strat["open_trades"] = max(0, strat.get("open_trades", 0) - 1)

    # ── Feature 4: Auto-redeem ──────────────────────────────────

    # ─────────────────────────────────────────────────────────────────
    # Aggregate recompute (Bug-10 fix, 2026-04-25)
    # ─────────────────────────────────────────────────────────────────

    def recompute_strategy_aggregates(self, strategy_name: str) -> dict:
        """Rebuild strategy.{realized_pnl, unrealized_pnl, total_pnl, wins,
        losses, open_trades, daily_pnl} from individual trade rows.

        Why: many code paths increment these fields incrementally
        (record_realized, sell_trade, etc.) and over time they drift from
        truth — particularly unrealized_pnl which doesn't auto-recompute
        when trades transition from open → sold/redeemed. After Bug-9
        backfill we saw 14 strategies with > $X drift in unrealized.

        Idempotent. Run periodically (every reconcile cycle is fine).
        """
        strat = self.state.get_strategy_state(strategy_name)
        trades = strat.get("trades", [])
        today = _now().strftime("%Y-%m-%d")

        realized = 0.0
        unrealized = 0.0
        daily_pnl = 0.0
        wins = 0
        losses = 0
        open_count = 0

        for t in trades:
            st = t.get("status")
            rpnl = t.get("realized_pnl")
            # Realized: any trade with explicit realized_pnl (closed)
            if rpnl is not None:
                try:
                    pnl = float(rpnl)
                except (TypeError, ValueError):
                    pnl = 0.0
                realized += pnl
                if pnl > 0:
                    wins += 1
                elif pnl < 0:
                    losses += 1
                closed_at = (t.get("redeemed_at") or t.get("sold_at") or "")
                if closed_at.startswith(today):
                    daily_pnl += pnl
            # Unrealized: open/filled positions, recomputed from MTM
            if (t.get("outcome") is None
                    and st in ("open", "filled", "dry_open", "dry_filled")):
                try:
                    entry = float(t.get("filled_price") or t.get("price") or 0)
                    mtm = float(t.get("last_mtm_price") or 0)
                    shares = float(t.get("shares") or 0)
                    size = float(t.get("size") or 0)
                except (TypeError, ValueError):
                    entry = mtm = shares = size = 0
                if shares == 0 and entry > 0:
                    shares = size / entry
                if entry > 0 and mtm > 0:
                    unrealized += (mtm - entry) * shares
                open_count += 1

        old_unreal = float(strat.get("unrealized_pnl") or 0)
        strat["realized_pnl"] = round(realized, 4)
        strat["unrealized_pnl"] = round(unrealized, 4)
        strat["total_pnl"] = round(realized + unrealized, 4)
        strat["wins"] = wins
        strat["losses"] = losses
        strat["open_trades"] = open_count
        strat["daily_pnl"] = round(daily_pnl, 4)

        if abs(old_unreal - unrealized) > 0.5:
            self.log.info(
                f"[PM][recompute] {strategy_name} unrealized "
                f"${old_unreal:+.2f}→${unrealized:+.2f} (drift fixed)"
            )

        return {
            "realized": round(realized, 4),
            "unrealized": round(unrealized, 4),
            "wins": wins, "losses": losses, "open": open_count,
            "drift": round(unrealized - old_unreal, 4),
        }

    # ─────────────────────────────────────────────────────────────────
    # Live-mode chain reconciliation (Bug-1 + Bug-3 fix, 2026-04-25)
    # ─────────────────────────────────────────────────────────────────

    def reconcile_chain_positions(self, strategy_name: str,
                                   proxy_address: Optional[str] = None) -> dict:
        """Sync state with on-chain positions via Polymarket data-api.

        Designed to fix two bugs we discovered during first live deployment:

        BUG-1: PositionManager.check_fills only does dry-run sim — live
               orders that fill on CLOB never get reflected in state, so
               check_and_redeem can't find them.

        BUG-3: After a container restart, the local state file has zero record of
               positions placed in earlier sessions, even though they're
               sitting on chain.

        For each on-chain position not properly tracked in state:
          - If matching `cancelled` trade exists (legacy fill bug) →
            upgrade status to `filled` and populate filled_price/shares
          - If matching `open` trade → upgrade to `filled` (got filled
            since last sim check)
          - If no matching trade → create synthetic LEGACY-IMPORT record

        Idempotent: safe to call repeatedly. Returns stats dict.
        """
        proxy = proxy_address or os.environ.get("POLYMARKET_PROXY_ADDRESS", "").strip()
        if not proxy:
            self.log.warning("[PM][reconcile-chain] POLYMARKET_PROXY_ADDRESS not set, skipping")
            return {"error": "no proxy address"}

        try:
            # Fetch chain positions via public data-api (no auth needed).
            # Retry here because a transient TLS/data-api failure would
            # otherwise skip the whole redeem/self-heal pass for this cycle.
            chain_positions = _fetch_data_api_positions(
                proxy, timeout=15, retries=3)
        except Exception as e:
            self.log.error(f"[PM][reconcile-chain] data-api fetch failed: {e}")
            return {"error": str(e), "imported": 0, "upgraded": 0}

        if not chain_positions:
            self.log.info(f"[PM][reconcile-chain] {strategy_name}: 0 chain positions, no-op")
            return {"imported": 0, "upgraded": 0, "total_chain": 0}

        strat_state = self.state.get_strategy_state(strategy_name)
        trades = strat_state.setdefault("trades", [])
        # Index trades by token_id (one position per token_id on Polymarket)
        by_token: dict[str, list[dict]] = {}
        for t in trades:
            tid = str(t.get("token_id") or "")
            if tid:
                by_token.setdefault(tid, []).append(t)

        imported = 0
        upgraded = 0
        already_synced = 0

        for pos in chain_positions:
            token_id = str(pos.get("asset", ""))
            if not token_id:
                continue

            chain_size = float(pos.get("size", 0) or 0)
            chain_cost = float(pos.get("initialValue", 0) or 0)
            chain_value = float(pos.get("currentValue", 0) or 0)
            avg_price = chain_cost / chain_size if chain_size > 0 and chain_cost > 0 else 0.0
            mtm_price = chain_value / chain_size if chain_size > 0 else 0.0
            outcome_label_raw = pos.get("outcome", "") or ""
            outcome_idx = pos.get("outcomeIndex", 0)
            # Bug-28 (2026-04-28): data-api returns "Yes"/"No" Title Case
            # (and category-specific labels like "Up"/"Down"/team names for
            # non-binary markets). Downstream code requires canonical
            # uppercase "YES"/"NO":
            #   - state.update_resolution skips trades where side_label not
            #     in ("YES","NO") → trades stay outcome=None forever
            #   - _redeem_trade computes outcome_index by exact "YES" check
            #     → wrong index → on-chain revert + lost gas
            # Normalize at the boundary AND map non-binary outcomes to
            # canonical labels by outcomeIndex (idx 0 → YES, idx 1 → NO).
            _label_upper = outcome_label_raw.strip().upper()
            if _label_upper in ("YES", "NO"):
                outcome_label = _label_upper
            else:
                # Non-binary outcome (e.g., "Up","Down","Team WE"). Map by
                # index so PM redeem logic still picks the right token side.
                outcome_label = "YES" if int(outcome_idx) == 0 else "NO"
            cid = pos.get("conditionId", "")
            redeemable = bool(pos.get("redeemable", False))
            negative_risk = bool(pos.get("negativeRisk", False))
            realized_chain = float(pos.get("realizedPnl", 0) or 0)

            matching = by_token.get(token_id, [])
            # Bug-29 (2026-04-28): split "we know about it" trades into
            # (still-holding, terminal). Old code refreshed unrealized_pnl
            # even on terminal trades, which produced phantom MTM on
            # closed positions when data-api still surfaced them for ~24h.
            # It also missed auto_redeemed/redeem_failed/cancel_failed (new
            # terminal statuses), causing those to fall to the synthesize
            # branch and create duplicate trade records.
            still_holding = [t for t in matching
                             if t.get("status") in ("filled", "dry_filled")]
            # A cancelled GTC can still have filled before the cancel landed
            # (or partially filled, then cancelled). If data-api says we hold
            # the token, the chain is authoritative: upgrade it to filled
            # before the generic terminal-status guard below.
            candidates = [t for t in matching
                          if t.get("status") in ("cancelled", "open", "dry_open",
                                                   "dry_cancelled",
                                                   "redeem_failed", "auto_redeemed",
                                                   "redeemed")
                          # Some legacy rows were marked sold without sold_at
                          # even though data-api still shows a redeemable dust
                          # position. Treat the chain as authoritative so the
                          # redeem cleanup path can burn/close the dust.
                          or (
                              t.get("status") == "sold"
                              and redeemable
                              and not t.get("sold_at")
                          )]
            candidates.sort(key=lambda t: t.get("time", ""), reverse=True)
            terminal_seen = [t for t in matching
                             if t.get("status") in TERMINAL_STATUSES]
            if still_holding:
                t = still_holding[0]
                if chain_cost <= 0 and chain_size > 0:
                    chain_cost = _fallback_chain_cost(t, chain_size)
                    avg_price = chain_cost / chain_size if chain_cost > 0 else 0.0
                md = t.setdefault("metadata", {})
                md["negativeRisk"] = negative_risk
                md["outcome_index"] = outcome_idx
                t["market_question"] = (
                    t.get("market_question")
                    or t.get("market")
                    or pos.get("title", "")
                )
                t["market"] = t.get("market") or t.get("market_question") or pos.get("title", "")
                t["size"] = round(chain_cost, 2)
                t["price"] = round(avg_price, 4)
                t["filled_price"] = round(avg_price, 4)
                t["shares"] = round(chain_size, 6)
                t["last_mtm_price"] = round(mtm_price, 4)
                t["last_mtm_time"] = _now_iso()
                t["unrealized_pnl"] = round(chain_value - chain_cost, 4)
                existing_sl = t.get("side_label", "")
                if existing_sl not in ("YES", "NO"):
                    t["side_label"] = outcome_label
                resolved_from_chain = False
                if redeemable and chain_value < 0.01:
                    resolved_from_chain = t.get("outcome") != "LOSS"
                    t["outcome"] = "LOSS"
                    if resolved_from_chain or not t.get("resolved_at"):
                        t["resolved_at"] = t.get("resolved_at") or _now_iso()
                elif redeemable and chain_value > chain_size * 0.99:
                    resolved_from_chain = t.get("outcome") != "WIN"
                    t["outcome"] = "WIN"
                    if resolved_from_chain or not t.get("resolved_at"):
                        t["resolved_at"] = t.get("resolved_at") or _now_iso()
                if resolved_from_chain:
                    self.log.info(
                        f"[PM][reconcile-chain] {strategy_name} marked "
                        f"filled→resolved: cid={cid[:20]}.. "
                        f"outcome={t.get('outcome')} mtm={mtm_price:.3f} "
                        f"[REDEEMABLE]"
                    )
                    upgraded += 1
                else:
                    already_synced += 1
                continue
            if candidates:
                t = candidates[0]
                if chain_cost <= 0 and chain_size > 0:
                    chain_cost = _fallback_chain_cost(t, chain_size)
                    avg_price = chain_cost / chain_size if chain_cost > 0 else 0.0
                old_status = t.get("status")
                prior_resolved_at = (
                    t.get("resolved_at")
                    or t.get("redeemed_at")
                    or t.get("sold_at")
                    or t.get("time")
                    or t.get("placed_at")
                    or _now_iso()
                )
                t["status"] = "filled"
                t["filled_at"] = t.get("filled_at") or t.get("time") or _now_iso()
                if old_status in ("redeem_failed", "auto_redeemed", "redeemed"):
                    t.pop("redeem_fail_count", None)
                    t.pop("redeemed_at", None)
                t["market_question"] = (
                    t.get("market_question")
                    or t.get("market")
                    or pos.get("title", "")
                )
                t["market"] = t.get("market") or t.get("market_question") or pos.get("title", "")
                t["size"] = round(chain_cost, 2)
                t["price"] = round(avg_price, 4)
                t["filled_price"] = round(avg_price, 4)
                t["shares"] = round(chain_size, 6)
                t["last_mtm_price"] = round(mtm_price, 4)
                t["last_mtm_time"] = _now_iso()
                # Bug-28: never overwrite an already-canonical side_label.
                # Only set if missing/non-canonical; the normalized
                # outcome_label above is always "YES" or "NO".
                existing_sl = t.get("side_label", "")
                if existing_sl not in ("YES", "NO"):
                    t["side_label"] = outcome_label
                t["unrealized_pnl"] = round(chain_value - chain_cost, 4)
                # Bug-18 (2026-04-27): backfill event_slug if missing so
                # v2's _has_open_on_event dedup keeps working after restart.
                # Without this, every container restart re-imports legacy
                # filled positions but strips dedup metadata, re-creating
                # the multi-bucket trap exposure for the next scan cycle.
                md = t.setdefault("metadata", {})
                md["negativeRisk"] = negative_risk
                md["outcome_index"] = outcome_idx
                if not md.get("event_slug"):
                    try:
                        from alpha_suite.strategies.bond_buyer import derive_event_key
                        md["event_slug"] = derive_event_key(t.get("market", "") or pos.get("title", ""))
                    except Exception:
                        pass
                # If chain marked redeemable, set outcome (PM redeem will pick up)
                if redeemable and chain_value < 0.01:
                    t["outcome"] = "LOSS"
                    t["resolved_at"] = prior_resolved_at
                elif redeemable and chain_value > chain_size * 0.99:
                    t["outcome"] = "WIN"
                    t["resolved_at"] = prior_resolved_at
                self.log.info(
                    f"[PM][reconcile-chain] {strategy_name} upgraded "
                    f"{old_status}→filled: cid={cid[:20]}.. "
                    f"entry={avg_price:.3f} shares={chain_size:.2f} "
                    f"mtm={mtm_price:.3f}{' [REDEEMABLE]' if redeemable else ''}"
                )
                upgraded += 1
            elif terminal_seen:
                # Closed trade still surfaced by data-api lag — leave alone.
                # Don't overwrite unrealized (should be 0 on terminal),
                # don't synthesize a duplicate.
                already_synced += 1
            else:
                # No state record — synthesize one (legacy import)
                now = _now_iso()
                new_trade = {
                    "time": now,
                    "placed_at": now,
                    "filled_at": now,
                    "status": "filled",
                    "order_type": "GTC",
                    "market": (pos.get("title", "") or "")[:80],
                    "market_question": (pos.get("title", "") or "")[:80],
                    "condition_id": cid,
                    "token_id": token_id,
                    "side": "BUY",
                    "side_label": outcome_label,
                    "price": round(avg_price, 4),
                    "filled_price": round(avg_price, 4),
                    "size": round(chain_cost, 2),
                    "shares": round(chain_size, 6),
                    "p_fair": 0,
                    "edge": 0,
                    "ev": 0,
                    "order_id": f"LEGACY-IMPORT-{token_id[:12]}",
                    "dry_run": False,
                    "success": True,
                    "error": "",
                    "metadata": {
                        "imported_from_chain": True,
                        "outcome_index": outcome_idx,
                        "negativeRisk": negative_risk,
                        # Bug-18: synthetic event_slug so dedup works on
                        # legacy-imported positions immediately.
                        "event_slug": _derive_event_key_safe(pos.get("title", "")),
                    },
                    "outcome": ("WIN" if redeemable and chain_value > chain_size * 0.99
                                 else "LOSS" if redeemable and chain_value < 0.01
                                 else None),
                    "realized_pnl": None,
                    "unrealized_pnl": round(chain_value - chain_cost, 4),
                    "last_mtm_price": round(mtm_price, 4),
                    "last_mtm_time": now,
                    "resolved_at": now if redeemable else "",
                }
                trades.append(new_trade)
                self.log.info(
                    f"[PM][reconcile-chain] {strategy_name} imported new: "
                    f"cid={cid[:20]}.. entry={avg_price:.3f} shares={chain_size:.2f}"
                )
                imported += 1

        chain_held_tokens = {str(p.get("asset", "")) for p in chain_positions
                             if str(p.get("asset", ""))}

        # Bug-39 (2026-05-03): self-heal filled winners that disappear from
        # data-api before local state sees an explicit resolution. Polymarket
        # can auto-redeem or collapse a 99c winning token off the positions API;
        # leaving it as filled keeps phantom positive MTM in Open Positions.
        # Only close clear winners (last_mtm_price >= 0.99) and only when the
        # token is absent from a successfully fetched chain snapshot.
        missing_winner_closed = 0
        for t in trades:
            if t.get("status") != "filled":
                continue
            if t.get("outcome") is not None:
                continue
            token_id = str(t.get("token_id") or "")
            if not token_id or token_id in chain_held_tokens:
                continue
            try:
                mtm_price = float(t.get("last_mtm_price") or 0)
            except (TypeError, ValueError):
                mtm_price = 0.0
            if mtm_price < 0.99:
                continue
            entry = _cost_basis(t)
            shares = float(t.get("shares", 0) or 0)
            pnl = round((1.0 - entry) * shares, 4) if shares > 0 else 0.0
            if t.get("realized_pnl") is None:
                t["realized_pnl"] = pnl
                self._record_realized(strategy_name, pnl, win=True)
            t["status"] = "auto_redeemed"
            t["outcome"] = "WIN"
            t["resolved_at"] = t.get("resolved_at") or _now_iso()
            t["redeemed_at"] = _now_iso()
            t["unrealized_pnl"] = 0.0
            missing_winner_closed += 1
            self.log.info(
                f"[PM][reconcile-chain] {strategy_name} auto-closed missing "
                f"winner: token={token_id[:16]} mtm={mtm_price:.3f} "
                f"realized={pnl:+.2f} — {(t.get('market') or '')[:50]}"
            )

        # Bug-30 (2026-04-28): self-heal phantom-sold trades. _sell_trade
        # marks status="sold" the moment CLOB returns an order_id, but
        # GTC orders don't always fill. If proxy still holds the token
        # >5min after sold_at, the SELL is stuck on the book — revert
        # status to "filled" so check_and_redeem and update_resolution
        # can still settle it correctly. Also clears wrongly-credited
        # realized_pnl; recompute_strategy_aggregates fixes totals.
        auto_closed = 0
        for t in trades:
            if t.get("outcome") != "WIN":
                continue
            if t.get("status") not in ("filled", "redeem_failed"):
                continue
            token_id = str(t.get("token_id") or "")
            if not token_id or token_id in chain_held_tokens:
                continue
            entry = _cost_basis(t)
            shares = float(t.get("shares", 0) or 0)
            pnl = round((1.0 - entry) * shares, 4) if shares > 0 else 0.0
            if t.get("realized_pnl") is None:
                t["realized_pnl"] = pnl
                self._record_realized(strategy_name, pnl, win=True)
            t["status"] = "auto_redeemed"
            t["redeemed_at"] = _now_iso()
            t["unrealized_pnl"] = 0.0
            t.pop("redeem_fail_count", None)
            auto_closed += 1

        reverted = 0
        for t in trades:
            if t.get("status") != "sold":
                continue
            if not t.get("token_id"):
                continue
            if str(t["token_id"]) not in chain_held_tokens:
                continue   # not on chain anymore = SELL really did fill
            sold_at = t.get("sold_at", "")
            if not sold_at:
                continue
            try:
                sold_dt = datetime.fromisoformat(sold_at.replace("Z", "+00:00"))
                age_sec = (datetime.now(timezone.utc) - sold_dt).total_seconds()
            except Exception:
                continue
            if age_sec < 300:
                continue   # under 5min — give CLOB time to match
            # Phantom: status=sold but proxy still has the shares
            stale_order_id = t.get("sold_order_id")
            self.log.warning(
                f"[PM][reconcile-chain] {strategy_name} REVERT phantom-sold: "
                f"token={t['token_id'][:16]} sold_at={sold_at[:19]} "
                f"age={age_sec/60:.0f}min — order resting unfilled. "
                f"Status sold→filled, clearing wrongly-credited "
                f"realized_pnl=${t.get('realized_pnl', 0)}. "
                f"Stale order {stale_order_id[:24] if stale_order_id else '?'} "
                f"will be cancelled by cancel_stale TTL=12h."
            )
            t["status"] = "filled"
            t["realized_pnl"] = None
            # Bug-33 (2026-04-28): track last revert time so the next
            # _process_resting_tp call applies a 30min cooldown — without
            # it, the same TP trigger fires immediately and stacks
            # another GTC SELL on top of the (still resting) old one.
            # The old order eventually clears via cancel_stale (12h TTL).
            t["sell_last_revert_at"] = _now_iso()
            t.pop("sold_at", None)
            t.pop("sold_price", None)
            t.pop("sold_reason", None)
            t.pop("sold_order_id", None)
            reverted += 1

        if imported or upgraded or reverted or auto_closed or missing_winner_closed:
            self.state.save()
        if reverted or auto_closed or missing_winner_closed:
            # Aggregator drifted; recompute will fix totals on next call
            self.recompute_strategy_aggregates(strategy_name)

        result = {
            "imported": imported,
            "upgraded": upgraded,
            "already_synced": already_synced,
            "phantom_reverted": reverted,
            "auto_closed": auto_closed,
            "missing_winner_closed": missing_winner_closed,
            "total_chain": len(chain_positions),
            "total_state_trades": len(trades),
        }
        self.log.info(
            f"[PM][reconcile-chain] {strategy_name} done: "
            f"imported={imported} upgraded={upgraded} synced={already_synced}"
            f"{' reverted='+str(reverted) if reverted else ''}"
            f"{' auto_closed='+str(auto_closed) if auto_closed else ''}"
            f"{' missing_winner_closed='+str(missing_winner_closed) if missing_winner_closed else ''}"
        )
        return result

    def check_and_redeem(self, strategies: list) -> int:
        """For each resolved trade still in holding status, call redeemPositions.

        In dry-run: marks status=dry_redeemed and locks realized_pnl based on
        outcome. In live: shells out to the configured redeem_position.py helper.

        Bug-17 (2026-04-27): live mode rate-limited to ONE on-chain TX
        per call. The signer EOA has a single nonce; submitting multiple
        redeem TXes back-to-back makes them all collide with "replacement
        transaction underpriced" except the first. Process one per cycle
        and let the tx confirm (~30s on Polygon) before submitting the
        next. Dry-run path is unbatched (synchronous, no nonce). LOSS
        redemptions don't send TXes either, so they all process freely.

        Bug-23 (2026-04-27): pre-flight via Polymarket data-api. If the
        proxy doesn't currently hold the cid, Polymarket already auto-
        redeemed it (or it never settled to our proxy). Sending a redeem
        TX would revert with "insufficient balance" and burn gas. Mark
        auto_redeemed and credit expected pnl without consuming the
        live-TX budget.
        """
        if not self.settings.redeem_enabled:
            return 0

        redeemed = 0
        live_tx_done = False  # at most one live on-chain redeem per cycle
        # Bug-23: clear cache at top so each check_and_redeem cycle starts
        # with a fresh data-api snapshot (TTL 60s for within-cycle reuse).
        self._held_cids_cache = None
        for strat in strategies:
            name = strat.name
            strat_state = self.state.get_strategy_state(name)
            for t in strat_state.get("trades", []):
                if t.get("outcome") not in ("WIN", "LOSS"):
                    continue
                status = t.get("status")
                if status in ("dry_redeemed", "redeemed", "redeem_failed",
                              "auto_redeemed",
                              "dry_sold", "sold", "dry_cancelled", "cancelled"):
                    continue
                is_live_win = (not getattr(strat, "dry_run", True)
                               and t.get("outcome") == "WIN")
                submits_live_tx = (
                    not getattr(strat, "dry_run", True)
                    and (
                        t.get("outcome") == "WIN"
                        or (
                            t.get("outcome") == "LOSS"
                            and self.settings.redeem_losses_onchain
                        )
                    )
                )
                # Bug-23 pre-flight: if proxy no longer holds this cid,
                # short-circuit to auto_redeemed. No on-chain TX, no
                # live-TX budget consumed.
                if is_live_win and not self._chain_position_still_held(t.get("condition_id", "")):
                    self._mark_auto_redeemed(strat, t)
                    redeemed += 1
                    continue
                # Live + WIN = will spawn an on-chain TX. Skip if we've
                # already submitted one this cycle (avoid nonce collision).
                # Bug-22 (2026-04-27): MUST lock on ATTEMPT, not success.
                # If subprocess times out (>120s) or returns rc!=0 with
                # a partial TX submission, the next iteration would
                # collide on nonce → 267 errors all over again. Lock
                # before _redeem_trade returns either way.
                if submits_live_tx and live_tx_done:
                    continue
                if submits_live_tx:
                    live_tx_done = True   # set BEFORE attempt
                if self._redeem_trade(strat, t):
                    redeemed += 1
        if redeemed:
            self.state.save()
        return redeemed

    def _chain_position_still_held(self, condition_id: str) -> bool:
        """Bug-23: returns True iff Polymarket data-api still lists the
        proxy as holding the position. Returns True (fail-open) on API
        error so we fall through to existing subprocess path. Caches the
        per-cycle response so iterating many trades only fetches once."""
        if not condition_id:
            return True
        proxy = os.environ.get("POLYMARKET_PROXY_ADDRESS", "").strip()
        if not proxy:
            return True

        cache = getattr(self, "_held_cids_cache", None)
        now = time.time()
        if cache and now - cache["fetched_at"] < 60:
            return condition_id.lower() in cache["cids"]

        try:
            # Bug-31 (2026-04-28): use sizeThreshold=0.0001 (filter zero-
            # balance historical positions) so the 500 cap covers ACTIVE
            # holdings only. Without this, an account with >500 lifetime
            # positions could have a still-held cid pushed off page 1.
            positions = _fetch_data_api_positions(
                proxy, size_threshold=0.0001, limit=500, timeout=10, retries=3)
        except Exception as e:
            self.log.warning(
                f"[PM][redeem-live] data-api preflight failed: {e} "
                f"— falling through to subprocess (existing behavior)")
            return True   # fail-open

        # Bug-31: warn if we hit the cap; auto-redeem decisions become
        # unsafe past 500 active holdings (we'd false-negative).
        if isinstance(positions, list) and len(positions) >= 500:
            self.log.error(
                f"[PM][redeem-live] data-api returned {len(positions)} "
                f"positions (cap=500). Some held cids may be invisible. "
                f"Failing open to subprocess to avoid false auto_redeemed.")
            return True   # fail-open

        cids = {(p.get("conditionId") or "").lower()
                for p in (positions or [])
                if (p.get("conditionId") or "")}
        self._held_cids_cache = {"fetched_at": now, "cids": cids}
        return condition_id.lower() in cids

    def _mark_auto_redeemed(self, strat, t: dict) -> None:
        """Bug-23: trade is WIN but proxy doesn't hold the position. Either
        Polymarket's UI auto-redeemed (USDC already received) or the order
        never actually settled to our proxy in the first place. Either way:
        no on-chain TX is needed. Credit the expected WIN payout
        (1.0 - entry) × shares as realized pnl and mark terminal."""
        entry = _cost_basis(t)
        shares = float(t.get("shares", 0) or 0)
        pnl = round((1.0 - entry) * shares, 4) if shares > 0 else 0.0
        market_short = (t.get("market") or "")[:40]
        already_counted = t.get("realized_pnl") is not None
        if not already_counted:
            t["realized_pnl"] = pnl
            self._record_realized(strat.name, pnl, win=True)
        t["status"] = "auto_redeemed"
        t["redeemed_at"] = _now_iso()
        t["unrealized_pnl"] = 0.0
        self.log.info(
            f"[PM][redeem-live] {strat.name} {market_short} outcome=WIN "
            f"AUTO-REDEEMED (not in data-api positions). "
            f"Credited realized={pnl:+.2f}, no on-chain TX."
        )

    def _redeem_trade(self, strat, t: dict) -> bool:
        """Redeem a single resolved trade. Returns True on success/dry-mark.

        Uses `_cost_basis(t)` for the PnL calculation so redemption PnL
        matches what was actually paid for the shares (filled_price),
        not the limit price.
        """
        dry = getattr(strat, "dry_run", True)
        outcome = t.get("outcome")
        entry = _cost_basis(t)
        shares = float(t.get("shares", 0) or 0)
        side_label = t.get("side_label", "")
        cid = t.get("condition_id", "")
        # Payout per share:
        #   WIN → 1.0 (we held the winning side)
        #   LOSS → 0.0 (our side lost)
        payout_per_share = 1.0 if outcome == "WIN" else 0.0
        pnl = round((payout_per_share - entry) * shares, 4) if shares > 0 else 0.0
        market_short = t.get("market", "")[:40]

        if dry:
            t["status"] = "dry_redeemed"
            t["redeemed_at"] = _now_iso()
            # Don't double-count realized_pnl if state.update_resolution already
            # recorded it at settlement time. Check existing trade realized_pnl:
            already_counted = t.get("realized_pnl") is not None
            if not already_counted:
                t["realized_pnl"] = pnl
                self._record_realized(strat.name, pnl, win=(outcome == "WIN"))
            self.log.info(
                f"[PM][redeem-dry] {strat.name} {market_short} "
                f"outcome={outcome} entry={entry:.3f} pnl={pnl:+.2f}"
            )
            return True

        # Live redeem: subprocess
        if not self.settings.redeem_script_path or not self.settings.redeem_venv_python:
            self.log.error(
                "[PM][redeem-live] redeem_script_path or redeem_venv_python "
                "not set in settings — cannot redeem"
            )
            return False

        # Determine outcome_index: 0 if we held YES/tokens[0], 1 if NO/tokens[1].
        # For WIN this is the winning side. For LOSS with on-chain cleanup
        # enabled, this is the losing token side to burn from the proxy wallet.
        outcome_index = 0 if side_label == "YES" else 1
        if outcome == "LOSS" and not self.settings.redeem_losses_onchain:
            # LOSS — skip on-chain redeem to save gas ($X payout) BUT
            # MUST still record the realized loss in state. Bug-9 (2026-04-25):
            # the original LOSS branch only set status+redeemed_at and forgot
            # both `realized_pnl` and `_record_realized()`, leaving the strategy
            # aggregator showing $X lost when wallet had real -$X / position
            # losses. This made the dashboard understate losses by tens of
            # dollars.
            already_counted = t.get("realized_pnl") is not None
            if not already_counted:
                t["realized_pnl"] = pnl   # pnl is negative for LOSS
                self._record_realized(strat.name, pnl, win=False)
            t["status"] = "redeemed"  # mark so we don't retry
            t["redeemed_at"] = _now_iso()
            t["unrealized_pnl"] = 0.0   # crystallized into realized
            self.log.info(
                f"[PM][redeem-live] {strat.name} {market_short} outcome=LOSS "
                f"realized={pnl:+.2f} (skip on-chain redeem, $X payout)"
            )
            return True

        # Bug-15 fix (2026-04-27): script's actual flags are
        #   --asset-token-id  (NOT --token-id)
        #   --neg-risk        (boolean flag, NOT "--is-neg-risk true")
        # The wrong flags caused argparse rc=2 → 989 retries spamming logs.
        cmd = [
            self.settings.redeem_venv_python,
            self.settings.redeem_script_path,
            "--condition-id", cid,
            "--outcome-index", str(outcome_index),
            "--size", str(shares),
        ]
        if t.get("token_id"):
            cmd.extend(["--asset-token-id", str(t["token_id"])])
        md = t.get("metadata") or {}
        if md.get("negativeRisk") is False:
            cmd.append("--no-neg-risk")
        else:
            cmd.append("--neg-risk")

        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except Exception as e:
            self.log.error(f"[PM][redeem-live] subprocess failed: {e}")
            return False

        ok = r.returncode == 0
        if ok:
            already_counted = t.get("realized_pnl") is not None
            if not already_counted:
                t["realized_pnl"] = pnl
                self._record_realized(strat.name, pnl, win=(outcome == "WIN"))
            t["status"] = "redeemed"
            t["redeemed_at"] = _now_iso()
            t["unrealized_pnl"] = 0.0
            self.log.info(
                f"[PM][redeem-live] {strat.name} {market_short} "
                f"outcome={outcome} pnl={pnl:+.2f}"
            )
        else:
            # Bug-16 (2026-04-27): also log stdout — redeem_position.py
            # writes JSON errors to stdout, not stderr. Without this, rc=1
            # failures looked identical (empty stderr) and were impossible
            # to diagnose without manual reproduction.
            stdout_excerpt = (r.stdout or "")[:300]
            stderr_excerpt = (r.stderr or "")[:200]
            self.log.warning(
                f"[PM][redeem-live] {strat.name} {market_short} "
                f"rc={r.returncode} "
                f"stdout={stdout_excerpt!r} stderr={stderr_excerpt!r}"
            )
            # Backoff: don't retry an unredeemable trade every cycle.
            # Track failure count and after 5 fails, mark as "redeem_failed"
            # so we stop spamming logs.
            t["redeem_fail_count"] = int(t.get("redeem_fail_count", 0) or 0) + 1
            if t["redeem_fail_count"] >= 5:
                t["status"] = "redeem_failed"
                t["redeemed_at"] = _now_iso()
                self.log.error(
                    f"[PM][redeem-live] {strat.name} {market_short} "
                    f"GIVING UP after {t['redeem_fail_count']} failures. "
                    f"Marked status=redeem_failed; manual redeem required."
                )
        return ok

    # ── Utility: mark legacy trades with placed_at/status ───────

    def backfill_legacy_trades(self, strategies: list) -> int:
        """One-time pass: give legacy trades (before status/placed_at existed)
        a reasonable default so new lifecycle code doesn't choke on them.

        Rules:
          * trades with outcome in (WIN, LOSS) → status="dry_redeemed" (dry) or
            "redeemed" (live), since they've already been settled historically.
          * trades with outcome=None and no status → status="dry_filled"
            (legacy behavior was instant fill) and placed_at=time.
        """
        touched = 0
        for strat in strategies:
            dry = getattr(strat, "dry_run", True)
            name = strat.name
            strat_state = self.state.get_strategy_state(name)
            for t in strat_state.get("trades", []):
                if t.get("placed_at") and t.get("status"):
                    continue
                touched += 1
                if not t.get("placed_at"):
                    t["placed_at"] = t.get("time", _now_iso())
                if not t.get("status"):
                    if t.get("outcome") in ("WIN", "LOSS"):
                        t["status"] = "dry_redeemed" if dry else "redeemed"
                        if not t.get("redeemed_at"):
                            t["redeemed_at"] = t.get("time", _now_iso())
                    else:
                        t["status"] = "dry_filled" if dry else "filled"
                        if not t.get("filled_at"):
                            t["filled_at"] = t.get("time", _now_iso())
                        if not t.get("filled_price"):
                            t["filled_price"] = float(t.get("price", 0) or 0)
        if touched:
            self.state.save()
            self.log.info(f"[PM][backfill] normalized {touched} legacy trades")
        return touched


# ──────────────────────────────────────────────────────────────────────
# Module-level helpers (Bug-18, 2026-04-27)
# ──────────────────────────────────────────────────────────────────────

def _derive_event_key_safe(question: str) -> str:
    """Wrapper around bond_buyer.derive_event_key with import guard.

    Used by reconcile_chain_positions to populate event_slug on legacy
    chain-imported trades — without this, every container restart
    re-creates Bug-13's multi-bucket dedup hole.

    Lazy import to avoid circular dependency at module-load time
    (bond_buyer imports from position_manager).
    """
    try:
        from alpha_suite.strategies.bond_buyer import derive_event_key
        return derive_event_key(question or "")
    except Exception:
        return ""
