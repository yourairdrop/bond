"""
Alpha Suite — Unified State Manager.

Manages persistent JSON state for all strategies. Provides atomic writes,
daily resets, trade recording, signal dedup, and dashboard data export.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Optional


# Default strategy names
STRATEGY_NAMES = ["arb", "bond", "whale", "signal", "latency", "coverage", "longshot"]

# Trim limits
MAX_TRADES_PER_STRATEGY = 500
# Equity curve points — at ~1 point per MTM cycle (~30s) this covers
# roughly 40 hours of history. Bumped 500→5000 (2026-04-28) so the
# dashboard can show a full multi-day equity curve and let users
# select any time window.
MAX_EQUITY_CURVE = 5000

# Statuses that mark a trade as closed out by PositionManager. Any code
# that iterates open positions (check_bankroll, update_unrealized,
# update_resolution, per-strategy cap checks in evaluate()) must exclude
# these — otherwise freed capital gets double-counted as still-open.
#
# Defined here (not in position_manager.py) so non-PM callers like
# check_bankroll don't need to import PM. PM re-exports this via
# `from .state import TERMINAL_STATUSES`.
TERMINAL_STATUSES = frozenset({
    "dry_sold",      "sold",
    "dry_cancelled", "cancelled",
    "dry_redeemed",  "redeemed",
    "dry_expired",   "expired",
    "dry_rejected",  "rejected",
    "cancel_failed",
    # Bug-23 (2026-04-27): position no longer held on-chain (Polymarket
    # auto-redeemed via UI sweeper, or never settled to proxy). Treated
    # terminal so MTM and open-cost stop counting it.
    "auto_redeemed",
    # redeem_failed already skipped in check_and_redeem; promoting to
    # terminal here keeps state.update_mtm and bankroll budgets consistent.
    "redeem_failed",
})


def _now_iso() -> str:
    """Current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _today_str() -> str:
    """Current UTC date as YYYY-MM-DD string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _parse_iso_ts(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _trade_ts(trade: dict, *keys: str) -> Optional[datetime]:
    for key in keys:
        dt = _parse_iso_ts(trade.get(key, ""))
        if dt:
            return dt
    return None


def _repair_trade_shape(trade: dict) -> None:
    """Repair legacy trade rows with missing display or cost-basis fields."""
    if not trade.get("market_question") and trade.get("market"):
        trade["market_question"] = trade.get("market")

    meta = trade.get("metadata") or {}
    try:
        shares = float(trade.get("shares") or 0.0)
    except (TypeError, ValueError):
        shares = 0.0
    try:
        stored_price = float(trade.get("price") or 0.0)
    except (TypeError, ValueError):
        stored_price = 0.0
    try:
        stored_filled = float(trade.get("filled_price") or 0.0)
    except (TypeError, ValueError):
        stored_filled = 0.0
    # Historical partial fills can retain requested USD notional while shares
    # reflect only the filled subset. That produces impossible binary prices
    # (>1.0) and corrupts realized/unrealized PnL. If we still have any sane
    # local price hint, restore cost basis from that hint × shares.
    if shares > 0 and (stored_price > 1.0 or stored_filled > 1.0):
        sane_entry = 0.0
        for candidate in (
            stored_filled if 0.0 < stored_filled <= 1.0 else 0.0,
            stored_price if 0.0 < stored_price <= 1.0 else 0.0,
            float(meta.get("vwap_price") or 0.0) if meta.get("vwap_price") is not None else 0.0,
        ):
            if 0.0 < candidate <= 1.0:
                sane_entry = candidate
                break
        if sane_entry > 0:
            trade["filled_price"] = round(sane_entry, 4)
            trade["price"] = round(sane_entry, 4)
            trade["size"] = round(sane_entry * shares, 2)

    status = str(trade.get("status") or "")
    if status not in {"open", "filled", "dry_open", "dry_filled"}:
        return
    if shares <= 0:
        return

    try:
        size = float(trade.get("size") or 0.0)
    except (TypeError, ValueError):
        size = 0.0
    if size <= 0:
        try:
            size = float(meta.get("sized_usd") or 0.0)
        except (TypeError, ValueError):
            size = 0.0
        if size > 0:
            trade["size"] = round(size, 2)

    try:
        entry = float(trade.get("filled_price") or 0.0)
    except (TypeError, ValueError):
        entry = 0.0
    if entry <= 0:
        try:
            entry = float(trade.get("price") or 0.0)
        except (TypeError, ValueError):
            entry = 0.0
    if entry <= 0 and size > 0 and shares > 0:
        entry = size / shares
    if entry > 0:
        trade["filled_price"] = round(entry, 4)
        if float(trade.get("price") or 0.0) <= 0:
            trade["price"] = round(entry, 4)

    try:
        mtm = float(trade.get("last_mtm_price") or 0.0)
    except (TypeError, ValueError):
        mtm = 0.0
    if mtm > 0 and size > 0 and shares > 0:
        trade["unrealized_pnl"] = round(mtm * shares - size, 4)


def _default_strategy_state() -> dict:
    """Default state block for a single strategy.

    PnL accounting (post-forensic-fix 2026-04-14):
      - `realized_pnl`: only from resolved markets. Never bumped at trade time.
      - `unrealized_pnl`: sum of mark-to-market across open positions.
      - `total_pnl`: realized + unrealized. Maintained for dashboard compat.
      - `wins`/`losses`: **resolved trades only**. A trade that is still open
        counts as neither.
    """
    return {
        "trades": [],
        "signals_today": 0,
        "daily_pnl": 0.0,          # realized-only, resets daily
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "total_pnl": 0.0,          # = realized + unrealized
        "total_trades": 0,
        "wins": 0,                 # resolved wins only
        "losses": 0,               # resolved losses only
        "open_trades": 0,
        "seen_signals": [],
        "last_run": "",
    }


def _default_state() -> dict:
    """Full default state structure."""
    return {
        "version": 2,
        "started_at": _now_iso(),
        "strategies": {name: _default_strategy_state() for name in STRATEGY_NAMES},
        "global": {
            "total_pnl": 0.0,
            "total_trades": 0,
            "daily_date": _today_str(),
            "daily_pnl": 0.0,
            "equity_curve": [],
            "cycle_num": 0,
        },
    }


class StateManager:
    """Unified state manager for all Alpha Suite strategies.

    Persists state to a JSON file with atomic writes (tmp + rename).
    Tracks trades, signals, P&L, and equity curve per strategy and globally.
    """

    def __init__(self, state_dir: str = "/app/state"):
        self.state_dir = state_dir
        self.path = os.path.join(state_dir, "alpha_suite_state.json")
        self.state = self._load()

    def _load(self) -> dict:
        """Load state from disk, returning defaults if missing or corrupt."""
        try:
            with open(self.path, "r") as f:
                data = json.load(f)

            # Ensure all expected keys exist (forward-compat)
            if "version" not in data or "strategies" not in data or "global" not in data:
                return _default_state()

            # Ensure all strategy slots exist
            for name in STRATEGY_NAMES:
                if name not in data["strategies"]:
                    data["strategies"][name] = _default_strategy_state()

            # Ensure global keys exist
            defaults_global = _default_state()["global"]
            for key, val in defaults_global.items():
                if key not in data["global"]:
                    data["global"][key] = val

            # Bug-28 (2026-04-28): canonical side_label across all trades.
            # data-api returns "Yes"/"No" Title Case, downstream code
            # (state.update_resolution, _redeem_trade) requires "YES"/"NO".
            # Normalize at every load so legacy/synthesized trades with
            # bad casing get auto-fixed before any mutation runs.
            for strat in data.get("strategies", {}).values():
                for t in strat.get("trades", []):
                    _repair_trade_shape(t)
                    sl = t.get("side_label")
                    if not sl:
                        continue
                    if sl in ("YES", "NO"):
                        continue
                    sl_upper = sl.strip().upper()
                    if sl_upper in ("YES", "NO"):
                        t["side_label"] = sl_upper
                    else:
                        # Non-binary outcome — derive from outcome_index
                        idx = (t.get("metadata") or {}).get("outcome_index")
                        if idx is not None:
                            t["side_label"] = "YES" if int(idx) == 0 else "NO"

            return data

        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
            return _default_state()

    def save(self) -> None:
        """Atomic write: write to temp file then rename.

        Trims trades to MAX_TRADES_PER_STRATEGY per strategy and
        equity_curve to MAX_EQUITY_CURVE entries.
        """
        # Trim trades per strategy
        for name in list(self.state.get("strategies", {})):
            strat = self.state["strategies"][name]
            if len(strat.get("trades", [])) > MAX_TRADES_PER_STRATEGY:
                strat["trades"] = strat["trades"][-MAX_TRADES_PER_STRATEGY:]
            # Trim seen_signals to last 200 for dedup
            if len(strat.get("seen_signals", [])) > 200:
                strat["seen_signals"] = strat["seen_signals"][-200:]

        # Trim global equity curve
        curve = self.state.get("global", {}).get("equity_curve", [])
        if len(curve) > MAX_EQUITY_CURVE:
            self.state["global"]["equity_curve"] = curve[-MAX_EQUITY_CURVE:]

        # Atomic write
        os.makedirs(self.state_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=self.state_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.state, f, indent=2, default=str)
            os.replace(tmp_path, self.path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def record_trade(self, strategy_name: str, trade_dict: dict) -> None:
        """Append a trade record. Does NOT touch P&L counters.

        **Post-forensic-fix 2026-04-14**: previously this method incremented
        `total_pnl` by `ev × size` at trade time, which is a theoretical
        expected value, not a realized P&L. That produced a "21 wins 0 losses
        / +$363" record for llm-signal that completely ignored actual market
        outcomes. Now:
          - The trade is appended as an OPEN position (`outcome=None`).
          - `total_trades` and `open_trades` are incremented.
          - `wins`/`losses`/`realized_pnl`/`unrealized_pnl`/`total_pnl` are
            only updated via `update_resolution()` and `update_unrealized()`.

        Args:
            strategy_name: Strategy key (e.g. 'arb', 'bond').
            trade_dict: Trade data dict. Expected keys:
                time, market, side, price, size, p_fair, edge, ev,
                order_id, dry_run, success, metadata.
        """
        strat = self.state["strategies"].setdefault(
            strategy_name, _default_strategy_state()
        )

        # Tag the trade as open: outcome & per-trade P&L not known yet.
        # Keep shares so mark-to-market is a single lookup.
        enriched = dict(trade_dict)
        enriched.setdefault("outcome", None)  # None=open, 'WIN'/'LOSS'=resolved
        enriched.setdefault("realized_pnl", None)
        enriched.setdefault("unrealized_pnl", 0.0)
        enriched.setdefault("last_mtm_time", "")
        enriched.setdefault("last_mtm_price", None)
        # PositionManager lifecycle fields. base.py always sets `status` and
        # `placed_at` for new trades; fall back here so legacy callers don't
        # break.
        enriched.setdefault("status", "dry_filled" if enriched.get("dry_run", True) else "filled")
        enriched.setdefault("placed_at", enriched.get("time", _now_iso()))
        try:
            price = float(enriched.get("price", 0))
            size = float(enriched.get("size", 0))
            if price > 0 and size > 0:
                enriched["shares"] = round(size / price, 6)
            else:
                enriched["shares"] = 0.0
        except (TypeError, ValueError):
            enriched["shares"] = 0.0

        strat["trades"].append(enriched)
        strat["total_trades"] = strat.get("total_trades", 0) + 1
        strat["open_trades"] = strat.get("open_trades", 0) + 1

        g = self.state["global"]
        g["total_trades"] = g.get("total_trades", 0) + 1
        # Equity curve only moves when realized/unrealized P&L changes, not at
        # trade entry. That update happens in update_resolution() and
        # update_unrealized().

    def update_unrealized(
        self,
        strategy_name: str,
        condition_id: str,
        current_side_price: float,
        token_id: str | None = None,
    ) -> float:
        """Mark-to-market update for open trades on a given market.

        Args:
            strategy_name: Strategy key.
            condition_id: Market condition id.
            current_side_price: The current market price of the SIDE the
                trade is holding (e.g. if the trade bought NO at 0.20 and
                NO is now trading at 0.35, pass 0.35).
            token_id: Optional — if provided, only update trades whose
                token_id matches. Prevents clobbering a sibling position
                holding the opposite token on the same market (e.g. an
                arb_scanner holding both YES and NO on the same cid).
                When None, legacy behavior: match cid only.

        Returns:
            Delta in unrealized P&L applied to this strategy (new - old).
        """
        strat = self.state["strategies"].setdefault(
            strategy_name, _default_strategy_state()
        )
        delta = 0.0
        now = _now_iso()

        for trade in strat.get("trades", []):
            if trade.get("outcome") is not None:
                continue  # already resolved
            if trade.get("status") in TERMINAL_STATUSES:
                continue  # already closed by PositionManager
            if trade.get("condition_id") != condition_id:
                continue
            if token_id is not None and trade.get("token_id", "") != token_id:
                continue
            shares = float(trade.get("shares") or 0)
            size = float(trade.get("size") or 0)
            if shares <= 0 or size <= 0:
                continue
            new_mtm = round(shares * float(current_side_price) - size, 4)
            prev = float(trade.get("unrealized_pnl") or 0)
            trade["unrealized_pnl"] = new_mtm
            trade["last_mtm_time"] = now
            trade["last_mtm_price"] = float(current_side_price)
            delta += new_mtm - prev

        if delta:
            strat["unrealized_pnl"] = round(
                strat.get("unrealized_pnl", 0.0) + delta, 4
            )
            strat["total_pnl"] = round(
                strat.get("realized_pnl", 0.0)
                + strat.get("unrealized_pnl", 0.0),
                4,
            )
            g = self.state["global"]
            g["total_pnl"] = round(
                sum(
                    s.get("realized_pnl", 0.0) + s.get("unrealized_pnl", 0.0)
                    for s in self.state["strategies"].values()
                ),
                4,
            )
            g.setdefault("equity_curve", []).append({
                "t": now,
                "pnl": g["total_pnl"],
            })
        return delta

    def update_resolution(
        self,
        strategy_name: str,
        condition_id: str,
        settlement: str,
    ) -> float:
        """Settle every open trade on a given market.

        Args:
            strategy_name: Strategy key.
            condition_id: Market condition id.
            settlement: 'YES' or 'NO' — the winning outcome of the market.

        Returns:
            Total realized P&L applied (new money that moved from unrealized
            into realized).
        """
        strat = self.state["strategies"].setdefault(
            strategy_name, _default_strategy_state()
        )
        total_delta_real = 0.0
        total_delta_unreal = 0.0
        today = _today_str()
        now = _now_iso()

        for trade in strat.get("trades", []):
            if trade.get("outcome") is not None:
                continue
            if trade.get("status") in TERMINAL_STATUSES:
                continue
            if trade.get("condition_id") != condition_id:
                continue

            shares = float(trade.get("shares") or 0)
            try:
                entry = float(trade.get("filled_price") or trade.get("price") or 0)
            except (TypeError, ValueError):
                entry = 0.0
            if shares <= 0 or entry <= 0:
                trade["outcome"] = "INVALID"
                continue

            # A trade "wins" if the market settles to the side we bought.
            # `side_label` MUST be set by the caller before this point.
            # We used to default missing side_label to "YES" with a
            # metadata-based heuristic — that defaulting caused $9,550 of
            # phantom wins in the Apr-18 rollback. Now we refuse to guess:
            # if the caller hasn't proven which side this trade holds, skip
            # settlement and leave it open so the next reconcile retries.
            side_label = trade.get("side_label")
            if side_label not in ("YES", "NO"):
                # Leave outcome=None so reconcile will pick it up next cycle,
                # hopefully after main.py has filled in side_label from the
                # market's yes_token/no_token.
                continue

            won = (side_label == settlement)
            payout = 1.0 if won else 0.0
            realized = round((payout - entry) * shares, 4)
            prev_unreal = float(trade.get("unrealized_pnl") or 0)

            trade["outcome"] = "WIN" if won else "LOSS"
            trade["realized_pnl"] = realized
            trade["resolved_at"] = now
            trade["settlement"] = settlement
            trade["unrealized_pnl"] = 0.0

            total_delta_real += realized
            total_delta_unreal -= prev_unreal
            if won:
                strat["wins"] = strat.get("wins", 0) + 1
            else:
                strat["losses"] = strat.get("losses", 0) + 1
            strat["open_trades"] = max(0, strat.get("open_trades", 0) - 1)

        if total_delta_real or total_delta_unreal:
            strat["realized_pnl"] = round(
                strat.get("realized_pnl", 0.0) + total_delta_real, 4
            )
            strat["unrealized_pnl"] = round(
                strat.get("unrealized_pnl", 0.0) + total_delta_unreal, 4
            )
            strat["total_pnl"] = round(
                strat["realized_pnl"] + strat["unrealized_pnl"], 4
            )

            # Daily P&L is realized only, date-aware.
            g = self.state["global"]
            if g.get("daily_date") != today:
                g["daily_date"] = today
                g["daily_pnl"] = 0.0
                for s in self.state["strategies"].values():
                    s["daily_pnl"] = 0.0
            strat["daily_pnl"] = round(
                strat.get("daily_pnl", 0.0) + total_delta_real, 4
            )
            g["daily_pnl"] = round(
                g.get("daily_pnl", 0.0) + total_delta_real, 4
            )

            # Recompute global total = sum of (realized + unrealized) across
            # strategies so dashboards always agree with the underlying books.
            g["total_pnl"] = round(
                sum(
                    s.get("realized_pnl", 0.0) + s.get("unrealized_pnl", 0.0)
                    for s in self.state["strategies"].values()
                ),
                4,
            )
            g.setdefault("equity_curve", []).append({
                "t": now,
                "pnl": g["total_pnl"],
            })

        return total_delta_real

    def recompute_totals(self) -> None:
        """Recompute every counter from the trades arrays (source of truth).

        Useful after manual state edits or after bulk back-filling resolutions.
        Never touches the trades themselves.

        Bug-34 (2026-04-28): old logic only counted realized_pnl when
        outcome ∈ ("WIN","LOSS"). Sold trades have outcome=None (we exited
        early before market resolution) but DO have valid realized_pnl,
        and those were silently dropped, making this recompute disagree
        with PositionManager.recompute_strategy_aggregates by ~$18.
        Now uses the same rule as PM: any non-None realized_pnl counts.
        """
        g_total = 0.0
        for strat in self.state.get("strategies", {}).values():
            realized = 0.0
            unrealized = 0.0
            wins = 0
            losses = 0
            open_n = 0
            total_n = 0
            for t in strat.get("trades", []):
                total_n += 1
                outcome = t.get("outcome")
                rpnl = t.get("realized_pnl")
                # Realized: any trade with explicit realized_pnl is closed,
                # whether outcome is WIN/LOSS (settled on chain) or None
                # (we sold early via TP / catastrophe / cancel). Mirrors
                # PositionManager.recompute_strategy_aggregates.
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
                # Unrealized: only currently open trades
                if (outcome is None
                        and t.get("status") in ("open", "filled",
                                                 "dry_open", "dry_filled")):
                    unrealized += float(t.get("unrealized_pnl") or 0)
                    open_n += 1
            strat["realized_pnl"] = round(realized, 4)
            strat["unrealized_pnl"] = round(unrealized, 4)
            strat["total_pnl"] = round(realized + unrealized, 4)
            strat["wins"] = wins
            strat["losses"] = losses
            strat["open_trades"] = open_n
            strat["total_trades"] = total_n
            g_total += realized + unrealized

        g = self.state["global"]
        g["total_pnl"] = round(g_total, 4)
        g["total_trades"] = sum(
            s.get("total_trades", 0) for s in self.state["strategies"].values()
        )

    def record_signal(self, strategy_name: str, signal_dict: dict) -> None:
        """Track a signal for dedup / counting.

        Args:
            strategy_name: Strategy key.
            signal_dict: Signal data. Should contain at least 'condition_id' or
                a unique identifier for dedup.
        """
        strat = self.state["strategies"].setdefault(
            strategy_name, _default_strategy_state()
        )
        strat["signals_today"] = strat.get("signals_today", 0) + 1

        # Store compact dedup key
        sig_id = signal_dict.get("condition_id", signal_dict.get("token_id", ""))
        if sig_id:
            strat.setdefault("seen_signals", []).append(sig_id)

    def get_strategy_state(self, name: str) -> dict:
        """Return the state dict for a specific strategy.

        Args:
            name: Strategy key (e.g. 'arb', 'bond').

        Returns:
            Strategy state dict (mutable reference).
        """
        return self.state["strategies"].setdefault(name, _default_strategy_state())

    def daily_reset(self) -> None:
        """Reset daily counters if the date has changed (UTC)."""
        today = _today_str()
        g = self.state["global"]

        if g.get("daily_date") != today:
            g["daily_date"] = today
            g["daily_pnl"] = 0.0

            for name in list(self.state.get("strategies", {})):
                strat = self.state["strategies"][name]
                strat["daily_pnl"] = 0.0
                strat["signals_today"] = 0
                strat["seen_signals"] = []

    def check_bankroll(self, strategy_name: str) -> bool:
        """Return True if the strategy still has capacity for new trades.

        Historical bug (pre-2026-04-19): only checked daily loss limit,
        letting dry-run over-leverage to ~3x the configured bankroll.
        For bond-buyer alone we ended up with 124 open × $30 = $3,690
        in-flight against a $1,250 allocation (297% over).

        Two gates now:
          1. Daily loss limit (unchanged): stop if daily_pnl <= -$100
          2. Open-exposure limit (new): reject new trades if currently
             deployed capital >= strategy's effective bankroll.

        Second gate reads `max_open_cost` from the strategy's state if
        set (defaults conservatively to $1,000 so un-configured strategies
        can still place a handful of trades).
        """
        strat = self.get_strategy_state(strategy_name)
        risk_epoch = _parse_iso_ts(strat.get("risk_epoch", ""))
        if risk_epoch:
            today = _today_str()
            daily_loss = 0.0
            for t in strat.get("trades", []):
                if not isinstance(t.get("realized_pnl"), (int, float)):
                    continue
                ts = _trade_ts(t, "resolved_at", "sold_at", "redeemed_at", "time")
                if not ts or ts < risk_epoch or ts.strftime("%Y-%m-%d") != today:
                    continue
                daily_loss += float(t.get("realized_pnl") or 0)
        else:
            daily_loss = float(strat.get("daily_pnl", 0) or 0)
        max_daily_loss = float(strat.get("max_daily_loss", 100.0) or 100.0)
        if daily_loss <= -abs(max_daily_loss):
            return False

        # Open-exposure gate. Must exclude trades that PositionManager has
        # closed (dry_sold / dry_cancelled / dry_redeemed etc.) — those have
        # outcome=None but their capital is already freed. Using only
        # "outcome is None" double-counted closed-but-unresolved trades,
        # causing bond-buyer to report "bankroll exhausted" even though
        # PM had just sold positions (Codex audit 2026-04-22 finding P1-2).
        open_cost = sum(
            float(t.get("size") or 0)
            for t in strat.get("trades", [])
            if t.get("outcome") is None
            and t.get("status") not in TERMINAL_STATUSES
            and (
                not risk_epoch
                or ((_trade_ts(t, "placed_at", "time") or datetime.min.replace(tzinfo=timezone.utc)) >= risk_epoch)
            )
        )
        cap = float(strat.get("max_open_cost", 1000.0))
        return open_cost < cap

    def get_bankroll(self, strategy_name: str, default: float = 1000.0) -> float:
        """Return effective bankroll for a strategy (initial allocation minus losses)."""
        strat = self.get_strategy_state(strategy_name)
        # Prefer state-configured bankroll if set (e.g. from docker env).
        configured = strat.get("bankroll")
        if configured is not None:
            return max(0, float(configured) + strat.get("total_pnl", 0))
        return max(0, default + strat.get("total_pnl", 0))

    def get_held_condition_ids(self, strategy_name: str) -> list[str]:
        """Return list of condition_ids for open trades (not yet resolved)."""
        strat = self.get_strategy_state(strategy_name)
        trades = strat.get("trades", [])
        # Trades without outcome/pnl are considered open
        return [
            t.get("condition_id", "")
            for t in trades
            if t.get("condition_id") and t.get("outcome") is None
        ]

    def get_dashboard_data(self) -> dict:
        """Return full state snapshot for dashboard rendering.

        Returns:
            Copy of the complete state dict.
        """
        return {
            "version": self.state.get("version", 2),
            "started_at": self.state.get("started_at", ""),
            "strategies": {
                name: {
                    "total_trades": s.get("total_trades", 0),
                    "open_trades": s.get("open_trades", 0),
                    "wins": s.get("wins", 0),
                    "losses": s.get("losses", 0),
                    "realized_pnl": s.get("realized_pnl", 0.0),
                    "unrealized_pnl": s.get("unrealized_pnl", 0.0),
                    "total_pnl": s.get("total_pnl", 0),
                    "daily_pnl": s.get("daily_pnl", 0),
                    "signals_today": s.get("signals_today", 0),
                    "recent_trades": s.get("trades", [])[-20:],
                }
                for name, s in self.state.get("strategies", {}).items()
            },
            "global": {
                "total_pnl": self.state["global"].get("total_pnl", 0),
                "total_trades": self.state["global"].get("total_trades", 0),
                "daily_pnl": self.state["global"].get("daily_pnl", 0),
                "daily_date": self.state["global"].get("daily_date", ""),
                "cycle_num": self.state["global"].get("cycle_num", 0),
                "equity_curve": self.state["global"].get("equity_curve", [])[-200:],
            },
        }
