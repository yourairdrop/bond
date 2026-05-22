"""
Alpha Suite v2 — BondBuyer with sibling dedup.

v1 data (Apr-18 dry-run, 397 resolved trades):
  - 93.8% winrate, realized +$241.54
  - BUT 38 sibling groups (same condition_id bought twice). Most are
    2W/0L, but the worst (cid=0x88617...) both legs lost = -$60 —
    one bad event chewed through 25% of total profit.

v2 fix:
  In evaluate(), after sizing a trade, check whether this strategy
  already has an OPEN trade on the same condition_id. If so, skip —
  don't double-up exposure on the same market.

That's the only behavioral change. Everything else inherits from
BondBuyer so the edge/threshold logic stays identical.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import List

from alpha_suite.strategies.bond_buyer import (
    BondBuyer,
    MIN_EDGE,
    MIN_EV,
    MIN_LIQUIDITY,
    STALE_SCAN_FLOOR,
    VWAP_SIZE,
)
from alpha_suite.base import Signal, Trade
from alpha_suite.state import TERMINAL_STATUSES
from alpha_suite.utils.api import fetch_orderbook, vwap_ask
from alpha_suite.utils.risk import ev_calc


class BondBuyerV2(BondBuyer):
    """Bond buyer that refuses to stack on a market it already holds.

    Live-mode overrides (2026-04-24): when the container sets BOND_V2_LIVE=1,
    this strategy flips to live trading. max_bet / bankroll_pct / max_daily_loss
    are also env-driven so the live container can use conservative values
    ($4 bet / 5% bankroll / $20 daily cap) without any code duplication.
    Dry-run container leaves the envs unset and inherits BondBuyer defaults.
    """

    name = "bond-buyer-v2"
    # Env-driven live toggle. Defaults to True (dry) so the dry-run container
    # gets the safe default; the live container sets BOND_V2_LIVE=1 to flip.
    dry_run = os.environ.get("BOND_V2_LIVE", "").strip() != "1"
    # Conservative caps when env vars are set; otherwise inherit from parent.
    max_bet = float(os.environ.get("BOND_V2_MAX_BET", "8.0"))
    bankroll_pct = float(os.environ.get("BOND_V2_BANKROLL_PCT", "0.25"))
    max_open_cost = float(os.environ.get(
        "BOND_V2_MAX_OPEN_COST",
        str(round(5000.0 * bankroll_pct, 2)),
    ))
    max_daily_loss = float(os.environ.get("BOND_V2_MAX_DAILY_LOSS", "40.0"))
    risk_epoch = os.environ.get("BOND_V2_RISK_EPOCH", "").strip()
    cancel_reentry_block_sec = float(os.environ.get(
        "BOND_V2_CANCEL_REENTRY_BLOCK_SEC",
        "21600",
    ))

    def _cancelled_live_trade_blocks_reentry(self, t: dict) -> bool:
        """Treat recent live cancels as still risky until chain sync settles.

        A live GTC cancel can race with a fill. If we immediately release the
        cid/event slot, the next scan can place another order before
        data-api/reconcile has surfaced the original fill. Keep the slot
        blocked for a few hours; this favors missing a re-entry over doubling
        exposure in one bucket.
        """
        if t.get("outcome") is not None:
            return False
        if t.get("status") not in {"cancelled", "cancel_failed"}:
            return False
        order_id = str(t.get("order_id") or "")
        if not order_id or order_id.startswith("DRY-"):
            return False
        raw = t.get("cancelled_at") or t.get("time") or t.get("placed_at")
        try:
            cancelled_at = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return True
        if cancelled_at.tzinfo is None:
            cancelled_at = cancelled_at.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - cancelled_at.astimezone(timezone.utc)).total_seconds()
        return age <= self.cancel_reentry_block_sec

    def _has_open_on_cid(self, condition_id: str) -> bool:
        """Return True if this strategy already holds an unresolved trade."""
        if not condition_id:
            return False
        strat = self.state.state.get("strategies", {}).get(self.name, {})
        for t in strat.get("trades", []):
            if t.get("outcome") is not None:
                continue
            if t.get("condition_id") == condition_id:
                if t.get("status") not in TERMINAL_STATUSES:
                    return True
                if self._cancelled_live_trade_blocks_reentry(t):
                    return True
        return False

    def _has_open_on_event(self, event_slug: str) -> bool:
        """Return True if this strategy already holds an unresolved trade
        on ANY market in the same event_slug. Catches multi-bucket sibling
        traps (Bug-13 fix, 2026-04-27).

        Example: weather temperature buckets for "Lagos on April 26" all
        share event_slug "highest-temperature-in-lagos-on-april-26-2026".
        Buying NO on multiple buckets guarantees ≥1 loss because the actual
        temperature must fall in exactly one bucket. Same root cause as
        multi-strike posts (already keyword-banned in v1's scan).

        Verified on 5-day data: 20/20 weather losses had sibling buckets we
        also bought NO on. Filtering would have prevented at least one loss
        per losing event.
        """
        if not event_slug:
            return False
        strat = self.state.state.get("strategies", {}).get(self.name, {})
        for t in strat.get("trades", []):
            if t.get("outcome") is not None:
                continue
            existing_slug = (t.get("metadata") or {}).get("event_slug")
            if existing_slug and existing_slug == event_slug:
                if t.get("status") not in TERMINAL_STATUSES:
                    return True
                if self._cancelled_live_trade_blocks_reentry(t):
                    return True
        return False

    @staticmethod
    def _is_weather_signal(signal: Signal) -> bool:
        return (signal.metadata or {}).get("category") == "weather"

    def _score_weather_signal(self, signal: Signal) -> tuple[float, float, float, float, float] | None:
        """Return a live-orderbook score for a weather sibling candidate.

        Higher tuple wins. The score is primarily live VWAP edge, then lower
        live price, then available liquidity, then Gamma volume. This keeps the
        one-position-per-event safety rule while avoiding "first Gamma row
        wins" within weather sibling buckets.
        """
        meta = signal.metadata or {}
        book = fetch_orderbook(signal.token_id)
        vwap_px, vwap_liq = vwap_ask(book.get("asks", []), VWAP_SIZE)
        if vwap_liq < MIN_LIQUIDITY:
            self.log.info(
                f"  [bond-v2] weather-choice reject liquidity "
                f"{vwap_liq:.1f} < {MIN_LIQUIDITY}: {signal.market_question[:50]}"
            )
            return None

        actual_price = vwap_px if vwap_px > 0 else signal.price
        if actual_price < STALE_SCAN_FLOOR:
            self.log.warning(
                f"  [bond-v2] weather-choice reject stale VWAP "
                f"{actual_price:.3f} < {STALE_SCAN_FLOOR}: "
                f"{signal.market_question[:50]}"
            )
            return None

        min_edge_required = float(meta.get("min_edge_required", MIN_EDGE))
        actual_edge = signal.p_fair - actual_price
        if actual_edge < min_edge_required:
            self.log.info(
                f"  [bond-v2] weather-choice reject edge "
                f"{actual_edge:.3f} < {min_edge_required:.3f}: "
                f"{signal.market_question[:50]}"
            )
            return None

        tick = float(meta.get("tick_size", 0.01) or 0.01)
        maker_limit = max(tick, round(actual_price - tick, 4))
        maker_ev = ev_calc(signal.p_fair, maker_limit)
        if maker_ev < MIN_EV:
            self.log.info(
                f"  [bond-v2] weather-choice reject EV "
                f"{maker_ev:.4f} < {MIN_EV:.4f}: "
                f"{signal.market_question[:50]}"
            )
            return None

        # Reuse this exact live book read in BondBuyer.evaluate() so selection
        # and execution checks are based on the same snapshot.
        meta["_v2_prechecked_vwap_price"] = actual_price
        meta["_v2_prechecked_vwap_liq"] = vwap_liq
        meta["_v2_prechecked_edge"] = actual_edge
        meta["_v2_prechecked_maker_ev"] = maker_ev

        try:
            volume = float(meta.get("volume") or 0.0)
        except (TypeError, ValueError):
            volume = 0.0
        return (maker_ev, actual_edge, -actual_price, vwap_liq, volume)

    def _preselect_signals(self, signals: List[Signal]) -> list[Signal]:
        """Dedup signals before parent evaluate() consumes max_open_cost.

        Non-weather markets keep the historical first-valid-candidate behavior.
        Weather markets are grouped by event_slug and scored with live VWAP so
        one sibling bucket per city/day is selected intentionally.
        """
        selected_by_index: list[tuple[int, Signal]] = []
        weather_groups: dict[str, list[tuple[int, Signal]]] = {}
        pre_batch_slugs: set[str] = set()
        pre_batch_cids: set[str] = set()

        for index, signal in enumerate(signals):
            cid = signal.condition_id
            if cid in pre_batch_cids or self._has_open_on_cid(cid):
                self.log.info(
                    f"  [bond-v2] pre-skip sibling dup on cid={cid[:20]}"
                )
                continue
            event_slug = (signal.metadata or {}).get("event_slug") or ""
            if event_slug and self._has_open_on_event(event_slug):
                self.log.info(
                    f"  [bond-v2] pre-skip same-event dup on slug={event_slug[:40]}"
                )
                continue

            if self._is_weather_signal(signal) and event_slug:
                weather_groups.setdefault(event_slug, []).append((index, signal))
                # Do not mark cid/slug as used yet. We first need to compare
                # all sibling buckets in the event and keep only the best one.
                continue

            if event_slug and event_slug in pre_batch_slugs:
                self.log.info(
                    f"  [bond-v2] pre-skip same-event dup on slug={event_slug[:40]}"
                )
                continue
            selected_by_index.append((index, signal))
            pre_batch_cids.add(cid)
            if event_slug:
                pre_batch_slugs.add(event_slug)

        for event_slug, grouped in weather_groups.items():
            if event_slug in pre_batch_slugs:
                self.log.info(
                    f"  [bond-v2] pre-skip weather event already selected "
                    f"slug={event_slug[:40]}"
                )
                continue
            scored: list[tuple[tuple[float, float, float, float, float], int, Signal]] = []
            for index, signal in grouped:
                score = self._score_weather_signal(signal)
                if score is not None:
                    scored.append((score, index, signal))
            if not scored:
                continue

            scored.sort(key=lambda row: row[0], reverse=True)
            best_score, best_index, best_signal = scored[0]
            selected_by_index.append((best_index, best_signal))
            pre_batch_cids.add(best_signal.condition_id)
            pre_batch_slugs.add(event_slug)

            if len(grouped) > 1:
                self.log.info(
                    f"  [bond-v2] weather-choice slug={event_slug[:40]} "
                    f"selected ev={best_score[0]:.4f} "
                    f"edge={best_score[1]:.3f} "
                    f"price={-best_score[2]:.3f} "
                    f"from {len(scored)}/{len(grouped)} live-eligible"
                )

        selected_by_index.sort(key=lambda row: row[0])
        return [signal for _, signal in selected_by_index]

    def evaluate(self, signals: List[Signal]) -> List[Trade]:
        """Size trades via parent logic, then dedupe by cid AND by event.

        Parent already filters by EV, liquidity, bankroll. We add three
        final gates:
          1. Same condition_id vs state (original v2 dedup)
          2. Same event_slug vs state (Bug-13)
          3. Same event_slug WITHIN this batch (Bug-21, 2026-04-27)

        Bug-21: state-write happens in execute() AFTER evaluate(). So
        within one evaluate() call processing 5 candidates with same
        event_slug, the first passes (state has nothing) and gets added
        to deduped[]. The remaining 4 ALSO pass (state still has nothing
        since first hasn't been recorded yet) → all 5 sibling buckets
        get placed. Track in-batch slugs to break this race.

        Bug-38 (2026-04-29): this dedup must happen BEFORE parent evaluate().
        Parent enforces max_open_cost while walking signals and increments its
        local current_open for every candidate it turns into a Trade. If v2
        waits until after parent evaluate() to drop duplicate cid/event trades,
        those doomed duplicates still consume the parent's cap and can block
        later valid candidates in the same scan. We still keep a post-parent
        dedup pass below as a defensive backstop for races/state changes.
        """
        filtered_signals = self._preselect_signals(signals)
        trades = super().evaluate(filtered_signals)
        deduped: list[Trade] = []
        batch_slugs: set[str] = set()    # slugs added in THIS evaluate()
        batch_cids: set[str] = set()     # cids added in THIS evaluate()
        for t in trades:
            cid = t.signal.condition_id
            if cid in batch_cids or self._has_open_on_cid(cid):
                self.log.info(
                    f"  [bond-v2] skip sibling dup on cid={cid[:20]}"
                )
                continue
            event_slug = (t.signal.metadata or {}).get("event_slug") or ""
            if event_slug and (event_slug in batch_slugs
                               or self._has_open_on_event(event_slug)):
                self.log.info(
                    f"  [bond-v2] skip same-event dup on slug={event_slug[:40]}"
                )
                continue
            deduped.append(t)
            batch_cids.add(cid)
            if event_slug:
                batch_slugs.add(event_slug)
        return deduped
