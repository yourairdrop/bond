"""
Alpha Suite v2 — Unified Polymarket Trading Bot.

Orchestrates multiple trading strategies in a single process.
Each strategy runs on its own interval, scanning for signals,
evaluating them into sized trades, and executing (dry-run or live).

Usage:
    python -m alpha_suite.main

Env vars:
    FREEZE_NEW_ENTRIES=1
        Skip scan/evaluate/execute for every strategy, but keep
        reconcile running. Use this when you want to stop opening
        new positions and let existing ones settle so you can read
        the "true" end-of-experiment PnL without new noise.
"""

import os
import json
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

from alpha_suite.state import StateManager, TERMINAL_STATUSES
from alpha_suite.base import live_buy_block_reason
from alpha_suite.utils.api import fetch_market_resolution, fetch_orderbook
from alpha_suite.utils.logger import setup_logger, log_event
from alpha_suite.position_manager import PositionManager, PositionManagerSettings

# How often to walk every open trade and refresh MTM / settle resolved ones.
RECONCILE_INTERVAL_SEC = 600   # 10 minutes

# Lifecycle check cadences (used by PositionManager).
#   fill sim     → every 30s (dry-run only)
#   cancel-stale → every 10s
#   sell-high    → every 60s
#   auto-redeem  → every 300s (after reconcile settles markets)
CANCEL_STALE_INTERVAL_SEC = 10
SELL_HIGH_INTERVAL_SEC = 60
REDEEM_INTERVAL_SEC = int(float(os.environ.get("REDEEM_INTERVAL_SEC", "300")))

# Freeze-mode flag: when set, strategies don't generate new trades,
# but reconcile keeps settling existing open positions.
FREEZE_NEW_ENTRIES = os.environ.get("FREEZE_NEW_ENTRIES", "").strip() in ("1", "true", "yes")

# Redeem script path (for live auto-redeem).
# Falls back to env var if the user provides a custom location.
_REDEEM_SCRIPT = os.environ.get(
    "REDEEM_SCRIPT_PATH",
    "/polymarket/shared/redeem_position.py",
)
_REDEEM_PYTHON = os.environ.get("REDEEM_VENV_PYTHON", sys.executable)
_PLACE_ORDER = os.path.join(
    os.environ.get("POLY_ROOT", "/polymarket"),
    "shared",
    "place_order.py",
)
_WALLET_SNAPSHOT_PATH = os.environ.get(
    "WALLET_SNAPSHOT_PATH",
    "/app/state/live_wallet_snapshot.json",
)

# Strategy imports — these will be implemented as separate modules.
# Uncomment as strategies are built:
# from alpha_suite.strategies.arb_scanner import ArbScanner
# from alpha_suite.strategies.bond_buyer import BondBuyer
# from alpha_suite.strategies.whale_copy import WhaleCopy
# from alpha_suite.strategies.llm_signal import LLMSignal
# from alpha_suite.strategies.latency_arb import LatencyArb
# from alpha_suite.strategies.coverage_arb import CoverageArb

# ── Graceful shutdown ──
RUNNING = True


def _handle_signal(sig, frame):
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    global RUNNING
    RUNNING = False


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def _load_strategies(state, log):
    """Instantiate all enabled strategies.

    Returns list of Strategy instances. Strategies that fail to import
    are logged and skipped — the bot continues with whatever loaded.
    """
    strategies = []

    # Strategy imports are kept dynamic so this public subset can expose
    # specific strategies without hard-coding a single production layout.
    strategy_classes = [
        # ── Bond family (high-prob 0.90-0.95 holds, taker) ──
        ("alpha_suite.strategies.bond_buyer", "BondBuyer"),
        ("alpha_suite.strategies_v2.bond_buyer_v2", "BondBuyerV2"),
        ("alpha_suite.strategies_v2.city_bond", "CityBond"),
        ("alpha_suite.strategies.bond_pro", "BondPro"),

        # ── Longshot family (cheap tail bets, 2-30c) ──
        ("alpha_suite.strategies.longshot", "Longshot"),
        ("alpha_suite.strategies_v2.longshot_v2", "LongshotV2"),
        # 2026-04-28 migrated from W4 alpha_suite (consolidation):
        # taker-style longshots without LLM screen, wider price band.
        # Backtest: WR=24% / ROI=+20% at max_price=0.30.
        # W4 dry-run: longshot-taker +$23.25 / longshot-taker-v2 +$22.22.
        ("alpha_suite.strategies.longshot_taker", "LongshotTaker"),
        ("alpha_suite.strategies_v2.longshot_taker_v2", "LongshotTakerV2"),

        # ── Arb (low-frequency but legitimate when triggered) ──
        ("alpha_suite.strategies.arb_scanner", "ArbScanner"),
        ("alpha_suite.strategies_v2.arb_scanner_v2", "ArbScannerV2"),

        # ── DROPPED 2026-04-28 (consolidation): see header above ──
        # ("alpha_suite.strategies.whale_copy", "WhaleCopy"),
        # ("alpha_suite.strategies_v2.whale_copy_v2", "WhaleCopyV2"),
        # ("alpha_suite.strategies.llm_signal", "LLMSignal"),
        # ("alpha_suite.strategies_v2.llm_signal_v2", "LLMSignalV2"),
        # ("alpha_suite.strategies.coverage_arb", "CoverageArb"),
        # ("alpha_suite.strategies_v2.coverage_arb_v2", "CoverageArbV2"),
        # ("alpha_suite.strategies_niche.lowliq_bond", "NicheLowLiqBond"),
        # ("alpha_suite.strategies_niche.multi_arb", "NicheMultiArb"),
        # ("alpha_suite.strategies.latency_arb", "LatencyArb"),
        # ("alpha_suite.strategies_niche.fresh", "NicheFreshLLM"),
    ]

    # Strategy whitelist (2026-04-24): when ALPHA_STRATEGY_WHITELIST env var
    # is set to a comma-separated list of class names, only those strategies
    # are loaded. Used by the live container (`docker-compose.live.yml`) to
    # run ONLY bond-buyer-v2 while the dry-run container runs everything.
    whitelist_raw = os.environ.get("ALPHA_STRATEGY_WHITELIST", "").strip()
    if whitelist_raw:
        allowed = {name.strip() for name in whitelist_raw.split(",") if name.strip()}
        before = len(strategy_classes)
        strategy_classes = [
            (mod, cls) for mod, cls in strategy_classes if cls in allowed
        ]
        log.info(
            f"ALPHA_STRATEGY_WHITELIST={whitelist_raw} — "
            f"filtered {before} → {len(strategy_classes)} strategies: "
            f"{[cls for _, cls in strategy_classes]}"
        )

    for module_path, class_name in strategy_classes:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            strategies.append(cls(state, log))
        except (ImportError, AttributeError) as e:
            log.warning(f"Strategy {class_name} not available: {e}")

    return strategies


def reconcile_open_positions(state: StateManager, log) -> dict:
    """Walk every open trade across every strategy.

    For each unique condition_id with open positions:
      1. Fetch the market via CLOB API (`fetch_market_resolution`)
      2. If closed: call `update_resolution(side='YES'|'NO')` so the trade
         moves from unrealized -> realized P&L.
      3. If still open: call `update_unrealized(side_price)` to refresh MTM.

    Returns a stats dict for logging.
    """
    stats = {
        "open": 0,
        "settled": 0,
        "mtm_updated": 0,
        "fetch_failures": 0,
        "unique_markets": 0,
    }

    # Group open trades by condition_id so we hit each market once.
    by_cid: dict[str, list[tuple[str, dict]]] = {}
    for name, strat in state.state.get("strategies", {}).items():
        for trade in strat.get("trades", []):
            if trade.get("outcome") is not None:
                continue
            if trade.get("status") in TERMINAL_STATUSES:
                continue
            cid = trade.get("condition_id")
            if not cid:
                continue
            by_cid.setdefault(cid, []).append((name, trade))

    stats["open"] = sum(len(rows) for rows in by_cid.values())
    stats["unique_markets"] = len(by_cid)

    if not by_cid:
        return stats

    log.info(
        f"[reconcile] walking {stats['open']} open trades "
        f"across {stats['unique_markets']} unique markets"
    )

    for cid, rows in by_cid.items():
        market = fetch_market_resolution(cid)
        if not market:
            stats["fetch_failures"] += 1
            continue

        if market["closed"]:
            settlement = "YES" if market["yes_winner"] else (
                "NO" if market["no_winner"] else None
            )
            if settlement is None:
                # Closed but neither side flagged winner — skip, don't poison
                # the data. Will retry next cycle.
                continue
            # CRITICAL: pre-set side_label for EVERY trade in this group
            # BEFORE calling update_resolution. update_resolution resolves
            # every open trade on the (strategy, condition_id) in one pass,
            # so siblings must already have correct side_label — otherwise
            # the first call settles sibling #2 using sibling #1's (possibly
            # stale) side_label path. Ground truth: token_id -> yes/no token.
            yes_tok = market.get("yes_token", "")
            no_tok = market.get("no_token", "")
            settleable: list[tuple[str, dict]] = []
            for strat_name, trade in rows:
                held = trade.get("token_id", "")
                if held and held == yes_tok:
                    trade["side_label"] = "YES"
                elif held and held == no_tok:
                    trade["side_label"] = "NO"
                else:
                    # Cannot prove which side we hold — refuse to settle.
                    # Better to leave open and retry than guess wrong.
                    log.warning(
                        f"[reconcile] cannot prove side for "
                        f"{strat_name} cid={cid[:16]} token={held[:16]} "
                        f"(yes={yes_tok[:16]} no={no_tok[:16]}); skipping"
                    )
                    continue
                settleable.append((strat_name, trade))
            # Now resolve each unique (strategy, cid) exactly once — all
            # siblings have correct side_label at this point.
            seen: set[str] = set()
            for strat_name, _ in settleable:
                if strat_name in seen:
                    continue
                seen.add(strat_name)
                state.update_resolution(strat_name, cid, settlement)
            stats["settled"] += len(settleable)
            continue

        # Still open → mark-to-market each holder, strictly by token_id.
        # If a trade's side can't be proven from its token_id, skip it —
        # don't guess (guessing is what caused the $9,550 phantom wins).
        yes_tok = market.get("yes_token", "")
        no_tok = market.get("no_token", "")
        for strat_name, trade in rows:
            held = trade.get("token_id", "")
            if held and held == yes_tok:
                side_label = "YES"
            elif held and held == no_tok:
                side_label = "NO"
            else:
                log.warning(
                    f"[reconcile] mtm skip: cannot prove side for "
                    f"{strat_name} cid={cid[:16]} token={held[:16]}"
                )
                continue
            trade["side_label"] = side_label

            current_side_price = (
                market["yes_price"] if side_label == "YES" else market["no_price"]
            )
            # If the market is mid-trading, "yes_price"/"no_price" are 0/1
            # only after it's closed — use orderbook midpoint instead.
            if not market["closed"] and current_side_price in (0, 1):
                token_id = yes_tok if side_label == "YES" else no_tok
                book = fetch_orderbook(token_id)
                bids = book.get("bids", [])
                asks = book.get("asks", [])
                if bids and asks:
                    try:
                        bb = float(bids[0]["price"])
                        ba = float(asks[0]["price"])
                        current_side_price = (bb + ba) / 2.0
                    except (KeyError, ValueError, TypeError):
                        pass
            # Pass token_id so update_unrealized can filter precisely and
            # not clobber a sibling trade's MTM with our price.
            state.update_unrealized(
                strat_name, cid, current_side_price, token_id=held
            )
            stats["mtm_updated"] += 1

    return stats


def write_live_wallet_snapshot(log) -> None:
    """Write CLOB V2 collateral balance for dashboard consumption.

    Dashboard intentionally does not receive live trading credentials. The
    trader already has them, so it snapshots the authenticated CLOB cash
    balance into the local state file; dashboard combines that cash with public
    data-api position values.
    """
    if not os.environ.get("POLYMARKET_PROXY_ADDRESS"):
        return

    now_iso = datetime.now(timezone.utc).isoformat()
    snapshot = {
        "ts": now_iso,
        "source": "clob-v2",
        "success": False,
    }
    try:
        proc = subprocess.run(
            [sys.executable, _PLACE_ORDER, "balance"],
            capture_output=True,
            text=True,
            timeout=20,
            env={**os.environ},
        )
        resp = json.loads(proc.stdout) if proc.stdout.strip() else {
            "success": False,
            "error": proc.stderr.strip() or "empty balance response",
        }
        if resp.get("success"):
            snapshot.update({
                "success": True,
                "collateral_balance": float(resp.get("collateral_balance") or 0.0),
                "raw_balance": resp.get("raw_balance"),
            })
        else:
            snapshot["error"] = resp.get("error") or "balance failed"
    except Exception as e:
        snapshot["error"] = str(e)

    # Do not let a transient CLOB/SSL failure erase the last known good cash.
    # The dashboard can safely label an old successful balance as stale; showing
    # cash as $0 because one balance call timed out is much worse.
    if not snapshot.get("success"):
        try:
            with open(_WALLET_SNAPSHOT_PATH) as f:
                prev = json.load(f)
            if prev.get("success"):
                prev["last_refresh_error"] = snapshot.get("error", "balance failed")
                prev["last_refresh_error_ts"] = now_iso
                snapshot = prev
        except Exception:
            pass

    try:
        tmp = _WALLET_SNAPSHOT_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(snapshot, f, indent=2)
        os.replace(tmp, _WALLET_SNAPSHOT_PATH)
    except Exception as e:
        log.warning(f"[wallet-snapshot] write failed: {e}")


def main():
    """Main entry point — runs the strategy loop until interrupted."""
    log = setup_logger("alpha-suite")
    state = StateManager()
    last_reconcile = 0.0
    last_cancel_stale = 0.0
    last_sell_high = 0.0
    last_redeem = 0.0

    log.info("=" * 60)
    log.info("Alpha Suite v2 starting")
    log.info("=" * 60)

    strategies = _load_strategies(state, log)

    # Position lifecycle manager — handles fill-sim / cancel-stale /
    # sell-high / auto-redeem across all strategies.
    pm_settings = PositionManagerSettings(
        fill_check_delay_sec=30,
        fill_check_interval_sec=30,
        redeem_enabled=True,
        redeem_script_path=_REDEEM_SCRIPT,
        redeem_venv_python=_REDEEM_PYTHON,
        redeem_losses_onchain=os.environ.get(
            "REDEEM_LOSSES_ONCHAIN", ""
        ).strip().lower() in ("1", "true", "yes"),
    )
    pm = PositionManager(state, log, pm_settings)
    # One-time normalize legacy trades so PM doesn't trip on missing fields.
    pm.backfill_legacy_trades(strategies)

    # Honor per-strategy "enabled" flag in state — lets us disable a
    # confirmed-loser strategy without touching code. Set via the state
    # file: strategies[NAME].enabled = False.
    for s in strategies:
        sv = state.state.get("strategies", {}).get(s.name, {})
        if sv.get("enabled") is False:
            s.enabled = False
            log.warning(f"[{s.name}] DISABLED via state.enabled=False")

    if not strategies:
        log.warning("No strategies loaded — running in idle mode")

    log.info(f"Strategies: {[s.name for s in strategies]}")
    log.info(f"DRY_RUN: {all(s.dry_run for s in strategies) if strategies else True}")
    log.info(f"State file: {state.path}")
    if FREEZE_NEW_ENTRIES:
        log.warning("=" * 60)
        log.warning("FREEZE_NEW_ENTRIES=1 — NO new trades will be placed.")
        log.warning("Reconcile still runs; existing opens will settle naturally.")
        log.warning("=" * 60)

    # ─── Chain reconciliation (Bug-1 + Bug-3 fix, 2026-04-25) ───
    # For LIVE strategies, query data-api to discover actual on-chain
    # positions and merge them into state. Without this, fills that
    # happened on CLOB never get tracked → check_and_redeem can't find
    # them → dead bonds pile up indefinitely.
    # Only runs for strategies that actually trade live (dry_run=False).
    live_strategies = [s for s in strategies if not s.dry_run]
    if live_strategies and os.environ.get("POLYMARKET_PROXY_ADDRESS"):
        log.info(f"[startup] reconciling chain positions for {len(live_strategies)} live strategies")
        for s in live_strategies:
            try:
                stats = pm.reconcile_chain_positions(s.name)
                log.info(f"[startup] {s.name}: {stats}")
            except Exception as e:
                log.error(f"[startup] reconcile_chain_positions failed for {s.name}: {e}")

    log_event("startup", {
        "strategies": [s.name for s in strategies],
        "dry_run": all(s.dry_run for s in strategies) if strategies else True,
        "state_path": state.path,
        "freeze_new_entries": FREEZE_NEW_ENTRIES,
    })

    cycle_count = 0

    while RUNNING:
        now = time.time()
        state.daily_reset()
        state.state["global"]["cycle_num"] += 1
        cycle_count += 1

        # Periodically reconcile every open trade across every strategy so
        # the dashboard reflects real PnL, not just open positions.
        if now - last_reconcile >= RECONCILE_INTERVAL_SEC:
            try:
                rstats = reconcile_open_positions(state, log)
                if rstats["unique_markets"] > 0:
                    log.info(
                        f"[reconcile] settled={rstats['settled']} "
                        f"mtm={rstats['mtm_updated']} "
                        f"fail={rstats['fetch_failures']} "
                        f"open={rstats['open']}"
                    )
            except Exception as e:
                log.error(f"[reconcile] FAILED: {e}", exc_info=True)

            # Periodic chain-position reconciliation for live strategies.
            # Catches fills that happened on CLOB since last cycle (Bug-1).
            if os.environ.get("POLYMARKET_PROXY_ADDRESS"):
                for s in strategies:
                    if s.dry_run:
                        continue
                    try:
                        cstats = pm.reconcile_chain_positions(s.name)
                        if cstats.get("upgraded", 0) or cstats.get("imported", 0):
                            log.info(
                                f"[reconcile-chain] {s.name}: "
                                f"upgraded={cstats.get('upgraded',0)} "
                                f"imported={cstats.get('imported',0)}"
                            )
                    except Exception as e:
                        log.warning(f"[reconcile-chain] {s.name} failed: {e}")

            # Recompute strategy aggregators from individual trades (Bug-10).
            # Without this, dashboard's strategy.unrealized_pnl drifts as
            # trades transition state but the field doesn't get refreshed.
            for s in strategies:
                try:
                    pm.recompute_strategy_aggregates(s.name)
                except Exception as e:
                    log.warning(f"[recompute] {s.name} failed: {e}")

            if live_strategies:
                write_live_wallet_snapshot(log)

            last_reconcile = now

        # ── Position lifecycle checks (PositionManager) ──────────────
        # Run each feature on its own cadence. PM is idempotent and cheap
        # when no trades need action.
        #
        # Order: fills FIRST so a just-filled trade can be evaluated for
        # sell-high in the same tick; cancel-stale & sell-high next;
        # redeem last (piggybacks on reconcile's resolution flags).

        # Fill simulator (dry-run only — PM skips if strategy is live)
        try:
            n = pm.check_fills(strategies)
            if n:
                log.info(f"[PM] filled {n} dry-run orders this tick")
        except Exception as e:
            log.error(f"[PM][fills] FAILED: {e}", exc_info=True)

        if now - last_cancel_stale >= CANCEL_STALE_INTERVAL_SEC:
            try:
                n = pm.check_cancel_stale(strategies)
                if n:
                    log.info(f"[PM] cancel-stale removed {n} orders")
            except Exception as e:
                log.error(f"[PM][cancel] FAILED: {e}", exc_info=True)
            last_cancel_stale = now

        if now - last_sell_high >= SELL_HIGH_INTERVAL_SEC:
            try:
                n = pm.check_sell_high(strategies)
                if n:
                    log.info(f"[PM] sell-high closed {n} positions")
            except Exception as e:
                log.error(f"[PM][sell] FAILED: {e}", exc_info=True)
            last_sell_high = now

        if now - last_redeem >= REDEEM_INTERVAL_SEC:
            try:
                n = pm.check_and_redeem(strategies)
                if n:
                    log.info(f"[PM] redeemed {n} settled trades")
            except Exception as e:
                log.error(f"[PM][redeem] FAILED: {e}", exc_info=True)
            last_redeem = now

        for strategy in strategies:
            if not strategy.enabled:
                continue

            if now < strategy.next_run:
                continue

            # FREEZE mode: skip scan/evaluate/execute so no new trades
            # are created. Reconcile (which runs above) still settles
            # existing opens and refreshes MTM.
            if FREEZE_NEW_ENTRIES:
                strategy.next_run = now + strategy.interval_sec
                continue

            if not getattr(strategy, "dry_run", True):
                block_reason = live_buy_block_reason()
                if block_reason:
                    log.warning(
                        f"[{strategy.name}][live-buy-block] skipping "
                        f"scan/evaluate/execute; recent CLOB restriction: "
                        f"{block_reason[:180]}"
                    )
                    strategy.next_run = now + strategy.interval_sec
                    continue

            cycle_start = time.time()
            try:
                # Phase 1: Scan for signals
                signals = strategy.scan()

                # Phase 2: Evaluate and size trades
                trades = strategy.evaluate(signals)

                # Phase 3: Execute trades
                results = strategy.execute(trades)

                elapsed = time.time() - cycle_start

                log.info(
                    f"[{strategy.name}] "
                    f"{len(signals)} signals -> "
                    f"{len(trades)} trades -> "
                    f"{len(results)} results "
                    f"({elapsed:.1f}s)"
                )

                log_event("cycle", {
                    "strategy": strategy.name,
                    "signals": len(signals),
                    "trades": len(trades),
                    "results": len(results),
                    "elapsed": round(elapsed, 1),
                    "cycle_num": cycle_count,
                })

            except Exception as e:
                elapsed = time.time() - cycle_start
                log.error(
                    f"[{strategy.name}] FAILED after {elapsed:.1f}s: {e}",
                    exc_info=True,
                )
                log_event("error", {
                    "strategy": strategy.name,
                    "error": str(e),
                    "elapsed": round(elapsed, 1),
                })

            # Schedule next run
            strategy.next_run = now + strategy.interval_sec

            # Flush state after each strategy (not just at end of full cycle).
            # Prevents a stuck strategy (e.g., LLM endpoint hanging) from
            # blocking every other strategy's trades from ever reaching disk.
            try:
                state.save()
            except Exception as e:
                log.error(f"[state.save] mid-cycle save failed: {e}")

        # Final flush at end of full cycle (covers any global-state changes)
        state.save()

        # Brief sleep to avoid busy-spinning
        time.sleep(1)

    # ── Shutdown ──
    log.info("Alpha Suite shutting down")
    log_event("shutdown", {"cycles": cycle_count})
    state.save()
    log.info("State saved. Goodbye.")


if __name__ == "__main__":
    main()
