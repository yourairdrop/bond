"""
Alpha Suite — Strategy Base Class and Data Models.

Defines the Signal/Trade/Result dataclasses and the Strategy base class
that all concrete strategies inherit from. Provides default dry-run
execution and subprocess-based live order placement.
"""

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from alpha_suite.state import StateManager


# ── Paths for order execution ──
_POLY_ROOT = os.environ.get("POLY_ROOT", "/polymarket")
_PLACE_ORDER = os.path.join(_POLY_ROOT, "shared", "place_order.py")
_VENV_PYTHON = sys.executable


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


_LIVE_CASH_RESERVE_USD = _float_env("LIVE_CASH_RESERVE_USD", 0.25)
_LIVE_BALANCE_TIMEOUT_SEC = _float_env("LIVE_BALANCE_TIMEOUT_SEC", 20.0)
_WALLET_SNAPSHOT_PATH = os.environ.get(
    "LIVE_WALLET_SNAPSHOT_PATH",
    os.path.join(os.environ.get("STATE_DIR", "/app/state"), "live_wallet_snapshot.json"),
)
_LIVE_BUY_BLOCK_MIN = _float_env("LIVE_BUY_BLOCK_MIN", 10.0)


def _live_buy_block_path() -> str:
    return os.environ.get(
        "LIVE_BUY_BLOCK_PATH",
        os.path.join(os.environ.get("STATE_DIR", "/app/state"), "live_buy_block.json"),
    )


def _is_geoblock_error(message: object) -> bool:
    text = str(message or "").lower()
    return "geoblock" in text or "trading restricted in your region" in text


def record_live_buy_block(reason: object) -> None:
    """Persist a live BUY pause after a CLOB geoblock failure.

    SELL/redeem paths must keep retrying existing inventory, but opening new
    positions while the same process cannot reliably exit them is unsafe.
    """
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "reason": str(reason or "")[:500],
    }
    path = _live_buy_block_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except OSError:
        pass


def live_buy_block_reason() -> Optional[str]:
    path = _live_buy_block_path()
    try:
        with open(path) as f:
            payload = json.load(f)
        ts_raw = payload.get("ts", "")
        ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - ts).total_seconds()
        if age <= _LIVE_BUY_BLOCK_MIN * 60:
            return str(payload.get("reason") or "live BUY paused")
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return None


# ══════════════════════════════════════════════════════════════════════
# Data Models
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Signal:
    """A trading signal identified by a strategy scan."""

    strategy: str
    market_question: str
    condition_id: str
    token_id: str
    side: str           # BUY or SELL
    price: float
    p_fair: float
    edge: float
    ev: float
    metadata: dict = field(default_factory=dict)


@dataclass
class Trade:
    """A sized trade ready for execution."""

    signal: Signal
    size_usd: float
    order_type: str = "GTC"   # GTC or FOK


@dataclass
class Result:
    """Outcome of an order placement attempt."""

    trade: Trade
    success: bool
    order_id: str
    error: str = ""


# ══════════════════════════════════════════════════════════════════════
# Strategy Base Class
# ══════════════════════════════════════════════════════════════════════

class Strategy:
    """Abstract base class for all Alpha Suite strategies.

    Subclasses must implement:
        - scan()     -> list[Signal]
        - evaluate() -> list[Trade]

    Optionally override:
        - execute()  -> list[Result]  (default handles dry-run / live)
    """

    # Override these in subclass
    name: str = "base"
    interval_sec: int = 300
    bankroll_pct: float = 0.25
    max_bet: float = 15.0
    max_daily_loss: float = 50.0
    dry_run: bool = True
    enabled: bool = True
    next_run: float = 0.0

    def __init__(self, state_manager: StateManager, logger: logging.Logger):
        """Initialize strategy with shared state and logger.

        Args:
            state_manager: Shared StateManager instance.
            logger: Configured logging.Logger.
        """
        self.state = state_manager
        self.log = logger
        strat = self.state.get_strategy_state(self.name)
        default_bankroll = round(5000.0 * float(self.bankroll_pct), 2)
        if strat.get("bankroll") is None:
            strat["bankroll"] = default_bankroll
        configured_max_open = getattr(self, "max_open_cost", None)
        if configured_max_open is not None:
            # Keep env-driven exposure caps live across restarts. Otherwise an
            # older persisted state value silently overrides docker-compose.
            strat["max_open_cost"] = round(float(configured_max_open), 2)
        elif strat.get("max_open_cost") is None:
            strat["max_open_cost"] = round(
                float(default_bankroll), 2
            )
        # Keep env-driven loss caps live across restarts. Older state files
        # often lack this field, and check_bankroll should not fall back to
        # the old global -$100 circuit breaker for small live bankrolls.
        strat["max_daily_loss"] = round(float(self.max_daily_loss), 2)
        risk_epoch = getattr(self, "risk_epoch", "")
        if risk_epoch:
            strat["risk_epoch"] = str(risk_epoch)

    def scan(self) -> List[Signal]:
        """Scan markets and return trading signals.

        Must be implemented by each strategy subclass.

        Returns:
            List of Signal objects representing potential trades.
        """
        raise NotImplementedError(f"{self.name}.scan() not implemented")

    def evaluate(self, signals: List[Signal]) -> List[Trade]:
        """Evaluate signals and produce sized trades.

        Must be implemented by each strategy subclass.
        Apply risk checks, Kelly sizing, dedup, etc.

        Args:
            signals: Signals from the scan() phase.

        Returns:
            List of Trade objects ready for execution.
        """
        raise NotImplementedError(f"{self.name}.evaluate() not implemented")

    def execute(self, trades: List[Trade]) -> List[Result]:
        """Execute trades: dry-run by default, live if dry_run=False.

        Records each trade to the state manager regardless of mode.

        Args:
            trades: Sized trades from the evaluate() phase.

        Returns:
            List of Result objects with execution outcomes.
        """
        results = []
        live_cash = None
        if not self.dry_run:
            live_cash = self._live_collateral_balance()
            if live_cash is None:
                live_cash = self._snapshot_collateral_balance()
            if live_cash is None:
                self.log.warning(
                    f"[{self.name}][cash-gate] unable to verify live "
                    "collateral; proceeding without local cash gate"
                )
            else:
                self.log.info(
                    f"[{self.name}][cash-gate] live collateral=${live_cash:.2f}, "
                    f"reserve=${_LIVE_CASH_RESERVE_USD:.2f}"
                )

        for trade in trades:
            if self.dry_run:
                result = Result(
                    trade=trade,
                    success=True,
                    order_id=f"DRY-{int(time.time() * 1000)}",
                )
                self.log.info(
                    f"[{self.name}][DRY] {trade.signal.side} "
                    f"${trade.size_usd:.2f} @ {trade.signal.price:.3f} — "
                    f"{trade.signal.market_question[:60]}"
                )
            else:
                side = str(trade.signal.side).upper()
                if side == "BUY":
                    block_reason = live_buy_block_reason()
                    if block_reason:
                        self.log.warning(
                            f"[{self.name}][live-buy-block] skip BUY "
                            f"${trade.size_usd:.2f} @ {trade.signal.price:.3f}; "
                            f"recent CLOB geoblock/order restriction: "
                            f"{block_reason[:180]} -- "
                            f"{trade.signal.market_question[:80]}"
                        )
                        continue
                if side == "BUY" and live_cash is not None:
                    required_cash = float(trade.size_usd) + _LIVE_CASH_RESERVE_USD
                    if live_cash < required_cash:
                        self.log.warning(
                            f"[{self.name}][cash-gate] skip BUY "
                            f"${trade.size_usd:.2f} @ {trade.signal.price:.3f}; "
                            f"live collateral ${live_cash:.2f} < required "
                            f"${required_cash:.2f} -- "
                            f"{trade.signal.market_question[:80]}"
                        )
                        continue
                result = self._place_real_order(trade)
                if side == "BUY" and result.success and live_cash is not None:
                    live_cash = max(0.0, live_cash - float(trade.size_usd))

            results.append(result)

            # Determine initial status for the lifecycle manager:
            #   failed live order → terminal rejected; do not let PM/reconcile
            #   treat an exchange/API reject as an open position.
            #   GTC orders → "open" (or "dry_open") — pending fill
            #   FOK orders → "filled" (or "dry_filled") — instant fill or nothing
            order_type = getattr(trade, "order_type", "GTC")
            if not result.success:
                initial_status = "dry_rejected" if self.dry_run else "rejected"
            elif order_type in ("FOK", "FAK"):
                initial_status = "dry_filled" if self.dry_run else "filled"
            else:
                initial_status = "dry_open" if self.dry_run else "open"

            now_iso = datetime.now(timezone.utc).isoformat()
            trade_record = {
                "time": now_iso,
                "placed_at": now_iso,                 # for PositionManager TTL
                "status": initial_status,              # PositionManager lifecycle
                "order_type": order_type,
                "market": trade.signal.market_question[:80],
                "condition_id": trade.signal.condition_id,
                "token_id": trade.signal.token_id,
                "side": trade.signal.side,
                "price": round(trade.signal.price, 4),
                "size": round(trade.size_usd, 2),
                "p_fair": round(trade.signal.p_fair, 4),
                "edge": round(trade.signal.edge, 4),
                "ev": round(trade.signal.ev, 4),
                "order_id": result.order_id,
                "dry_run": self.dry_run,
                "success": result.success,
                "error": result.error,
                "metadata": trade.signal.metadata,
            }
            # If the order came back filled immediately (FOK, or live instant
            # fill), stamp filled_at/filled_price so the PM doesn't trip over
            # a "filled" status with no filled_at.
            if initial_status in ("dry_filled", "filled"):
                trade_record["filled_at"] = now_iso
                trade_record["filled_price"] = round(trade.signal.price, 4)

            self.state.record_trade(self.name, trade_record)

            # Record signal for dedup
            if result.success:
                self.state.record_signal(self.name, {
                    "condition_id": trade.signal.condition_id,
                    "token_id": trade.signal.token_id,
                    "time": datetime.now(timezone.utc).isoformat(),
                })

        return results

    def _snapshot_collateral_balance(self) -> Optional[float]:
        """Read the latest wallet snapshot written by main.py."""
        try:
            with open(_WALLET_SNAPSHOT_PATH, "r") as f:
                snapshot = json.load(f)
            if not snapshot.get("success"):
                return None
            return float(snapshot.get("collateral_balance"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _live_collateral_balance(self) -> Optional[float]:
        """Fetch authenticated CLOB V2 collateral balance for live cash gate."""
        if os.environ.get("FORCE_DRY_RUN"):
            return None
        if not os.environ.get("POLYMARKET_PROXY_ADDRESS"):
            return None

        try:
            proc = subprocess.run(
                [_VENV_PYTHON, _PLACE_ORDER, "balance"],
                capture_output=True,
                text=True,
                timeout=_LIVE_BALANCE_TIMEOUT_SEC,
                env={**os.environ},
            )
            if not proc.stdout.strip():
                self.log.warning(
                    f"[{self.name}][cash-gate] empty balance response: "
                    f"{proc.stderr[:160]!r}"
                )
                return None
            resp = json.loads(proc.stdout)
            if not resp.get("success"):
                err = str(resp.get("error") or "")
                if _is_geoblock_error(err):
                    record_live_buy_block(err)
                self.log.warning(
                    f"[{self.name}][cash-gate] balance failed: "
                    f"{err[:160]}"
                )
                return None
            return float(resp.get("collateral_balance"))
        except (subprocess.TimeoutExpired, json.JSONDecodeError, TypeError, ValueError) as e:
            self.log.warning(f"[{self.name}][cash-gate] balance check failed: {e}")
            return None

    def _place_real_order(self, trade: Trade) -> Result:
        """Place a real order via place_order.py subprocess.

        Calls the shared order placement script with appropriate arguments.
        Parses the JSON response to determine success.

        Args:
            trade: The Trade to execute.

        Returns:
            Result with success status, order ID, and any error message.
        """
        # Env-level kill switch. If FORCE_DRY_RUN is set in the environment
        # (docker-compose, shell, etc.) we refuse to place real orders no
        # matter what `self.dry_run` is — previously a single accidental
        # `self.dry_run = False` in strategy code would have sent a live
        # order. Matches W1's FORCE_DRY_RUN guard for parity across groups.
        if os.environ.get("FORCE_DRY_RUN"):
            self.log.warning(
                f"[{self.name}][DRY-GUARD] FORCE_DRY_RUN env set; refusing "
                f"to place live order for {trade.signal.token_id[:16]}. "
                f"Returning synthetic DRY result."
            )
            return Result(
                trade=trade,
                success=True,
                order_id=f"DRY-GUARD-{int(time.time() * 1000)}",
                error="",
            )
        cmd = [
            _VENV_PYTHON, _PLACE_ORDER, "limit",
            "--token-id", trade.signal.token_id,
            "--side", trade.signal.side,
            "--price", str(round(trade.signal.price, 4)),
            "--size", str(round(trade.size_usd, 2)),
            "--order-type", trade.order_type,
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                env={**os.environ},
            )

            if proc.stdout.strip():
                resp = json.loads(proc.stdout)
            else:
                resp = {"success": False, "error": proc.stderr.strip() or "Empty response"}

            success = resp.get("success", False)
            order_id = resp.get("order_id", resp.get("orderID", ""))
            error = "" if success else resp.get("error", "Unknown error")
            if not success and _is_geoblock_error(error):
                record_live_buy_block(error)

            status = "OK" if success else "FAIL"
            self.log.info(
                f"[{self.name}][{status}] {trade.signal.side} "
                f"${trade.size_usd:.2f} @ {trade.signal.price:.3f} — "
                f"{trade.signal.market_question[:60]}"
            )

            return Result(
                trade=trade,
                success=success,
                order_id=str(order_id),
                error=error,
            )

        except subprocess.TimeoutExpired:
            self.log.error(f"[{self.name}] Order timed out for {trade.signal.token_id}")
            return Result(trade=trade, success=False, order_id="", error="Timeout")

        except json.JSONDecodeError as e:
            self.log.error(f"[{self.name}] Invalid JSON from place_order: {e}")
            return Result(trade=trade, success=False, order_id="", error=f"JSON parse: {e}")

        except Exception as e:
            self.log.error(f"[{self.name}] Order execution error: {e}")
            return Result(trade=trade, success=False, order_id="", error=str(e))

    # ── Live-mode helpers used by PositionManager ──────────────────
    #
    # These are opt-in subprocess wrappers for cancel / sell. Strategies
    # that go live inherit them; dry-run-only strategies never call them.
    # They follow the same FORCE_DRY_RUN guard and subprocess pattern as
    # `_place_real_order`.

    def cancel_live_order(self, order_id: str) -> bool:
        """Cancel a live order by ID via shared/place_order.py subprocess.

        Returns True on success, False on failure. Respects FORCE_DRY_RUN.
        """
        if os.environ.get("FORCE_DRY_RUN"):
            self.log.warning(
                f"[{self.name}][DRY-GUARD] FORCE_DRY_RUN — refusing to cancel "
                f"live order {order_id[:24]}"
            )
            return False

        cmd = [_VENV_PYTHON, _PLACE_ORDER, "cancel", "--order-id", order_id]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=30, env={**os.environ})
            if proc.stdout.strip():
                resp = json.loads(proc.stdout)
                return bool(resp.get("success", False))
            self.log.warning(
                f"[{self.name}] cancel empty stdout, stderr={proc.stderr[:120]!r}"
            )
            return False
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
            self.log.error(f"[{self.name}] cancel failed for {order_id[:24]}: {e}")
            return False

    def place_live_sell(self, token_id: str, price: float, size: float,
                        order_type: str = "GTC") -> Optional[str]:
        """Place a live SELL order via shared/place_order.py.

        Used by PositionManager's sell-high path. Default changed FOK→GTC
        2026-04-25 because FOK was failing systematically:
          - PM observes best_bid >= TP at moment T
          - Subprocess spawn takes 1-3s, by moment T+2 best_bid may have moved
          - FOK then fails (needs immediate match at full size)
        GTC instead rests on book; PM cancel-stale will re-evaluate next tick.

        Returns order_id string on success, None on failure. Respects
        FORCE_DRY_RUN.

        Args:
            token_id: CLOB token ID we're holding (must be the same token we
                      bought — sells a long position).
            price: Target price. For FOK this is the minimum acceptable price.
            size: USD size to sell (place_order.py interprets --size as USD).
            order_type: GTC / GTD / FOK / FAK. Default FOK.
        """
        if os.environ.get("FORCE_DRY_RUN"):
            self.log.warning(
                f"[{self.name}][DRY-GUARD] FORCE_DRY_RUN — refusing live SELL"
            )
            return None

        cmd = [
            _VENV_PYTHON, _PLACE_ORDER, "limit",
            "--token-id", token_id,
            "--side", "SELL",
            "--price", str(round(price, 4)),
            "--size", str(round(size, 2)),
            "--order-type", order_type,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=30, env={**os.environ})
            if proc.stdout.strip():
                try:
                    resp = json.loads(proc.stdout)
                except json.JSONDecodeError:
                    resp = {}
                if resp.get("success"):
                    oid = resp.get("order_id") or resp.get("orderID") or ""
                    return str(oid) if oid else None
                # Surface the real error from stdout (place_order.py emits
                # JSON to stdout for both success and failure cases).
                err = resp.get("error") or proc.stdout[:300]
                self.last_live_order_error = str(err)
                if _is_geoblock_error(err):
                    record_live_buy_block(err)
                self.log.warning(
                    f"[{self.name}] SELL failed: {err!r}"
                )
                return None
            self.log.warning(
                f"[{self.name}] SELL failed (empty stdout): "
                f"stderr={proc.stderr[:200]!r}"
            )
            self.last_live_order_error = proc.stderr[:200] or "empty stdout"
            return None
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
            self.last_live_order_error = str(e)
            self.log.error(f"[{self.name}] SELL exception: {e}")
            return None
