"""
CityBond dry-run strategy.

Forks the current BondBuyerV2 live logic, but changes admission control for
weather temperature markets: p_fair is recalculated from a city-level Open-
Meteo forecast distribution instead of using the generic bond prior.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import List

from alpha_suite.base import Signal
from alpha_suite.engines.city_weather_model import estimate_city_weather_probability
from alpha_suite.strategies import bond_buyer as bb
from alpha_suite.strategies.bond_buyer import MIN_EDGE, MIN_EV
from alpha_suite.strategies_v2.bond_buyer_v2 import BondBuyerV2
from alpha_suite.utils.api import fetch_gamma_markets
from alpha_suite.utils.risk import ev_calc


class CityBond(BondBuyerV2):
    """Dry-run city-calibrated weather bond strategy."""

    name = "分城市bond"
    dry_run = True
    max_bet = float(os.environ.get("CITY_BOND_MAX_BET", "6.0"))
    bankroll_pct = float(os.environ.get("CITY_BOND_BANKROLL_PCT", "0.10"))
    max_open_cost = float(os.environ.get("CITY_BOND_MAX_OPEN_COST", "120.0"))
    max_daily_loss = float(os.environ.get("CITY_BOND_MAX_DAILY_LOSS", "40.0"))
    risk_epoch = os.environ.get("CITY_BOND_RISK_EPOCH", "").strip()

    def scan(self) -> List[Signal]:
        self.log.info("-- [CITY-BOND] Scanning city weather temperature markets --")
        self._seen_bonds.clear()

        if not self.state.check_bankroll(self.name):
            self.log.info("  [city-bond] bankroll exhausted")
            return []

        now = datetime.now(timezone.utc)
        params = {
            "active": "true",
            "closed": "false",
            "end_date_min": (
                now - timedelta(hours=bb.WEATHER_GAMMA_LOOKBACK_HOURS)
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_date_max": (
                now + timedelta(days=bb.BOND_EXPIRY_DAYS)
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "order": "endDate",
            "ascending": "true",
        }
        markets = fetch_gamma_markets(params, max_pages=bb.BOND_GAMMA_MAX_PAGES)
        deep_pages = max(bb.BOND_GAMMA_MAX_PAGES, bb.BOND_DEEP_SCAN_MAX_PAGES)
        if deep_pages > bb.BOND_GAMMA_MAX_PAGES and len(markets) >= bb.BOND_GAMMA_MAX_PAGES * 100:
            extra = fetch_gamma_markets(
                params,
                max_pages=deep_pages - bb.BOND_GAMMA_MAX_PAGES,
                start_offset=bb.BOND_GAMMA_MAX_PAGES * 100,
            )
            markets.extend(extra)
            self.log.info(
                f"  [city-bond] deep scan fetched {len(extra)} additional markets "
                f"(pages={deep_pages})"
            )

        raw_signals: list[Signal] = []
        counts: dict[str, int] = {}
        for m in markets:
            parsed = self._parse_market(m)
            if not parsed:
                counts["parse_fail"] = counts.get("parse_fail", 0) + 1
                continue
            if parsed.get("volume", 0) < bb.MIN_VOLUME:
                counts["volume_lt_min"] = counts.get("volume_lt_min", 0) + 1
                continue
            q_full = parsed.get("question") or ""
            category = bb.classify_bond_question(q_full)
            if category != "weather":
                counts[f"not_weather_{category}"] = counts.get(f"not_weather_{category}", 0) + 1
                continue
            if not bb.is_weather_temperature_market(q_full):
                counts["weather_not_high_temp"] = counts.get("weather_not_high_temp", 0) + 1
                continue
            if parsed.get("closed") or not parsed.get("active", True):
                counts["weather_closed"] = counts.get("weather_closed", 0) + 1
                continue
            if not parsed.get("accepting_orders", True):
                counts["weather_not_accepting"] = counts.get("weather_not_accepting", 0) + 1
                continue

            end_str = parsed.get("end_date", "")
            try:
                end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                if end_dt.tzinfo is None:
                    end_dt = end_dt.replace(tzinfo=timezone.utc)
                end_dt = end_dt.astimezone(timezone.utc)
            except (ValueError, TypeError):
                counts["bad_end_date"] = counts.get("bad_end_date", 0) + 1
                continue

            game_start_dt = bb._parse_dt(parsed.get("game_start_time"))
            clock_value, clock_dt, clock_source = bb.weather_entry_clock(
                q_full, now, end_dt, game_start_dt
            )
            if clock_value is None:
                counts["weather_clock_unknown"] = counts.get("weather_clock_unknown", 0) + 1
                continue
            if clock_source == "game_start_age":
                if not (bb.WEATHER_LOCAL_DAY_MIN_HOURS <= clock_value <= bb.WEATHER_LOCAL_DAY_MAX_HOURS):
                    counts["weather_time_reject"] = counts.get("weather_time_reject", 0) + 1
                    continue
            else:
                if clock_dt and clock_dt < now + timedelta(hours=1):
                    counts["weather_too_close"] = counts.get("weather_too_close", 0) + 1
                    continue
                if not (bb.WEATHER_BOND_MIN_HOURS <= clock_value <= bb.WEATHER_BOND_MAX_HOURS):
                    counts["weather_time_reject"] = counts.get("weather_time_reject", 0) + 1
                    continue

            best_idx = -1
            best_price = 0.0
            for i, price in enumerate(parsed.get("prices", [])):
                if bb.BOND_MIN_PRICE <= price <= bb.WEATHER_BOND_MAX_PRICE and price > best_price:
                    best_price = price
                    best_idx = i
            if best_idx < 0:
                counts["no_price_90_95"] = counts.get("no_price_90_95", 0) + 1
                continue

            outcome = parsed["outcomes"][best_idx]
            raw_signals.append(Signal(
                strategy=self.name,
                market_question=q_full,
                condition_id=parsed["condition_id"],
                token_id=parsed["tokens"][best_idx],
                side="BUY",
                price=best_price,
                p_fair=0.5,
                edge=0.0,
                ev=0.0,
                metadata={
                    "outcome": outcome,
                    "outcome_index": best_idx,
                    "hours_to_expiry": round(clock_value, 1),
                    "end_date": parsed.get("end_date", ""),
                    "gamma_hours_to_expiry": round((end_dt - now).total_seconds() / 3600, 1),
                    "volume": parsed.get("volume", 0),
                    "neg_risk": parsed.get("neg_risk", False),
                    "event_slug": parsed.get("event_slug", ""),
                    "tick_size": float(parsed.get("orderPriceMinTickSize", 0.01) or 0.01),
                    "category": "weather",
                    "weather_city": bb.weather_city_key(q_full),
                    "weather_city_adjustment": bb.weather_city_adjustment(q_full),
                    "weather_expected_close": (
                        bb.expected_weather_close_dt(q_full, end_dt).isoformat()
                        if bb.expected_weather_close_dt(q_full, end_dt)
                        else ""
                    ),
                    "weather_clock_source": clock_source,
                    "weather_local_day_age": round(clock_value, 1) if clock_source == "game_start_age" else None,
                    "weather_hours_to_close": round(clock_value, 1) if clock_source != "game_start_age" else None,
                    "weather_game_start": parsed.get("game_start_time", ""),
                    "accepting_orders": parsed.get("accepting_orders", True),
                },
            ))

        if counts:
            summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            self.log.info(f"  [city-bond] scan rejects: {summary}")

        signals: list[Signal] = []
        min_edge = float(os.environ.get("CITY_BOND_MIN_EDGE", str(max(MIN_EDGE, 0.02))))
        min_ev = float(os.environ.get("CITY_BOND_MIN_EV", str(MIN_EV)))

        for signal in raw_signals:
            meta = dict(signal.metadata or {})
            if meta.get("category") != "weather":
                continue
            outcome = str(meta.get("outcome") or "").strip()
            estimate = estimate_city_weather_probability(signal.market_question, outcome)
            if estimate is None:
                self.log.info(
                    f"  [city-bond] skip no forecast: {signal.market_question[:70]}"
                )
                continue

            p_fair = float(estimate.p_outcome)
            edge = p_fair - float(signal.price)
            ev = ev_calc(p_fair, signal.price)
            if edge < min_edge:
                self.log.info(
                    f"  [city-bond] skip edge {edge:.3f} < {min_edge:.3f}: "
                    f"{signal.market_question[:70]}"
                )
                continue
            if ev < min_ev:
                self.log.info(
                    f"  [city-bond] skip EV {ev:.4f} < {min_ev:.4f}: "
                    f"{signal.market_question[:70]}"
                )
                continue

            meta.update({
                "city_bond": True,
                "city_model_source": estimate.source,
                "city_model_p_yes": estimate.p_yes,
                "city_model_p_outcome": estimate.p_outcome,
                "city_model_forecast_temp": estimate.forecast_temp,
                "city_model_forecast_unit": estimate.forecast_unit,
                "city_model_sigma": estimate.sigma,
                "city_model_kind": estimate.kind,
                "city_model_lo": estimate.lo,
                "city_model_hi": estimate.hi,
                "city_model_target_date": estimate.target_date,
                "pre_city_prior_p_fair": signal.p_fair,
                "min_edge_required": max(
                    float(meta.get("min_edge_required", min_edge) or min_edge),
                    min_edge,
                ),
            })

            signals.append(Signal(
                strategy=self.name,
                market_question=signal.market_question,
                condition_id=signal.condition_id,
                token_id=signal.token_id,
                side=signal.side,
                price=signal.price,
                p_fair=round(p_fair, 4),
                edge=round(edge, 4),
                ev=round(ev, 4),
                metadata=meta,
            ))
            self.log.info(
                f"  [city-bond] {outcome}={signal.price:.3f} "
                f"p={p_fair:.3f} ev={ev:.4f} city={estimate.city} "
                f"fcst={estimate.forecast_temp:.1f}{estimate.forecast_unit} "
                f"| {signal.market_question[:60]}"
            )

        self.log.info(f"  [city-bond] Done: {len(signals)} city-calibrated signals")
        return signals
