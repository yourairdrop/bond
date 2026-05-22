"""CLI and orchestration for the Alpha Suite backtest."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from alpha_suite.backtesting.data import HistoricalDataClient
from alpha_suite.backtesting.metrics import compute_trade_stats, log_strategy_results
from alpha_suite.backtesting.strategies import (
    run_bond_backtest,
    run_coverage_backtest,
    run_llm_signal_backtest,
    run_multi_arb_backtest,
)


def default_log_path() -> str:
    """Return the repo-local default log path."""
    return str(Path(__file__).resolve().parents[2] / "state" / "backtest_alpha_output.txt")


@dataclass(frozen=True)
class BacktestConfig:
    """User-configurable backtest settings."""

    days_back: int = 60
    max_pages: int = 50
    fee_rate: float = 0.02
    log_path: str = default_log_path()
    llm_sample_size: int = 40
    include_llm: bool = True
    max_multi_events: int = 200
    max_weather_events: int = 100
    history_workers: int = 32


class BacktestLogger:
    """Tiny dual sink logger for stdout + file output."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8")

    def log(self, message: str) -> None:
        print(message, flush=True)
        self.handle.write(message + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def run_backtest(config: BacktestConfig) -> dict[str, list]:
    """Run the full backtest and write the human-readable report."""
    logger = BacktestLogger(config.log_path)
    try:
        log = logger.log
        client = HistoricalDataClient(log=log)

        log("=" * 70)
        log(f"  ALPHA SUITE v2 - {config.days_back}-DAY BACKTEST")
        log("=" * 70)

        log("\n--- Fetching data ---")
        markets = client.fetch_closed_markets(config.days_back, config.max_pages)
        log(f"  Total markets: {len(markets)}")
        events = client.fetch_closed_events(config.days_back, config.max_pages)
        log(f"  Total events: {len(events)}")

        log("\n--- Running strategies ---")
        bond = run_bond_backtest(
            markets,
            client,
            fee_rate=config.fee_rate,
            history_workers=config.history_workers,
            log=log,
        )
        log_strategy_results(log, "BOND BUYER", bond)

        arb = run_multi_arb_backtest(
            events,
            client,
            fee_rate=config.fee_rate,
            max_events=config.max_multi_events,
            history_workers=config.history_workers,
            log=log,
        )
        log_strategy_results(log, "ARB SCANNER", arb)

        llm = run_llm_signal_backtest(
            markets,
            client,
            fee_rate=config.fee_rate,
            llm_sample_size=config.llm_sample_size,
            include_llm=config.include_llm,
            history_workers=config.history_workers,
            log=log,
        )
        log_strategy_results(log, "LLM SIGNAL", llm)

        coverage = run_coverage_backtest(
            events,
            client,
            fee_rate=config.fee_rate,
            max_weather_events=config.max_weather_events,
            history_workers=config.history_workers,
            log=log,
        )
        log_strategy_results(log, "COVERAGE ARB", coverage)

        all_trades = bond + arb + llm + coverage
        log("\n" + "=" * 70)
        log("  COMBINED RESULTS")
        log("=" * 70)
        if all_trades:
            stats = compute_trade_stats(all_trades)
            log(
                f"  Total: {stats['n_trades']} trades, "
                f"{stats['wins']}W/{stats['losses']}L ({stats['win_rate']:.1f}%)"
            )
            log(
                f"  PnL: ${stats['total_pnl']:+.2f} | Cost: ${stats['total_cost']:.2f} "
                f"| ROI: {stats['roi']:+.1f}%"
            )
            log("\n  By Strategy:")
            for strategy_name, trades in (
                ("bond", bond),
                ("multi_arb", arb),
                ("llm_signal", llm),
                ("coverage", coverage),
            ):
                if not trades:
                    continue
                sub = compute_trade_stats(trades)
                log(
                    f"    {strategy_name:15s}: {sub['wins']}/{sub['n_trades']} won, "
                    f"PnL=${sub['total_pnl']:+.2f}, ROI={sub['roi']:+.1f}%"
                )

        log("\nDone.")
        return {
            "bond": bond,
            "multi_arb": arb,
            "llm_signal": llm,
            "coverage": coverage,
        }
    finally:
        logger.close()


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days-back", type=int, default=60, help="number of days of closed data to backtest")
    parser.add_argument("--max-pages", type=int, default=50, help="maximum Gamma pages to fetch per endpoint")
    parser.add_argument("--fee-rate", type=float, default=0.02, help="entry fee rate applied to notional")
    parser.add_argument("--log-path", default=default_log_path(), help="path for the text report")
    parser.add_argument("--llm-sample-size", type=int, default=40, help="maximum LLM-evaluated markets")
    parser.add_argument("--skip-llm", action="store_true", help="disable the LLM backtest")
    parser.add_argument("--max-multi-events", type=int, default=200, help="max multi-outcome events to inspect")
    parser.add_argument("--max-weather-events", type=int, default=100, help="max weather events to inspect")
    parser.add_argument("--history-workers", type=int, default=32, help="parallel workers for price-history prefetch")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = build_parser().parse_args(argv)
    config = BacktestConfig(
        days_back=args.days_back,
        max_pages=args.max_pages,
        fee_rate=args.fee_rate,
        log_path=args.log_path,
        llm_sample_size=args.llm_sample_size,
        include_llm=not args.skip_llm,
        max_multi_events=args.max_multi_events,
        max_weather_events=args.max_weather_events,
        history_workers=args.history_workers,
    )
    run_backtest(config)
    return 0
