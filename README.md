# Alpha Suite Public Notes

Public-facing research notes for a Polymarket bond-style strategy framework.

This repository contains a public-facing, sanitized subset of the framework.

It does **not** include:

- private keys
- wallet credentials
- live deployment secrets
- production state files
- the current higher-margin strategy logic

It does include:

- a high-level description of the bond-style framework
- the weather bond strategy code path
- the weather-bond backtest system
- the city-calibrated weather bond dry-run variant
- a Chinese strategy article
- a Chinese usage note
- a Chinese Twitter/X thread draft
- a boundary note on what can and cannot be claimed from the historical data

It is organized so the folder can be uploaded directly as a standalone public repository.

## Files

- `README_CN.md`
  - Chinese entrypoint
- `alpha_suite/`
  - sanitized weather bond strategy code and research utilities
- `alpha_suite/backtest.py`
  - CLI entrypoint for the public backtest runner
- `alpha_suite/backtesting/`
  - public backtest system used to replay bond-style ideas on historical markets
- `docs/01_PUBLIC_STRATEGY_PLAN.md`
  - release plan and public positioning
- `docs/02_STRATEGY_ARTICLE_CN.md`
  - long-form Chinese article focused on strategy method
- `docs/03_HOW_TO_USE_CN.md`
  - how to use the framework as research, not as a turnkey live system
- `docs/04_TWITTER_THREAD_CN.md`
  - Chinese Twitter/X thread draft
- `docs/05_DATA_BASIS_AND_BOUNDARY.md`
  - wording boundaries and evidence standards

## Positioning

The correct positioning for this repository is:

> a real research framework that ran, produced a stable profitable window for a period of time, and was later retired in favor of a different direction.

This repository should be read as a methodology and strategy release, not as a plug-and-play live trading system.

## Strategy Roles

This repository currently exposes three strategy implementations:

- `alpha_suite/strategies/bond_buyer.py`
  - the original `BondBuyer` framework
- `alpha_suite/strategies_v2/bond_buyer_v2.py`
  - the later `BondBuyerV2` framework, which became the main working branch
- `alpha_suite/strategies_v2/city_bond.py`
  - a city-calibrated dry-run variant built on top of `BondBuyerV2`

The public strategy content is primarily about the weather bond line of research.

## Backtesting

The public repo now includes the historical backtest subsystem:

- `python -m alpha_suite.backtest --help`

This backtest layer is included as research infrastructure, not as a claim that historical replay is sufficient for live deployment.
