#!/usr/bin/env python3
"""Alpha Suite v2 backtest entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from alpha_suite.backtesting.runner import main


if __name__ == "__main__":
    raise SystemExit(main())
