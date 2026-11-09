"""Smoke tests for the pure-pandas rule modules (no TA-Lib / vectorbt / network needed).

Run:  pytest -q
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# rules that depend only on pandas/numpy
PURE_RULES = ["bbi", "boll", "demark", "divergence", "fib", "gmma", "ichimoku",
              "kdj", "ma", "macd", "obv", "rsi", "rvwap", "volume_price", "wvad"]


def make_ohlcv(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({
        "Open": close * (1 + rng.normal(0, 0.003, n)),
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Volume": rng.integers(100_000, 1_000_000, n).astype(float),
    }, index=idx)


@pytest.mark.parametrize("name", PURE_RULES)
def test_signal_shape_and_values(name):
    mod = importlib.import_module(f"tech_score.rules.{name}")
    df = make_ohlcv()
    s = mod.signal(df)
    assert isinstance(s, pd.Series)
    assert s.index.equals(df.index)
    assert set(s.unique()) <= {-1, 0, 1}
    assert mod.FAMILY and isinstance(mod.EPISODE, int)


@pytest.mark.parametrize("name", PURE_RULES)
def test_no_lookahead(name):
    """A signal on bar t must not change when bars after t are removed."""
    mod = importlib.import_module(f"tech_score.rules.{name}")
    df = make_ohlcv()
    full = mod.signal(df)
    cut = 350
    truncated = mod.signal(df.iloc[:cut])
    pd.testing.assert_series_equal(full.iloc[:cut], truncated, check_names=False)


def test_gmma_fires_once_per_regime_change():
    """Regression: `~bool.shift(1).fillna(False)` became an int invert on pandas>=3.

    A monotonically rising series puts the short EMAs above the long ones for
    good, so GMMA must emit exactly one buy, not one per bar.
    """
    from tech_score.rules import gmma
    n = 300
    df = make_ohlcv(n)
    df["Close"] = np.linspace(100, 200, n)
    s = gmma.signal(df)
    assert (s == 1).sum() == 1
    assert (s == -1).sum() == 0


def test_demark_setup_nine_on_straight_decline():
    from tech_score.rules import demark
    n = 60
    df = make_ohlcv(n)
    df["Close"] = np.linspace(100, 50, n)
    s = demark.signal(df)
    # bars 0-3 have no c[-4]; the down run starts at bar 4 and hits 9 at bar 12
    assert s.iloc[12] == 1
    assert (s == 1).sum() == 1
    assert (s == -1).sum() == 0
