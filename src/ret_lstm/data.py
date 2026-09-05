"""Price/return loaders: synthetic AR returns (default) or yfinance daily bars."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def make_synthetic_returns(
    n_bars: int = 1200,
    seed: int = 42,
    ar_coef: float = 0.08,
    vol: float = 0.012,
    start: str = "2018-01-01",
) -> pd.DataFrame:
    """
    Synthetic daily close/returns with mild AR(1) drift in returns.

    Citation: generated in-process for reproducible offline research;
    not market data. Parameters chosen for ~1.2% daily vol (equity-like).
    """
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, vol, size=n_bars)
    rets = np.empty(n_bars, dtype=float)
    rets[0] = eps[0]
    for t in range(1, n_bars):
        rets[t] = ar_coef * rets[t - 1] + eps[t]
    idx = pd.bdate_range(start=start, periods=n_bars)
    close = 100.0 * np.cumprod(1.0 + rets)
    return pd.DataFrame({"close": close, "return": rets}, index=idx)


def load_yfinance_returns(
    ticker: str = "SPY",
    start: str = "2018-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Daily adjusted close and simple returns via yfinance.

    Citation: Yahoo Finance market data accessed through the `yfinance`
    Python package (https://github.com/ranaroussi/yfinance). For research
    only; subject to Yahoo terms of use.
    """
    import yfinance as yf

    kwargs = {"start": start, "auto_adjust": True, "progress": False}
    if end is not None:
        kwargs["end"] = end
    hist = yf.download(ticker, **kwargs)
    if hist is None or hist.empty:
        raise RuntimeError(f"yfinance returned empty history for {ticker}")
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    close = hist["Close"].astype(float).dropna()
    rets = close.pct_change()
    out = pd.DataFrame({"close": close, "return": rets}).dropna()
    if len(out) < 100:
        raise RuntimeError(f"insufficient bars for {ticker}: {len(out)}")
    return out


def load_returns(
    source: str = "synthetic",
    ticker: str = "SPY",
    n_bars: int = 1200,
    seed: int = 42,
    start: str = "2018-01-01",
    end: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """
    Load returns; fall back to synthetic if yfinance fails.

    Returns (frame, note) where note documents the data provenance.
    """
    if source == "synthetic":
        df = make_synthetic_returns(n_bars=n_bars, seed=seed, start=start)
        note = (
            f"synthetic AR(1) returns n={len(df)} seed={seed} "
            "(in-process; not market data)"
        )
        return df, note

    try:
        df = load_yfinance_returns(ticker=ticker, start=start, end=end)
        note = (
            f"yfinance daily {ticker} n={len(df)} start={start} "
            "(Yahoo Finance via yfinance; research use)"
        )
        return df, note
    except Exception as exc:  # noqa: BLE001 — intentional fallback
        df = make_synthetic_returns(n_bars=n_bars, seed=seed, start=start)
        note = (
            f"yfinance failed ({type(exc).__name__}: {exc}); "
            f"fell back to synthetic AR(1) n={len(df)} seed={seed}"
        )
        return df, note
