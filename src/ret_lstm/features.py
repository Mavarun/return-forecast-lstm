"""Lagged return/vol features and next-bar direction labels (no look-ahead)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def realized_vol(returns: pd.Series, window: int = 21) -> pd.Series:
    """Trailing realized volatility (std of past `window` returns, shifted)."""
    return returns.rolling(window, min_periods=window).std()


def build_feature_matrix(
    returns: pd.Series,
    n_lags: int = 5,
    vol_window: int = 21,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build lagged return + vol features and next-bar targets.

    Features at time t use only information available at t (lags of return
    and trailing vol ending at t). Label is sign of return at t+1.
    Forward return series is also returned for PnL evaluation.

    Returns
    -------
    X : DataFrame of features aligned to valid rows
    y_dir : Series of {0, 1} direction labels (1 = next return > 0)
    fwd_ret : Series of next-bar simple returns
    """
    if n_lags < 1:
        raise ValueError("n_lags must be >= 1")
    r = returns.astype(float)
    cols: dict[str, pd.Series] = {}
    for lag in range(1, n_lags + 1):
        # lag k at index t is return[t-k] — past only
        cols[f"ret_lag{lag}"] = r.shift(lag)
    vol = realized_vol(r, window=vol_window)
    cols["vol"] = vol
    # vol of lagged returns (optional second vol scale)
    cols["vol_lag1"] = vol.shift(1)

    X = pd.DataFrame(cols, index=r.index)
    fwd_ret = r.shift(-1)  # next bar return — label only, never as feature
    y_dir = (fwd_ret > 0).astype(int)

    valid = X.notna().all(axis=1) & fwd_ret.notna()
    X = X.loc[valid]
    y_dir = y_dir.loc[valid]
    fwd_ret = fwd_ret.loc[valid]
    return X, y_dir, fwd_ret


def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    fwd: np.ndarray,
    seq_len: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sliding windows of length `seq_len` ending at each label index.

    Window X[i - seq_len + 1 : i + 1] predicts y[i] / fwd[i].
    No future rows enter the window.
    """
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    n = len(X)
    if n < seq_len:
        raise ValueError(f"need at least seq_len={seq_len} rows, got {n}")
    xs, ys, fs = [], [], []
    for i in range(seq_len - 1, n):
        xs.append(X[i - seq_len + 1 : i + 1])
        ys.append(y[i])
        fs.append(fwd[i])
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.int64),
        np.asarray(fs, dtype=np.float64),
    )
