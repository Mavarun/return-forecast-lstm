"""Feature shape and no look-ahead checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ret_lstm.data import make_synthetic_returns
from ret_lstm.features import build_feature_matrix, make_sequences


def test_feature_shapes_and_alignment():
    df = make_synthetic_returns(n_bars=300, seed=0)
    X, y, fwd = build_feature_matrix(df["return"], n_lags=5, vol_window=21)
    assert len(X) == len(y) == len(fwd)
    assert X.shape[1] == 5 + 2  # lags + vol + vol_lag1
    assert set(y.unique()).issubset({0, 1})
    # first valid index after max lag/vol warmup
    assert X.index[0] >= df.index[21]


def test_no_lookahead_in_lags():
    """ret_lag1 at t must equal return[t-1], never return[t] or later."""
    rng = np.random.default_rng(1)
    rets = pd.Series(rng.normal(0, 0.01, 80), index=pd.RangeIndex(80))
    X, y, fwd = build_feature_matrix(rets, n_lags=3, vol_window=5)
    for t in X.index:
        assert X.loc[t, "ret_lag1"] == rets.loc[t - 1]
        assert X.loc[t, "ret_lag2"] == rets.loc[t - 2]
        assert X.loc[t, "ret_lag3"] == rets.loc[t - 3]
        # label is next return
        assert fwd.loc[t] == rets.loc[t + 1]
        # features must not equal contemporaneous or future return
        assert X.loc[t, "ret_lag1"] != rets.loc[t]


def test_make_sequences_no_future():
    X = np.arange(20 * 3, dtype=float).reshape(20, 3)
    y = np.arange(20)
    fwd = np.arange(20) * 0.01
    Xs, ys, fs = make_sequences(X, y, fwd, seq_len=5)
    assert Xs.shape == (16, 5, 3)
    # last window ends at row 19
    np.testing.assert_array_equal(Xs[-1], X[15:20])
    assert ys[-1] == y[19]
    assert fs[-1] == fwd[19]
