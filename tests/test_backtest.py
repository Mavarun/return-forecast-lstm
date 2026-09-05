"""Cost drag and walk-forward train/test barrier."""

from __future__ import annotations

import numpy as np
import pytest

from ret_lstm.backtest import (
    WalkForwardConfig,
    _barrier_check,
    signals_from_pred,
    walk_forward,
)
from ret_lstm.data import make_synthetic_returns
from ret_lstm.features import build_feature_matrix
from ret_lstm.metrics import strategy_pnl, summarize_pnl


def test_costs_reduce_pnl_vs_zero_cost():
    rng = np.random.default_rng(7)
    n = 200
    # flipping positions every bar maximizes turnover
    pos = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)])
    fwd = rng.normal(0.0005, 0.01, size=n)
    pnl0 = strategy_pnl(pos, fwd, cost_bps=0.0)
    pnl_c = strategy_pnl(pos, fwd, cost_bps=10.0)
    assert np.sum(pnl_c) < np.sum(pnl0)
    s0 = summarize_pnl(pos, fwd, cost_bps=0.0)
    sc = summarize_pnl(pos, fwd, cost_bps=10.0)
    assert sc["total_pnl"] < s0["total_pnl"]


def test_barrier_rejects_overlap():
    _barrier_check(100, 100)  # ok: contiguous
    with pytest.raises(ValueError, match="barrier"):
        _barrier_check(100, 99)


def test_walk_forward_train_test_barrier():
    df = make_synthetic_returns(n_bars=900, seed=3)
    X, y, fwd = build_feature_matrix(df["return"], n_lags=5, vol_window=21)
    cfg = WalkForwardConfig(
        train_size=250,
        test_size=50,
        step=50,
        seq_len=8,
        rnn_epochs=2,
        rnn_hidden=8,
        cost_bps=5.0,
    )
    res = walk_forward(
        X.values.astype(float),
        y.values.astype(int),
        fwd.values.astype(float),
        model_name="baseline",
        cfg=cfg,
    )
    assert len(res.folds) >= 2
    for f in res.folds:
        assert f.test_start == f.train_end
        assert f.test_end > f.test_start
        assert f.test_start >= f.train_end


def test_signals_modes():
    pred = np.array([1, 0, 1, 0])
    assert list(signals_from_pred(pred, "long_flat")) == [1.0, 0.0, 1.0, 0.0]
    assert list(signals_from_pred(pred, "long_short")) == [1.0, -1.0, 1.0, -1.0]
