"""Costed long/flat or long/short signals and walk-forward evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

from ret_lstm.features import make_sequences
from ret_lstm.metrics import summarize_pnl
from ret_lstm.models import RNNClassifier, RNNTrainConfig, build_baseline


Mode = Literal["long_flat", "long_short"]
ModelName = Literal["baseline", "lstm", "gru"]


def signals_from_pred(
    y_pred: np.ndarray,
    mode: Mode = "long_flat",
) -> np.ndarray:
    """Map class predictions {0,1} to positions."""
    y = np.asarray(y_pred).ravel().astype(int)
    if mode == "long_flat":
        return y.astype(float)  # 1 long, 0 flat
    if mode == "long_short":
        return np.where(y == 1, 1.0, -1.0)
    raise ValueError(f"unknown mode: {mode}")


@dataclass
class WalkForwardConfig:
    train_size: int = 400
    test_size: int = 80
    step: int = 80
    seq_len: int = 10
    n_lags: int = 5
    mode: Mode = "long_flat"
    cost_bps: float = 5.0
    baseline_kind: str = "logistic"
    rnn_hidden: int = 16
    rnn_epochs: int = 12
    rnn_batch: int = 64
    seed: int = 42
    val_frac: float = 0.15


@dataclass
class FoldResult:
    fold: int
    train_end: int
    test_start: int
    test_end: int
    metrics_costed: dict[str, float]
    metrics_uncosted: dict[str, float]
    is_metrics_costed: dict[str, float] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    model: str
    folds: list[FoldResult]
    oos_costed: dict[str, float]
    oos_uncosted: dict[str, float]
    is_costed: dict[str, float]


def _barrier_check(train_end: int, test_start: int) -> None:
    """Enforce strict train/test barrier (no overlap)."""
    if test_start < train_end:
        raise ValueError(
            f"look-ahead barrier violated: test_start={test_start} < train_end={train_end}"
        )


def _last_step_features(X_seq: np.ndarray) -> np.ndarray:
    """Flatten last timestep features for sklearn baseline."""
    return X_seq[:, -1, :]


def walk_forward(
    X: np.ndarray,
    y: np.ndarray,
    fwd: np.ndarray,
    model_name: ModelName = "lstm",
    cfg: Optional[WalkForwardConfig] = None,
) -> WalkForwardResult:
    """
    Expanding or rolling walk-forward: train on [0, train_end), test on
    [test_start, test_end) with test_start == train_end (contiguous, no leak).

    For RNN models, sequences are built first; indices refer to sequence rows.
    """
    cfg = cfg or WalkForwardConfig()
    X_seq, y_seq, fwd_seq = make_sequences(X, y, fwd, seq_len=cfg.seq_len)
    n = len(X_seq)
    if n < cfg.train_size + cfg.test_size:
        raise ValueError(
            f"not enough sequence rows ({n}) for train={cfg.train_size} "
            f"+ test={cfg.test_size}"
        )

    folds: list[FoldResult] = []
    oos_pos: list[np.ndarray] = []
    oos_fwd: list[np.ndarray] = []
    oos_y: list[np.ndarray] = []
    oos_pred: list[np.ndarray] = []
    is_pos: list[np.ndarray] = []
    is_fwd: list[np.ndarray] = []
    is_y: list[np.ndarray] = []
    is_pred: list[np.ndarray] = []

    fold_i = 0
    train_end = cfg.train_size
    while train_end + cfg.test_size <= n:
        test_start = train_end
        test_end = train_end + cfg.test_size
        _barrier_check(train_end, test_start)

        X_tr = X_seq[:train_end]
        y_tr = y_seq[:train_end]
        fwd_tr = fwd_seq[:train_end]
        X_te = X_seq[test_start:test_end]
        y_te = y_seq[test_start:test_end]
        fwd_te = fwd_seq[test_start:test_end]

        n_val = max(1, int(len(X_tr) * cfg.val_frac))
        X_fit, y_fit = X_tr[:-n_val], y_tr[:-n_val]
        X_val, y_val = X_tr[-n_val:], y_tr[-n_val:]

        if model_name == "baseline":
            pipe = build_baseline(kind=cfg.baseline_kind)  # type: ignore[arg-type]
            pipe.fit(_last_step_features(X_fit), y_fit)
            pred_te = pipe.predict(_last_step_features(X_te))
            pred_tr = pipe.predict(_last_step_features(X_tr))
        else:
            kind = "lstm" if model_name == "lstm" else "gru"
            rnn = RNNClassifier(
                n_features=X_seq.shape[-1],
                hidden=cfg.rnn_hidden,
                kind=kind,  # type: ignore[arg-type]
                config=RNNTrainConfig(
                    epochs=cfg.rnn_epochs,
                    batch_size=cfg.rnn_batch,
                    seed=cfg.seed + fold_i,
                ),
            )
            rnn.fit(X_fit, y_fit, X_val, y_val)
            pred_te = rnn.predict(X_te)
            pred_tr = rnn.predict(X_tr)

        pos_te = signals_from_pred(pred_te, mode=cfg.mode)
        pos_tr = signals_from_pred(pred_tr, mode=cfg.mode)

        m_c = summarize_pnl(pos_te, fwd_te, cost_bps=cfg.cost_bps, y_true=y_te, y_pred=pred_te)
        m_u = summarize_pnl(pos_te, fwd_te, cost_bps=0.0, y_true=y_te, y_pred=pred_te)
        m_is = summarize_pnl(
            pos_tr, fwd_tr, cost_bps=cfg.cost_bps, y_true=y_tr, y_pred=pred_tr
        )

        folds.append(
            FoldResult(
                fold=fold_i,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                metrics_costed=m_c,
                metrics_uncosted=m_u,
                is_metrics_costed=m_is,
            )
        )
        oos_pos.append(pos_te)
        oos_fwd.append(fwd_te)
        oos_y.append(y_te)
        oos_pred.append(pred_te)
        is_pos.append(pos_tr)
        is_fwd.append(fwd_tr)
        is_y.append(y_tr)
        is_pred.append(pred_tr)

        fold_i += 1
        train_end += cfg.step

    if not folds:
        raise RuntimeError("no walk-forward folds produced")

    pos_o = np.concatenate(oos_pos)
    fwd_o = np.concatenate(oos_fwd)
    y_o = np.concatenate(oos_y)
    pred_o = np.concatenate(oos_pred)
    # IS aggregate uses last fold's train window (honest: report fold-mean too)
    pos_i = is_pos[-1]
    fwd_i = is_fwd[-1]
    y_i = is_y[-1]
    pred_i = is_pred[-1]

    return WalkForwardResult(
        model=model_name,
        folds=folds,
        oos_costed=summarize_pnl(
            pos_o, fwd_o, cost_bps=cfg.cost_bps, y_true=y_o, y_pred=pred_o
        ),
        oos_uncosted=summarize_pnl(
            pos_o, fwd_o, cost_bps=0.0, y_true=y_o, y_pred=pred_o
        ),
        is_costed=summarize_pnl(
            pos_i, fwd_i, cost_bps=cfg.cost_bps, y_true=y_i, y_pred=pred_i
        ),
    )


def result_to_dict(res: WalkForwardResult) -> dict[str, Any]:
    return {
        "model": res.model,
        "n_folds": len(res.folds),
        "oos_costed": res.oos_costed,
        "oos_uncosted": res.oos_uncosted,
        "is_costed": res.is_costed,
        "fold_oos_dir_acc": [
            f.metrics_costed.get("dir_accuracy", float("nan")) for f in res.folds
        ],
        "fold_oos_sharpe_costed": [
            f.metrics_costed.get("sharpe", float("nan")) for f in res.folds
        ],
    }
