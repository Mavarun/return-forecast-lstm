"""Model smoke tests: shapes and fit/predict on tiny tensors."""

from __future__ import annotations

import numpy as np

from ret_lstm.models import RNNClassifier, RNNTrainConfig, TinyRNN, build_baseline
import torch


def test_baseline_fit_predict():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))
    y = (X[:, 0] > 0).astype(int)
    pipe = build_baseline("logistic")
    pipe.fit(X[:80], y[:80])
    pred = pipe.predict(X[80:])
    assert pred.shape == (20,)
    assert set(np.unique(pred)).issubset({0, 1})


def test_tiny_rnn_forward_shape():
    m = TinyRNN(n_features=3, hidden=8, kind="lstm")
    x = torch.randn(4, 10, 3)
    out = m(x)
    assert out.shape == (4,)


def test_rnn_classifier_fit_predict():
    rng = np.random.default_rng(2)
    n, seq, f = 120, 6, 3
    X = rng.normal(size=(n, seq, f)).astype(np.float32)
    y = (X[:, -1, 0] > 0).astype(int)
    clf = RNNClassifier(
        n_features=f,
        hidden=8,
        kind="gru",
        config=RNNTrainConfig(epochs=3, batch_size=32, patience=2, seed=2),
    )
    clf.fit(X[:90], y[:90], X[90:100], y[90:100])
    pred = clf.predict(X[100:])
    proba = clf.predict_proba(X[100:])
    assert pred.shape == (20,)
    assert proba.shape == (20, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
