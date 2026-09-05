"""Tiny LSTM/GRU (PyTorch) and logistic/ridge sklearn baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BaselineKind = Literal["logistic", "ridge"]
RnnKind = Literal["lstm", "gru"]


def build_baseline(kind: BaselineKind = "logistic", C: float = 1.0) -> Pipeline:
    """StandardScaler + logistic or ridge classifier on flattened features."""
    if kind == "logistic":
        clf = LogisticRegression(C=C, max_iter=500, random_state=42)
    elif kind == "ridge":
        clf = RidgeClassifier(alpha=1.0 / C if C else 1.0)
    else:
        raise ValueError(f"unknown baseline kind: {kind}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


class TinyRNN(nn.Module):
    """CPU-friendly single-layer LSTM or GRU + linear head for binary logits."""

    def __init__(
        self,
        n_features: int,
        hidden: int = 16,
        kind: RnnKind = "lstm",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.kind = kind
        rnn_cls = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, features)
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        last = self.drop(last)
        return self.head(last).squeeze(-1)


@dataclass
class RNNTrainConfig:
    epochs: int = 12
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 4
    device: str = "cpu"
    seed: int = 42


class RNNClassifier:
    """Fit/predict wrapper around TinyRNN with early stopping on val loss."""

    def __init__(
        self,
        n_features: int,
        hidden: int = 16,
        kind: RnnKind = "lstm",
        dropout: float = 0.1,
        config: Optional[RNNTrainConfig] = None,
    ) -> None:
        self.n_features = n_features
        self.hidden = hidden
        self.kind = kind
        self.dropout = dropout
        self.config = config or RNNTrainConfig()
        self.model: Optional[TinyRNN] = None
        self.feat_mean_: Optional[np.ndarray] = None
        self.feat_std_: Optional[np.ndarray] = None

    def _set_seed(self) -> None:
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

    def _scale_fit(self, X: np.ndarray) -> np.ndarray:
        # X: (n, seq, f) — fit mean/std on train timesteps
        flat = X.reshape(-1, X.shape[-1])
        self.feat_mean_ = flat.mean(axis=0)
        self.feat_std_ = flat.std(axis=0)
        self.feat_std_ = np.where(self.feat_std_ < 1e-8, 1.0, self.feat_std_)
        return (X - self.feat_mean_) / self.feat_std_

    def _scale_transform(self, X: np.ndarray) -> np.ndarray:
        assert self.feat_mean_ is not None and self.feat_std_ is not None
        return (X - self.feat_mean_) / self.feat_std_

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "RNNClassifier":
        self._set_seed()
        cfg = self.config
        device = torch.device(cfg.device)
        Xs = self._scale_fit(X_train)
        self.model = TinyRNN(
            n_features=self.n_features,
            hidden=self.hidden,
            kind=self.kind,
            dropout=self.dropout,
        ).to(device)
        opt = torch.optim.Adam(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        loss_fn = nn.BCEWithLogitsLoss()

        Xt = torch.tensor(Xs, dtype=torch.float32, device=device)
        yt = torch.tensor(y_train.astype(np.float32), device=device)

        has_val = X_val is not None and y_val is not None and len(X_val) > 0
        if has_val:
            Xv = torch.tensor(
                self._scale_transform(X_val), dtype=torch.float32, device=device
            )
            yv = torch.tensor(y_val.astype(np.float32), device=device)

        best_state = None
        best_val = float("inf")
        stale = 0
        n = len(Xt)

        for _epoch in range(cfg.epochs):
            self.model.train()
            perm = torch.randperm(n, device=device)
            for start in range(0, n, cfg.batch_size):
                idx = perm[start : start + cfg.batch_size]
                opt.zero_grad()
                logits = self.model(Xt[idx])
                loss = loss_fn(logits, yt[idx])
                loss.backward()
                opt.step()

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    vloss = float(loss_fn(self.model(Xv), yv).item())
                if vloss < best_val - 1e-5:
                    best_val = vloss
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                    stale = 0
                else:
                    stale += 1
                    if stale >= cfg.patience:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        device = torch.device(self.config.device)
        Xs = self._scale_transform(X)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(
                torch.tensor(Xs, dtype=torch.float32, device=device)
            )
            p = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
