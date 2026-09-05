#!/usr/bin/env python3
"""Run the LSTM vs baseline walk-forward research slice; dump metrics JSON/table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# allow running without install
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ret_lstm.backtest import WalkForwardConfig, result_to_dict, walk_forward
from ret_lstm.data import load_returns
from ret_lstm.features import build_feature_matrix


def _fmt(x: float) -> str:
    if x != x:  # NaN
        return "nan"
    return f"{x:.4f}"


def print_table(rows: list[dict]) -> None:
    headers = [
        "model",
        "split",
        "costed",
        "dir_acc",
        "hit_rate",
        "sharpe",
        "total_pnl",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        print(
            "| "
            + " | ".join(
                [
                    str(r["model"]),
                    str(r["split"]),
                    str(r["costed"]),
                    _fmt(r["dir_acc"]),
                    _fmt(r["hit_rate"]),
                    _fmt(r["sharpe"]),
                    _fmt(r["total_pnl"]),
                ]
            )
            + " |"
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["synthetic", "yfinance"], default="synthetic")
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--n-bars", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-lags", type=int, default=5)
    p.add_argument("--vol-window", type=int, default=21)
    p.add_argument("--seq-len", type=int, default=10)
    p.add_argument("--train-size", type=int, default=400)
    p.add_argument("--test-size", type=int, default=80)
    p.add_argument("--step", type=int, default=80)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--mode", choices=["long_flat", "long_short"], default="long_flat")
    p.add_argument("--baseline", default="logistic", choices=["logistic", "ridge"])
    p.add_argument("--rnn-hidden", type=int, default=16)
    p.add_argument("--rnn-epochs", type=int, default=12)
    p.add_argument("--models", nargs="+", default=["baseline", "lstm", "gru"])
    p.add_argument("--json", action="store_true", help="print full JSON dump")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional path to write metrics JSON",
    )
    args = p.parse_args()

    df, note = load_returns(
        source=args.source,
        ticker=args.ticker,
        n_bars=args.n_bars,
        seed=args.seed,
    )
    print(f"Data: {note}")
    X, y, fwd = build_feature_matrix(
        df["return"], n_lags=args.n_lags, vol_window=args.vol_window
    )
    print(
        f"Features: n={len(X)} cols={list(X.columns)} "
        f"pos_rate={y.mean():.3f}"
    )

    cfg = WalkForwardConfig(
        train_size=args.train_size,
        test_size=args.test_size,
        step=args.step,
        seq_len=args.seq_len,
        n_lags=args.n_lags,
        mode=args.mode,
        cost_bps=args.cost_bps,
        baseline_kind=args.baseline,
        rnn_hidden=args.rnn_hidden,
        rnn_epochs=args.rnn_epochs,
        seed=args.seed,
    )

    Xa = X.values.astype(np.float64)
    ya = y.values.astype(np.int64)
    fa = fwd.values.astype(np.float64)

    all_results = {}
    table_rows = []
    for name in args.models:
        print(f"\n=== walk-forward: {name} ===")
        res = walk_forward(Xa, ya, fa, model_name=name, cfg=cfg)  # type: ignore[arg-type]
        d = result_to_dict(res)
        all_results[name] = d
        print(
            f"folds={d['n_folds']} "
            f"OOS dir_acc(costed meta)={d['oos_costed'].get('dir_accuracy', float('nan')):.4f} "
            f"OOS sharpe costed={d['oos_costed']['sharpe']:.4f} "
            f"uncosted={d['oos_uncosted']['sharpe']:.4f}"
        )
        for split, key, costed in [
            ("IS", "is_costed", True),
            ("OOS", "oos_costed", True),
            ("OOS", "oos_uncosted", False),
        ]:
            m = d[key]
            table_rows.append(
                {
                    "model": name,
                    "split": split,
                    "costed": "yes" if costed else "no",
                    "dir_acc": m.get("dir_accuracy", float("nan")),
                    "hit_rate": m.get("hit_rate", float("nan")),
                    "sharpe": m.get("sharpe", float("nan")),
                    "total_pnl": m.get("total_pnl", float("nan")),
                }
            )

    print("\n## Metrics table")
    print_table(table_rows)
    print(
        "\nDisclaimer: research slice only — not live PnL, not a claim of alpha. "
        "OOS after costs is the score that matters; IS will look better."
    )

    payload = {
        "data_note": note,
        "config": {
            "source": args.source,
            "n_bars": args.n_bars,
            "seed": args.seed,
            "n_lags": args.n_lags,
            "vol_window": args.vol_window,
            "seq_len": args.seq_len,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "step": args.step,
            "cost_bps": args.cost_bps,
            "mode": args.mode,
            "baseline": args.baseline,
            "rnn_hidden": args.rnn_hidden,
            "rnn_epochs": args.rnn_epochs,
            "models": args.models,
        },
        "results": all_results,
        "table": table_rows,
    }
    if args.json:
        print(json.dumps(payload, indent=2, default=float))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, default=float))
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
