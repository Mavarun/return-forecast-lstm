# return-forecast-lstm

Research slice: tiny PyTorch LSTM/GRU vs logistic/ridge baseline for next-bar **directional** return forecast, scored with **costed** walk-forward PnL — not RMSE alone.

This does **not** claim live PnL or alpha.

## Hypothesis

1. Lagged return/vol features fed to a small LSTM/GRU can beat a logistic/ridge baseline on directional accuracy in a walk-forward split.
2. Strategy PnL after spread+slippage costs is the real score — not just RMSE or accuracy.
3. OOS costed Sharpe / hit rate will likely be weaker than in-sample; report both honestly.

## Method

- **Data (default):** synthetic AR(1) daily returns generated in-process (`seed=42`). Optional `--source yfinance` pulls Yahoo Finance daily bars via [`yfinance`](https://github.com/ranaroussi/yfinance) (cite Yahoo Finance / yfinance; falls back to synthetic on failure).
- **Features:** lags 1..`n_lags` of return + trailing realized vol (and vol lag-1). Labels = sign of next-bar return. No contemporaneous/future return in features.
- **Models:** `StandardScaler` + logistic/ridge (last-step features); tiny 1-layer LSTM or GRU (`hidden=16`, short `seq_len=10`) + linear logit head (CPU-friendly).
- **Backtest:** walk-forward contiguous folds (`train_size` → `test_size`, step=`test_size`). Positions: long/flat (default) or long/short from predicted direction. Costs: `cost_bps` charged on `|Δposition|` each bar (spread+slippage proxy).
- **Metrics:** directional accuracy, hit rate, total PnL, annualized Sharpe — reported **IS vs OOS** and **costed vs uncosted**.

## Defaults

| Knob | Default |
|------|---------|
| source | synthetic AR(1), `n_bars=1200`, `seed=42` |
| features | `n_lags=5`, `vol_window=21`, `seq_len=10` |
| walk-forward | `train_size=400`, `test_size=80`, `step=80` |
| costs | `5` bps per unit turnover |
| mode | `long_flat` |
| LSTM/GRU | `hidden=16`, `epochs=12`, Adam |

## Metrics (local synthetic run)

Command: `python scripts/run_lstm_slice.py --source synthetic`  
Settings: defaults above; models `baseline` (logistic), `lstm`, `gru`.

| model | split | costed | dir_acc | hit_rate | sharpe | total_pnl |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | IS | yes | 0.5010 | 0.4960 | -0.8424 | -0.3332 |
| baseline | OOS | yes | 0.4944 | 0.5000 | -1.0692 | -0.3490 |
| baseline | OOS | no | 0.4944 | 0.5000 | -0.6970 | -0.2270 |
| lstm | IS | yes | 0.5115 | 0.5222 | 0.3937 | 0.1291 |
| lstm | OOS | yes | 0.4736 | 0.4566 | -0.9981 | -0.2735 |
| lstm | OOS | no | 0.4736 | 0.4566 | -0.8416 | -0.2300 |
| gru | IS | yes | 0.4846 | 0.4728 | -1.0599 | -0.5091 |
| gru | OOS | yes | 0.4694 | 0.4685 | -1.2234 | -0.4497 |
| gru | OOS | no | 0.4694 | 0.4685 | -1.0033 | -0.3687 |

**Reading:** On this AR(1) synthetic series, neither LSTM nor GRU beats the logistic baseline OOS on directional accuracy (~0.47–0.49, near chance). LSTM **IS** costed Sharpe looks positive while **OOS** turns negative — consistent with hypothesis (3). Costs reduce OOS total PnL vs the uncosted column for every model.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest
python scripts/run_lstm_slice.py
python scripts/run_lstm_slice.py --json --out artifacts/metrics.json
# optional market data (network; cite Yahoo Finance / yfinance):
python scripts/run_lstm_slice.py --source yfinance --ticker SPY
```

## Package layout

```
src/ret_lstm/   data, features, models, backtest, metrics
scripts/        run_lstm_slice.py
tests/          shapes, no look-ahead, cost drag, walk-forward barrier
```

## Limits / why it can fail

- Mild AR(1) synthetic signal is weak; equity daily direction is close to coin-flip after costs.
- Tiny net + short sequences underfit or overfit fold noise; no hyperparameter search.
- Long/flat with 5 bps turnover tax kills low-edge strategies.
- Walk-forward uses expanding train prefix; regime shifts still hurt OOS.
- yfinance gaps/adjustments and survivorship are out of scope.
- **Not** production trading research; no portfolio constraints, borrow, or execution model beyond a flat bps cost.
