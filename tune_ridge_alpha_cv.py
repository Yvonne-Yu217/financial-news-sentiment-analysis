"""
tune_ridge_alpha_cv.py
======================
Honest pre-OOS cross-validation for the Ridge regularization parameter alpha
used in Table 7 (Out-of-Sample Predictive Performance).

Design (referee-proof):
  * The full sample is 2014-01-02 to 2026-04-30 (T = 2,982 trading days).
  * The OOS window starts at trading day 1,512 (= 2020-03-23) and runs to
    the end of sample (1,467 OOS predictions).
  * To avoid look-ahead bias, alpha is tuned EXCLUSIVELY on data ENDING at
    trading day 1,511 (i.e., the last day BEFORE the OOS window opens).
  * Within that pre-OOS window, we run an expanding-window 5-fold time-series
    cross-validation (sklearn.TimeSeriesSplit) and pick the alpha that minimizes
    pooled mean squared prediction error, averaged across folds and across
    the five indices (one global alpha, matching paper's headline spec).
  * Grid: alpha in {10, 30, 100, 300, 1000, 3000} -- deliberately wider and
    coarser than the post-hoc grid that produced alpha=500.
  * Random seeds fixed; no peeking at post-2020 data.

Outputs:
  * results/cv_alpha_results.csv   -- per-fold MSE for every (alpha, index)
  * results/cv_alpha_summary.csv   -- pooled mean MSE per alpha; locked choice
  * stdout: a one-line summary of the locked alpha and the CV protocol

Usage:
  cd /Volumes/Data_Drive/research0322226/financial-news-sentiment-analysis
  python tune_ridge_alpha_cv.py
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Constants ───────────────────────────────────────────────────────────────
DATA = "results/merged_market_sentiment_data_old.csv"
OUTPUT_DIR = "results"
INDICES   = ["CSI300", "SHCOMP", "SZCOMP", "ChiNext", "CSI500"]
PRED_COLS = ["PhotoPes_std_lag3", "PhotoPes_std_lag4", "PhotoPes_std_lag5"]

# Pre-OOS sample boundary: last training day before OOS window opens.
# The paper defines INIT = 1512 trading days as the initial estimation window.
# CV must use ONLY rows [0, 1511], inclusive.
PRE_OOS_END_IDX = 1512  # exclusive upper bound -> indices 0..1511 in CV

# CV protocol
N_SPLITS = 5
ALPHA_GRID = [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]


def load_pre_oos(path: str) -> pd.DataFrame:
    """Load the merged dataset and return ONLY rows before the OOS window."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df[df["Date"] >= "2014-01-01"].copy().reset_index(drop=True)
    pre_oos = df.iloc[:PRE_OOS_END_IDX].copy()
    print(f"Full sample:        {len(df):>5} rows ({df['Date'].min().date()} -> {df['Date'].max().date()})")
    print(f"Pre-OOS for tuning: {len(pre_oos):>5} rows ({pre_oos['Date'].min().date()} -> {pre_oos['Date'].max().date()})")
    return pre_oos


def cv_one_index(df_pre_oos: pd.DataFrame, idx: str, alphas: list) -> pd.DataFrame:
    """
    Expanding-window TimeSeriesSplit CV on the pre-OOS sample for one index.
    Returns a DataFrame with columns [alpha, fold, mse].
    """
    col_y = f"{idx}_log_returns"
    cols = [col_y] + PRED_COLS
    data = df_pre_oos[cols].dropna().reset_index(drop=True).values  # n x 4
    y_all = data[:, 0]
    X_all = data[:, 1:]

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    rows = []
    for alpha in alphas:
        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_all)):
            X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
            X_va, y_va = X_all[va_idx], y_all[va_idx]
            mdl = Ridge(alpha=alpha, fit_intercept=True, random_state=SEED)
            mdl.fit(X_tr, y_tr)
            mse = float(np.mean((mdl.predict(X_va) - y_va) ** 2))
            rows.append({"index": idx, "alpha": alpha, "fold": fold, "mse": mse, "n_train": len(tr_idx), "n_val": len(va_idx)})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_pre = load_pre_oos(DATA)

    print("\nRunning expanding-window TimeSeriesSplit CV (n_splits=%d)" % N_SPLITS)
    print("Grid: alpha in", ALPHA_GRID)
    print("Predictors:", PRED_COLS)
    print("-" * 70)

    all_rows = pd.concat(
        [cv_one_index(df_pre, idx, ALPHA_GRID) for idx in INDICES],
        ignore_index=True,
    )
    all_rows.to_csv(os.path.join(OUTPUT_DIR, "cv_alpha_results.csv"), index=False)

    # Pool: mean MSE across folds and indices for each alpha (one global alpha).
    summary = (
        all_rows
        .groupby("alpha", as_index=False)["mse"]
        .mean()
        .rename(columns={"mse": "pooled_mean_mse"})
    )
    # Per-index mean MSE table for inspection.
    per_index = (
        all_rows
        .groupby(["alpha", "index"], as_index=False)["mse"]
        .mean()
        .pivot(index="alpha", columns="index", values="mse")
        .reset_index()
    )
    summary = summary.merge(per_index, on="alpha")

    # Lock alpha = argmin pooled_mean_mse
    locked_idx = summary["pooled_mean_mse"].idxmin()
    locked_alpha = int(summary.loc[locked_idx, "alpha"])

    summary.to_csv(os.path.join(OUTPUT_DIR, "cv_alpha_summary.csv"), index=False)

    print("\nCV summary (pooled mean MSE across folds and indices):")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6e}" if isinstance(x, float) else str(x)))

    print("\n" + "=" * 70)
    print(f"LOCKED ALPHA = {locked_alpha}")
    print(f"CV protocol: TimeSeriesSplit({N_SPLITS}-fold) on rows 0..{PRE_OOS_END_IDX-1}")
    print(f"             ({df_pre['Date'].min().date()} -> {df_pre['Date'].max().date()})")
    print(f"             Random seed = {SEED}; this value is now FROZEN.")
    print(f"             OOS exercise (rows {PRE_OOS_END_IDX}..end) uses alpha={locked_alpha}")
    print("=" * 70)
    print("\nNext step: rerun paper_table8_oos.py with RIDGE_ALPHA = %d" % locked_alpha)


if __name__ == "__main__":
    main()
