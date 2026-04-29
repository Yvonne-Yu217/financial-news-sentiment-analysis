"""
oos_rolling_standardization.py
==============================
Audit & fix: re-run the OOS Ridge/OLS exercise using ROLLING-WINDOW
standardization of PhotoPes / TextPes (avoids look-ahead bias).

The paper currently uses full-sample standardization in
13merge_data_and_calculate_returns_oldmodel.py (line 252):
    mu, sigma = df[col].mean(), df[col].std()

This means the OOS predictors at time t depend on the FUTURE mean and
std of PhotoPes — a violation of strict OOS protocol.

Fix: at each OOS forecast date t, standardize PhotoPes_{t-3,4,5} using
mean and std computed from data up to t-1 only.

Compare:
- Headline (full-sample standardize, current paper): R²_OOS_chinext = 0.267%
- Honest (rolling standardize): ?
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from scipy.stats.mstats import winsorize

DATA = "results/merged_market_sentiment_data_old.csv"
INDICES = ["CSI300", "SHCOMP", "SZCOMP", "ChiNext", "CSI500"]
INIT = 1512


def cw(a, pm, pb):
    f = (a - pb) ** 2 - (a - pm) ** 2 + (pm - pb) ** 2
    return f.mean() / (f.std() / np.sqrt(len(f)))


def r2pct(a, pm, pb):
    return (1 - np.mean((a - pm) ** 2) / np.mean((a - pb) ** 2)) * 100


def run_oos_with_rolling_std(df, idx):
    """OOS exercise with rolling-window standardization of PhotoPes lags."""
    col_y = f"{idx}_log_returns"
    photopes_raw = df["PhotoPes"].copy()
    a, pm, pb = [], [], []

    # data has lag columns for PhotoPes_lag3, lag4, lag5 — but those are FROM RAW,
    # not standardized. We need to standardize them at each t using info up to t-1.
    # Easier path: at each t, build rolling-standardized PhotoPes lags from raw series.
    for t in range(INIT, len(df)):
        # Rolling-std PhotoPes using only data up to t-1
        history = photopes_raw.iloc[:t].dropna()
        if len(history) < 60:
            a.append(np.nan); pm.append(np.nan); pb.append(np.nan); continue

        # Winsorize at 1%, then standardize using rolling
        win = pd.Series(winsorize(history.values, limits=[0.01, 0.01]))
        mu, sigma = win.mean(), win.std()
        if sigma == 0: sigma = 1.0
        # Get the t-3, t-4, t-5 PhotoPes values (raw), winsorize them against rolling thresholds
        # For simplicity: standardize raw lags using rolling mu/sigma
        try:
            x_lag3 = (df.iloc[t - 3]["PhotoPes"] - mu) / sigma
            x_lag4 = (df.iloc[t - 4]["PhotoPes"] - mu) / sigma
            x_lag5 = (df.iloc[t - 5]["PhotoPes"] - mu) / sigma
        except Exception:
            a.append(np.nan); pm.append(np.nan); pb.append(np.nan); continue

        # Build training set using ROLLING-standardized lags (each t' has its own mu/sigma)
        # For tractability: use the SAME rolling mu/sigma for the whole training window
        tr = df.iloc[:t].dropna(subset=[col_y, "PhotoPes"])
        # Apply rolling-fixed std to training lags too
        tr_lag3 = (tr["PhotoPes"].shift(3) - mu) / sigma
        tr_lag4 = (tr["PhotoPes"].shift(4) - mu) / sigma
        tr_lag5 = (tr["PhotoPes"].shift(5) - mu) / sigma
        Xtr = pd.concat([tr_lag3, tr_lag4, tr_lag5], axis=1).dropna()
        ytr = tr.loc[Xtr.index, col_y]
        if len(Xtr) < 100 or pd.isna(x_lag3) or pd.isna(x_lag4) or pd.isna(x_lag5):
            a.append(np.nan); pm.append(np.nan); pb.append(np.nan); continue

        mdl = LinearRegression()
        mdl.fit(Xtr.values, ytr.values)
        pred = mdl.predict([[x_lag3, x_lag4, x_lag5]])[0]
        bench = ytr.mean()
        actual = df.iloc[t][col_y]
        if pd.isna(actual):
            a.append(np.nan); pm.append(np.nan); pb.append(np.nan); continue

        a.append(actual); pm.append(pred); pb.append(bench)

    a = np.array(a); pm = np.array(pm); pb = np.array(pb)
    valid = ~(np.isnan(a) | np.isnan(pm) | np.isnan(pb))
    return a[valid], pm[valid], pb[valid]


def sig(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def main():
    df = pd.read_csv(DATA, parse_dates=["Date"])
    df = df[df["Date"] >= "2014-01-01"].copy().reset_index(drop=True)
    print("=" * 80)
    print("OOS audit: rolling-window standardization vs full-sample standardization")
    print("=" * 80)
    print()
    print(f"{'Index':<10} {'R2_OOS%':>12} {'CW_t':>8} {'p':>8} {'sig':>5}  N")
    print("-" * 60)

    for idx in INDICES:
        a, pm, pb = run_oos_with_rolling_std(df, idx)
        r = r2pct(a, pm, pb)
        t = cw(a, pm, pb)
        p = 1 - norm.cdf(t)
        print(f"{idx:<10} {r:>11.4f}% {t:>8.3f} {p:>8.4f} {sig(p):>5}  {len(a)}")

    print()
    print("Compare to current paper (full-sample standardization, line 252 of merge script):")
    print("  ChiNext R^2_OOS = 0.267%  t=2.118  p=0.017 ** (in current Table 7)")


if __name__ == "__main__":
    main()
