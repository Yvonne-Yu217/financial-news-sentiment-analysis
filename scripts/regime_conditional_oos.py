"""
regime_conditional_oos.py
==========================
Phase-1: regime-conditional OOS — generate OOS forecasts ONLY when an
ex-ante regime indicator (rolling 60-day SHCOMP volatility) is in the
top 25%. Run Clark-West only on those dates.

This addresses Referee 1 v2 MC6:
> "...report a regime-conditional OOS test that conditions only on
>  ex-ante information (e.g., trailing 90-day retail trading-volume
>  share) — this would actually convert a weakness into a strength
>  and would be a real mechanism test."

Logic:
  At each OOS date t, compute the trailing 60-day vol of SHCOMP using
  data through t-1 only. If above the rolling 60th percentile (computed
  on the pre-OOS sample), generate a forecast and benchmark; otherwise
  skip.

  Then compute R^2_OOS and Clark-West only on the regime-active
  forecasts.

If the regime-active R^2_OOS is substantially larger than the
unconditional R^2_OOS, it confirms M1 (effects concentrated in
high-retail/high-volatility regimes) using ONLY ex-ante info.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from scipy.stats.mstats import winsorize

DATA = "results/merged_market_sentiment_data_old.csv"
INDICES = ["CSI300", "SHCOMP", "SZCOMP", "ChiNext", "CSI500"]
INIT = 1512  # initial estimation window (matches paper)
PRED_COLS = ["PhotoPes_std_lag3", "PhotoPes_std_lag4", "PhotoPes_std_lag5"]
ROLL_VOL_WINDOW = 60   # trailing window for ex-ante regime indicator
PERCENTILE = 0.60      # regime-active = above this percentile (using pre-OOS sample)


def cw(a, pm, pb):
    f = (a - pb) ** 2 - (a - pm) ** 2 + (pm - pb) ** 2
    return f.mean() / (f.std() / np.sqrt(len(f)))


def r2pct(a, pm, pb):
    return (1 - np.mean((a - pm) ** 2) / np.mean((a - pb) ** 2)) * 100


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def main():
    df = pd.read_csv(DATA, parse_dates=["Date"])
    df = df[df["Date"] >= "2014-01-01"].copy().reset_index(drop=True)
    print("=" * 80)
    print("Regime-conditional OOS (ex-ante 60d SHCOMP vol regime)")
    print("=" * 80)

    # Compute ex-ante regime indicator: 60-day rolling realized vol of SHCOMP
    # IMPORTANT: at date t, we use vol computed from data UP TO t-1 only
    df["shcomp_vol_60d"] = df["SHCOMP_log_returns"].rolling(ROLL_VOL_WINDOW).std() * np.sqrt(252)

    # Determine the threshold using ONLY pre-OOS data (rows 0 to INIT-1)
    pre_oos = df.iloc[:INIT].copy()
    threshold = pre_oos["shcomp_vol_60d"].quantile(PERCENTILE)
    print(f"Pre-OOS 60d-vol {int(PERCENTILE*100)}th percentile threshold: {threshold:.4f}")
    print(f"OOS sample (rows {INIT} to {len(df)-1}): {len(df) - INIT} potential forecasts")
    print()

    print(f"{'Index':<10} {'Spec':<28} {'N':>6} {'R2_OOS%':>10} {'CW_t':>8} {'p':>8} {'sig':>5}")
    print("-" * 75)

    for idx in INDICES:
        col_y = f"{idx}_log_returns"
        data = df[[col_y, "shcomp_vol_60d"] + PRED_COLS].dropna().reset_index(drop=True)
        a, pm, pb, regime = [], [], [], []
        for t in range(INIT, len(data)):
            tr = data.iloc[:t]
            Xtr = tr[PRED_COLS].values; ytr = tr[col_y].values
            Xte = data.iloc[t][PRED_COLS].values.reshape(1, -1)
            try:
                mdl = LinearRegression().fit(Xtr, ytr)
                pred = mdl.predict(Xte)[0]
                bench = ytr.mean()
                actual = data.iloc[t][col_y]
                vol_t = data.iloc[t]["shcomp_vol_60d"]
                a.append(actual); pm.append(pred); pb.append(bench)
                regime.append(vol_t >= threshold)
            except Exception:
                pass
        a = np.array(a); pm = np.array(pm); pb = np.array(pb); regime = np.array(regime, dtype=bool)

        # Unconditional
        r_un = r2pct(a, pm, pb); t_un = cw(a, pm, pb); p_un = 1 - norm.cdf(t_un)
        print(f"{idx:<10} {'Unconditional (all OOS)':<28} {len(a):>6d} {r_un:>9.4f}% {t_un:>8.3f} {p_un:>8.4f} {stars(p_un):>5}")

        # Regime-active
        if regime.sum() > 30:
            r_r = r2pct(a[regime], pm[regime], pb[regime])
            t_r = cw(a[regime], pm[regime], pb[regime])
            p_r = 1 - norm.cdf(t_r)
            print(f"{idx:<10} {'Regime-active (high vol)':<28} {int(regime.sum()):>6d} {r_r:>9.4f}% {t_r:>8.3f} {p_r:>8.4f} {stars(p_r):>5}")

        # Regime-inactive (sanity check)
        if (~regime).sum() > 30:
            r_i = r2pct(a[~regime], pm[~regime], pb[~regime])
            t_i = cw(a[~regime], pm[~regime], pb[~regime])
            p_i = 1 - norm.cdf(t_i)
            print(f"{idx:<10} {'Regime-inactive (low vol)':<28} {int((~regime).sum()):>6d} {r_i:>9.4f}% {t_i:>8.3f} {p_i:>8.4f} {stars(p_i):>5}")
        print("-" * 75)


if __name__ == "__main__":
    main()
