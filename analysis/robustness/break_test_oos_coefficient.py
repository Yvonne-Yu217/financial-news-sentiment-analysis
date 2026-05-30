"""
break_test_oos_coefficient.py
==============================
Tests for structural breaks in the time-varying ChiNext PhotoPes_{t-3}
predictive coefficient over the full sample 2014-2026.

Procedure:
  1. Estimate a 90-day expanding-window OLS regression of ChiNext returns on
     PhotoPes_{t-3} only (single-predictor, to isolate the parameter of
     interest). The output is a series of beta_hat_{t} over t in [90, T].
  2. Run two break tests on this beta_hat series:
       (a) Bai-Perron multiple-break detection via the `ruptures` library
           with PELT and L2 cost (no fixed number of breaks);
       (b) CUSUM-of-squares break test on the residuals of the
           full-sample OLS regression (sm.stats.diagnostic).
  3. Report the detected break dates and discuss whether they line up with
     known macro/regulatory events.

Outputs:
  results/breaks_chinext_coef.csv  -- the rolling beta_hat series and
                                       binary break-flag column.
  Console: detected break dates and their alignment to known events.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.recursive_ls import RecursiveLS
import ruptures as rpt

np.random.seed(42)

DATA = "results/merged_market_sentiment_data_old.csv"
OUTPUT = "results/breaks_chinext_coef.csv"
WINDOW = 252  # 1 trading-year rolling window for the time-varying coefficient

def load():
    df = pd.read_csv(DATA, parse_dates=["Date"])
    df = df[(df["Date"].dt.year >= 2014)].copy()
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def rolling_coef_series(df, ret_col="ChiNext_log_returns",
                       pred_col="PhotoPes_std_lag3", window=WINDOW):
    """Single-predictor rolling OLS of ret on PhotoPes_lag3."""
    sub = df[["Date", ret_col, pred_col]].dropna().reset_index(drop=True)
    n = len(sub)
    betas, dates, ts = [], [], []
    for i in range(window, n):
        chunk = sub.iloc[i-window:i]
        X = sm.add_constant(chunk[pred_col].values)
        y = chunk[ret_col].values
        try:
            m = sm.OLS(y, X).fit()
            betas.append(m.params[1])
            ts.append(m.tvalues[1])
        except Exception:
            betas.append(np.nan); ts.append(np.nan)
        dates.append(sub.iloc[i]["Date"])
    return pd.DataFrame({"Date": dates, "beta_lag3": betas, "t_lag3": ts})


def detect_breaks_pelt(beta_series, pen=10):
    """PELT change-point detection on the beta series.
    `pen` is the L2 cost penalty - higher means fewer breakpoints."""
    signal = beta_series.values.reshape(-1, 1)
    algo = rpt.Pelt(model="l2", min_size=60).fit(signal)
    bkps = algo.predict(pen=pen)
    # `bkps` includes the end-of-series; drop it
    return bkps[:-1]


def cusum_of_squares(residuals, alpha=0.05):
    """CUSUM-of-squares (Brown, Durbin, Evans 1975); returns max statistic and
    boundary, plus break detected (yes/no)."""
    e2 = residuals ** 2
    T = len(e2)
    s = np.cumsum(e2)
    fr = s / s[-1]
    line = np.linspace(0, 1, T + 1)[1:]
    devs = fr - line
    stat_max = np.max(np.abs(devs))
    # Asymptotic boundary at 5% from BDE table:
    # crit_05 = 0.948 / sqrt(T/2) for symmetric CUSUM-sq
    crit = 0.948 / np.sqrt(T / 2)
    return {"max_stat": stat_max, "crit_05": crit, "reject_null": stat_max > crit}


def main():
    os.makedirs("results", exist_ok=True)
    df = load()

    print("=" * 78)
    print("Structural break test on ChiNext PhotoPes_{t-3} predictive coefficient")
    print("=" * 78)
    print(f"Rolling window: {WINDOW} trading days  (~one calendar year)")

    coef_df = rolling_coef_series(df)
    print(f"Rolling beta series: {len(coef_df)} estimates from {coef_df['Date'].iloc[0].date()} to {coef_df['Date'].iloc[-1].date()}")
    print(f"Beta summary: mean = {coef_df['beta_lag3'].mean():.6f}, std = {coef_df['beta_lag3'].std():.6f}")
    print(f"             min  = {coef_df['beta_lag3'].min():.6f} on {coef_df.loc[coef_df['beta_lag3'].idxmin(), 'Date'].date()}")
    print(f"             max  = {coef_df['beta_lag3'].max():.6f} on {coef_df.loc[coef_df['beta_lag3'].idxmax(), 'Date'].date()}")

    # ── Bai-Perron via PELT ───────────────────────────────────────────────────
    print("\n--- (a) Bai-Perron multiple-break detection (PELT, L2, min_size=60) ---")
    for pen in [1, 5, 10, 25]:
        breaks = detect_breaks_pelt(coef_df["beta_lag3"], pen=pen)
        if breaks:
            break_dates = [coef_df.iloc[b]["Date"].date() for b in breaks if b < len(coef_df)]
            print(f"  Penalty={pen:>3}: {len(break_dates)} break(s) at {break_dates}")
        else:
            print(f"  Penalty={pen:>3}: no breaks detected")

    # ── CUSUM-of-squares ──────────────────────────────────────────────────────
    print("\n--- (b) CUSUM-of-squares on full-sample regression residuals ---")
    sub = df[["Date", "ChiNext_log_returns", "PhotoPes_std_lag3"]].dropna().reset_index(drop=True)
    X = sm.add_constant(sub["PhotoPes_std_lag3"].values)
    y = sub["ChiNext_log_returns"].values
    full = sm.OLS(y, X).fit()
    cusum = cusum_of_squares(full.resid, alpha=0.05)
    print(f"  Max |CUSUM-sq deviation| = {cusum['max_stat']:.4f}")
    print(f"  5% boundary (asymptotic) = {cusum['crit_05']:.4f}")
    if cusum["reject_null"]:
        print(f"  REJECT null of constant variance (break in volatility, not necessarily in beta)")
    else:
        print(f"  FAIL TO REJECT null of constant variance")

    # ── Recursive least squares CUSUM (canonical Brown-Durbin-Evans on coefficients) ──
    print("\n--- (c) Recursive-LS CUSUM (Brown-Durbin-Evans 1975) on the coefficient ---")
    rls = RecursiveLS(y, X).fit()
    cusum_obj = rls.cusum
    cusum_t_05 = 0.948  # 5% asymptotic boundary multiplier
    sigma_hat = np.sqrt(rls.scale)
    bound = cusum_t_05 * np.sqrt(len(y))
    max_cs = np.nanmax(np.abs(cusum_obj))
    print(f"  Max |recursive-LS CUSUM| = {max_cs:.4f}")
    print(f"  5% asymptotic boundary    = {bound:.4f}")
    print(f"  {'REJECT' if max_cs > bound else 'FAIL TO REJECT'} null of constant beta")

    # ── Save artifacts ────────────────────────────────────────────────────────
    breaks10 = detect_breaks_pelt(coef_df["beta_lag3"], pen=10)
    coef_df["pelt_break_pen10"] = 0
    for b in breaks10:
        if b < len(coef_df):
            coef_df.loc[b, "pelt_break_pen10"] = 1
    coef_df.to_csv(OUTPUT, index=False)
    print(f"\nSaved rolling-coefficient + break flags to {OUTPUT}")

    # Map detected breaks to known events
    print("\n--- Calendar interpretation of detected breaks (PELT pen=10) ---")
    if breaks10:
        for b in breaks10:
            if b < len(coef_df):
                d = coef_df.iloc[b]["Date"]
                # Pre-event regime average and post-event regime average
                pre = coef_df.iloc[max(0, b-126):b]["beta_lag3"].mean()
                post = coef_df.iloc[b:min(len(coef_df), b+126)]["beta_lag3"].mean()
                print(f"  {d.date()}:  pre-break avg beta = {pre:.6f}  -->  post-break avg beta = {post:.6f}")
    else:
        print("  No breaks at this penalty level; coefficient is statistically homogeneous.")


if __name__ == "__main__":
    main()
