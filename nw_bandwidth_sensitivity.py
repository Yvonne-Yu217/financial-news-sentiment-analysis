"""
nw_bandwidth_sensitivity.py
===========================
Reports the Newey-West HAC t-statistic for the headline ChiNext PhotoPes_{t-3}
coefficient under a grid of bandwidth choices, plus three automatic-bandwidth
selectors (Andrews 1991 plug-in, Newey-West 1994 plug-in, and the standard
floor(4*(T/100)^(2/9)) rule of thumb).

Motivation: referee 2 flagged that the paper's reported NW(5) bandwidth
exactly matches the regression's 5-lag right-hand-side structure -- a
knife-edge choice. This script produces a robustness panel for §5.

Output: results/nw_bandwidth_sensitivity.csv  +  printed table.
"""
import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats.mstats import winsorize

DATA = "results/merged_market_sentiment_data_old.csv"
OUTPUT = "results/nw_bandwidth_sensitivity.csv"

INDICES = ["CSI300", "SHCOMP", "SZCOMP", "ChiNext", "CSI500"]
N_LAGS = 5
SEED = 42
np.random.seed(SEED)


def build_design(df, ret_col, photopes_col="PhotoPes"):
    """Build the headline regression design: 5 lags of PhotoPes, R, R^2 + day-of-week."""
    df = df.copy()
    # Apply the same 1%-winsorization + standardization as in the paper's main spec
    pp = winsorize(df[photopes_col].values, limits=[0.01, 0.01])
    pp = (pp - np.mean(pp)) / np.std(pp)
    df["PhotoPes_std"] = pp
    df["R"] = df[ret_col]
    df["R2"] = df[ret_col] ** 2

    cols = {}
    for lag in range(1, N_LAGS + 1):
        cols[f"PhotoPes_lag{lag}"] = df["PhotoPes_std"].shift(lag)
        cols[f"R_lag{lag}"] = df["R"].shift(lag)
        cols[f"R2_lag{lag}"] = df["R2"].shift(lag)
    # day-of-week (Monday is base)
    dow = df.index.dayofweek
    cols["weekday_Tue"] = (dow == 1).astype(int)
    cols["weekday_Wed"] = (dow == 2).astype(int)
    cols["weekday_Thu"] = (dow == 3).astype(int)
    cols["weekday_Fri"] = (dow == 4).astype(int)

    X_df = pd.DataFrame(cols, index=df.index)
    X_df = sm.add_constant(X_df)
    y = df["R"].copy()
    valid = X_df.notna().all(axis=1) & y.notna()
    return X_df[valid], y[valid]


def newey_west_1994_lag(T):
    """NW(1994) data-based bandwidth: floor(4*(T/100)^(2/9))."""
    return int(np.floor(4 * (T / 100) ** (2 / 9)))


def andrews_1991_lag(residuals):
    """
    Andrews (1991) AR(1) plug-in bandwidth for Bartlett kernel.
    For an AR(1) with parameter rho:
       optimal bandwidth = ceil(1.1447 * (alpha(1) * T)^(1/3))
    where alpha(1) = 4*rho^2/(1-rho)^2/(1+rho)^2.
    """
    e = np.asarray(residuals)
    rho = np.corrcoef(e[:-1], e[1:])[0, 1]
    if abs(rho) > 0.999:
        rho = np.sign(rho) * 0.999
    alpha1 = 4 * rho**2 / ((1 - rho) ** 2 * (1 + rho) ** 2)
    T = len(e)
    return int(np.ceil(1.1447 * (alpha1 * T) ** (1 / 3)))


def fit_with_bandwidth(X, y, bw):
    return sm.OLS(y, X).fit(cov_type="HAC",
                            cov_kwds={"maxlags": bw, "use_correction": True})


def run_one(df, ret_col, idx_label):
    X, y = build_design(df, ret_col)
    T = len(y)
    results = []

    for bw in [5, 10, 15, 22]:
        m = fit_with_bandwidth(X, y, bw)
        coef = m.params["PhotoPes_lag3"]
        t = m.tvalues["PhotoPes_lag3"]
        p = m.pvalues["PhotoPes_lag3"]
        results.append({"index": idx_label, "selector": f"NW({bw})", "bandwidth": bw,
                        "coef_lag3": coef, "tstat_lag3": t, "pval_lag3": p, "T": T})

    # NW(1994) auto
    bw_nw94 = newey_west_1994_lag(T)
    m_nw94 = fit_with_bandwidth(X, y, bw_nw94)
    results.append({"index": idx_label, "selector": "NW(1994) auto", "bandwidth": bw_nw94,
                    "coef_lag3": m_nw94.params["PhotoPes_lag3"],
                    "tstat_lag3": m_nw94.tvalues["PhotoPes_lag3"],
                    "pval_lag3": m_nw94.pvalues["PhotoPes_lag3"], "T": T})

    # Andrews(1991) plug-in: needs first-pass residuals to estimate rho.
    m5 = fit_with_bandwidth(X, y, 5)  # any bandwidth gives same point estimate; use 5 to get residuals
    bw_a91 = andrews_1991_lag(m5.resid)
    m_a91 = fit_with_bandwidth(X, y, bw_a91)
    results.append({"index": idx_label, "selector": "Andrews(1991) plug-in", "bandwidth": bw_a91,
                    "coef_lag3": m_a91.params["PhotoPes_lag3"],
                    "tstat_lag3": m_a91.tvalues["PhotoPes_lag3"],
                    "pval_lag3": m_a91.pvalues["PhotoPes_lag3"], "T": T})

    return results


def main():
    df = pd.read_csv(DATA, parse_dates=["Date"])
    df = df[(df["Date"].dt.year >= 2014) & (df["Date"].dt.year <= 2026)].copy()
    df = df.sort_values("Date").set_index("Date")

    all_rows = []
    for idx in INDICES:
        ret_col = f"{idx}_log_returns"
        if ret_col not in df.columns:
            ret_col = ret_col.replace("CSI300", "CSI300")  # noop, just safety
        all_rows.extend(run_one(df, ret_col, idx))

    out = pd.DataFrame(all_rows)
    os.makedirs("results", exist_ok=True)
    out.to_csv(OUTPUT, index=False)

    # Pretty-print headline ChiNext panel
    chi = out[out["index"] == "ChiNext"]
    print("=" * 78)
    print("Newey-West Bandwidth Sensitivity for ChiNext PhotoPes_{t-3}")
    print("(headline result from Table 4)")
    print("=" * 78)
    print(f"{'Selector':<22} {'Bandwidth':>10} {'Coefficient':>14} {'t-stat':>10} {'p-value':>10}")
    print("-" * 78)
    for _, row in chi.iterrows():
        sig = "***" if row["pval_lag3"] < 0.01 else "**" if row["pval_lag3"] < 0.05 else "*" if row["pval_lag3"] < 0.10 else ""
        print(f"{row['selector']:<22} {int(row['bandwidth']):>10} {row['coef_lag3']:>14.6f} "
              f"{row['tstat_lag3']:>10.3f} {row['pval_lag3']:>9.4f} {sig}")
    print("=" * 78)
    print(f"Saved per-index full panel to {OUTPUT}")
    print("\nFull panel (all 5 indices x 6 bandwidth selectors):")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))


if __name__ == "__main__":
    main()
