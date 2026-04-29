"""
clustered_se_pooled.py
=======================
Phase-1: pooled regression across 5 indices with clustered SE by date.

Stack the per-index regression into a panel:
  Y_{i,t} = α_i + β·PhotoPes_{t-3} + γ·TextPes_{t-3} + (lagged R, R^2 specific
            to index i) + DOW dummies + ε_{i,t}

Cluster SE by date (since within-day cross-index residual correlation is
non-trivial — all 5 indices share macro shocks).

Compare pooled β_PhotoPes_{t-3} t-statistic to per-index NW(5) HAC t-stats.
This addresses Referee 2 v2 minor / clustered-SE concern.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = "results/merged_market_sentiment_data_old.csv"
INDICES = ["CSI300", "SHCOMP", "SZCOMP", "ChiNext", "CSI500"]
N_LAGS = 5

np.random.seed(42)


def load_panel():
    """Stack the 5-index data into long format with one row per (date, index)."""
    df = pd.read_csv(DATA, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    panel_rows = []
    for idx in INDICES:
        g = df[["Date", "PhotoPes_std_lag1", "PhotoPes_std_lag2", "PhotoPes_std_lag3",
                "PhotoPes_std_lag4", "PhotoPes_std_lag5",
                "TextPes_std_lag1", "TextPes_std_lag2", "TextPes_std_lag3",
                "TextPes_std_lag4", "TextPes_std_lag5",
                "weekday_Tue", "weekday_Wed", "weekday_Thu", "weekday_Fri",
                f"{idx}_log_returns"] + [f"{idx}_log_returns_lag{l}" for l in range(1, 6)]
                                       + [f"{idx}_log_returns_sq_lag{l}" for l in range(1, 6)]].copy()
        g = g.rename(columns={f"{idx}_log_returns": "Y"})
        for lag in range(1, 6):
            g = g.rename(columns={f"{idx}_log_returns_lag{lag}": f"R_lag{lag}",
                                   f"{idx}_log_returns_sq_lag{lag}": f"R2_lag{lag}"})
        g["index_name"] = idx
        panel_rows.append(g)
    return pd.concat(panel_rows, ignore_index=True)


def build_design(df, hi_vol_only=False):
    """Build pooled design matrix."""
    if hi_vol_only:
        # We need to compute SHCOMP rolling vol per date — easiest is to compute on the
        # original wide dataset and then merge.
        wide = pd.read_csv(DATA, parse_dates=["Date"])
        wide["shcomp_vol_20d"] = wide["SHCOMP_log_returns"].rolling(20).std() * np.sqrt(252)
        threshold = wide["shcomp_vol_20d"].quantile(0.75)
        wide["high_vol"] = wide["shcomp_vol_20d"] >= threshold
        df = df.merge(wide[["Date", "high_vol"]], on="Date", how="left")
        df = df[df["high_vol"]].copy()

    cols = {}
    # Sentiment lags 1-5
    for lag in range(1, N_LAGS + 1):
        cols[f"PhotoPes_lag{lag}"] = df[f"PhotoPes_std_lag{lag}"]
        cols[f"TextPes_lag{lag}"]  = df[f"TextPes_std_lag{lag}"]
        cols[f"R_lag{lag}"]        = df[f"R_lag{lag}"]
        cols[f"R2_lag{lag}"]       = df[f"R2_lag{lag}"]
    # Index fixed effects
    for idx in INDICES[1:]:  # skip first index as baseline
        cols[f"FE_{idx}"] = (df["index_name"] == idx).astype(int)
    cols["weekday_Tue"] = df["weekday_Tue"]
    cols["weekday_Wed"] = df["weekday_Wed"]
    cols["weekday_Thu"] = df["weekday_Thu"]
    cols["weekday_Fri"] = df["weekday_Fri"]
    X = pd.DataFrame(cols, index=df.index)
    X = sm.add_constant(X)
    y = df["Y"]
    valid = X.notna().all(axis=1) & y.notna()
    return X[valid], y[valid], df.loc[valid, "Date"]


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def main():
    panel = load_panel()
    print("=" * 80)
    print("Pooled cross-index regression with cluster-by-date SE")
    print("=" * 80)
    print(f"Panel: {len(panel)} obs ({panel['index_name'].nunique()} indices × ~"
          f"{int(len(panel)/panel['index_name'].nunique())} dates)")
    print()

    # Full-sample
    X, y, dates = build_design(panel, hi_vol_only=False)
    m_full = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": dates})
    bp = m_full.params["PhotoPes_lag3"]; tp = m_full.tvalues["PhotoPes_lag3"]; pp = m_full.pvalues["PhotoPes_lag3"]
    bt = m_full.params["TextPes_lag3"];  tt = m_full.tvalues["TextPes_lag3"];  pt = m_full.pvalues["TextPes_lag3"]
    print(f"FULL SAMPLE pooled (cluster-by-date SE), N = {len(y)}:")
    print(f"  PhotoPes_t-3: β = {bp:.5f}, t = {tp:.2f}, p = {pp:.4f} {stars(pp)}")
    print(f"  TextPes_t-3:  β = {bt:.5f}, t = {tt:.2f}, p = {pt:.4f} {stars(pt)}")
    print()

    # High-vol subsample
    X, y, dates = build_design(panel, hi_vol_only=True)
    m_hi = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": dates})
    bp = m_hi.params["PhotoPes_lag3"]; tp = m_hi.tvalues["PhotoPes_lag3"]; pp = m_hi.pvalues["PhotoPes_lag3"]
    bt = m_hi.params["TextPes_lag3"];  tt = m_hi.tvalues["TextPes_lag3"];  pt = m_hi.pvalues["TextPes_lag3"]
    print(f"HIGH-VOL SUBSAMPLE pooled (cluster-by-date SE), N = {len(y)}:")
    print(f"  PhotoPes_t-3: β = {bp:.5f}, t = {tp:.2f}, p = {pp:.4f} {stars(pp)}")
    print(f"  TextPes_t-3:  β = {bt:.5f}, t = {tt:.2f}, p = {pt:.4f} {stars(pt)}")

    print()
    print("=" * 80)
    print("Reference (per-index NW(5) from G1, ChiNext only):")
    print("  PhotoPes_t-3 on ChiNext: β = -0.00329, t = -2.41, p = 0.016 **")


if __name__ == "__main__":
    main()
