"""
tetlock_orthogonalization.py
=============================
Phase-1 必补 2: Tetlock (2008) - style orthogonalization.

Question: does PhotoPes carry predictive info orthogonal to TextPes,
or is it just a transformation of TextPes?

Procedure:
  Step 1: regress PhotoPes_t on TextPes lags (t, t-1, t-2). Take residual
          PhotoPes^orth_t.
  Step 2: substitute PhotoPes^orth_{t-1..5} for PhotoPes_{t-1..5} in the
          high-vol joint regression and re-test.

If PhotoPes^orth_{t-3} on ChiNext is still 5%-significant → image carries
INDEPENDENT info, the headline claim is robust to text-information overlap.
If not → headline ChiNext effect could be reflecting text sentiment via
some image-text correlation.

This addresses the v2 referee 1 MC1 / MC7 concern about whether image
sentiment is a separable contribution from text sentiment.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats.mstats import winsorize

DATA = "results/merged_market_sentiment_data_old.csv"
INDICES = ["CSI300", "SHCOMP", "SZCOMP", "ChiNext", "CSI500"]
N_LAGS = 5
NW_BW = 5

np.random.seed(42)


def load():
    df = pd.read_csv(DATA, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def compute_orth_photopes(df):
    """Step 1: regress PhotoPes_t on TextPes_t (and lags 1, 2). Take residual."""
    df = df.copy()
    df["PhotoPes_w"] = winsorize(df["PhotoPes"].fillna(df["PhotoPes"].mean()).values,
                                  limits=[0.01, 0.01])
    df["TextPes_w"] = winsorize(df["TextPes"].fillna(df["TextPes"].mean()).values,
                                 limits=[0.01, 0.01])
    df["TextPes_w_l1"] = df["TextPes_w"].shift(1)
    df["TextPes_w_l2"] = df["TextPes_w"].shift(2)
    valid = df[["PhotoPes_w", "TextPes_w", "TextPes_w_l1", "TextPes_w_l2"]].notna().all(axis=1)
    X = sm.add_constant(df.loc[valid, ["TextPes_w", "TextPes_w_l1", "TextPes_w_l2"]])
    y = df.loc[valid, "PhotoPes_w"]
    m = sm.OLS(y, X).fit()
    print(f"Step 1 (orthogonalization regression):")
    print(f"  PhotoPes = {m.params['const']:.4f} + "
          f"{m.params['TextPes_w']:+.4f}·TextPes + "
          f"{m.params['TextPes_w_l1']:+.4f}·TextPes_l1 + "
          f"{m.params['TextPes_w_l2']:+.4f}·TextPes_l2")
    print(f"  R² = {m.rsquared:.4f}")
    print(f"  Interpretation: {m.rsquared*100:.2f}% of PhotoPes variance is explained by TextPes (and 2 lags).")
    print(f"  Residual = {(1-m.rsquared)*100:.2f}% is orthogonal to text sentiment.")

    df.loc[valid, "PhotoPes_orth"] = y.values - m.predict(X).values
    # Standardize the residual
    df["PhotoPes_orth_std"] = (df["PhotoPes_orth"] - df["PhotoPes_orth"].mean()) / df["PhotoPes_orth"].std()
    for lag in range(1, N_LAGS + 1):
        df[f"PhotoPes_orth_std_lag{lag}"] = df["PhotoPes_orth_std"].shift(lag)
    return df


def define_high_vol(df, percentile=0.75, window=20):
    rv = df["SHCOMP_log_returns"].rolling(window).std() * np.sqrt(252)
    return rv >= rv.quantile(percentile)


def build_design(df, ret_col, hi_mask, photopes_prefix="PhotoPes_orth_std"):
    cols = {}
    for lag in range(1, N_LAGS + 1):
        cols[f"PhotoPes_lag{lag}"] = df[f"{photopes_prefix}_lag{lag}"]
        cols[f"TextPes_lag{lag}"]  = df[f"TextPes_std_lag{lag}"]
        cols[f"R_lag{lag}"]   = df[f"{ret_col.split('_log_')[0]}_log_returns_lag{lag}"]
        cols[f"R2_lag{lag}"]  = df[f"{ret_col.split('_log_')[0]}_log_returns_sq_lag{lag}"]
    cols["weekday_Tue"] = df["weekday_Tue"]
    cols["weekday_Wed"] = df["weekday_Wed"]
    cols["weekday_Thu"] = df["weekday_Thu"]
    cols["weekday_Fri"] = df["weekday_Fri"]
    X = pd.DataFrame(cols, index=df.index)
    X = sm.add_constant(X)
    y = df[ret_col]
    valid = X.notna().all(axis=1) & y.notna() & hi_mask.reindex(X.index).fillna(False)
    return X[valid], y[valid]


def fit(X, y):
    return sm.OLS(y, X).fit(cov_type="HAC",
                            cov_kwds={"maxlags": NW_BW, "use_correction": True})


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def main():
    df = load()
    print("=" * 80)
    print("Tetlock-style orthogonalization: PhotoPes^orth = PhotoPes - proj(TextPes)")
    print("=" * 80)
    df = compute_orth_photopes(df)
    hi_mask = define_high_vol(df)

    print()
    print("Step 2: high-vol joint regression with PhotoPes^orth_{t-3} replacing PhotoPes_{t-3}")
    print("=" * 80)
    print(f"{'Index':<10} {'PhotoPes^orth_t-3':>18} {'t':>7} {'p':>7} {'sig':>5} | "
          f"{'TextPes_t-3':>14} {'t':>7} {'p':>7} {'sig':>5}")
    print("-" * 100)

    for idx in INDICES:
        ret_col = f"{idx}_log_returns"
        X, y = build_design(df, ret_col, hi_mask, photopes_prefix="PhotoPes_orth_std")
        m = fit(X, y)
        bp = m.params["PhotoPes_lag3"]; tp = m.tvalues["PhotoPes_lag3"]; pp = m.pvalues["PhotoPes_lag3"]
        bt = m.params["TextPes_lag3"];  tt = m.tvalues["TextPes_lag3"];  pt = m.pvalues["TextPes_lag3"]
        print(f"{idx:<10} {bp:>18.5f} {tp:>7.2f} {pp:>7.4f} {stars(pp):>5} | "
              f"{bt:>14.5f} {tt:>7.2f} {pt:>7.4f} {stars(pt):>5}")


if __name__ == "__main__":
    main()
