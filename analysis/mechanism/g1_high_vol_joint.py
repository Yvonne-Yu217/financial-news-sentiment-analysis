"""
g1_high_vol_joint.py
====================
Phase-0 Gate G1: High-volatility subsample joint regression of
PhotoPes_{t-3} + TextPes_{t-3} on each of 5 indices, to test the
"task-division" main-axis hypothesis from the professor's memo.

Pattern the new main axis predicts:
  ChiNext:        PhotoPes sig, TextPes NOT sig
  SZCOMP/CSI500:  TextPes sig, PhotoPes NOT sig (or weaker)
  CSI300/SHCOMP:  both NOT sig (institution-dominated)

If actual pattern is muddier (e.g., both sig on ChiNext), need to
fall back to a weaker "relative-dominance" framing.

Spec:
  - Subsample: top 25% of 20-day rolling SHCOMP volatility
  - Joint regression with BOTH PhotoPes_{t-1..5} AND TextPes_{t-1..5}
    (full lag structure as in paper Table 5 Panel A)
  - HAC standard errors with Newey-West bandwidth=5
  - Day-of-week dummies + lagged returns + lagged squared returns

Outputs:
  - Per-index PhotoPes_{t-3} / TextPes_{t-3} coefficient + t-stat + p-value
  - Wald test 1: cross-index PhotoPes_{ChiNext} vs PhotoPes_{CSI300}
  - Wald test 2: within-index PhotoPes_{t-3} vs TextPes_{t-3} dominance
  - Verdict: does the pattern support task division?
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats.mstats import winsorize
from scipy import stats as scipy_stats

DATA = "results/merged_market_sentiment_data_old.csv"
INDICES = ["CSI300", "SHCOMP", "SZCOMP", "ChiNext", "CSI500"]
NAMES = {"CSI300": "CSI 300", "SHCOMP": "SHCOMP", "SZCOMP": "SZCOMP",
         "ChiNext": "ChiNext", "CSI500": "CSI 500"}
N_LAGS = 5
NW_BW = 5
VOL_PERCENTILE = 0.75  # top 25% as professor specified

np.random.seed(42)


def load_data():
    df = pd.read_csv(DATA, parse_dates=["Date"])
    df = df[df["Date"].dt.year >= 2014].copy().sort_values("Date").reset_index(drop=True)
    df = df.set_index("Date")
    return df


def winsorize_and_standardize(s):
    arr = winsorize(s.values, limits=[0.01, 0.01])
    return (arr - np.mean(arr)) / np.std(arr)


def define_high_vol_regime(df, percentile=0.75, window=20):
    """Top quartile of 20-day rolling SHCOMP vol = high-vol regime."""
    rv = df["SHCOMP_log_returns"].rolling(window).std() * np.sqrt(252)
    threshold = rv.quantile(percentile)
    return rv >= threshold


def build_joint_design(df, ret_col, high_vol_mask):
    """Joint regression: lags of PhotoPes + TextPes + R + R^2 + day-of-week."""
    df = df.copy()
    df["PhotoPes_std"] = winsorize_and_standardize(df["PhotoPes"].fillna(df["PhotoPes"].mean()))
    df["TextPes_std"] = winsorize_and_standardize(df["TextPes"].fillna(df["TextPes"].mean()))
    df["R"] = df[ret_col]
    df["R2"] = df[ret_col] ** 2

    cols = {}
    for lag in range(1, N_LAGS + 1):
        cols[f"PhotoPes_lag{lag}"] = df["PhotoPes_std"].shift(lag)
        cols[f"TextPes_lag{lag}"] = df["TextPes_std"].shift(lag)
        cols[f"R_lag{lag}"] = df["R"].shift(lag)
        cols[f"R2_lag{lag}"] = df["R2"].shift(lag)
    dow = df.index.dayofweek
    cols["weekday_Tue"] = (dow == 1).astype(int)
    cols["weekday_Wed"] = (dow == 2).astype(int)
    cols["weekday_Thu"] = (dow == 3).astype(int)
    cols["weekday_Fri"] = (dow == 4).astype(int)

    X = pd.DataFrame(cols, index=df.index)
    X = sm.add_constant(X)
    y = df["R"]
    valid = X.notna().all(axis=1) & y.notna() & high_vol_mask.reindex(X.index).fillna(False)
    return X[valid], y[valid]


def fit_joint(X, y):
    return sm.OLS(y, X).fit(cov_type="HAC",
                            cov_kwds={"maxlags": NW_BW, "use_correction": True})


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def main():
    df = load_data()
    high_vol = define_high_vol_regime(df, percentile=VOL_PERCENTILE)
    n_high = int(high_vol.sum())
    print("=" * 80)
    print("G1: High-Volatility Joint Regression — Task Division Test")
    print("=" * 80)
    print(f"High-vol mask: top {int((1-VOL_PERCENTILE)*100)}% of 20-day rolling SHCOMP vol")
    print(f"Threshold (annualized): {df['SHCOMP_log_returns'].rolling(20).std().quantile(VOL_PERCENTILE)*np.sqrt(252):.4f}")
    print(f"High-vol days available: {n_high} of {len(df)}")
    print()

    # Fit each index
    results = {}
    print(f"{'Index':<10} {'PhotoPes_t-3':>14} {'t':>7} {'p':>7} {'sig':>5} | "
          f"{'TextPes_t-3':>14} {'t':>7} {'p':>7} {'sig':>5}  N")
    print("-" * 100)
    for idx in INDICES:
        X, y = build_joint_design(df, f"{idx}_log_returns", high_vol)
        m = fit_joint(X, y)
        b_p = m.params["PhotoPes_lag3"]; t_p = m.tvalues["PhotoPes_lag3"]; pv_p = m.pvalues["PhotoPes_lag3"]
        b_t = m.params["TextPes_lag3"];  t_t = m.tvalues["TextPes_lag3"];  pv_t = m.pvalues["TextPes_lag3"]
        results[idx] = {"model": m, "n": len(y),
                        "PhotoPes_b": b_p, "PhotoPes_t": t_p, "PhotoPes_p": pv_p,
                        "TextPes_b": b_t,  "TextPes_t": t_t,  "TextPes_p": pv_t}
        print(f"{NAMES[idx]:<10} {b_p:>14.5f} {t_p:>7.2f} {pv_p:>7.4f} {stars(pv_p):>5} | "
              f"{b_t:>14.5f} {t_t:>7.2f} {pv_t:>7.4f} {stars(pv_t):>5}  {len(y)}")

    # ─────────────────────────────────────────────────────────────────────────
    # Test pattern adherence to task-division hypothesis
    # ─────────────────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("Pattern check: does this match TASK DIVISION (Φ at ChiNext, T at SZCOMP/CSI500)?")
    print("=" * 80)
    print(f"{'Index':<10} {'pattern':<30} {'task-div predicts'}")
    print("-" * 80)
    for idx in INDICES:
        r = results[idx]
        actual_p = "Φ sig" if r["PhotoPes_p"] < 0.10 else "Φ null"
        actual_t = "T sig" if r["TextPes_p"] < 0.10 else "T null"
        actual = f"{actual_p}, {actual_t}"
        if idx == "ChiNext":
            predicted = "Φ sig + T null  (pure)"
        elif idx in ("SZCOMP", "CSI500"):
            predicted = "T sig + Φ null  (pure)"
        else:
            predicted = "both null  (institutional)"
        match = "✓" if (
            (idx == "ChiNext" and r["PhotoPes_p"] < 0.10 and r["TextPes_p"] >= 0.10) or
            (idx in ("SZCOMP", "CSI500") and r["TextPes_p"] < 0.10 and r["PhotoPes_p"] >= 0.10) or
            (idx in ("CSI300", "SHCOMP") and r["PhotoPes_p"] >= 0.10 and r["TextPes_p"] >= 0.10)
        ) else "✗"
        print(f"{NAMES[idx]:<10} {actual:<30} {predicted}  [{match}]")

    # ─────────────────────────────────────────────────────────────────────────
    # Within-index dominance test: is PhotoPes_t-3 different from TextPes_t-3?
    # ─────────────────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("Within-index dominance: PhotoPes_t-3 vs TextPes_t-3 (Wald test β_P = β_T)")
    print("=" * 80)
    print(f"{'Index':<10} {'β_P':>10} {'β_T':>10} {'β_P-β_T':>10} {'F':>7} {'p':>7}  {'dominant':<20}")
    print("-" * 80)
    for idx in INDICES:
        m = results[idx]["model"]
        # Wald: PhotoPes_lag3 - TextPes_lag3 = 0
        wald = m.t_test("PhotoPes_lag3 = TextPes_lag3")
        b_p = results[idx]["PhotoPes_b"]
        b_t = results[idx]["TextPes_b"]
        diff = b_p - b_t
        f_stat = wald.statistic.item() ** 2
        p_val = wald.pvalue
        if abs(b_p) > abs(b_t) * 1.20 and results[idx]["PhotoPes_p"] < 0.10:
            dominant = "Φ dominant"
        elif abs(b_t) > abs(b_p) * 1.20 and results[idx]["TextPes_p"] < 0.10:
            dominant = "T dominant"
        else:
            dominant = "tied / both weak"
        print(f"{NAMES[idx]:<10} {b_p:>10.5f} {b_t:>10.5f} {diff:>10.5f} {f_stat:>7.3f} {p_val:>7.4f}  {dominant}")

    # ─────────────────────────────────────────────────────────────────────────
    # Cross-index test: is PhotoPes_t-3 stronger on ChiNext than CSI 300?
    # (Critical for 'cross-sectional task division' claim)
    # ─────────────────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("Cross-index strength: PhotoPes_t-3 ChiNext vs CSI 300 (z-test on diff)")
    print("=" * 80)
    chi = results["ChiNext"]
    csi300 = results["CSI300"]
    diff_b = chi["PhotoPes_b"] - csi300["PhotoPes_b"]
    se_chi = chi["PhotoPes_b"] / chi["PhotoPes_t"]
    se_csi = csi300["PhotoPes_b"] / csi300["PhotoPes_t"]
    se_diff = np.sqrt(se_chi ** 2 + se_csi ** 2)  # assumes independent (rough; ignores cross-equation cov)
    z = diff_b / se_diff
    p_diff = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
    print(f"  PhotoPes_t-3 on ChiNext: β = {chi['PhotoPes_b']:.5f}, t = {chi['PhotoPes_t']:.2f}")
    print(f"  PhotoPes_t-3 on CSI300:  β = {csi300['PhotoPes_b']:.5f}, t = {csi300['PhotoPes_t']:.2f}")
    print(f"  Difference (ChiNext - CSI300): {diff_b:.5f}")
    print(f"  z-stat (rough, independent SE): {z:.2f}, two-sided p ≈ {p_diff:.4f}")
    print(f"  (Note: a proper SUR with stacked panel + cluster-by-day is needed for definitive test.)")

    # ─────────────────────────────────────────────────────────────────────────
    # Verdict
    # ─────────────────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    chi_p_sig = results["ChiNext"]["PhotoPes_p"] < 0.05
    chi_t_null = results["ChiNext"]["TextPes_p"] >= 0.10
    sz_t_sig = results["SZCOMP"]["TextPes_p"] < 0.05
    sz_p_null = results["SZCOMP"]["PhotoPes_p"] >= 0.10
    cs_t_sig = results["CSI500"]["TextPes_p"] < 0.05
    cs_p_null = results["CSI500"]["PhotoPes_p"] >= 0.10

    pure_pattern = chi_p_sig and chi_t_null and (
        (sz_t_sig and sz_p_null) or (cs_t_sig and cs_p_null)
    )
    print(f"Pure task-division pattern (sharp split): {'YES ✓' if pure_pattern else 'NO ✗'}")
    print(f"  - ChiNext PhotoPes 5%-sig:           {chi_p_sig}")
    print(f"  - ChiNext TextPes NOT 10%-sig:       {chi_t_null}")
    print(f"  - SZCOMP TextPes 5%-sig + PhotoPes null:    {sz_t_sig and sz_p_null}")
    print(f"  - CSI500 TextPes 5%-sig + PhotoPes null:    {cs_t_sig and cs_p_null}")

    # Weak pattern: relative-dominance
    chi_p_dom = abs(results["ChiNext"]["PhotoPes_b"]) > abs(results["ChiNext"]["TextPes_b"])
    sz_t_dom = abs(results["SZCOMP"]["TextPes_b"]) > abs(results["SZCOMP"]["PhotoPes_b"])
    cs_t_dom = abs(results["CSI500"]["TextPes_b"]) > abs(results["CSI500"]["PhotoPes_b"])
    weak_pattern = chi_p_dom and (sz_t_dom or cs_t_dom)
    print(f"\nWeak relative-dominance pattern (Φ stronger on ChiNext, T stronger on mid-cap): "
          f"{'YES ✓' if weak_pattern else 'NO ✗'}")
    print(f"  - ChiNext: |β_P| > |β_T|:  {chi_p_dom}  ({results['ChiNext']['PhotoPes_b']:.5f} vs {results['ChiNext']['TextPes_b']:.5f})")
    print(f"  - SZCOMP:  |β_T| > |β_P|:  {sz_t_dom}   ({results['SZCOMP']['TextPes_b']:.5f} vs {results['SZCOMP']['PhotoPes_b']:.5f})")
    print(f"  - CSI500:  |β_T| > |β_P|:  {cs_t_dom}   ({results['CSI500']['TextPes_b']:.5f} vs {results['CSI500']['PhotoPes_b']:.5f})")

    print()
    if pure_pattern:
        print(">>> G1 PASS (strong): Pure task-division supported. Proceed to G2/G3.")
    elif weak_pattern:
        print(">>> G1 PASS (weak): Relative-dominance supported. Main axis works as 'differential")
        print("    sensitivity' framing rather than 'pure specialization'. Proceed to G2/G3,")
        print("    but soften abstract from 'task division' to 'differential modality strength'.")
    else:
        print(">>> G1 FAIL: Neither pure nor weak pattern detected. Main-axis pivot not supported.")
        print("    Consider professor's fallback: ChiNext-only + image + high-vol + single-lag.")


if __name__ == "__main__":
    main()
