"""
g1_v3_high_vol_joint.py
========================
Phase-0 Gate G1 on v3 indices: 上证50, 沪深300, 中证500, 中证1000, 创业板综指.

High-volatility subsample joint regression:
  R_t = α + β1·PhotoPes_{t-1..5} + β2·TextPes_{t-1..5}
        + β3·R_{t-1..5} + β4·R²_{t-1..5} + DOW dummies + ε_t

Subsample: top 25% of 20-day rolling SSE50 vol (using 上证50 as the
"market" since 上证综指 is removed in v3; SSE50 has decent overlap
with original SHCOMP for stress regime identification).

Verdict criterion: ranking of |β_PhotoPes_t-3| should be monotone
along v3 size gradient (SSE50 < CSI300 < CSI500 < CSI1000 < ChiNext).
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm

DATA = "paper_v3_csi_family/data/merged_v3.csv"
INDICES = ["SSE50", "CSI300", "CSI500", "CSI1000", "ChiNext"]
N_LAGS = 5
NW_BW = 5

np.random.seed(42)


def load():
    df = pd.read_csv(DATA, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def define_high_vol(df, percentile=0.75, window=20, market_col="SSE50_log_returns"):
    rv = df[market_col].rolling(window).std() * np.sqrt(252)
    threshold = rv.quantile(percentile)
    return rv >= threshold, threshold


def build_design(df, ret_col, hi_vol_mask):
    cols = {}
    for lag in range(1, N_LAGS + 1):
        cols[f"PhotoPes_lag{lag}"] = df[f"PhotoPes_std_lag{lag}"]
        cols[f"TextPes_lag{lag}"]  = df[f"TextPes_std_lag{lag}"]
        cols[f"R_lag{lag}"]   = df[f"{ret_col.split('_log_')[0]}_log_returns_lag{lag}"]
        cols[f"R2_lag{lag}"]  = df[f"{ret_col.split('_log_')[0]}_log_returns_sq_lag{lag}"]
    cols["weekday_Tue"] = df["weekday_Tue"]
    cols["weekday_Wed"] = df["weekday_Wed"]
    cols["weekday_Thu"] = df["weekday_Thu"]
    cols["weekday_Fri"] = df["weekday_Fri"]
    X = pd.DataFrame(cols)
    X = sm.add_constant(X)
    y = df[ret_col]
    valid = X.notna().all(axis=1) & y.notna() & hi_vol_mask
    return X[valid], y[valid]


def fit(X, y):
    return sm.OLS(y, X).fit(cov_type="HAC",
                            cov_kwds={"maxlags": NW_BW, "use_correction": True})


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def main():
    df = load()
    hi_mask, threshold = define_high_vol(df)
    n_hi = int(hi_mask.sum())
    print("=" * 95)
    print("G1 v3: High-Volatility Joint Regression — CSI Size-Gradient Family")
    print("=" * 95)
    print(f"Sample: {df['Date'].min().date()} to {df['Date'].max().date()}, n={len(df)}")
    print(f"High-vol regime: top 25% of 20-day rolling SSE50 vol "
          f"(annualized threshold {threshold*np.sqrt(252):.4f})")
    print(f"High-vol days: {n_hi}\n")

    print(f"{'Index':<10} {'PhotoPes_t-3':>15} {'t':>7} {'p':>7} {'sig':>5} | "
          f"{'TextPes_t-3':>15} {'t':>7} {'p':>7} {'sig':>5}  N")
    print("-" * 105)

    results = []
    for idx in INDICES:
        ret_col = f"{idx}_log_returns"
        X, y = build_design(df, ret_col, hi_mask.reindex(df.index).fillna(False))
        m = fit(X, y)
        bp = m.params["PhotoPes_lag3"]; tp = m.tvalues["PhotoPes_lag3"]; pp = m.pvalues["PhotoPes_lag3"]
        bt = m.params["TextPes_lag3"];  tt = m.tvalues["TextPes_lag3"];  pt = m.pvalues["TextPes_lag3"]
        results.append({
            "idx": idx, "n": len(y),
            "Photo_b": bp, "Photo_t": tp, "Photo_p": pp,
            "Text_b": bt,  "Text_t": tt,  "Text_p": pt,
        })
        print(f"{idx:<10} {bp:>15.5f} {tp:>7.2f} {pp:>7.4f} {stars(pp):>5} | "
              f"{bt:>15.5f} {tt:>7.2f} {pt:>7.4f} {stars(pt):>5}  {len(y)}")

    # Monotonicity check: |β_PhotoPes_t-3| should grow as we move from SSE50 → ChiNext
    print()
    print("=" * 95)
    print("Monotonicity check: |PhotoPes_t-3| coefficient along v3 size gradient")
    print("=" * 95)
    print(f"{'Index':<10} {'|β_PhotoPes|':>14} {'rank (5=largest)':>18}")
    abs_b = [(r["idx"], abs(r["Photo_b"])) for r in results]
    abs_b_sorted = sorted(abs_b, key=lambda x: -x[1])  # largest first
    rank_map = {idx: i+1 for i, (idx, _) in enumerate(abs_b_sorted)}
    expected = {"ChiNext": 1, "CSI1000": 2, "CSI500": 3, "CSI300": 4, "SSE50": 5}
    print("-" * 95)
    for idx, b in abs_b:
        match = "✓" if rank_map[idx] == expected[idx] else f"✗ (expected #{expected[idx]})"
        print(f"{idx:<10} {b:>14.5f} {rank_map[idx]:>18} {match}")

    print()
    correct = sum(1 for idx in INDICES if rank_map[idx] == expected[idx])
    print(f"Monotonicity: {correct}/5 ranks match expected order ChiNext > CSI1000 > CSI500 > CSI300 > SSE50")
    if correct == 5:
        print(">>> G1 v3 PASS (perfect monotone): cleanest possible gradient.")
    elif correct >= 4:
        print(">>> G1 v3 PASS (strong): one position swap.")
    else:
        print(">>> G1 v3 FAIL: gradient not clean. Reassess.")


if __name__ == "__main__":
    main()
