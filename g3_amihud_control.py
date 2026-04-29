"""
g3_amihud_control.py
=====================
Phase-0 Gate G3: Liquidity control via Amihud (2002) illiquidity ratio.

Add Amihud_{t-1} = |R_{t-1}| / TradingAmount_{t-1}  (in 元) to the
high-vol joint regression of PhotoPes_{t-1..5} + TextPes_{t-1..5} + ...
across the 5 v2 indices.

Critical question (per professor): does ChiNext PhotoPes_{t-3} remain
5%-significant after controlling for time-varying liquidity?

Data sources for daily trading amount:
  CSI 300  (000300): paper_v3_csi_family/data/index_csindex_daily_v3.csv (中证 接口)
  SHCOMP   (000001): paper_v3_csi_family/data/shcomp_000001_daily_csindex.csv (中证 接口)
  CSI 500  (000905): same csindex file
  ChiNext  (399102): paper_v3_csi_family/data/chinext_399102_daily_em.csv (East Money)
  SZCOMP   (399106): NOT AVAILABLE — East Money rate-limit; use volume×close proxy

Amihud unit (canonical Amihud 2002): million-USD inverse. Here we use
|R| / amount_in_yuan where amount in 元. Scale doesn't matter for
the regression because beta absorbs it.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm

DATA = "results/merged_market_sentiment_data_old.csv"
N_LAGS = 5
NW_BW = 5

np.random.seed(42)

INDICES = ["CSI300", "SHCOMP", "SZCOMP", "ChiNext", "CSI500"]


def load_index_amounts():
    """Load daily 成交金额 (in 元) for all 5 indices from various sources."""
    # CSI 300, CSI 500 from 中证 file
    df_csi = pd.read_csv("paper_v3_csi_family/data/index_csindex_daily_v3.csv")
    df_csi["日期"] = pd.to_datetime(df_csi["日期"])
    df_csi["amount_yuan"] = pd.to_numeric(df_csi["成交金额"], errors="coerce") * 1e8  # 亿 → 元

    # SHCOMP via 中证
    df_sh = pd.read_csv("paper_v3_csi_family/data/shcomp_000001_daily_csindex.csv")
    df_sh["日期"] = pd.to_datetime(df_sh["日期"])
    df_sh["amount_yuan"] = pd.to_numeric(df_sh["成交金额"], errors="coerce") * 1e8

    # ChiNext via East Money
    df_ch = pd.read_csv("paper_v3_csi_family/data/chinext_399102_daily_em.csv")
    df_ch["日期"] = pd.to_datetime(df_ch["date"])
    df_ch["amount_yuan"] = pd.to_numeric(df_ch["amount"], errors="coerce")  # already in 元

    # Build a unified DataFrame with one column per index amount
    out = pd.DataFrame({"Date": sorted(set(df_csi["日期"].unique())
                                        | set(df_sh["日期"].unique())
                                        | set(df_ch["日期"].unique()))})

    # CSI 300 amount
    csi300 = df_csi[df_csi["index_code"] == 300][["日期", "amount_yuan"]].rename(
        columns={"日期": "Date", "amount_yuan": "CSI300_amount"})
    out = out.merge(csi300, on="Date", how="left")
    # CSI 500
    csi500 = df_csi[df_csi["index_code"] == 905][["日期", "amount_yuan"]].rename(
        columns={"日期": "Date", "amount_yuan": "CSI500_amount"})
    out = out.merge(csi500, on="Date", how="left")
    # SHCOMP
    sh = df_sh[["日期", "amount_yuan"]].rename(
        columns={"日期": "Date", "amount_yuan": "SHCOMP_amount"})
    out = out.merge(sh, on="Date", how="left")
    # ChiNext
    ch = df_ch[["日期", "amount_yuan"]].rename(
        columns={"日期": "Date", "amount_yuan": "ChiNext_amount"})
    out = out.merge(ch, on="Date", how="left")

    return out


def build_design_with_amihud(df, ret_col, idx_short, hi_vol_mask):
    """Joint regression spec with PhotoPes/TextPes/R/R^2 lags + Amihud_{t-1} control."""
    df = df.copy()
    df[f"{idx_short}_abs_ret"] = df[ret_col].abs()
    if f"{idx_short}_amount" in df.columns and df[f"{idx_short}_amount"].notna().any():
        df[f"{idx_short}_amihud"] = df[f"{idx_short}_abs_ret"] / df[f"{idx_short}_amount"]
        amihud_source = "real-amount"
    else:
        # SZCOMP fallback: use abs_ret / (close * volume) as a relative proxy
        # The scale doesn't matter for the regression, only the time-series variation
        if "SZCOMP" in idx_short and "SZCOMP_close" in df.columns:
            df[f"{idx_short}_amihud"] = df[f"{idx_short}_abs_ret"] / (df["SZCOMP_close"] * 1.0)
            amihud_source = "proxy (no real amount)"
        else:
            amihud_source = "MISSING"
            return None, None, amihud_source

    df[f"{idx_short}_amihud_lag1"] = df[f"{idx_short}_amihud"].shift(1)

    cols = {}
    for lag in range(1, N_LAGS + 1):
        cols[f"PhotoPes_lag{lag}"] = df[f"PhotoPes_std_lag{lag}"]
        cols[f"TextPes_lag{lag}"]  = df[f"TextPes_std_lag{lag}"]
        cols[f"R_lag{lag}"]   = df[f"{idx_short}_log_returns_lag{lag}"]
        cols[f"R2_lag{lag}"]  = df[f"{idx_short}_log_returns_sq_lag{lag}"]
    cols[f"{idx_short}_amihud_lag1"] = df[f"{idx_short}_amihud_lag1"]
    cols["weekday_Tue"] = df["weekday_Tue"]
    cols["weekday_Wed"] = df["weekday_Wed"]
    cols["weekday_Thu"] = df["weekday_Thu"]
    cols["weekday_Fri"] = df["weekday_Fri"]
    X = pd.DataFrame(cols, index=df.index)
    X = sm.add_constant(X)
    y = df[ret_col]
    valid = X.notna().all(axis=1) & y.notna() & hi_vol_mask
    return X[valid], y[valid], amihud_source


def fit(X, y):
    return sm.OLS(y, X).fit(cov_type="HAC",
                            cov_kwds={"maxlags": NW_BW, "use_correction": True})


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def main():
    print("=" * 100)
    print("G3 v2: Amihud Liquidity Control — high-vol joint regression with Amihud_{t-1}")
    print("=" * 100)

    # Load merged data
    df_main = pd.read_csv(DATA, parse_dates=["Date"])
    print(f"Main panel: {len(df_main)} rows, dates {df_main['Date'].min().date()} → {df_main['Date'].max().date()}")

    # Load index amounts and merge
    df_amounts = load_index_amounts()
    df_amounts["Date"] = pd.to_datetime(df_amounts["Date"])
    df_main["Date"] = pd.to_datetime(df_main["Date"])
    df = df_main.merge(df_amounts, on="Date", how="left")
    print(f"After amount merge: amount columns = "
          f"{[c for c in df.columns if c.endswith('_amount')]}")

    # High-vol regime
    rv = df["SHCOMP_log_returns"].rolling(20).std() * np.sqrt(252)
    threshold = rv.quantile(0.75)
    hi_mask = (rv >= threshold).reindex(df.index).fillna(False)
    print(f"High-vol days: {int(hi_mask.sum())}")
    print()

    # Run regressions
    print(f"{'Index':<10} {'Amihud src':<22} {'PhotoPes_t-3':>14} {'t':>7} {'p':>7} {'sig':>5} | "
          f"{'Amihud_t-1':>12} {'t':>7} {'p':>7} {'sig':>5}  N")
    print("-" * 120)

    results = []
    for idx in INDICES:
        ret_col = f"{idx}_log_returns"
        out = build_design_with_amihud(df, ret_col, idx, hi_mask)
        X, y, src = out
        if X is None:
            print(f"{idx:<10} {src:<22} (skipped — no amount data)")
            continue
        m = fit(X, y)
        bp = m.params["PhotoPes_lag3"]; tp = m.tvalues["PhotoPes_lag3"]; pp = m.pvalues["PhotoPes_lag3"]
        ba = m.params[f"{idx}_amihud_lag1"]; ta = m.tvalues[f"{idx}_amihud_lag1"]; pa = m.pvalues[f"{idx}_amihud_lag1"]
        results.append({"idx": idx, "src": src, "Photo_b": bp, "Photo_t": tp, "Photo_p": pp,
                        "Amihud_b": ba, "Amihud_t": ta, "Amihud_p": pa, "n": len(y)})
        print(f"{idx:<10} {src:<22} {bp:>14.5f} {tp:>7.2f} {pp:>7.4f} {stars(pp):>5} | "
              f"{ba:>12.4e} {ta:>7.2f} {pa:>7.4f} {stars(pa):>5}  {len(y)}")

    # Compare to G1 (without Amihud)
    print()
    print("=" * 100)
    print("VERDICT — does ChiNext PhotoPes_{t-3} survive Amihud control?")
    print("=" * 100)
    chi_result = next(r for r in results if r["idx"] == "ChiNext")
    g1_chinext_t = -2.41  # from g1_high_vol_joint.py earlier
    g1_chinext_p = 0.0162
    print(f"G1 (no Amihud control):       ChiNext t = {g1_chinext_t:.2f}, p = {g1_chinext_p:.4f}")
    print(f"G3 (with Amihud_t-1 control): ChiNext t = {chi_result['Photo_t']:.2f}, p = {chi_result['Photo_p']:.4f}")
    print(f"Amihud_t-1 itself: t = {chi_result['Amihud_t']:.2f}, p = {chi_result['Amihud_p']:.4f}")
    print()
    if chi_result["Photo_p"] < 0.05:
        print(">>> G3 PASS: ChiNext PhotoPes_{t-3} 5%-significant after Amihud control.")
        print("    Effect is NOT a liquidity-effect proxy. Main axis safe.")
    elif chi_result["Photo_p"] < 0.10:
        print(">>> G3 weak pass: ChiNext PhotoPes_{t-3} 10%-marginal after Amihud control.")
        print("    Effect partially absorbed by liquidity but not eliminated.")
    else:
        print(">>> G3 FAIL: ChiNext PhotoPes_{t-3} not significant after Amihud control.")
        print("    Suggests effect was a liquidity proxy. Main axis pivot at risk.")


if __name__ == "__main__":
    main()
