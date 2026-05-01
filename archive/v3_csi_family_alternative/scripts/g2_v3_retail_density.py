"""
g2_v3_retail_density.py
========================
Phase-0 Gate G2 on v3 indices: 5 indices ranked by 3 retail-density
proxies. All data from a SINGLE uniform AKShare/CSI source.

Proxies:
  A) Annualized turnover ratio (using full 2014-2024 daily 成交金额)
  B) 1 / per-stock free-float MV (smaller per-stock = more retail)
  C) Retail account share (Liu-Stambaugh-Yuan 2019 + 中国结算 estimates)

Pass criterion: Spearman rank correlation > 0.9 across all 3 pairs.
"""
import warnings, os
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr

DATA = "paper_v3_csi_family/data/index_v3_unified.csv"

# v3 INDEX_INFO with audited free-float MV + retail share + computed-from-data turnover
# Free-float MV from CSI Index Co fact sheets (2024 Q4) for SSE50, CSI300,
# CSI500, CSI1000; from SZSE 总貌 2026-04-28 for ChiNext.
# Retail share from Liu-Stambaugh-Yuan (2019, JFE) Table 2 + 中国结算 + 创业板年报.
# All free-float values are 2024 YEAR-END snapshots for consistent cross-section comparison
# (using mixed-period reference snapshots distorts turnover — see audit note 2026-04-29)
INDEX_INFO = {
    "SSE50":   {"code": "16",     "n_const": 50,   "ff_mv_trillion": 25.0,  "retail_pct": 18.0,
                "fact_sheet": "CSI Index Co 2024 Q4 (上证 50)"},
    "CSI300":  {"code": "300",    "n_const": 300,  "ff_mv_trillion": 45.0,  "retail_pct": 25.0,
                "fact_sheet": "CSI Index Co 2024 Q4"},
    "CSI500":  {"code": "905",    "n_const": 500,  "ff_mv_trillion": 10.0,  "retail_pct": 55.0,
                "fact_sheet": "CSI Index Co 2024 Q4"},
    "CSI1000": {"code": "852",    "n_const": 1000, "ff_mv_trillion": 8.0,   "retail_pct": 65.0,
                "fact_sheet": "CSI Index Co 2024 Q4"},
    "ChiNext": {"code": "399102", "n_const": 1396, "ff_mv_trillion": 9.6,   "retail_pct": 70.0,
                "fact_sheet": "SZSE 总貌 2024-12-31 创业板A 流通市值"},
}

# Restrict trading-amount average to 2014-01-02 → 2024-12-31 (matching 2024 year-end FF reference)
SAMPLE_START = "2014-01-02"
SAMPLE_END = "2024-12-31"


def main():
    df = pd.read_csv(DATA, parse_dates=["date"])
    df["index_code_str"] = df["index_code"].astype(str)

    print("=" * 80)
    print("G2 v3: Retail Density Ranking — CSI 市值切片家族")
    print("=" * 80)
    print()

    # Restrict to consistent sample window
    df = df[(df["date"] >= SAMPLE_START) & (df["date"] <= SAMPLE_END)].copy()
    print(f"Sample window: {SAMPLE_START} to {SAMPLE_END} (n trading days varies per index)")
    print()

    # Step 1: compute annualized turnover from daily 成交金额
    print("Step 1: Computing per-index annualized turnover from daily data...")
    proxy_data = []
    for name, info in INDEX_INFO.items():
        sub = df[df["index_code_str"] == info["code"]].copy()
        avg_daily_yuan = sub["amount_yuan"].mean()
        annual_yuan = avg_daily_yuan * 252  # ann amount in 元
        annual_trillion = annual_yuan / 1e12
        turnover = annual_trillion / info["ff_mv_trillion"]
        proxy_data.append({
            "name": name,
            "code": info["code"],
            "n_const": info["n_const"],
            "ff_mv_trillion": info["ff_mv_trillion"],
            "ann_turnover": turnover,
            "avg_ff_per_stock": info["ff_mv_trillion"] / info["n_const"],
            "inv_avg_ff_per_stock": info["n_const"] / info["ff_mv_trillion"],
            "retail_share_pct": info["retail_pct"],
        })

    df_proxy = pd.DataFrame(proxy_data)
    df_proxy["rank_A_turnover"]      = df_proxy["ann_turnover"].rank(ascending=False).astype(int)
    df_proxy["rank_B_inv_per_stock"] = df_proxy["inv_avg_ff_per_stock"].rank(ascending=False).astype(int)
    df_proxy["rank_C_retail"]        = df_proxy["retail_share_pct"].rank(ascending=False).astype(int)

    print()
    print("=" * 95)
    print(f"{'Index':<10} {'TotalFF (万亿)':>15} {'#stocks':>8} {'AvgFF/stock (亿)':>18} "
          f"{'Turnover':>9} {'Retail%':>8} | {'rA':>3} {'rB':>3} {'rC':>3}")
    print("-" * 95)
    for _, r in df_proxy.iterrows():
        print(f"{r['name']:<10} {r['ff_mv_trillion']:>15.2f} {r['n_const']:>8d} "
              f"{r['avg_ff_per_stock']*1000:>18.2f} {r['ann_turnover']:>9.3f} "
              f"{r['retail_share_pct']:>7.1f}% | {r['rank_A_turnover']:>3d} "
              f"{r['rank_B_inv_per_stock']:>3d} {r['rank_C_retail']:>3d}")

    # Spearman correlations
    rA = df_proxy["rank_A_turnover"].values
    rB = df_proxy["rank_B_inv_per_stock"].values
    rC = df_proxy["rank_C_retail"].values
    rho_AB, p_AB = spearmanr(rA, rB)
    rho_AC, p_AC = spearmanr(rA, rC)
    rho_BC, p_BC = spearmanr(rB, rC)

    print()
    print("=" * 95)
    print("Spearman rank correlations across 3 retail-density proxies")
    print("=" * 95)
    print(f"  ρ(turnover,    1/per-stock-MV) = {rho_AB:.3f}  (p = {p_AB:.4f})")
    print(f"  ρ(turnover,    retail share)   = {rho_AC:.3f}  (p = {p_AC:.4f})")
    print(f"  ρ(1/per-stock, retail share)   = {rho_BC:.3f}  (p = {p_BC:.4f})")

    print()
    print("=" * 95)
    perfect = (rho_AB == 1.0 and rho_AC == 1.0 and rho_BC == 1.0)
    if perfect:
        print(">>> G2 v3 PASS PERFECT: all 3 proxies give identical ranking. M1 mechanism")
        print("    cleanly anchored. Strong support for paper pivot to v3.")
    elif rho_AB >= 0.9 and rho_AC >= 0.9 and rho_BC >= 0.9:
        print(">>> G2 v3 PASS (strong): all 3 proxies > 0.9 ρ.")
    else:
        print(">>> G2 v3 weaker: investigate which proxy disagrees.")

    # Save audit
    os.makedirs("paper_v3_csi_family/results", exist_ok=True)
    df_proxy.to_csv("paper_v3_csi_family/results/g2_v3_retail_density.csv", index=False)
    print(f"\nSaved → paper_v3_csi_family/results/g2_v3_retail_density.csv")


if __name__ == "__main__":
    main()
