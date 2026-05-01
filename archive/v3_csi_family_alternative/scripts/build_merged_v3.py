"""
build_merged_v3.py
===================
Build the v3 merged daily panel:
  - 5 indices: 上证50 (000016), 沪深300 (000300), 中证500 (000905),
               中证1000 (000852), 创业板综指 (399102)
  - PhotoPes / TextPes: same as v2 (already computed in
    results/merged_market_sentiment_data_old.csv)

For each index we compute log returns, R^2, and 5 lags. PhotoPes/TextPes
are standardized (winsorize 1% + z-score) and lagged 5 periods, matching
the v2 pipeline exactly.

Output: paper_v3_csi_family/data/merged_v3.csv
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize

V2_MERGED = "results/merged_market_sentiment_data_old.csv"
V3_INDEX = "paper_v3_csi_family/data/index_v3_unified.csv"
OUTPUT = "paper_v3_csi_family/data/merged_v3.csv"

# v3 indices in same order as v2 (for column-wise comparability)
V3_MAP = {
    "16":     "SSE50",       # 上证 50
    "300":    "CSI300",      # 沪深 300
    "905":    "CSI500",      # 中证 500
    "852":    "CSI1000",     # 中证 1000
    "399102": "ChiNext",     # 创业板综指
}


def load_v3_indices():
    """Load v3 unified index data, pivot to wide format with one column per index."""
    df = pd.read_csv(V3_INDEX, parse_dates=["date"])
    df["index_code_str"] = df["index_code"].astype(str)
    out = pd.DataFrame({"Date": sorted(df["date"].unique())})
    for code, name in V3_MAP.items():
        g = df[df["index_code_str"] == code][["date", "close", "amount_yuan"]].rename(
            columns={"date": "Date", "close": f"{name}_close", "amount_yuan": f"{name}_amount"}
        )
        out = out.merge(g, on="Date", how="left")
    return out


def compute_returns_lags(df):
    """Compute log returns, squared returns, and 5 lags for each v3 index."""
    for name in V3_MAP.values():
        df[f"{name}_log_returns"] = np.log(df[f"{name}_close"] / df[f"{name}_close"].shift(1))
        df[f"{name}_log_returns_sq"] = df[f"{name}_log_returns"] ** 2
        for lag in range(1, 6):
            df[f"{name}_log_returns_lag{lag}"] = df[f"{name}_log_returns"].shift(lag)
            df[f"{name}_log_returns_sq_lag{lag}"] = df[f"{name}_log_returns_sq"].shift(lag)
    return df


def attach_sentiment(df):
    """Pull PhotoPes, TextPes, and their standardized + lagged versions from v2 file."""
    v2 = pd.read_csv(V2_MERGED, parse_dates=["Date"])
    sentiment_cols = [
        "PhotoPes", "TextPes",
        "PhotoPes_std", "TextPes_std",
    ] + [f"PhotoPes_std_lag{i}" for i in range(1, 6)] + [f"TextPes_std_lag{i}" for i in range(1, 6)]
    sentiment_cols = [c for c in sentiment_cols if c in v2.columns]
    v2 = v2[["Date"] + sentiment_cols]
    return df.merge(v2, on="Date", how="left")


def add_weekday_dummies(df):
    """Add Tuesday-Friday dummies (Monday is base)."""
    dow = df["Date"].dt.dayofweek
    df["weekday_Tue"] = (dow == 1).astype(int)
    df["weekday_Wed"] = (dow == 2).astype(int)
    df["weekday_Thu"] = (dow == 3).astype(int)
    df["weekday_Fri"] = (dow == 4).astype(int)
    return df


def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    print("=" * 70)
    print("Building v3 merged dataset")
    print("=" * 70)

    df = load_v3_indices()
    print(f"v3 index panel: {len(df)} rows, "
          f"date range {df['Date'].min().date()} → {df['Date'].max().date()}")

    df = compute_returns_lags(df)
    df = attach_sentiment(df)
    df = add_weekday_dummies(df)

    # Filter to 2014+ where sentiment data is available
    df = df[df["Date"] >= "2014-01-02"].reset_index(drop=True)

    df.to_csv(OUTPUT, index=False)
    print(f"Saved {len(df)} rows × {len(df.columns)} cols → {OUTPUT}")

    # Audit per-index data availability
    print("\nPer-index trading-day count after merge:")
    for name in V3_MAP.values():
        n_close = df[f"{name}_close"].notna().sum()
        n_ret = df[f"{name}_log_returns"].notna().sum()
        n_sentiment = df[f"PhotoPes"].notna().sum() if "PhotoPes" in df else 0
        print(f"  {name:<10}: close={n_close}, returns={n_ret}, sentiment={n_sentiment}")


if __name__ == "__main__":
    main()
