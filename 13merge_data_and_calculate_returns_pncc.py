#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 13 — 合并五大指数、情感指标，计算对数收益率，生成回归数据集

数据来源：
  data/000300_000001_399106_dataset/TRD_Index.xlsx
      → SHCOMP (000001)、CSI300 (000300)、SZCOMP (399106 深证综合)
  data/399006_000905_{时段}/IDX_Idxtrd.xlsx  (4段拼接)
      → ChiNext (399006)、CSI500 (000905)
  results/weighted_photopes.csv
  results/weighted_textpes.csv

输出：
  results/merged_market_sentiment_data.csv   ← 回归用主数据集
"""

import os, sys, logging, warnings
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# 0. 路径配置
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

TRD_INDEX_FILE   = os.path.join(DATA_DIR, "000300_000001_399106_dataset", "TRD_Index.xlsx")
IDX_DIRS = [
    os.path.join(DATA_DIR, "399006_000905_2013_2016"),
    os.path.join(DATA_DIR, "399006_000905_2017_2020"),
    os.path.join(DATA_DIR, "399006_000905_2021_2024"),
    os.path.join(DATA_DIR, "399006_000905_2025_2026"),
]

PHOTOPES_FILE = os.path.join(BASE_DIR, "results", "weighted_photopes_pncc.csv")
TEXTPES_FILE  = os.path.join(BASE_DIR, "results", "weighted_textpes.csv")
OUTPUT_FILE   = os.path.join(BASE_DIR, "results", "merged_market_sentiment_data_pncc.csv")

# 论文五大指数映射（代码 → 列名）
TRD_INDEX_MAP = {1: "SHCOMP", 300: "CSI300", 399106: "SZCOMP"}   # 论文: 399106 = 深证综合 = SZCOMP
IDX_CODE_MAP  = {905: "CSI500", 399006: "ChiNext"}                # 399006 = 创业板指, 000905 = 中证500

FIVE_INDICES  = ["SHCOMP", "CSI300", "SZCOMP", "ChiNext", "CSI500"]   # 论文顺序
SENTIMENT_COLS = ["PhotoPes", "TextPes", "WeightedPhotoPes", "WeightedTextPes"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. 前置检查
# ─────────────────────────────────────────────────────────────────────────────
def preflight_check() -> bool:
    ok = True
    logging.info("=== 前置检查 ===")

    for path, label in [
        (TRD_INDEX_FILE,  "000300_000001_399106 TRD_Index"),
        (PHOTOPES_FILE,   "weighted_photopes.csv"),
        (TEXTPES_FILE,    "weighted_textpes.csv"),
    ]:
        if not os.path.exists(path):
            logging.error(f"文件不存在: {path}  [{label}]"); ok = False
        else:
            logging.info(f"  ✓ {label}")

    for d in IDX_DIRS:
        f = os.path.join(d, "IDX_Idxtrd.xlsx")
        if not os.path.exists(f):
            logging.error(f"文件不存在: {f}"); ok = False
        else:
            logging.info(f"  ✓ {os.path.basename(d)}/IDX_Idxtrd.xlsx")

    # 列名快速检查
    try:
        df_p = pd.read_csv(PHOTOPES_FILE, nrows=2)
        df_t = pd.read_csv(TEXTPES_FILE,  nrows=2)
        for col in ["news_date", "PhotoPes"]:
            if col not in df_p.columns: logging.error(f"weighted_photopes 缺列 '{col}'"); ok = False
        for col in ["news_date", "TextPes"]:
            if col not in df_t.columns: logging.error(f"weighted_textpes 缺列 '{col}'"); ok = False
    except Exception as e:
        logging.error(f"情感文件读取失败: {e}"); ok = False

    logging.info("前置检查" + ("通过 ✓\n" if ok else "失败 ✗"))
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# 2A. 加载 TRD_Index（SHCOMP / CSI300 / SZCOMP）
# ─────────────────────────────────────────────────────────────────────────────
def load_trd_index() -> pd.DataFrame:
    """
    读取 000300_000001_399106_dataset/TRD_Index.xlsx
    格式：skiprows=[1,2]（跳过中文名行和单位行），列名由第0行提供
    返回: wide DataFrame, index=Date, cols=[SHCOMP, CSI300, SZCOMP]
    """
    logging.info(f"加载 TRD_Index → {os.path.basename(TRD_INDEX_FILE)}")
    df = pd.read_excel(TRD_INDEX_FILE, skiprows=[1, 2])
    df["Indexcd"]  = pd.to_numeric(df["Indexcd"],  errors="coerce").astype("Int64")
    df["Trddt"]    = pd.to_datetime(df["Trddt"],   errors="coerce")
    df["Clsindex"] = pd.to_numeric(df["Clsindex"], errors="coerce")
    df = df.dropna(subset=["Indexcd", "Trddt", "Clsindex"])

    # 只保留论文 3 个目标指数
    df = df[df["Indexcd"].isin(TRD_INDEX_MAP)]
    df["Name"] = df["Indexcd"].map(TRD_INDEX_MAP)

    wide = (df.pivot_table(index="Trddt", columns="Name", values="Clsindex", aggfunc="last")
              .sort_index()
              .reset_index()
              .rename(columns={"Trddt": "Date"}))

    found = [c for c in FIVE_INDICES[:3] if c in wide.columns]
    logging.info(f"  TRD_Index 指数: {found}, 行数={len(wide)}, "
                 f"日期={wide['Date'].min().date()}–{wide['Date'].max().date()}")
    return wide


# ─────────────────────────────────────────────────────────────────────────────
# 2B. 加载 IDX_Idxtrd（ChiNext / CSI500，4段拼接）
# ─────────────────────────────────────────────────────────────────────────────
def load_idx_idxtrd() -> pd.DataFrame:
    """
    读取 4 个时间分段的 IDX_Idxtrd.xlsx，垂直拼接后 pivot 成宽格式。
    格式：skiprows=[1,2]（行0=中文名，行1=单位）
    列名映射：Indexcd, Idxtrd01→Trddt, Idxtrd05→Clsindex
    """
    logging.info("加载 IDX_Idxtrd（ChiNext + CSI500，4段拼接）")
    frames = []
    for d in IDX_DIRS:
        path = os.path.join(d, "IDX_Idxtrd.xlsx")
        df = pd.read_excel(path, skiprows=[1, 2])
        # 标准化列名
        df.columns = ["Indexcd", "Trddt", "Opnindex", "Hiindex", "Loindex", "Clsindex"]
        df["Indexcd"]  = pd.to_numeric(df["Indexcd"],  errors="coerce").astype("Int64")
        df["Trddt"]    = pd.to_datetime(df["Trddt"],   errors="coerce")
        df["Clsindex"] = pd.to_numeric(df["Clsindex"], errors="coerce")
        df = df.dropna(subset=["Indexcd", "Trddt", "Clsindex"])
        df = df[df["Indexcd"].isin(IDX_CODE_MAP)]
        frames.append(df)
        logging.info(f"  {os.path.basename(d)}: {len(df)} 行, codes={df['Indexcd'].unique().tolist()}")

    combined = pd.concat(frames, ignore_index=True)
    combined["Name"] = combined["Indexcd"].map(IDX_CODE_MAP)

    wide = (combined.pivot_table(index="Trddt", columns="Name", values="Clsindex", aggfunc="last")
                    .sort_index()
                    .reset_index()
                    .rename(columns={"Trddt": "Date"}))

    # 去重（时段边界可能重叠）
    wide = wide.drop_duplicates(subset=["Date"])

    found = [c for c in ["ChiNext", "CSI500"] if c in wide.columns]
    logging.info(f"  IDX 拼接后: {found}, 行数={len(wide)}, "
                 f"日期={wide['Date'].min().date()}–{wide['Date'].max().date()}")
    return wide


# ─────────────────────────────────────────────────────────────────────────────
# 3. 合并五大指数 → 计算对数收益率
# ─────────────────────────────────────────────────────────────────────────────
def build_index_returns(trd_df: pd.DataFrame, idx_df: pd.DataFrame) -> pd.DataFrame:
    """
    外连接五大指数价格序列，以 TRD_Index 的交易日历为主，
    补入 ChiNext/CSI500，计算各指数的对数日收益率和简单收益率。
    """
    logging.info("合并五大指数并计算收益率")
    # outer merge：保留所有交易日（两套数据应基本重合）
    df = pd.merge(trd_df, idx_df, on="Date", how="outer").sort_values("Date").reset_index(drop=True)

    # 填充偶发缺口（非交易日导致的单边缺口）
    price_cols = [c for c in FIVE_INDICES if c in df.columns]
    df[price_cols] = df[price_cols].ffill()

    # 计算收益率（按指数分组，确保不跨指数串连）
    for name in price_cols:
        df[f"{name}_log_returns"] = np.log(df[name] / df[name].shift(1))
        df[f"{name}_returns"]     = df[name].pct_change()   # 保留 pct_change 供回归 notebook 兼容

    missing = [c for c in FIVE_INDICES if c not in df.columns]
    if missing:
        logging.warning(f"以下指数未找到: {missing}")

    logging.info(f"  五大指数合并后: {len(df)} 行, "
                 f"日期={df['Date'].min().date()}–{df['Date'].max().date()}")
    logging.info(f"  可用收益率列: {[f'{n}_log_returns' for n in price_cols]}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. 加载情感数据
# ─────────────────────────────────────────────────────────────────────────────
def load_sentiment() -> pd.DataFrame:
    logging.info("加载情感数据")
    df_p = pd.read_csv(PHOTOPES_FILE)
    df_p["news_date"] = pd.to_datetime(df_p["news_date"], errors="coerce")
    df_p = df_p.dropna(subset=["news_date"]).sort_values("news_date")

    df_t = pd.read_csv(TEXTPES_FILE)
    df_t["news_date"] = pd.to_datetime(df_t["news_date"], errors="coerce")
    df_t = df_t.dropna(subset=["news_date"]).sort_values("news_date")

    photo_keep = ["news_date"] + [c for c in ["PhotoPes","PhotoPes_likelihood","WeightedPhotoPes"]
                                  if c in df_p.columns]
    text_keep  = ["news_date"] + [c for c in ["TextPes","TextPes_likelihood","WeightedTextPes"]
                                  if c in df_t.columns]

    df = pd.merge(df_p[photo_keep], df_t[text_keep], on="news_date", how="outer").sort_values("news_date")
    logging.info(f"  情感数据: {len(df)} 行, 列={df.columns.tolist()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. 合并指数 + 情感（以交易日为基准，左连接情感，前向填充缺口）
# ─────────────────────────────────────────────────────────────────────────────
def merge_all(index_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    logging.info("合并指数与情感数据")
    sent = sentiment_df.rename(columns={"news_date": "Date"})
    merged = pd.merge(index_df, sent, on="Date", how="left").sort_values("Date").reset_index(drop=True)

    sent_cols = [c for c in SENTIMENT_COLS if c in merged.columns]
    merged[sent_cols] = merged[sent_cols].ffill().bfill()

    # 样本期内剩余缺失警告
    window = merged[(merged["Date"] >= "2014-01-01") & (merged["Date"] <= "2024-12-31")]
    for col in sent_cols:
        n = window[col].isna().sum()
        if n:
            logging.warning(f"  {col}: 样本期内仍有 {n} 缺失值，用全局均值填充")
            merged[col] = merged[col].fillna(merged[col].mean())

    logging.info(f"  合并后: {len(merged)} 行")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 6. 1% Winsorization + z-score 标准化
# ─────────────────────────────────────────────────────────────────────────────
def winsorize_standardize(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Winsorization(1%) + 标准化")
    for col in SENTIMENT_COLS:
        if col not in df.columns:
            continue
        mask = df[col].notna()
        arr = df.loc[mask, col].values.copy()
        df.loc[mask, col] = np.array(winsorize(arr, limits=[0.01, 0.01]))
        mu, sigma = df[col].mean(), df[col].std()
        df[f"{col}_std"] = (df[col] - mu) / sigma
        logging.info(f"  {col}: μ={mu:.4f}, σ={sigma:.4f}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. Lag 1-5：情感（原始+标准化）、对数收益率、收益率平方
# ─────────────────────────────────────────────────────────────────────────────
def create_lags(df: pd.DataFrame, n_lags: int = 5) -> pd.DataFrame:
    logging.info(f"生成 Lag 1–{n_lags}")
    df = df.sort_values("Date").reset_index(drop=True)

    # 情感滞后
    lag_sent = [c for c in df.columns
                if c in SENTIMENT_COLS or c in [f"{s}_std" for s in SENTIMENT_COLS]]
    for col in lag_sent:
        for lag in range(1, n_lags + 1):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # 对数收益率及平方滞后
    for col in [c for c in df.columns if c.endswith("_log_returns")]:
        sq = f"{col}_sq"
        df[sq] = df[col] ** 2
        for lag in range(1, n_lags + 1):
            df[f"{col}_lag{lag}"]  = df[col].shift(lag)
            df[f"{sq}_lag{lag}"]   = df[sq].shift(lag)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 8. 星期虚拟变量（周二–周五，周一为基准）
# ─────────────────────────────────────────────────────────────────────────────
def add_weekday_dummies(df: pd.DataFrame) -> pd.DataFrame:
    # Date.dt.dayofweek: Mon=0 … Fri=4
    df["_dow"] = df["Date"].dt.dayofweek + 1   # Mon=1 … Fri=5
    for code, name in {2:"Tue", 3:"Wed", 4:"Thu", 5:"Fri"}.items():
        df[f"weekday_{name}"] = (df["_dow"] == code).astype(int)
    return df.drop(columns=["_dow"])


# ─────────────────────────────────────────────────────────────────────────────
# 9. 描述性统计（Table 1 草稿）
# ─────────────────────────────────────────────────────────────────────────────
def print_descriptive_stats(df: pd.DataFrame):
    sample = df[(df["Date"] >= "2014-01-01") & (df["Date"] <= "2024-12-31")].copy()
    print("\n" + "="*70)
    print("描述性统计  (2014–2024 样本期)")
    print("="*70)

    # Panel A: 情感指标
    print("\nPanel A: 情感指标")
    sent_raw = [c for c in ["PhotoPes", "TextPes"] if c in sample.columns]
    if sent_raw:
        stats = sample[sent_raw].agg(["count","mean","median",
                                      lambda x: x.quantile(0.25),
                                      lambda x: x.quantile(0.75),
                                      "std"])
        stats.index = ["N","Mean","Median","P25","P75","Std"]
        print(stats.round(4).T.to_string())

    # Panel B: 指数日收益率
    print("\nPanel B: 五大指数对数日收益率")
    ret_cols = [f"{idx}_log_returns" for idx in FIVE_INDICES if f"{idx}_log_returns" in sample.columns]
    if ret_cols:
        stats = sample[ret_cols].agg(["count","mean","median",
                                      lambda x: x.quantile(0.25),
                                      lambda x: x.quantile(0.75),
                                      "std"])
        stats.index = ["N","Mean","Median","P25","P75","Std"]
        print(stats.round(6).T.to_string())

    # 总样本期新闻天数
    if "PhotoPes" in sample.columns:
        news_days = sample["PhotoPes"].notna().sum()
        print(f"\n情感数据覆盖交易日天数 (2014-2024): {news_days}")
    print("="*70)


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────
def main():
    logging.info("="*60)
    logging.info("Script 13: 五大指数 + 情感 合并，生成回归数据集")
    logging.info("="*60 + "\n")

    if not preflight_check():
        sys.exit(1)

    # 2. 加载指数
    trd_df  = load_trd_index()
    idx_df  = load_idx_idxtrd()

    # 3. 合并五大指数，计算收益率
    index_df = build_index_returns(trd_df, idx_df)

    # 4. 加载情感
    sentiment_df = load_sentiment()

    # 5. 合并
    merged = merge_all(index_df, sentiment_df)

    # 6. Winsorize + 标准化（在生成滞后前，确保滞后值也是处理后的）
    merged = winsorize_standardize(merged)

    # 7. 生成 Lag 1-5
    merged = create_lags(merged, n_lags=5)

    # 8. 星期虚拟变量
    merged = add_weekday_dummies(merged)

    # 9. 保存（保留 2013 数据供滞后计算，回归时按年份筛选）
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    merged.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    logging.info(f"\n回归数据集已保存: {OUTPUT_FILE}")
    logging.info(f"  行数={len(merged)}, 列数={len(merged.columns)}")
    logging.info(f"  日期范围: {merged['Date'].min().date()} – {merged['Date'].max().date()}")

    # 10. 描述性统计
    print_descriptive_stats(merged)

    # 11. 样本验证：首个所有 lag-5 完整的日期
    lag5_cols = [c for c in merged.columns if "_lag5" in c]
    first_full_df = merged[(merged["Date"] >= "2014-01-01") &
                           merged[lag5_cols].notna().all(axis=1)]
    if len(first_full_df):
        first_date = first_full_df["Date"].iloc[0].date()
        logging.info(f"\n首个 Lag-5 全部完整的日期: {first_date}")
        logging.info(f"  2014-2024 样本期内完整行数: {len(first_full_df[first_full_df['Date']<='2024-12-31'])}")

    # 12. 前5行关键列预览
    key = (["Date"] +
           [f"{n}_log_returns" for n in FIVE_INDICES if f"{n}_log_returns" in merged.columns] +
           ["PhotoPes","TextPes"] +
           ["PhotoPes_lag3","TextPes_lag3"] +
           ["SHCOMP_log_returns_lag3","ChiNext_log_returns_lag3"])
    key = [c for c in key if c in merged.columns]
    preview = first_full_df[key].head(5)
    print("\n前5行回归数据预览（首个完整 Lag-5 日期起）：")
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", "{:.6f}".format):
        print(preview.to_string(index=False))

    logging.info("\n========== 完成 ==========")


if __name__ == "__main__":
    main()
