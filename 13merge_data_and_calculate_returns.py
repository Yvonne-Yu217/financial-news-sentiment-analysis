#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并市场指数、文本情感和图片情感数据，并计算收益率

输入：
  - indicator_dataset/TRD_Index.xlsx  ← CSMAR宽格式指数数据
  - results/weighted_photopes.csv
  - results/weighted_textpes.csv

核心逻辑：
  1. 读取 TRD_Index.xlsx（长格式），pivot 成宽格式，每列一个指数
  2. 对 Clsindex 计算对数日收益率 R_t = ln(P_t / P_{t-1})
  3. 利用 2013-12-30 起始数据（早于样本期2014），使 Lag 1-5 在 2014年初无缺失
  4. 将情感指标按交易日期合并（左连接到指数数据，前向填充缺口）
  5. 对 PhotoPes / TextPes 执行 1% Winsorization
  6. 计算标准化情感 (z-score)
  7. 生成 Lag 1-5：情感、R_t、R²_t
  8. 生成星期虚拟变量（周二–周五，周一为基准组）
  9. 输出回归数据集到 results/merged_market_sentiment_data.csv
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# =============================================================================
# 0. 文件路径配置
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INDEX_FILE    = os.path.join(BASE_DIR, "indicator_dataset", "TRD_Index.xlsx")
PHOTOPES_FILE = os.path.join(BASE_DIR, "results", "weighted_photopes.csv")
TEXTPES_FILE  = os.path.join(BASE_DIR, "results", "weighted_textpes.csv")
OUTPUT_FILE   = os.path.join(BASE_DIR, "results", "merged_market_sentiment_data.csv")

# CSMAR 指数代码映射（整数→论文名称）
# 描述文件：000001=上证综指, 000300=沪深300, 399001=深证成份指数
# 399106=深证综合指数, 000903/399903=中证100
INDEX_CODE_MAP = {
    1:      "SHCOMP",    # 000001 上证综合指数
    300:    "CSI300",    # 000300 沪深300
    399001: "SZCOMP",   # 399001 深证成份指数
    399106: "SZSE",     # 399106 深证综合指数
    903:    "CSI100",   # 000903 中证100
}
# 论文回归使用的主要情感列名
SENTIMENT_COLS = ["PhotoPes", "TextPes", "WeightedPhotoPes", "WeightedTextPes"]


# =============================================================================
# 1. 前置检查：列名与日期格式
# =============================================================================
def preflight_check():
    """
    检查 weighted_photopes.csv 与 TRD_Index.xlsx 的列名及日期格式是否匹配。
    返回 True 则继续执行，False 则终止。
    """
    logging.info("=== 前置检查 ===")
    ok = True

    # — 检查 PhotoPes
    try:
        df_p = pd.read_csv(PHOTOPES_FILE, nrows=3)
        logging.info(f"weighted_photopes.csv 列名: {df_p.columns.tolist()}")
        if "news_date" not in df_p.columns:
            logging.error("weighted_photopes.csv 缺少 'news_date' 列！")
            ok = False
        else:
            sample_date = str(df_p["news_date"].iloc[0])
            pd.to_datetime(sample_date)   # 触发格式错误
            logging.info(f"  日期示例: {sample_date} → 格式正常")
    except Exception as e:
        logging.error(f"weighted_photopes.csv 读取失败: {e}")
        ok = False

    # — 检查 TextPes
    try:
        df_t = pd.read_csv(TEXTPES_FILE, nrows=3)
        logging.info(f"weighted_textpes.csv 列名: {df_t.columns.tolist()}")
        if "news_date" not in df_t.columns:
            logging.error("weighted_textpes.csv 缺少 'news_date' 列！")
            ok = False
    except Exception as e:
        logging.error(f"weighted_textpes.csv 读取失败: {e}")
        ok = False

    # — 检查 TRD_Index
    try:
        # skiprows=[1,2]：保留第0行为列名，跳过第1行(中文名)和第2行(单位说明)
        df_idx = pd.read_excel(INDEX_FILE, skiprows=[1, 2], nrows=3)
        logging.info(f"TRD_Index.xlsx 列名: {df_idx.columns.tolist()}")
        for col in ["Indexcd", "Trddt", "Clsindex"]:
            if col not in df_idx.columns:
                logging.error(f"TRD_Index.xlsx 缺少列 '{col}'！")
                ok = False
        if "Trddt" in df_idx.columns:
            sample_trddt = str(df_idx["Trddt"].iloc[0])
            pd.to_datetime(sample_trddt)
            logging.info(f"  Trddt 示例: {sample_trddt} → 格式正常")
    except Exception as e:
        logging.error(f"TRD_Index.xlsx 读取失败: {e}")
        ok = False

    if ok:
        logging.info("前置检查通过。\n")
    else:
        logging.error("前置检查失败，请修正后重新运行。")
    return ok


# =============================================================================
# 2. 加载 TRD_Index.xlsx → 宽格式 + 对数收益率
# =============================================================================
def load_index_data() -> pd.DataFrame:
    """
    读取 TRD_Index.xlsx（长格式），pivot 成宽格式，计算对数日收益率。

    返回 DataFrame，索引为日期，列为各指数的 {name}_log_returns 及原始收盘价 {name}。
    数据从 2013-12-30 开始，确保 2014 年初 Lag 1-5 无缺失。
    """
    logging.info("加载 TRD_Index.xlsx …")
    # 第 0 行为英文列头，第 1 行为中文名(跳过)，第 2 行为单位说明(跳过)
    df = pd.read_excel(INDEX_FILE, skiprows=[1, 2])

    # 转换类型
    df["Trddt"]    = pd.to_datetime(df["Trddt"], errors="coerce")
    df["Clsindex"] = pd.to_numeric(df["Clsindex"], errors="coerce")
    df["Indexcd"]  = pd.to_numeric(df["Indexcd"],  errors="coerce")

    # 删除无效行
    df = df.dropna(subset=["Trddt", "Clsindex", "Indexcd"])
    df["Indexcd"] = df["Indexcd"].astype(int)

    # 只保留已映射的指数
    df = df[df["Indexcd"].isin(INDEX_CODE_MAP)]
    df["index_name"] = df["Indexcd"].map(INDEX_CODE_MAP)

    available = df["index_name"].unique().tolist()
    logging.info(f"可用指数: {available}")

    # Pivot → 每列一个指数的收盘价
    wide = df.pivot_table(index="Trddt", columns="index_name", values="Clsindex", aggfunc="last")
    wide = wide.sort_index()

    # 计算对数日收益率 R_t = ln(P_t / P_{t-1})
    for name in available:
        if name in wide.columns:
            wide[f"{name}_log_returns"] = np.log(wide[name] / wide[name].shift(1))

    # 同时保留简单百分比收益率，供回归 notebook 的现有管道兼容
    for name in available:
        if name in wide.columns:
            wide[f"{name}_returns"] = wide[name].pct_change()

    wide.index.name = "Date"
    wide = wide.reset_index()

    logging.info(f"指数数据: {len(wide)} 行，日期范围 {wide['Date'].min().date()} – {wide['Date'].max().date()}")
    return wide


# =============================================================================
# 3. 加载情感数据
# =============================================================================
def load_sentiment_data() -> pd.DataFrame:
    """
    读取 weighted_photopes.csv 和 weighted_textpes.csv，合并后返回。
    日期列统一命名为 news_date。
    """
    logging.info("加载情感数据 …")

    df_photo = pd.read_csv(PHOTOPES_FILE)
    df_photo["news_date"] = pd.to_datetime(df_photo["news_date"], errors="coerce")
    df_photo = df_photo.dropna(subset=["news_date"]).sort_values("news_date")

    df_text = pd.read_csv(TEXTPES_FILE)
    df_text["news_date"] = pd.to_datetime(df_text["news_date"], errors="coerce")
    df_text = df_text.dropna(subset=["news_date"]).sort_values("news_date")

    # 选取需要的列
    photo_keep = ["news_date"] + [c for c in ["PhotoPes", "PhotoPes_likelihood",
                                               "WeightedPhotoPes"] if c in df_photo.columns]
    text_keep  = ["news_date"] + [c for c in ["TextPes",  "TextPes_likelihood",
                                               "WeightedTextPes"]  if c in df_text.columns]

    df_sentiment = pd.merge(
        df_photo[photo_keep], df_text[text_keep],
        on="news_date", how="outer"
    ).sort_values("news_date")

    logging.info(f"情感数据: {len(df_sentiment)} 行，列: {df_sentiment.columns.tolist()}")
    return df_sentiment


# =============================================================================
# 4. 合并：指数（交易日为基准） + 情感（前向填充缺口）
# =============================================================================
def merge_datasets(index_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    以交易日历（来自 TRD_Index）为基准，左连接情感数据。
    非交易日的情感数据通过前向填充处理，使交易日始终有情感值。
    """
    logging.info("合并指数与情感数据 …")

    sentiment_df = sentiment_df.rename(columns={"news_date": "Date"})

    # 左连接：保留所有交易日
    merged = pd.merge(index_df, sentiment_df, on="Date", how="left")
    merged = merged.sort_values("Date").reset_index(drop=True)

    # 情感列：前向填充（新闻缺失的交易日用前一天），再后向填充（极少数开头缺口）
    sent_cols_present = [c for c in SENTIMENT_COLS
                         if c in merged.columns]
    if sent_cols_present:
        merged[sent_cols_present] = merged[sent_cols_present].ffill().bfill()

    # 统计缺失情况
    sample_window = merged[(merged["Date"] >= "2014-01-01") & (merged["Date"] <= "2024-12-31")]
    for col in sent_cols_present:
        n_miss = sample_window[col].isna().sum()
        if n_miss:
            logging.warning(f"  {col}: 样本期内仍有 {n_miss} 个缺失值，用均值填充")
            col_mean = merged[col].mean()
            merged[col] = merged[col].fillna(col_mean)

    logging.info(f"合并后: {len(merged)} 行")
    return merged


# =============================================================================
# 5. Winsorization（1%）+ 标准化
# =============================================================================
def winsorize_and_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    对 SENTIMENT_COLS 中存在的列执行：
      - 1% Winsorization（上下各 1%）
      - z-score 标准化，生成 {col}_std 列
    """
    logging.info("执行 1% Winsorization 和标准化 …")
    for col in SENTIMENT_COLS:
        if col not in df.columns:
            continue
        # 仅对有效（非 NaN）部分做 winsorize
        valid_mask = df[col].notna()
        arr = df.loc[valid_mask, col].values.copy()
        arr_wins = np.array(winsorize(arr, limits=[0.01, 0.01]))
        df.loc[valid_mask, col] = arr_wins

        # z-score
        mu, sigma = df[col].mean(), df[col].std()
        df[f"{col}_std"] = (df[col] - mu) / sigma
        logging.info(f"  {col}: mean={mu:.4f}, std={sigma:.4f} (after winsorize)")
    return df


# =============================================================================
# 6. 生成滞后项 Lag 1-5
# =============================================================================
def create_lags(df: pd.DataFrame, n_lags: int = 5) -> pd.DataFrame:
    """
    为情感指标（原始值 + 标准化值）、对数收益率 R_t 及 R²_t 生成 lag 1–n_lags。
    数据按日期排序，使用 .shift() 确保正确对齐。
    """
    logging.info(f"生成 Lag 1–{n_lags} …")
    df = df.sort_values("Date").reset_index(drop=True)

    # 情感列（原始 + 标准化）
    lag_sentiment = [c for c in df.columns
                     if c in SENTIMENT_COLS or c in [f"{s}_std" for s in SENTIMENT_COLS]]

    for col in lag_sentiment:
        for lag in range(1, n_lags + 1):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # 对数收益率及其平方的滞后
    log_ret_cols = [c for c in df.columns if c.endswith("_log_returns")]
    for col in log_ret_cols:
        sq_col = f"{col}_sq"
        df[sq_col] = df[col] ** 2
        for lag in range(1, n_lags + 1):
            df[f"{col}_lag{lag}"]    = df[col].shift(lag)
            df[f"{sq_col}_lag{lag}"] = df[sq_col].shift(lag)

    return df


# =============================================================================
# 7. 星期虚拟变量（周二–周五，周一为基准组）
# =============================================================================
def add_weekday_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daywk 列：1=Mon … 5=Fri（与 CSMAR 编码一致）
    如果 Daywk 列存在则直接用，否则从 Date 推算。
    创建 weekday_Tue / Wed / Thu / Fri 四个 0/1 虚拟变量。
    """
    if "Daywk" in df.columns:
        df["_dow"] = pd.to_numeric(df["Daywk"], errors="coerce")
    else:
        df["_dow"] = df["Date"].dt.dayofweek + 1   # Mon=1 … Fri=5

    day_names = {2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri"}
    for code, name in day_names.items():
        df[f"weekday_{name}"] = (df["_dow"] == code).astype(int)
    df = df.drop(columns=["_dow"], errors="ignore")
    return df


# =============================================================================
# 8. 主流程
# =============================================================================
def main():
    logging.info("========== 脚本 13：合并数据并计算收益率 ==========\n")

    # 0. 前置检查
    if not preflight_check():
        sys.exit(1)

    # 1. 加载指数数据
    index_df = load_index_data()

    # 2. 加载情感数据
    sentiment_df = load_sentiment_data()

    # 3. 合并
    merged = merge_datasets(index_df, sentiment_df)

    # 4. Winsorize + 标准化（在生成滞后前操作，确保滞后值也是 winsorize 后的结果）
    merged = winsorize_and_standardize(merged)

    # 5. 生成 Lag 1-5
    merged = create_lags(merged, n_lags=5)

    # 6. 星期虚拟变量
    merged = add_weekday_dummies(merged)

    # 7. 保存完整数据（含 2013-12-30 起的数据，供回归时按年筛选）
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    merged.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    logging.info(f"\n回归数据集已保存: {OUTPUT_FILE}")
    logging.info(f"  总行数: {len(merged)}，列数: {len(merged.columns)}")
    logging.info(f"  日期范围: {merged['Date'].min().date()} – {merged['Date'].max().date()}")

    # 8. 输出验证样本（前 10 行，筛选关键列）
    key_cols = (
        ["Date"]
        + [c for c in merged.columns if c.endswith("_log_returns")]
        + [c for c in merged.columns if c in ["PhotoPes", "TextPes"]]
        + [c for c in merged.columns if "_lag1" in c or "_lag2" in c or "_lag3" in c]
    )
    key_cols = [c for c in key_cols if c in merged.columns]

    # 2014 年初行（验证滞后项是否来自 2013 年底数据）
    sample = merged[merged["Date"] >= "2014-01-01"][key_cols].head(5)
    print("\n========== 样本预览（2014 年初前 5 行，验证滞后项无缺失） ==========")
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", "{:.6f}".format):
        print(sample.to_string(index=False))

    # 检查 2014-01-02（第一个交易日）的 lag 5 是否无 NaN
    first_row = merged[merged["Date"] >= "2014-01-02"].iloc[0]
    log_ret_lag5_cols = [c for c in merged.columns if "_log_returns_lag5" in c]
    missing_lag5 = first_row[log_ret_lag5_cols].isna().sum()
    if missing_lag5 == 0:
        logging.info("验证通过：2014-01-02 的所有 R_t Lag-5 均无缺失值。")
    else:
        logging.warning(f"注意：2014-01-02 的 {missing_lag5} 个 R_t Lag-5 仍有缺失。")

    logging.info("\n========== 完成 ==========")


if __name__ == "__main__":
    main()
