#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regression_tables.py — 复现论文 §4.1–§4.4 回归结果

生成表格：
  Table 2  — PhotoPes 对市场收益率的影响 (§4.1)
  Table 3  — TextPes  对市场收益率的影响 (§4.2)
  Table 4  — 联合回归 + 交互项 (§4.3)
  Table 5  — PhotoPes/TextPes 条件分析 / 极端期 vs 非极端期 (§4.4)

回归方程（以 PhotoPes 单变量为例）：
  R_t = β1·L5(PhotoPes) + β2·L5(R_t) + β3·L5(R²_t) + β4·X_t + ε_t
  其中 X_t = {常数, 周二, 周三, 周四, 周五}

标准误：Newey-West HAC（与论文一致）
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats.mstats import winsorize

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_FILE   = os.path.join(BASE_DIR, "results", "merged_market_sentiment_data_old.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "results", "regression_output_oldmodel")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIVE_INDICES = ["SHCOMP", "CSI300", "SZCOMP", "ChiNext", "CSI500"]
COL_HEADERS  = ["SHCOMP", "CSI 300", "SZCOMP", "ChiNext", "CSI 500"]

N_LAGS      = 5
START_YEAR  = 2014
END_YEAR    = 2026

# ─────────────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
    df = df[(df["Date"].dt.year >= START_YEAR) &
            (df["Date"].dt.year <= END_YEAR)].copy()
    df = df.sort_values("Date").reset_index(drop=True)
    df.set_index("Date", inplace=True)
    print(f"样本加载：{len(df)} 行, {df.index.min().date()} – {df.index.max().date()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 工具：构建回归矩阵
# ─────────────────────────────────────────────────────────────────────────────
def build_X(df, sentiment_col, index_name, extra_cols=None):
    """
    构建单个情感指标的回归矩阵 X：
      - 情感滞后 t-1 … t-5（标准化后的值）
      - R_t 滞后 t-1 … t-5
      - R²_t 滞后 t-1 … t-5
      - 星期虚拟变量
      - 常数项
    返回 (X, y, valid_mask)
    """
    std_col = f"{sentiment_col}_std"
    ret_col = f"{index_name}_log_returns"
    week_cols = [c for c in df.columns if c.startswith("weekday_")]

    cols = {}
    # 情感滞后
    for lag in range(1, N_LAGS + 1):
        key = f"{std_col}_lag{lag}"
        if key not in df.columns:
            # 计算备用
            df[key] = df[std_col].shift(lag)
        cols[f"sent_lag{lag}"] = df[key]

    # 收益率滞后
    for lag in range(1, N_LAGS + 1):
        key = f"{ret_col}_lag{lag}"
        if key not in df.columns:
            df[key] = df[ret_col].shift(lag)
        cols[f"ret_lag{lag}"] = df[key]

    # 收益率平方滞后
    sq_col = f"{ret_col}_sq"
    if sq_col not in df.columns:
        df[sq_col] = df[ret_col] ** 2
    for lag in range(1, N_LAGS + 1):
        key = f"{sq_col}_lag{lag}"
        if key not in df.columns:
            df[key] = df[sq_col].shift(lag)
        cols[f"ret_sq_lag{lag}"] = df[key]

    # 星期虚拟变量
    for wc in week_cols:
        cols[wc] = df[wc]

    # 额外列（用于联合回归）
    if extra_cols:
        for k, v in extra_cols.items():
            cols[k] = v

    X_df = pd.DataFrame(cols, index=df.index)
    X_df = sm.add_constant(X_df)
    y = df[ret_col].copy()

    valid = X_df.notna().all(axis=1) & y.notna()
    return X_df[valid], y[valid]


def run_ols(X, y, n_lags_hac=5):
    """Newey-West HAC 标准误的 OLS 回归"""
    model = sm.OLS(y, X).fit(cov_type="HAC",
                              cov_kwds={"maxlags": n_lags_hac,
                                        "use_correction": True})
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 提取结果：返回 (coef, tstat, pval) for each lag
# ─────────────────────────────────────────────────────────────────────────────
def extract_lags(model, lag_prefix, n_lags=N_LAGS):
    coefs, tstats, pvals = [], [], []
    for lag in range(1, n_lags + 1):
        key = f"{lag_prefix}{lag}"
        if key in model.params:
            coefs.append(model.params[key])
            tstats.append(model.tvalues[key])
            pvals.append(model.pvalues[key])
        else:
            coefs.append(np.nan)
            tstats.append(np.nan)
            pvals.append(np.nan)
    return coefs, tstats, pvals


def sig_stars(pval):
    if pval <= 0.01: return "***"
    if pval <= 0.05: return "**"
    if pval <= 0.10: return "*"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# 表格打印（Paper 风格）
# ─────────────────────────────────────────────────────────────────────────────
def print_table(title, rows, col_labels, sep=True):
    col_w = 16
    hdr = f"{'Indicators':<22}" + "".join(f"{c:>{col_w}}" for c in col_labels)
    print(f"\n{'='*len(hdr)}")
    print(title)
    print(f"{'='*len(hdr)}")
    print(hdr)
    print("-"*len(hdr))
    for row_label, values in rows:
        if row_label == "---":
            print("-"*len(hdr))
            continue
        val_str = "".join(f"{v:>{col_w}}" if isinstance(v, str) else f"{v:>{col_w}.4f}"
                          for v in values)
        print(f"{row_label:<22}{val_str}")
    print("="*len(hdr))


def save_table_csv(title_tag, rows, col_labels):
    path = os.path.join(OUTPUT_DIR, f"{title_tag}.csv")
    records = []
    for row_label, values in rows:
        if row_label == "---": continue
        rec = {"Indicator": row_label}
        for col, val in zip(col_labels, values):
            rec[col] = val
        records.append(rec)
    pd.DataFrame(records).to_csv(path, index=False)
    print(f"  → 已保存: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# §4.1 / §4.2 单情感指标回归（Table 2 / Table 3）
# ─────────────────────────────────────────────────────────────────────────────
def run_single_sentiment(df, sentiment_col, table_num, table_title):
    print(f"\n{'#'*60}")
    print(f"  {table_title}")
    print(f"{'#'*60}")

    rows = []
    model_stats = {"R2": [], "Adj_R2": [], "N": []}
    lag_results = {f"lag{i}": {"coefs":[], "tstats":[], "pvals":[]} for i in range(1, N_LAGS+1)}

    for idx_name in FIVE_INDICES:
        ret_col = f"{idx_name}_log_returns"
        if ret_col not in df.columns:
            print(f"  跳过 {idx_name}（列不存在）")
            for i in range(1, N_LAGS+1):
                lag_results[f"lag{i}"]["coefs"].append(np.nan)
                lag_results[f"lag{i}"]["tstats"].append(np.nan)
                lag_results[f"lag{i}"]["pvals"].append(np.nan)
            model_stats["R2"].append(np.nan)
            model_stats["Adj_R2"].append(np.nan)
            model_stats["N"].append(np.nan)
            continue

        X, y = build_X(df, sentiment_col, idx_name)
        model = run_ols(X, y)
        coefs, tstats, pvals = extract_lags(model, "sent_lag")
        for i in range(N_LAGS):
            lag_results[f"lag{i+1}"]["coefs"].append(coefs[i])
            lag_results[f"lag{i+1}"]["tstats"].append(tstats[i])
            lag_results[f"lag{i+1}"]["pvals"].append(pvals[i])
        model_stats["R2"].append(model.rsquared)
        model_stats["Adj_R2"].append(model.rsquared_adj)
        model_stats["N"].append(int(model.nobs))

    # 格式化输出行
    for lag_num in range(1, N_LAGS+1):
        lk = f"lag{lag_num}"
        coef_row, tstat_row = [], []
        for c, t, p in zip(lag_results[lk]["coefs"],
                           lag_results[lk]["tstats"],
                           lag_results[lk]["pvals"]):
            if np.isnan(c):
                coef_row.append("—"); tstat_row.append("—")
            else:
                stars = sig_stars(p)
                coef_row.append(f"{c:.4f}{stars}")
                tstat_row.append(f"({t:.2f})")
        rows.append((f"{sentiment_col}_t-{lag_num}", coef_row))
        rows.append(("", tstat_row))

    # Sum(t-3 to t-5)
    rows.append(("---", []))
    sum_coef, sum_tstat = [], []
    for idx_i, idx_name in enumerate(FIVE_INDICES):
        ret_col = f"{idx_name}_log_returns"
        if ret_col not in df.columns:
            sum_coef.append("—"); sum_tstat.append("—"); continue
        X, y = build_X(df, sentiment_col, idx_name)
        model = run_ols(X, y)
        # Test H0: β3+β4+β5=0
        R_vec = np.zeros(len(model.params))
        for lag in [3, 4, 5]:
            k = f"sent_lag{lag}"
            if k in model.params.index:
                R_vec[model.params.index.get_loc(k)] = 1
        try:
            f_test = model.f_test(R_vec.reshape(1, -1))
            sum_c = sum(model.params.get(f"sent_lag{lag}", 0) for lag in [3,4,5])
            sum_t = float(np.sqrt(f_test.fvalue)) * np.sign(sum_c)
            sum_p = float(f_test.pvalue)
            sum_coef.append(f"{sum_c:.4f}{sig_stars(sum_p)}")
            sum_tstat.append(f"({sum_t:.2f})")
        except Exception:
            sum_coef.append("—"); sum_tstat.append("—")

    rows.append(("Sum(t-3 to t-5)", sum_coef))
    rows.append(("", sum_tstat))

    # Sum(t-4 to t-5)
    sum45_coef, sum45_tstat = [], []
    for idx_i, idx_name in enumerate(FIVE_INDICES):
        ret_col = f"{idx_name}_log_returns"
        if ret_col not in df.columns:
            sum45_coef.append("—"); sum45_tstat.append("—"); continue
        X, y = build_X(df, sentiment_col, idx_name)
        model = run_ols(X, y)
        R_vec = np.zeros(len(model.params))
        for lag in [4, 5]:
            k = f"sent_lag{lag}"
            if k in model.params.index:
                R_vec[model.params.index.get_loc(k)] = 1
        try:
            f_test = model.f_test(R_vec.reshape(1, -1))
            sum_c = sum(model.params.get(f"sent_lag{lag}", 0) for lag in [4,5])
            sum_t = float(np.sqrt(f_test.fvalue)) * np.sign(sum_c)
            sum_p = float(f_test.pvalue)
            sum45_coef.append(f"{sum_c:.4f}{sig_stars(sum_p)}")
            sum45_tstat.append(f"({sum_t:.2f})")
        except Exception:
            sum45_coef.append("—"); sum45_tstat.append("—")

    rows.append(("Sum(t-4 to t-5)", sum45_coef))
    rows.append(("", sum45_tstat))

    rows.append(("---", []))
    rows.append(("R²",     [f"{v:.4f}" if not np.isnan(v) else "—" for v in model_stats["R2"]]))
    rows.append(("Adj R²", [f"{v:.4f}" if not np.isnan(v) else "—" for v in model_stats["Adj_R2"]]))
    rows.append(("N",      [str(int(v)) if not np.isnan(v) else "—" for v in model_stats["N"]]))

    print_table(table_title, rows, COL_HEADERS)
    save_table_csv(f"table{table_num}_{sentiment_col}", rows, COL_HEADERS)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# §4.3 联合回归 + 交互项（Table 4）
# ─────────────────────────────────────────────────────────────────────────────
def run_joint_interaction(df):
    title = "Table 4 — PhotoPes, TextPes and Their Interaction Effect (2014–2026)"
    print(f"\n{'#'*60}\n  {title}\n{'#'*60}")

    rows_photo, rows_text, rows_inter = [], [], []
    model_stats = {"R2": [], "Adj_R2": [], "N": []}

    for idx_name in FIVE_INDICES:
        ret_col = f"{idx_name}_log_returns"
        if ret_col not in df.columns:
            for lst in [rows_photo, rows_text, rows_inter]:
                lst.append(np.nan)
            model_stats["R2"].append(np.nan)
            continue

        # 构建联合 X：包含 PhotoPes 和 TextPes 各 5 阶滞后 + 交互项
        ps_col = "PhotoPes_std"
        ts_col = "TextPes_std"

        extra = {}
        for lag in range(1, N_LAGS+1):
            k = f"{ts_col}_lag{lag}"
            if k not in df.columns:
                df[k] = df[ts_col].shift(lag)
            extra[f"text_lag{lag}"] = df[k]

        # 交互项 t-3
        for sub in ["_std_lag3", "_lag3"]:
            pk = f"PhotoPes{sub}"
            tk = f"TextPes{sub}"
            if pk in df.columns and tk in df.columns:
                df["inter_t3"] = df[pk] * df[tk]
                break
        if "inter_t3" not in df.columns:
            df["inter_t3"] = np.nan
        extra["inter_t3"] = df["inter_t3"]

        X, y = build_X(df, "PhotoPes", idx_name, extra_cols=extra)
        model = run_ols(X, y)

        coefs_p, tstats_p, pvals_p = extract_lags(model, "sent_lag")
        coefs_t, tstats_t, pvals_t = extract_lags(model, "text_lag")

        rows_photo.append((coefs_p, tstats_p, pvals_p))
        rows_text.append((coefs_t, tstats_t, pvals_t))

        # Interaction
        ic = model.params.get("inter_t3", np.nan)
        it = model.tvalues.get("inter_t3", np.nan)
        ip = model.pvalues.get("inter_t3", np.nan)
        rows_inter.append((ic, it, ip))

        model_stats["R2"].append(model.rsquared)
        model_stats["Adj_R2"].append(model.rsquared_adj)
        model_stats["N"].append(int(model.nobs))

    # Format table
    all_rows = []
    all_rows.append(("Panel A: PhotoPes", [""] * 5))
    for lag_num in range(1, N_LAGS + 1):
        cr, tr = [], []
        for i, idx_name in enumerate(FIVE_INDICES):
            if i >= len(rows_photo) or isinstance(rows_photo[i], float):
                cr.append("—"); tr.append("—"); continue
            c = rows_photo[i][0][lag_num-1]
            t = rows_photo[i][1][lag_num-1]
            p = rows_photo[i][2][lag_num-1]
            cr.append(f"{c:.4f}{sig_stars(p)}" if not np.isnan(c) else "—")
            tr.append(f"({t:.2f})" if not np.isnan(t) else "—")
        all_rows.append((f"PhotoPes_t-{lag_num}", cr))
        all_rows.append(("", tr))

    all_rows.append(("---", []))
    all_rows.append(("Panel B: TextPes", [""] * 5))
    for lag_num in range(1, N_LAGS + 1):
        cr, tr = [], []
        for i, idx_name in enumerate(FIVE_INDICES):
            if i >= len(rows_text) or isinstance(rows_text[i], float):
                cr.append("—"); tr.append("—"); continue
            c = rows_text[i][0][lag_num-1]
            t = rows_text[i][1][lag_num-1]
            p = rows_text[i][2][lag_num-1]
            cr.append(f"{c:.4f}{sig_stars(p)}" if not np.isnan(c) else "—")
            tr.append(f"({t:.2f})" if not np.isnan(t) else "—")
        all_rows.append((f"TextPes_t-{lag_num}", cr))
        all_rows.append(("", tr))

    all_rows.append(("---", []))
    all_rows.append(("Panel C: Interaction", [""] * 5))
    ic_row, it_row = [], []
    for i in range(len(FIVE_INDICES)):
        if i >= len(rows_inter) or isinstance(rows_inter[i][0], float) and np.isnan(rows_inter[i][0]):
            ic_row.append("—"); it_row.append("—"); continue
        ic, it, ip = rows_inter[i]
        ic_row.append(f"{ic:.4f}{sig_stars(ip)}" if not np.isnan(ic) else "—")
        it_row.append(f"({it:.2f})" if not np.isnan(it) else "—")
    all_rows.append(("(Photo×Text)_t-3", ic_row))
    all_rows.append(("", it_row))
    all_rows.append(("---", []))
    all_rows.append(("R²",     [f"{v:.4f}" if not np.isnan(v) else "—" for v in model_stats["R2"]]))
    all_rows.append(("Adj R²", [f"{v:.4f}" if not np.isnan(v) else "—" for v in model_stats["Adj_R2"]]))
    all_rows.append(("N",      [str(int(v)) if not np.isnan(v) else "—" for v in model_stats["N"]]))

    print_table(title, all_rows, COL_HEADERS)
    save_table_csv("table4_joint", all_rows, COL_HEADERS)


# ─────────────────────────────────────────────────────────────────────────────
# §4.4 条件分析（Table 5）
# ─────────────────────────────────────────────────────────────────────────────
def _run_panel_regression(sub_df, df_full):
    """Run joint regression on a subsample, return row_data dict per index."""
    row_data = {}
    for idx_name in FIVE_INDICES:
        ret_col = f"{idx_name}_log_returns"
        if ret_col not in sub_df.columns:
            row_data[idx_name] = {}
            continue
        extra = {}
        ts_col = "TextPes_std"
        sub = sub_df.copy()
        for lag in range(1, N_LAGS+1):
            k = f"{ts_col}_lag{lag}"
            if k not in sub.columns:
                sub[k] = sub[ts_col].shift(lag)
            extra[f"text_lag{lag}"] = sub[k]
        for sub_suf in ["_std_lag3", "_lag3"]:
            pk = f"PhotoPes{sub_suf}"
            tk = f"TextPes{sub_suf}"
            if pk in sub.columns and tk in sub.columns:
                sub["inter_t3"] = sub[pk] * sub[tk]
                break
        if "inter_t3" not in sub.columns:
            sub["inter_t3"] = np.nan
        extra["inter_t3"] = sub["inter_t3"]
        try:
            X, y = build_X(sub, "PhotoPes", idx_name, extra_cols=extra)
            if len(y) < 30:
                row_data[idx_name] = {}
                continue
            model = run_ols(X, y)
            coefs_p, tstats_p, pvals_p = extract_lags(model, "sent_lag")
            coefs_t, tstats_t, pvals_t = extract_lags(model, "text_lag")
            ic = model.params.get("inter_t3", np.nan)
            it = model.tvalues.get("inter_t3", np.nan)
            ip = model.pvalues.get("inter_t3", np.nan)
            row_data[idx_name] = {
                "photo": (coefs_p, tstats_p, pvals_p),
                "text":  (coefs_t, tstats_t, pvals_t),
                "inter": (ic, it, ip),
                "r2": model.rsquared, "n": int(model.nobs)
            }
        except Exception:
            row_data[idx_name] = {}
    return row_data


def _panel_to_csv_rows(row_data, panel_label):
    """Convert row_data to list of (row_label, values) for CSV output."""
    rows = []
    rows.append((panel_label, [""] * 5))
    for src, src_lbl in [("photo", "PhotoPes"), ("text", "TextPes")]:
        for lag_num in range(1, N_LAGS + 1):
            cr, tr = [], []
            for idx_name in FIVE_INDICES:
                d = row_data.get(idx_name, {})
                if not d or src not in d:
                    cr.append("—"); tr.append("—"); continue
                c = d[src][0][lag_num - 1]
                t = d[src][1][lag_num - 1]
                p = d[src][2][lag_num - 1]
                cr.append(f"{c:.4f}{sig_stars(p)}" if not np.isnan(c) else "—")
                tr.append(f"({t:.2f})" if not np.isnan(t) else "—")
            rows.append((f"{src_lbl}_t-{lag_num}", cr))
            rows.append(("", tr))
    # Interaction
    ic_r, it_r = [], []
    for idx_name in FIVE_INDICES:
        d = row_data.get(idx_name, {})
        if not d or "inter" not in d:
            ic_r.append("—"); it_r.append("—"); continue
        ic, it, ip = d["inter"]
        ic_r.append(f"{ic:.4f}{sig_stars(ip)}" if not np.isnan(ic) else "—")
        it_r.append(f"({it:.2f})" if not np.isnan(it) else "—")
    rows.append(("(Photo×Text)_t-3", ic_r))
    rows.append(("", it_r))
    rows.append(("---", []))
    r2_r = [f"{row_data[i]['r2']:.4f}" if row_data.get(i) and "r2" in row_data[i] else "—"
            for i in FIVE_INDICES]
    n_r = [str(row_data[i]["n"]) if row_data.get(i) and "n" in row_data[i] else "—"
           for i in FIVE_INDICES]
    rows.append(("R²", r2_r))
    rows.append(("N", n_r))
    return rows


def run_conditional_analysis(df):
    title = "Table 5 — Conditional Impact of PhotoPes and TextPes (2014–2026)"
    print(f"\n{'#'*60}\n  {title}\n{'#'*60}")

    # 定义极端期：PhotoPes 在全样本 10% / 90% 分位数之外
    threshold_lo = df["PhotoPes"].quantile(0.10)
    threshold_hi = df["PhotoPes"].quantile(0.90)
    extreme_mask = (df["PhotoPes"] <= threshold_lo) | (df["PhotoPes"] >= threshold_hi)
    df_ext  = df[extreme_mask].copy()
    df_norm = df[~extreme_mask].copy()
    print(f"  极端期行数: {len(df_ext)}  |  非极端期行数: {len(df_norm)}")
    print(f"  极端期阈值: [{threshold_lo:.4f}, {threshold_hi:.4f}]")

    all_csv_rows = []
    for panel_label, sub_df in [("Panel A: Extreme PhotoPes (E=1)", df_ext),
                                 ("Panel B: Non-Extreme PhotoPes (E=0)", df_norm)]:
        print(f"\n  {panel_label}")
        row_data = _run_panel_regression(sub_df, df)

        # Print condensed
        print(f"\n  {'Indicator':<26}" + "".join(f"{c:>14}" for c in COL_HEADERS))
        print("  " + "-"*96)
        for src, src_lbl in [("photo","PhotoPes"), ("text","TextPes")]:
            for lag_num in range(1, N_LAGS+1):
                cr, tr = [], []
                for idx_name in FIVE_INDICES:
                    d = row_data.get(idx_name, {})
                    if not d or src not in d:
                        cr.append("—"); tr.append("—"); continue
                    c = d[src][0][lag_num-1]
                    t = d[src][1][lag_num-1]
                    p = d[src][2][lag_num-1]
                    cr.append(f"{c:.4f}{sig_stars(p)}" if not np.isnan(c) else "—")
                    tr.append(f"({t:.2f})" if not np.isnan(t) else "—")
                print(f"  {src_lbl}_t-{lag_num:<19}" + "".join(f"{v:>14}" for v in cr))
                print(f"  {'':26}" + "".join(f"{v:>14}" for v in tr))

        # Interaction row
        ic_r, it_r = [], []
        for idx_name in FIVE_INDICES:
            d = row_data.get(idx_name, {})
            if not d or "inter" not in d:
                ic_r.append("—"); it_r.append("—"); continue
            ic, it, ip = d["inter"]
            ic_r.append(f"{ic:.4f}{sig_stars(ip)}" if not np.isnan(ic) else "—")
            it_r.append(f"({it:.2f})" if not np.isnan(it) else "—")
        print(f"  {'(Photo×Text)_t-3':<26}" + "".join(f"{v:>14}" for v in ic_r))
        print(f"  {'':26}" + "".join(f"{v:>14}" for v in it_r))

        # R² and N
        r2_r = [f"{row_data[i]['r2']:.4f}" if row_data.get(i) and "r2" in row_data[i] else "—"
                for i in FIVE_INDICES]
        n_r  = [str(row_data[i]["n"]) if row_data.get(i) and "n" in row_data[i] else "—"
                for i in FIVE_INDICES]
        print("  " + "-"*96)
        print(f"  {'R²':<26}" + "".join(f"{v:>14}" for v in r2_r))
        print(f"  {'N':<26}" + "".join(f"{v:>14}" for v in n_r))

        # Collect CSV rows
        all_csv_rows.extend(_panel_to_csv_rows(row_data, panel_label))
        all_csv_rows.append(("---", []))

    save_table_csv("table5_conditional", all_csv_rows, COL_HEADERS)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 描述性统计 (Table 1)
# ─────────────────────────────────────────────────────────────────────────────
def run_descriptive_stats(df):
    print(f"\n{'#'*60}")
    print("  Table 1 — Descriptive Statistics (2014–2026)")
    print(f"{'#'*60}")
    sample = df.reset_index() if df.index.name == "Date" else df.copy()
    sample["Date"] = pd.to_datetime(sample["Date"])
    sample = sample[(sample["Date"].dt.year >= 2014) & (sample["Date"].dt.year <= 2026)].copy()

    print("\nPanel A: Sentiment Indicators (raw, before winsorize)")
    sent_raw_cols = []
    for c in ["PhotoPes", "TextPes"]:
        # Use the winsorized version from the file (matches paper's reported stats)
        if c in sample.columns:
            sent_raw_cols.append(c)
    if sent_raw_cols:
        for c in sent_raw_cols:
            s = sample[c].dropna()
            print(f"  {c:<20} N={len(s)}  Mean={s.mean():.4f}  Median={s.median():.4f}"
                  f"  P25={s.quantile(.25):.4f}  P75={s.quantile(.75):.4f}  Std={s.std():.4f}")

    print("\nPanel B: Market Index Log Returns")
    for idx_name in FIVE_INDICES:
        col = f"{idx_name}_log_returns"
        if col in sample.columns:
            s = sample[col].dropna()
            print(f"  {idx_name:<10} N={len(s)}  Mean={s.mean():.4f}  Median={s.median():.4f}"
                  f"  P25={s.quantile(.25):.4f}  P75={s.quantile(.75):.4f}  Std={s.std():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print(" 论文回归复现：Table 1–5")
    print("="*60)

    df = load_data()

    # Table 1: 描述性统计
    run_descriptive_stats(df)

    # Table 2: PhotoPes 单变量回归
    run_single_sentiment(df, "PhotoPes", 2,
                         "Table 2 — Impact of PhotoPes on Market Returns (2014–2026)")

    # Table 3: TextPes 单变量回归
    run_single_sentiment(df, "TextPes", 3,
                         "Table 3 — Impact of TextPes on Market Returns (2014–2026)")

    # Table 4: 联合回归 + 交互项
    run_joint_interaction(df)

    # Table 5: 条件分析
    run_conditional_analysis(df)

    print(f"\n所有结果已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
