#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


INDEX_RETURN_COLS = [
    "SHCOMP_log_returns",
    "CSI300_log_returns",
    "SZCOMP_log_returns",
    "ChiNext_log_returns",
    "CSI500_log_returns",
]


def winsorize_series(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    lo = s.quantile(lower_q)
    hi = s.quantile(upper_q)
    return s.clip(lower=lo, upper=hi)


def standardize_series(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return s * 0.0
    return (s - s.mean()) / std


def add_weekday_dummies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).copy()
    out["weekday"] = out["Date"].dt.weekday  # Monday=0
    out["weekday_Tue"] = (out["weekday"] == 1).astype(int)
    out["weekday_Wed"] = (out["weekday"] == 2).astype(int)
    out["weekday_Thu"] = (out["weekday"] == 3).astype(int)
    out["weekday_Fri"] = (out["weekday"] == 4).astype(int)
    return out


def build_lagged_regression_frame(df: pd.DataFrame, sentiment_col: str, return_col: str) -> pd.DataFrame:
    x = df[["Date", sentiment_col, return_col, "weekday_Tue", "weekday_Wed", "weekday_Thu", "weekday_Fri"]].copy()
    x = x.dropna(subset=[sentiment_col, return_col]).copy()

    # Paper setting: 1% winsorize + standardize sentiment indicator.
    sent = winsorize_series(x[sentiment_col])
    x["sent_z"] = standardize_series(sent)

    # Return and squared return lags (1..5).
    x["ret"] = x[return_col]
    x["ret_sq"] = x["ret"] ** 2

    for k in range(1, 6):
        x[f"sent_lag{k}"] = x["sent_z"].shift(k)
        x[f"ret_lag{k}"] = x["ret"].shift(k)
        x[f"ret_sq_lag{k}"] = x["ret_sq"].shift(k)

    cols = ["ret"] + [f"sent_lag{k}" for k in range(1, 6)] + [f"ret_lag{k}" for k in range(1, 6)] + [
        f"ret_sq_lag{k}" for k in range(1, 6)
    ] + ["weekday_Tue", "weekday_Wed", "weekday_Thu", "weekday_Fri"]
    return x.dropna(subset=cols).copy()


def run_one_regression(df: pd.DataFrame, sentiment_col: str, return_col: str):
    reg_df = build_lagged_regression_frame(df, sentiment_col, return_col)
    y = reg_df["ret"]
    x_cols = [f"sent_lag{k}" for k in range(1, 6)] + [f"ret_lag{k}" for k in range(1, 6)] + [
        f"ret_sq_lag{k}" for k in range(1, 6)
    ] + ["weekday_Tue", "weekday_Wed", "weekday_Thu", "weekday_Fri"]
    X = sm.add_constant(reg_df[x_cols], has_constant="add")

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    # Joint tests consistent with paper tables.
    t3_t5 = model.t_test("sent_lag3 + sent_lag4 + sent_lag5 = 0")
    t4_t5 = model.t_test("sent_lag4 + sent_lag5 = 0")

    return model, reg_df, t3_t5, t4_t5


def main():
    parser = argparse.ArgumentParser(description="Validate paper-style L5 lag regressions and PhotoPes_{t-3} significance")
    parser.add_argument("--input", default="results/merged_market_sentiment_data.csv", help="Merged market-sentiment csv path")
    parser.add_argument("--output-dir", default="results/regression_validation", help="Output folder")
    parser.add_argument("--sentiment-col", default="PhotoPes", help="Sentiment column used for lag regressions")
    parser.add_argument("--start-date", default="2014-01-01")
    parser.add_argument("--end-date", default="2026-12-31")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df = add_weekday_dummies(df)
    df = df[(df["Date"] >= pd.to_datetime(args.start_date)) & (df["Date"] <= pd.to_datetime(args.end_date))].copy()

    if args.sentiment_col not in df.columns:
        raise ValueError(f"Sentiment column not found: {args.sentiment_col}")

    available_returns = [c for c in INDEX_RETURN_COLS if c in df.columns]
    if not available_returns:
        raise ValueError("No target return columns found in merged dataset")

    rows = []
    for ret_col in available_returns:
        model, reg_df, t3_t5, t4_t5 = run_one_regression(df, args.sentiment_col, ret_col)
        params = model.params
        pvals = model.pvalues

        row = {
            "return_col": ret_col,
            "n_obs": int(model.nobs),
            "r2": float(model.rsquared),
            "adj_r2": float(model.rsquared_adj),
            "coef_t1": float(params.get("sent_lag1", np.nan)),
            "p_t1": float(pvals.get("sent_lag1", np.nan)),
            "coef_t2": float(params.get("sent_lag2", np.nan)),
            "p_t2": float(pvals.get("sent_lag2", np.nan)),
            "coef_t3": float(params.get("sent_lag3", np.nan)),
            "p_t3": float(pvals.get("sent_lag3", np.nan)),
            "coef_t4": float(params.get("sent_lag4", np.nan)),
            "p_t4": float(pvals.get("sent_lag4", np.nan)),
            "coef_t5": float(params.get("sent_lag5", np.nan)),
            "p_t5": float(pvals.get("sent_lag5", np.nan)),
            "sum_t3_t5": float(params.get("sent_lag3", 0.0) + params.get("sent_lag4", 0.0) + params.get("sent_lag5", 0.0)),
            "sum_t3_t5_p": float(np.asarray(t3_t5.pvalue).item()),
            "sum_t4_t5": float(params.get("sent_lag4", 0.0) + params.get("sent_lag5", 0.0)),
            "sum_t4_t5_p": float(np.asarray(t4_t5.pvalue).item()),
            "photo_t3_negative_and_sig_10pct": bool((params.get("sent_lag3", np.nan) < 0) and (pvals.get("sent_lag3", 1.0) < 0.10)),
            "photo_t3_negative_and_sig_5pct": bool((params.get("sent_lag3", np.nan) < 0) and (pvals.get("sent_lag3", 1.0) < 0.05)),
            "photo_t3_negative_and_sig_1pct": bool((params.get("sent_lag3", np.nan) < 0) and (pvals.get("sent_lag3", 1.0) < 0.01)),
        }
        rows.append(row)

        model_txt = out_dir / f"ols_{args.sentiment_col}_{ret_col}.txt"
        model_txt.write_text(model.summary().as_text() + "\n", encoding="utf-8")

    result_df = pd.DataFrame(rows).sort_values("return_col")
    csv_out = out_dir / f"lag_regression_summary_{args.sentiment_col}.csv"
    result_df.to_csv(csv_out, index=False, encoding="utf-8")

    key_out = out_dir / f"photo_t3_significance_check_{args.sentiment_col}.txt"
    lines = []
    lines.append(f"sentiment_col={args.sentiment_col}")
    lines.append(f"date_range={args.start_date}..{args.end_date}")
    lines.append("PhotoPes_{t-3} check by index:")
    for _, r in result_df.iterrows():
        lines.append(
            f"- {r['return_col']}: coef_t3={r['coef_t3']:.6g}, p_t3={r['p_t3']:.6g}, "
            f"neg&sig(10/5/1)=({r['photo_t3_negative_and_sig_10pct']}/{r['photo_t3_negative_and_sig_5pct']}/{r['photo_t3_negative_and_sig_1pct']})"
        )
    key_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"saved_summary={csv_out}")
    print(f"saved_key_check={key_out}")
    print(result_df[["return_col", "coef_t3", "p_t3", "photo_t3_negative_and_sig_5pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
