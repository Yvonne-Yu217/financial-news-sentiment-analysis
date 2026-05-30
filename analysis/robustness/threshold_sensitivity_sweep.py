"""
threshold_sensitivity_sweep.py
==============================

Reproduces the supplementary file `threshold_sensitivity.csv` promised in
§4.1 of the paper:

    "Robustness across thresholds in {50, 60, 70, 75, 80, 90} is reported
     in the supplementary file threshold_sensitivity.csv ..."

For each percentile q in {0.50, 0.60, 0.70, 0.75, 0.80, 0.90}:
  - Define high-volatility days as vol20 > q-th percentile of vol20, where
    vol20 is the 20-day rolling std of SHCOMP log returns (same definition
    as `paper_table5_conditional.py`).
  - Estimate the same conditional regression as Table 5 on the high-vol
    subsample for each of 5 indices (CSI300, SHCOMP, SZCOMP, ChiNext,
    CSI500) with NW(5) HAC standard errors.
  - Report PhotoPes_{t-i} and TextPes_{t-i} coefficient, t-statistic,
    p-value, and N for lags i in 1..5.

Output: analysis/robustness/threshold_sensitivity.csv (long format)

This script is read-only with respect to the merged data file. Run from
the repository root.
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = "results/merged_market_sentiment_data_old.csv"
OUT = "analysis/robustness/threshold_sensitivity.csv"

INDICES = ["CSI300", "SHCOMP", "SZCOMP", "ChiNext", "CSI500"]
PRED_LAGS = [f"{s}_std_lag{i}" for s in ["PhotoPes", "TextPes"] for i in range(1, 6)]
THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90]


def load():
    df = pd.read_csv(DATA, parse_dates=["Date"])
    df = df[df["Date"] >= "2014-01-01"].copy()
    df["vol20"] = df["SHCOMP_log_returns"].rolling(20).std()
    return df


def run_subsample(sub, df_full):
    """Run NW(5) regression on each index, return dict of (results, N)."""
    wk = [c for c in df_full.columns if "weekday" in c]
    out = {}
    for idx in INDICES:
        y = f"{idx}_log_returns"
        lag_r = [f"{idx}_log_returns_lag{i}" for i in range(1, 6)]
        lag_r2 = [f"{idx}_log_returns_sq_lag{i}" for i in range(1, 6)]
        xcols = [c for c in PRED_LAGS + lag_r + lag_r2 + wk if c in sub.columns]
        data = sub[[y] + xcols].dropna()
        if len(data) < 30:
            out[idx] = (None, len(data))
            continue
        X = sm.add_constant(data[xcols])
        res = sm.OLS(data[y], X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        out[idx] = (res, len(data))
    return out


def main():
    df = load()
    rows = []

    for q in THRESHOLDS:
        thr_val = df["vol20"].quantile(q)
        sub = df[df["vol20"] > thr_val].copy()
        results = run_subsample(sub, df)

        for idx in INDICES:
            res, n = results[idx]
            if res is None:
                continue
            for pred in PRED_LAGS:
                rows.append(
                    {
                        "threshold_pctile": int(q * 100),
                        "threshold_vol20_value": round(thr_val, 6),
                        "index": idx,
                        "predictor": pred,
                        "coef": round(res.params[pred], 6),
                        "tstat": round(res.tvalues[pred], 3),
                        "pval": round(res.pvalues[pred], 4),
                        "N_highvol": n,
                    }
                )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT, index=False)

    print("=" * 70)
    print("Threshold-sensitivity sweep complete")
    print("=" * 70)
    print(f"Output written: {OUT}")
    print(f"Rows: {len(out_df)}  (6 thresholds x 5 indices x 10 predictor lags)")
    print()

    # Quick summary: show PhotoPes_lag3 ChiNext result at each threshold
    headline = out_df[
        (out_df["index"] == "ChiNext") & (out_df["predictor"] == "PhotoPes_std_lag3")
    ][["threshold_pctile", "threshold_vol20_value", "coef", "tstat", "pval", "N_highvol"]]
    print("Headline check: PhotoPes_{t-3} on ChiNext, across thresholds:")
    print(headline.to_string(index=False))


if __name__ == "__main__":
    main()
