"""
placebo_image_count.py
=======================
Phase-1 Placebo (per professor's memo §4.8): replace PhotoPes with daily
image count (a sentiment-free signal). If the placebo signal also produces
a ChiNext reversal effect, the original effect is "any high-variance daily
signal" rather than visual sentiment specifically.

Placebo signals tested:
  PhotoPes_count_std_t-3 = std-normalized log(daily image count)
  PhotoPes_negCount_std_t-3 = std-normalized daily count of negative-classified images

Run the same high-vol joint regression with the placebo signal in place
of PhotoPes_std. Expected: placebo NOT significant on ChiNext.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = "results/merged_market_sentiment_data_old.csv"
PHOTOPES_CSV = "results/weighted_photopes_old.csv"
INDICES = ["CSI300", "SHCOMP", "SZCOMP", "ChiNext", "CSI500"]
N_LAGS = 5
NW_BW = 5

np.random.seed(42)


def load_with_placebo():
    df = pd.read_csv(DATA, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    p = pd.read_csv(PHOTOPES_CSV, parse_dates=["news_date"])
    p = p[["news_date", "total_images", "negative_count"]].rename(columns={"news_date": "Date"})
    df = df.merge(p, on="Date", how="left")
    # Placebo 1: log(daily image count) — pure activity signal
    df["log_n_images"] = np.log(df["total_images"].clip(lower=1))
    df["log_n_images_std"] = (df["log_n_images"] - df["log_n_images"].mean()) / df["log_n_images"].std()
    # Placebo 2: log(daily NEGATIVE-image count) — signal that contains both sentiment AND volume
    df["log_neg_count"] = np.log(df["negative_count"].clip(lower=1))
    df["log_neg_count_std"] = (df["log_neg_count"] - df["log_neg_count"].mean()) / df["log_neg_count"].std()
    # Lags
    for lag in range(1, N_LAGS + 1):
        df[f"placebo_imgcount_lag{lag}"] = df["log_n_images_std"].shift(lag)
        df[f"placebo_negcount_lag{lag}"] = df["log_neg_count_std"].shift(lag)
    return df


def define_high_vol(df, percentile=0.75, window=20):
    rv = df["SHCOMP_log_returns"].rolling(window).std() * np.sqrt(252)
    return rv >= rv.quantile(percentile)


def build_design(df, ret_col, hi_mask, photopes_prefix):
    """Use 'photopes_prefix' instead of PhotoPes_std."""
    cols = {}
    for lag in range(1, N_LAGS + 1):
        cols[f"PhotoPes_lag{lag}"] = df[f"{photopes_prefix}_lag{lag}"]
        cols[f"TextPes_lag{lag}"]  = df[f"TextPes_std_lag{lag}"]
        cols[f"R_lag{lag}"]   = df[f"{ret_col.split('_log_')[0]}_log_returns_lag{lag}"]
        cols[f"R2_lag{lag}"]  = df[f"{ret_col.split('_log_')[0]}_log_returns_sq_lag{lag}"]
    cols["weekday_Tue"] = df["weekday_Tue"]
    cols["weekday_Wed"] = df["weekday_Wed"]
    cols["weekday_Thu"] = df["weekday_Thu"]
    cols["weekday_Fri"] = df["weekday_Fri"]
    X = pd.DataFrame(cols, index=df.index)
    X = sm.add_constant(X)
    y = df[ret_col]
    valid = X.notna().all(axis=1) & y.notna() & hi_mask.reindex(X.index).fillna(False)
    return X[valid], y[valid]


def fit(X, y):
    return sm.OLS(y, X).fit(cov_type="HAC",
                            cov_kwds={"maxlags": NW_BW, "use_correction": True})


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def main():
    df = load_with_placebo()
    print("=" * 90)
    print("Placebo test: replace PhotoPes_std with sentiment-free signals in joint regression")
    print("=" * 90)
    print()
    print("Signal A: log(total_images)_std — pure NEWS-VOLUME signal")
    print("Signal B: log(negative_count)_std — contains both sentiment AND volume info")
    print("Signal C: PhotoPes_std (baseline) — pure SENTIMENT (= negative_count / total_images)")
    print()

    hi_mask = define_high_vol(df)

    print(f"{'Index':<10} {'Signal':<32} {'lag3 coef':>12} {'t':>7} {'p':>7} {'sig':>5}")
    print("-" * 80)

    for idx in INDICES:
        ret_col = f"{idx}_log_returns"

        # Baseline: real PhotoPes
        X, y = build_design(df, ret_col, hi_mask, photopes_prefix="PhotoPes_std")
        m = fit(X, y)
        b = m.params["PhotoPes_lag3"]; t = m.tvalues["PhotoPes_lag3"]; p = m.pvalues["PhotoPes_lag3"]
        print(f"{idx:<10} {'PhotoPes_std (baseline)':<32} {b:>12.5f} {t:>7.2f} {p:>7.4f} {stars(p):>5}")

        # Placebo A: image count
        X, y = build_design(df, ret_col, hi_mask, photopes_prefix="placebo_imgcount")
        m = fit(X, y)
        b = m.params["PhotoPes_lag3"]; t = m.tvalues["PhotoPes_lag3"]; p = m.pvalues["PhotoPes_lag3"]
        print(f"{idx:<10} {'PLACEBO: log(n_images)':<32} {b:>12.5f} {t:>7.2f} {p:>7.4f} {stars(p):>5}")

        # Placebo B: log neg count
        X, y = build_design(df, ret_col, hi_mask, photopes_prefix="placebo_negcount")
        m = fit(X, y)
        b = m.params["PhotoPes_lag3"]; t = m.tvalues["PhotoPes_lag3"]; p = m.pvalues["PhotoPes_lag3"]
        print(f"{idx:<10} {'PLACEBO: log(negative_count)':<32} {b:>12.5f} {t:>7.2f} {p:>7.4f} {stars(p):>5}")
        print("-" * 80)

    print()
    print("Interpretation:")
    print("  If placebo signals also produce ChiNext 5%-significant negative coefficients,")
    print("  the PhotoPes effect is a 'news-activity proxy' rather than visual sentiment.")
    print("  If placebos are NULL on ChiNext, the sentiment-specific information of PhotoPes")
    print("  is what carries the predictive content.")


if __name__ == "__main__":
    main()
