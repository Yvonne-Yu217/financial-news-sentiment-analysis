"""
r2_log_news_count_control.py
=============================
Phase-1 R2 (per professor's memo §5.2): add log(n_t) and its lag to the
high-vol joint regression. Tests whether the PhotoPes effect is a
news-volume artifact (high-vol days happen to coincide with high news
volume, which mechanically inflates sentiment effects).

Procedure:
  Add log(total_articles_t-1) and log(total_images_t-1) as controls
  in the same high-vol joint regression. Re-test ChiNext PhotoPes_{t-3}.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = "results/merged_market_sentiment_data_old.csv"
TEXTPES_CSV = "results/weighted_textpes.csv"
PHOTOPES_CSV = "results/weighted_photopes_old.csv"
INDICES = ["CSI300", "SHCOMP", "SZCOMP", "ChiNext", "CSI500"]
N_LAGS = 5
NW_BW = 5

np.random.seed(42)


def load_with_news_counts():
    df = pd.read_csv(DATA, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    # Daily article count (text)
    text = pd.read_csv(TEXTPES_CSV, parse_dates=["news_date"])
    text = text[["news_date", "total_count"]].rename(
        columns={"news_date": "Date", "total_count": "n_articles"})
    # Daily image count
    photo = pd.read_csv(PHOTOPES_CSV, parse_dates=["news_date"])
    photo = photo[["news_date", "total_images"]].rename(
        columns={"news_date": "Date", "total_images": "n_images"})
    df = df.merge(text, on="Date", how="left")
    df = df.merge(photo, on="Date", how="left")
    df["log_n_articles"] = np.log(df["n_articles"].clip(lower=1))
    df["log_n_images"] = np.log(df["n_images"].clip(lower=1))
    df["log_n_articles_lag1"] = df["log_n_articles"].shift(1)
    df["log_n_images_lag1"] = df["log_n_images"].shift(1)
    return df


def define_high_vol(df, percentile=0.75, window=20):
    rv = df["SHCOMP_log_returns"].rolling(window).std() * np.sqrt(252)
    return rv >= rv.quantile(percentile)


def build_design(df, ret_col, hi_mask, with_news=True):
    cols = {}
    for lag in range(1, N_LAGS + 1):
        cols[f"PhotoPes_lag{lag}"] = df[f"PhotoPes_std_lag{lag}"]
        cols[f"TextPes_lag{lag}"]  = df[f"TextPes_std_lag{lag}"]
        cols[f"R_lag{lag}"]   = df[f"{ret_col.split('_log_')[0]}_log_returns_lag{lag}"]
        cols[f"R2_lag{lag}"]  = df[f"{ret_col.split('_log_')[0]}_log_returns_sq_lag{lag}"]
    if with_news:
        cols["log_n_articles_lag1"] = df["log_n_articles_lag1"]
        cols["log_n_images_lag1"] = df["log_n_images_lag1"]
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
    df = load_with_news_counts()
    print("=" * 80)
    print("R2: high-vol joint regression with log(n_articles) + log(n_images) controls")
    print("=" * 80)
    print(f"Daily n_articles: mean={df['n_articles'].mean():.0f}, std={df['n_articles'].std():.0f}")
    print(f"Daily n_images:   mean={df['n_images'].mean():.0f}, std={df['n_images'].std():.0f}")
    print()

    hi_mask = define_high_vol(df)

    # Verify high-vol days have different news volume than low-vol days
    hi_days = df.loc[hi_mask, ["n_articles", "n_images"]].mean()
    lo_days = df.loc[~hi_mask, ["n_articles", "n_images"]].mean()
    print("Mean daily volumes:")
    print(f"  High-vol: {int(hi_days['n_articles'])} articles, {int(hi_days['n_images'])} images")
    print(f"  Low-vol:  {int(lo_days['n_articles'])} articles, {int(lo_days['n_images'])} images")
    print()

    # Run regression both with and without news controls
    print(f"{'Index':<10} {'Spec':<28} {'PhotoPes_t-3':>14} {'t':>7} {'p':>7} {'sig':>5} | {'log_n_art_l1 t':>14} {'log_n_img_l1 t':>14}")
    print("-" * 120)

    for idx in INDICES:
        ret_col = f"{idx}_log_returns"
        # Without news control (baseline)
        X, y = build_design(df, ret_col, hi_mask, with_news=False)
        m_base = fit(X, y)
        bp = m_base.params["PhotoPes_lag3"]; tp = m_base.tvalues["PhotoPes_lag3"]; pp = m_base.pvalues["PhotoPes_lag3"]
        print(f"{idx:<10} {'baseline (no n control)':<28} {bp:>14.5f} {tp:>7.2f} {pp:>7.4f} {stars(pp):>5} |  -            -")

        # With news control
        X, y = build_design(df, ret_col, hi_mask, with_news=True)
        m_news = fit(X, y)
        bp = m_news.params["PhotoPes_lag3"]; tp = m_news.tvalues["PhotoPes_lag3"]; pp = m_news.pvalues["PhotoPes_lag3"]
        ta = m_news.tvalues["log_n_articles_lag1"]
        ti = m_news.tvalues["log_n_images_lag1"]
        print(f"{idx:<10} {'+ log_n_art + log_n_img':<28} {bp:>14.5f} {tp:>7.2f} {pp:>7.4f} {stars(pp):>5} | {ta:>14.2f} {ti:>14.2f}")
        print("-" * 120)


if __name__ == "__main__":
    main()
