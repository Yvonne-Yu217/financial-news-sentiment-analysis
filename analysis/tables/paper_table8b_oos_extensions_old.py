"""
paper_table8b_oos_extensions_old.py
Quick-run extensions to Table 8 (OOS predictability):
  1. Sub-sample OOS R^2 (split OOS predictions into 3 equal-length windows)
  2. Certainty-equivalent (CER) gain at gamma = 3 and gamma = 5 for a
     mean-variance investor switching from historical-mean to PhotoPes forecast.
Same data + model spec as paper_table8_oos.py (Ridge alpha=500, lags 3-5).
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, json, os
from sklearn.linear_model import Ridge
from scipy.stats import norm

DATA = "results/merged_market_sentiment_data_old.csv"
INDICES    = ['CSI300', 'SHCOMP', 'SZCOMP', 'ChiNext', 'CSI500']
COL_LABELS = ['CSI 300', 'SHCOMP', 'SZCOMP', 'ChiNext', 'CSI 500']
PRED_COLS  = ['PhotoPes_std_lag3', 'PhotoPes_std_lag4', 'PhotoPes_std_lag5']
INIT       = 1512
RIDGE_ALPHA = 500
TRADING_DAYS = 252


def clark_west(actual, pred_m, pred_b):
    f = (actual - pred_b) ** 2 - (actual - pred_m) ** 2 + (pred_m - pred_b) ** 2
    n = len(f)
    return f.mean() / (f.std() / np.sqrt(n))


def r2_oos(actual, pred_m, pred_b):
    return (1 - np.mean((actual - pred_m) ** 2) / np.mean((actual - pred_b) ** 2)) * 100


def expanding_oos(df, idx):
    col_y = f'{idx}_log_returns'
    data = df[['Date', col_y] + PRED_COLS].dropna().reset_index(drop=True)
    n = len(data)
    dates, a, pm, pb, sigma2 = [], [], [], [], []
    for t in range(INIT, n):
        tr = data.iloc[:t]
        Xtr = tr[PRED_COLS].values
        ytr = tr[col_y].values
        Xte = data.iloc[t][PRED_COLS].values.reshape(1, -1)
        yte = data.iloc[t][col_y]
        mdl = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
        mdl.fit(Xtr, ytr)
        pm.append(mdl.predict(Xte)[0])
        pb.append(ytr.mean())
        sigma2.append(ytr.var())
        a.append(yte)
        dates.append(data.iloc[t]['Date'])
    return pd.DataFrame({
        'Date': pd.to_datetime(dates),
        'actual': a,
        'pred_model': pm,
        'pred_bench': pb,
        'sigma2': sigma2,
    })


def cer_for_strategy(actual, mu_hat, sigma2, gamma, w_min=0.0, w_max=1.5):
    """Campbell-Thompson (2008) mean-variance investor CER, daily units."""
    w = (mu_hat / (gamma * sigma2))
    w = np.clip(w, w_min, w_max)
    r_p = w * actual
    return r_p.mean() - 0.5 * gamma * r_p.var()


def cer_gain(oos, gamma):
    actual = oos['actual'].values
    sigma2 = oos['sigma2'].values
    cer_m = cer_for_strategy(actual, oos['pred_model'].values, sigma2, gamma)
    cer_b = cer_for_strategy(actual, oos['pred_bench'].values, sigma2, gamma)
    delta_daily = cer_m - cer_b
    delta_annual_pct = delta_daily * TRADING_DAYS * 100
    return cer_m * TRADING_DAYS * 100, cer_b * TRADING_DAYS * 100, delta_annual_pct


def split_oos_three_windows(oos):
    """Split predictions into three equal-length sub-periods by date order."""
    n = len(oos)
    edges = [0, n // 3, 2 * n // 3, n]
    return [oos.iloc[edges[i]:edges[i + 1]].reset_index(drop=True) for i in range(3)]


def sig(p):
    return '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else ''))


if __name__ == '__main__':
    df = pd.read_csv(DATA, parse_dates=['Date'])
    df = df[df['Date'] >= '2014-01-01'].copy().reset_index(drop=True)

    sub_results = {}
    cer_results = {}
    print("=" * 90)
    print("Table 8b: Sub-sample OOS R^2 + CER gain (gamma=3, 5)")
    print("Spec: Ridge(alpha=500), predictors = PhotoPes_std lags 3,4,5; init window 1512 days")
    print("=" * 90)

    for idx in INDICES:
        print(f"\n--- {idx} ---")
        oos = expanding_oos(df, idx)
        full_r2 = r2_oos(oos['actual'].values, oos['pred_model'].values, oos['pred_bench'].values)
        full_t = clark_west(oos['actual'].values, oos['pred_model'].values, oos['pred_bench'].values)
        full_p = 1 - norm.cdf(full_t)
        print(f"Full OOS  : N={len(oos)}, R2={full_r2:.4f}% t={full_t:.3f} p={full_p:.4f} {sig(full_p)}")

        windows = split_oos_three_windows(oos)
        sub_results[idx] = {'full': {'N': int(len(oos)), 'R2': full_r2, 't_CW': full_t, 'p': full_p}}
        for k, sub in enumerate(windows, 1):
            r2 = r2_oos(sub['actual'].values, sub['pred_model'].values, sub['pred_bench'].values)
            t = clark_west(sub['actual'].values, sub['pred_model'].values, sub['pred_bench'].values)
            p = 1 - norm.cdf(t)
            d_start = sub['Date'].iloc[0].strftime('%Y-%m-%d')
            d_end = sub['Date'].iloc[-1].strftime('%Y-%m-%d')
            print(f"Window {k} : {d_start} -> {d_end}  N={len(sub):4d}  R2={r2:7.4f}%  t={t:6.3f}  p={p:.4f} {sig(p)}")
            sub_results[idx][f'window{k}'] = {
                'start': d_start, 'end': d_end, 'N': int(len(sub)),
                'R2': r2, 't_CW': t, 'p': p,
            }

        cer_results[idx] = {}
        for gamma in (3, 5):
            cer_m, cer_b, dcer = cer_gain(oos, gamma)
            print(f"CER (gamma={gamma}): model={cer_m:.3f}%  bench={cer_b:.3f}%  delta={dcer:+.3f}% /yr")
            cer_results[idx][f'gamma{gamma}'] = {
                'cer_model_pct_yr': cer_m, 'cer_bench_pct_yr': cer_b, 'delta_pct_yr': dcer,
            }

    print("\n" + "=" * 90)
    print("CER gain (annualised %) summary:")
    header = f"{'Index':10s}" + ''.join(f"  gamma={g:<5d}" for g in (3, 5))
    print(header)
    for idx in INDICES:
        row = f"{idx:10s}" + ''.join(
            f"  {cer_results[idx][f'gamma{g}']['delta_pct_yr']:+8.3f}" for g in (3, 5)
        )
        print(row)

    out_dir = 'results/regression_output_oldmodel'
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'table8b_subsample_oos.json'), 'w') as f:
        json.dump(sub_results, f, indent=2)
    with open(os.path.join(out_dir, 'table8b_cer_gain.json'), 'w') as f:
        json.dump(cer_results, f, indent=2)
    print(f"\nSaved JSON to: {out_dir}/table8b_subsample_oos.json")
    print(f"Saved JSON to: {out_dir}/table8b_cer_gain.json")
