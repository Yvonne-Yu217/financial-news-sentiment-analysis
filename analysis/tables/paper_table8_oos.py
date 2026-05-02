"""
paper_table8_oos.py
Reproduces Table 8: Out-of-Sample Predictive Performance (2014-2026)
Model: Expanding-window Ridge (alpha=500), predictors = PhotoPes_std lags 3,4,5
Init window: 1512 days (~6 years, Jan 2014 - Mar 2020)
OOS period: Mar 2020 - Apr 2026, N=1467 predictions
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from scipy.stats import norm

DATA = "results/merged_market_sentiment_data_old.csv"

INDICES    = ['CSI300', 'SHCOMP', 'SZCOMP', 'ChiNext', 'CSI500']
COL_LABELS = ['CSI 300', 'SHCOMP', 'SZCOMP', 'ChiNext', 'CSI 500']
PRED_COLS  = ['PhotoPes_std_lag3', 'PhotoPes_std_lag4', 'PhotoPes_std_lag5']
INIT       = 1512
RIDGE_ALPHA = 500

def clark_west(actual, pred_m, pred_b):
    f = (actual - pred_b)**2 - (actual - pred_m)**2 + (pred_m - pred_b)**2
    n = len(f)
    return f.mean() / (f.std() / np.sqrt(n))

def r2_oos(actual, pred_m, pred_b):
    return (1 - np.mean((actual - pred_m)**2) / np.mean((actual - pred_b)**2)) * 100

def run_oos(df, idx):
    col_y = f'{idx}_log_returns'
    data  = df[[col_y] + PRED_COLS].dropna().reset_index(drop=True)
    n     = len(data)
    a, pm, pb = [], [], []
    for t in range(INIT, n):
        tr  = data.iloc[:t]
        Xtr = tr[PRED_COLS].values; ytr = tr[col_y].values
        Xte = data.iloc[t][PRED_COLS].values.reshape(1, -1)
        yte = data.iloc[t][col_y]
        mdl = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
        mdl.fit(Xtr, ytr)
        pm.append(mdl.predict(Xte)[0])
        pb.append(ytr.mean())
        a.append(yte)
    a, pm, pb = np.array(a), np.array(pm), np.array(pb)
    r2 = r2_oos(a, pm, pb)
    t  = clark_west(a, pm, pb)
    p  = 1 - norm.cdf(t)
    return {'R2_OOS': round(r2, 4), 't_CW': round(t, 3), 'p': round(p, 4), 'N': len(a)}

def sig(p):
    return '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else ''))

if __name__ == '__main__':
    df = pd.read_csv(DATA, parse_dates=['Date'])
    df = df[df['Date'] >= '2014-01-01'].copy().reset_index(drop=True)

    print("=" * 70)
    print("Table 8: Out-of-Sample Predictive Performance (2014-2026)")
    print(f"Model: Ridge(alpha={RIDGE_ALPHA}), predictors={PRED_COLS}")
    print(f"Init window: {INIT} days, OOS: Mar 2020 - Apr 2026")
    print("=" * 70)

    results = {}
    for idx in INDICES:
        print(f"  Running OOS for {idx}...", end=' ', flush=True)
        results[idx] = run_oos(df, idx)
        r = results[idx]
        print(f"R2={r['R2_OOS']:.4f}% t={r['t_CW']:.3f} p={r['p']:.4f} {sig(r['p'])}")

    print()
    print(f"{'':20}" + "".join(f"  {l:>10}" for l in COL_LABELS))
    r2_row = f"{'R2_OOS (%)':20}" + "".join(
        f"  {results[i]['R2_OOS']:>7.4f}{sig(results[i]['p']):>3}" for i in INDICES)
    t_row  = f"{'MSPE-adj t':20}" + "".join(
        f"  {results[i]['t_CW']:>10.3f}" for i in INDICES)
    p_row  = f"{'p-value':20}" + "".join(
        f"  {results[i]['p']:>10.4f}" for i in INDICES)
    n_row  = f"{'N':20}" + "".join(
        f"  {results[i]['N']:>10d}" for i in INDICES)
    print(r2_row); print(t_row); print(p_row); print(n_row)

    print("\n--- Paper verification ---")
    expected = {
        'CSI300':  (0.04, 1.07), 'SHCOMP': (0.06, 1.22),
        'SZCOMP':  (0.08, 1.38), 'ChiNext':(0.29, 2.02), 'CSI500': (0.07, 1.38)
    }
    for idx in INDICES:
        r = results[idx]; e = expected[idx]
        print(f"{idx}: R2={r['R2_OOS']:.4f}% (paper ~{e[0]:.2f}%)  "
              f"t={r['t_CW']:.3f} (paper ~{e[1]:.2f})")

    import os, json
    os.makedirs('results/regression_output_oldmodel', exist_ok=True)
    with open('results/regression_output_oldmodel/table8_oos.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved: results/regression_output_oldmodel/table8_oos.json")
