"""
paper_table5_conditional.py
Reproduces Table 5: Conditional Impact by Market Volatility State (2014-2026)
Split: High vol = top 25% of 20-day rolling SHCOMP std dev
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import statsmodels.api as sm

DATA = "results/merged_market_sentiment_data_old.csv"
OUT  = "results/regression_output_oldmodel/table5_conditional_vol.csv"

INDICES = ['CSI300', 'SHCOMP', 'SZCOMP', 'ChiNext', 'CSI500']
COL_LABELS = ['CSI 300', 'SHCOMP', 'SZCOMP', 'ChiNext', 'CSI 500']
PRED_LAGS = [f'{s}_std_lag{i}' for s in ['PhotoPes','TextPes'] for i in range(1,6)]
WK = None  # filled after load

def load():
    df = pd.read_csv(DATA, parse_dates=['Date'])
    df = df[df['Date'] >= '2014-01-01'].copy()
    df['vol20'] = df['SHCOMP_log_returns'].rolling(20).std()
    df['high_vol'] = (df['vol20'] > df['vol20'].quantile(0.75)).astype(int)
    return df

def sig(t):
    a = abs(t)
    return '***' if a > 2.576 else ('**' if a > 1.96 else ('*' if a > 1.645 else ''))

def run_panel(sub, df_full):
    wk = [c for c in df_full.columns if 'weekday' in c]
    results = {}
    for idx in INDICES:
        y = f'{idx}_log_returns'
        lag_r  = [f'{idx}_log_returns_lag{i}' for i in range(1,6)]
        lag_r2 = [f'{idx}_log_returns_sq_lag{i}' for i in range(1,6)]
        xcols  = [c for c in PRED_LAGS + lag_r + lag_r2 + wk if c in sub.columns]
        data   = sub[[y] + xcols].dropna()
        X      = sm.add_constant(data[xcols])
        res    = sm.OLS(data[y], X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
        results[idx] = (res, len(data))
    return results

def print_panel(results, label):
    n = results['CSI300'][1]
    print(f"\n{label} (N={n})")
    print(f"{'':22}" + "".join(f"  {l:>10}" for l in COL_LABELS))
    preds = [
        ('PhotoPes_std_lag1','PhotoPes_t-1'), ('PhotoPes_std_lag2','PhotoPes_t-2'),
        ('PhotoPes_std_lag3','PhotoPes_t-3'), ('PhotoPes_std_lag4','PhotoPes_t-4'),
        ('PhotoPes_std_lag5','PhotoPes_t-5'),
        ('TextPes_std_lag1', 'TextPes_t-1'),  ('TextPes_std_lag2', 'TextPes_t-2'),
        ('TextPes_std_lag3', 'TextPes_t-3'),  ('TextPes_std_lag4', 'TextPes_t-4'),
        ('TextPes_std_lag5', 'TextPes_t-5'),
    ]
    for col, name in preds:
        crow = f"{name:<22}"
        trow = f"{'':22}"
        for idx in INDICES:
            res, _ = results[idx]
            c = res.params[col]; t = res.tvalues[col]
            crow += f"  {c:+.4f}{sig(t):<3}"
            trow += f"  ({t:+.2f})     "
        print(crow); print(trow)
    r2row = f"{'R2':<22}" + "".join(f"  {results[i][0].rsquared:.4f}     " for i in INDICES)
    print(r2row)

def save_csv(res_high, n_high, res_low, n_low):
    rows = []
    preds = [f'{s}_std_lag{i}' for s in ['PhotoPes','TextPes'] for i in range(1,6)]
    for panel, panel_res, n in [('High Vol', res_high, n_high), ('Low Vol', res_low, n_low)]:
        for pred in preds:
            row = {'Panel': panel, 'Predictor': pred}
            for idx in INDICES:
                res, _ = panel_res[idx]
                row[f'{idx}_coef'] = round(res.params[pred], 6)
                row[f'{idx}_tstat'] = round(res.tvalues[pred], 3)
                row[f'{idx}_pval'] = round(res.pvalues[pred], 4)
            rows.append(row)
    import os; os.makedirs('results/regression_output_oldmodel', exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nSaved: {OUT}")

if __name__ == '__main__':
    df = load()
    hi = df[df['high_vol'] == 1]
    lo = df[df['high_vol'] == 0]
    res_h = run_panel(hi, df)
    res_l = run_panel(lo, df)
    n_h = res_h['CSI300'][1]; n_l = res_l['CSI300'][1]
    print("=" * 70)
    print("Table 5: Conditional Impact by Market Volatility State (2014-2026)")
    print("=" * 70)
    print(f"Vol threshold (75th pct): {df['vol20'].quantile(0.75):.5f}")
    print_panel(res_h, f"Panel A: High Volatility (N={n_h})")
    print_panel(res_l, f"Panel B: Low Volatility (N={n_l})")
    save_csv(res_h, n_h, res_l, n_l)
