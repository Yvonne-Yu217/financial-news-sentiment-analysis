"""
Conditional analysis: High vs Low Market Volatility
Replaces the current PhotoPes-extreme conditioning with a volatility-state conditioning.
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import statsmodels.api as sm

df = pd.read_csv('/Volumes/Data_Drive/research0322226/financial-news-sentiment-analysis/results/merged_market_sentiment_data_old.csv', parse_dates=['Date'])
df = df[df['Date'] >= '2014-01-01'].copy()

# High vol = top 25% of 20-day rolling SHCOMP std dev
df['vol20'] = df['SHCOMP_log_returns'].rolling(20).std()
vol_thresh = df['vol20'].quantile(0.75)
df['high_vol'] = (df['vol20'] > vol_thresh).astype(int)
print(f"Vol threshold (75th pct): {vol_thresh:.5f}")
print(f"High vol obs: {df['high_vol'].sum()}, Low vol obs: {(df['high_vol']==0).sum()}")

INDICES = ['CSI300', 'SHCOMP', 'SZCOMP', 'ChiNext', 'CSI500']
PRED_LAGS = [f'{s}_std_lag{i}' for s in ['PhotoPes','TextPes'] for i in range(1,6)]
WK = [c for c in df.columns if 'weekday' in c]

def run_regression_table(sub):
    results = {}
    n_obs = None
    for idx in INDICES:
        y = f'{idx}_log_returns'
        lag_r  = [f'{idx}_log_returns_lag{i}' for i in range(1,6)]
        lag_r2 = [f'{idx}_log_returns_sq_lag{i}' for i in range(1,6)]
        xcols = PRED_LAGS + lag_r + lag_r2 + WK
        xcols = [c for c in xcols if c in sub.columns]
        data = sub[[y] + xcols].dropna()
        if n_obs is None:
            n_obs = len(data)
        X = sm.add_constant(data[xcols])
        res = sm.OLS(data[y], X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
        results[idx] = res
    return results, n_obs

def star(t):
    a = abs(t)
    return '***' if a > 2.576 else ('**' if a > 1.96 else ('*' if a > 1.645 else ''))

def print_table(results, n_obs, label):
    preds_to_show = [
        ('PhotoPes_std_lag1', 'PhotoPes$_{t-1}$'),
        ('PhotoPes_std_lag2', 'PhotoPes$_{t-2}$'),
        ('PhotoPes_std_lag3', 'PhotoPes$_{t-3}$'),
        ('PhotoPes_std_lag4', 'PhotoPes$_{t-4}$'),
        ('PhotoPes_std_lag5', 'PhotoPes$_{t-5}$'),
        ('TextPes_std_lag1',  'TextPes$_{t-1}$'),
        ('TextPes_std_lag2',  'TextPes$_{t-2}$'),
        ('TextPes_std_lag3',  'TextPes$_{t-3}$'),
        ('TextPes_std_lag4',  'TextPes$_{t-4}$'),
        ('TextPes_std_lag5',  'TextPes$_{t-5}$'),
    ]
    print(f"\n{label} (N={n_obs})")
    header = f"{'Predictor':<22}" + "".join(f"  {i:>12}" for i in INDICES)
    print(header)
    for col, plabel in preds_to_show:
        coef_row = f"{plabel:<22}"
        tstat_row = f"{'':22}"
        for idx in INDICES:
            res = results[idx]
            c = res.params[col]; t = res.tvalues[col]
            s = star(t)
            coef_row  += f"  {c:+.5f}{s:<3}"
            tstat_row += f"  ({t:+.2f})     "
        print(coef_row)
        print(tstat_row)
    # R2
    r2_row = f"{'R2':<22}"
    for idx in INDICES:
        r2_row += f"  {results[idx].rsquared:.4f}     "
    print(r2_row)
    print(f"N = {n_obs}")

# ── Run both panels ──
res_high, n_high = run_regression_table(df[df['high_vol'] == 1])
res_low,  n_low  = run_regression_table(df[df['high_vol'] == 0])

print_table(res_high, n_high, "Panel A: High Volatility Periods (top 25% of 20-day vol)")
print_table(res_low,  n_low,  "Panel B: Low Volatility Periods (bottom 75%)")

# ── LaTeX snippet for paper ──
print("\n\n=== LaTeX TABLE ===")
print(r"""\begin{table}[p]
    \centering
    \caption{Conditional Impact of PhotoPes and TextPes by Market Volatility State (2014--2026)}
    \label{tab:sentiment_significant_periods}
    \setlength{\tabcolsep}{3pt}
    \footnotesize
    \begin{tabular}{lccccc}
    \toprule
    Indicators & \textit{CSI 300} & \textit{SHCOMP} & \textit{SZCOMP} & \textit{ChiNext} & \textit{CSI 500} \\""")

for panel_results, panel_n, panel_label in [
    (res_high, n_high, r'Panel A: High Volatility Periods ($\sigma_t > $ 75th percentile, N=NHIGH)'),
    (res_low,  n_low,  r'Panel B: Low Volatility Periods ($\sigma_t \leq $ 75th percentile, N=NLOW)'),
]:
    panel_label = panel_label.replace('NHIGH', str(panel_n)).replace('NLOW', str(panel_n))
    print(r"    \midrule")
    print(f"    \\multicolumn{{6}}{{l}}{{\\textit{{{panel_label}}}}} \\\\")
    print(r"    \midrule")

    preds = [
        ('PhotoPes_std_lag1', 'PhotoPes$_{t-1}$'),
        ('PhotoPes_std_lag2', 'PhotoPes$_{t-2}$'),
        ('PhotoPes_std_lag3', 'PhotoPes$_{t-3}$'),
        ('PhotoPes_std_lag4', 'PhotoPes$_{t-4}$'),
        ('PhotoPes_std_lag5', 'PhotoPes$_{t-5}$'),
        ('TextPes_std_lag1',  'TextPes$_{t-1}$'),
        ('TextPes_std_lag2',  'TextPes$_{t-2}$'),
        ('TextPes_std_lag3',  'TextPes$_{t-3}$'),
        ('TextPes_std_lag4',  'TextPes$_{t-4}$'),
        ('TextPes_std_lag5',  'TextPes$_{t-5}$'),
    ]
    for col, plabel in preds:
        coefs = []; tstats = []
        for idx in INDICES:
            res = panel_results[idx]
            coefs.append(res.params[col])
            tstats.append(res.tvalues[col])
        cs = " & ".join(f"{c:.4f}{star(t)}" for c,t in zip(coefs,tstats))
        ts = " & ".join(f"({t:.2f})" for t in tstats)
        print(f"    {plabel} & {cs} \\\\")
        print(f"         & {ts} \\\\")

    r2s = " & ".join(f"{panel_results[i].rsquared:.4f}" for i in INDICES)
    print(f"    R\\textsuperscript{{2}} & {r2s} \\\\")
    print(f"    N & {panel_n} & {panel_n} & {panel_n} & {panel_n} & {panel_n} \\\\")

print(r"""    \bottomrule
    \end{tabular}
    \vspace{0.2cm}
    \footnotesize
    \textit{Note:} Market volatility state is defined by the 20-day rolling standard deviation of SHCOMP log returns; high volatility = top 25th percentile. t-statistics in parentheses. *, **, *** denote significance at 10\%, 5\%, 1\% levels. Standard errors are Newey-West HAC with 5 lags.
\end{table}""")
