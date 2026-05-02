"""
paper_table9_robustness.py
Reproduces Table 9: Robustness Checks for PhotoPes -> Market Returns
Panel A: Without winsorization (raw PhotoPes, not winsorized)
Panel B: GARCH(1,1) volatility-adjusted returns
Panel C: Remove extreme market returns (top/bottom 0.5% of SHCOMP)
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import statsmodels.api as sm
from arch import arch_model
from scipy.stats.mstats import winsorize

DATA = "results/merged_market_sentiment_data_old.csv"
OUT  = "results/regression_output_oldmodel/table9_robustness.csv"

INDICES    = ['CSI300', 'SHCOMP', 'SZCOMP', 'ChiNext', 'CSI500']
COL_LABELS = ['CSI 300', 'SHCOMP', 'SZCOMP', 'ChiNext', 'CSI 500']
N_LAGS     = 5

def sig(t):
    a = abs(t)
    return '***' if a > 2.576 else ('**' if a > 1.96 else ('*' if a > 1.645 else ''))

def build_controls(df, idx):
    wk    = [c for c in df.columns if 'weekday' in c]
    lag_r = [f'{idx}_log_returns_lag{i}' for i in range(1, N_LAGS+1)]
    lag_r2= [f'{idx}_log_returns_sq_lag{i}' for i in range(1, N_LAGS+1)]
    return lag_r + lag_r2 + wk

def run_panel(df, y_col_map, photo_cols, label):
    """
    df: dataframe
    y_col_map: dict {idx -> dependent variable column}
    photo_cols: list of PhotoPes lag column names to use as predictors
    """
    rows = []
    n_obs = None
    for idx in INDICES:
        y_col = y_col_map[idx]
        controls = [c for c in build_controls(df, idx) if c in df.columns]
        xcols = photo_cols + controls
        xcols = [c for c in xcols if c in df.columns]
        data  = df[[y_col] + xcols].dropna()
        if n_obs is None:
            n_obs = len(data)
        X   = sm.add_constant(data[xcols])
        res = sm.OLS(data[y_col], X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
        rows.append((idx, res, n_obs))
    return rows

def print_panel(rows, panel_label, photo_cols):
    n = rows[0][2]
    print(f"\n{panel_label} (N={n})")
    print(f"{'':22}" + "".join(f"  {l:>10}" for l in COL_LABELS))
    lag_labels = [f'PhotoPes_t-{i}' for i in range(1, N_LAGS+1)]
    for col, name in zip(photo_cols, lag_labels):
        crow = f"{name:<22}"; trow = f"{'':22}"
        for idx, res, _ in rows:
            c = res.params.get(col, np.nan)
            t = res.tvalues.get(col, np.nan)
            crow += f"  {c:+.4f}{sig(t):<3}"
            trow += f"  ({t:+.2f})     "
        print(crow); print(trow)
    r2_row = f"{'R2':<22}" + "".join(f"  {r.rsquared:.4f}     " for _, r, _ in rows)
    adj_row= f"{'Adj R2':<22}" + "".join(f"  {r.rsquared_adj:.4f}     " for _, r, _ in rows)
    n_row  = f"{'N':<22}" + "".join(f"  {n:>10d}" for _, _, n in rows)
    print(r2_row); print(adj_row); print(n_row)

def save_rows_to_records(rows, panel, photo_cols):
    records = []
    for col in photo_cols:
        rec = {'Panel': panel, 'Predictor': col}
        for idx, res, n in rows:
            rec[f'{idx}_coef']  = round(res.params.get(col, np.nan), 6)
            rec[f'{idx}_tstat'] = round(res.tvalues.get(col, np.nan), 3)
            rec[f'{idx}_pval']  = round(res.pvalues.get(col, np.nan), 4)
            rec['N'] = n
        records.append(rec)
    return records

if __name__ == '__main__':
    df = pd.read_csv(DATA, parse_dates=['Date'])
    df = df[df['Date'] >= '2014-01-01'].copy()

    print("=" * 70)
    print("Table 9: Robustness Checks (PhotoPes -> Market Returns, 2014-2026)")
    print("=" * 70)

    all_records = []

    # ── Panel A: Without winsorization ────────────────────────────────────
    # Re-standardize raw PhotoPes without winsorization
    df['PhotoPes_raw_std'] = (df['PhotoPes'] - df['PhotoPes'].mean()) / df['PhotoPes'].std()
    for i in range(1, N_LAGS+1):
        df[f'PhotoPes_raw_lag{i}'] = df['PhotoPes_raw_std'].shift(i)

    photo_cols_a = [f'PhotoPes_raw_lag{i}' for i in range(1, N_LAGS+1)]
    y_map_a = {idx: f'{idx}_log_returns' for idx in INDICES}
    rows_a = run_panel(df, y_map_a, photo_cols_a, 'Panel A')
    # Print with original names
    print_panel(rows_a, "Panel A: Without Winsorization", photo_cols_a)
    all_records += save_rows_to_records(rows_a, 'A_NoWinsorize', photo_cols_a)

    # ── Panel B: GARCH(1,1) volatility-adjusted returns ───────────────────
    print("\nPanel B: Fitting GARCH(1,1) for each index...")
    garch_y_map = {}
    for idx in INDICES:
        col = f'{idx}_log_returns'
        ret = df[col].dropna() * 100  # arch expects percent scale
        gm  = arch_model(ret, vol='GARCH', p=1, q=1, dist='Normal', rescale=False)
        gres = gm.fit(disp='off')
        cond_vol = gres.conditional_volatility / 100
        # Align with df
        adj = pd.Series(df[col].values / cond_vol.reindex(df.index).values,
                        index=df.index, name=f'{idx}_garch_adj')
        df[f'{idx}_garch_adj'] = adj
        garch_y_map[idx] = f'{idx}_garch_adj'
        print(f"  {idx} done")

    rows_b = run_panel(df, garch_y_map,
                       [f'PhotoPes_std_lag{i}' for i in range(1, N_LAGS+1)], 'Panel B')
    print_panel(rows_b, "Panel B: GARCH(1,1) Volatility-Adjusted Returns",
                [f'PhotoPes_std_lag{i}' for i in range(1, N_LAGS+1)])
    all_records += save_rows_to_records(rows_b, 'B_GARCH',
                                        [f'PhotoPes_std_lag{i}' for i in range(1, N_LAGS+1)])

    # ── Panel C: Remove extreme returns (0.5% tails of SHCOMP) ───────────
    q_lo = df['SHCOMP_log_returns'].quantile(0.005)
    q_hi = df['SHCOMP_log_returns'].quantile(0.995)
    df_c = df[(df['SHCOMP_log_returns'] >= q_lo) & (df['SHCOMP_log_returns'] <= q_hi)].copy()
    print(f"\nPanel C: After removing extreme SHCOMP returns (<{q_lo:.4f}, >{q_hi:.4f})")
    print(f"         Dropped {len(df) - len(df_c)} rows, N={len(df_c)}")

    rows_c = run_panel(df_c, {idx: f'{idx}_log_returns' for idx in INDICES},
                       [f'PhotoPes_std_lag{i}' for i in range(1, N_LAGS+1)], 'Panel C')
    print_panel(rows_c, "Panel C: Remove Extreme Market Returns (0.5% tails)",
                [f'PhotoPes_std_lag{i}' for i in range(1, N_LAGS+1)])
    all_records += save_rows_to_records(rows_c, 'C_NoExtreme',
                                        [f'PhotoPes_std_lag{i}' for i in range(1, N_LAGS+1)])

    import os
    os.makedirs('results/regression_output_oldmodel', exist_ok=True)
    pd.DataFrame(all_records).to_csv(OUT, index=False)
    print(f"\nSaved: {OUT}")
