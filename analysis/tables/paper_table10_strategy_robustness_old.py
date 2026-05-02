"""
paper_table10_strategy_robustness_old.py
Hyperparameter robustness for the sentiment trading strategy (Table 6).

Headline params used in main text:  W=90, R=5, w_p=0.80, (Q_LO, Q_HI)=(0.20, 0.80)
Grid scan:
    Photo weight w_p   in {0.00, 0.25, 0.50, 0.75, 0.80, 1.00}            (6)
    Rolling window W   in {30, 60, 90, 120, 252}                          (5)
    Rebalance freq R   in {1, 3, 5, 10, 21}                               (5)
    Quantile pair (QL,QH) in {(.10,.90), (.20,.80), (.30,.70)}            (3)
    Indices            in {CSI300, SHCOMP, SZCOMP, ChiNext, CSI500}       (5)

Total = 6*5*5*3*5 = 2,250 backtests.  Each backtest is evaluated under three
transaction-cost regimes (TC = 0, 30, 50 bps round-trip).  For each (cell, TC)
we compute (i) raw Sharpe, (ii) max drawdown, (iii) FF3 alpha t-stat (using
the CUFE three-factor file with HAC-5 SE).

Reports per index:
  - headline percentile in Sharpe distribution
  - share of grid points beating B&H benchmark on Sharpe at TC=0 / 30 / 50
  - share of grid points with positive FF3 alpha at TC=0 / 30 bps
  - bootstrap 95% CI for headline-vs-bench Sharpe difference

Outputs:
  results/regression_output_oldmodel/table10_strategy_robustness_summary.json
  results/regression_output_oldmodel/table10_strategy_robustness_full.csv
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, json, os, itertools
import statsmodels.api as sm

DATA   = "results/merged_market_sentiment_data_old.csv"
FACTOR = "three_four_five_factor_daily/fivefactor_daily.csv"
OUT_DIR = "results/regression_output_oldmodel"

INDICES = ['CSI300', 'SHCOMP', 'SZCOMP', 'ChiNext', 'CSI500']
W_PHOTO_GRID = [0.0, 0.25, 0.50, 0.75, 0.80, 1.00]
WINDOW_GRID  = [30, 60, 90, 120, 252]
REBAL_GRID   = [1, 3, 5, 10, 21]
QUANTILE_GRID = [(0.10, 0.90), (0.20, 0.80), (0.30, 0.70)]
TC_GRID      = [0.0, 0.0030, 0.0050]   # round-trip costs in decimal
HEADLINE = dict(w_p=0.80, W=90, R=5, Q_LO=0.20, Q_HI=0.80)
TC_HEADLINE = 0.0030


def backtest(bt, w_p, W, R, q_lo, q_hi, tc):
    """Run one backtest. Returns Series of net daily returns (after TC)."""
    bt = bt.copy()
    bt['combined'] = w_p * bt['PhotoPes_std_lag3'] + (1 - w_p) * bt['TextPes_std_lag3']
    bt['roll_lo']  = bt['combined'].rolling(W, min_periods=W).quantile(q_lo)
    bt['roll_hi']  = bt['combined'].rolling(W, min_periods=W).quantile(q_hi)
    bt['raw_sig']  = 0.0
    bt.loc[bt['combined'] > bt['roll_hi'], 'raw_sig'] = -1.0
    bt.loc[bt['combined'] < bt['roll_lo'], 'raw_sig'] =  1.0

    cur, days, pos = 0.0, 0, []
    for i in range(len(bt)):
        if days >= R and bt['raw_sig'].iloc[i] != 0:
            cur = bt['raw_sig'].iloc[i]; days = 0
        days += 1; pos.append(cur)
    bt['position'] = pos
    bt['gross_ret'] = bt['position'].shift(1) * bt['r']

    # Transaction cost: |Δposition| * tc charged on the day of switch
    bt['cost'] = bt['position'].diff().abs().fillna(0) * tc
    bt['net_ret'] = bt['gross_ret'] - bt['cost']
    return bt['net_ret'], bt['position']


def metrics(rets, rf=None):
    """Sharpe is computed on excess returns: (r-rf)/std(r). Vol uses raw r."""
    rets = rets.dropna()
    if len(rets) == 0 or rets.std() == 0:
        return dict(ann_ret=0.0, ann_vol=0.0, sharpe=0.0, max_dd=0.0)
    if rf is not None:
        rf_aligned = rf.reindex(rets.index).fillna(method='ffill').fillna(0.0)
        excess = rets - rf_aligned
    else:
        excess = rets
    ann_ret_excess = excess.mean() * 252
    ann_ret_total  = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sharpe  = ann_ret_excess / ann_vol
    cum     = (1 + rets).cumprod()
    max_dd  = (cum / cum.cummax() - 1).min()
    return dict(ann_ret=ann_ret_total, ann_vol=ann_vol,
                sharpe=sharpe, max_dd=max_dd)


def ff3_alpha(strat_ret_series, dates, fac):
    """Run FF3 regression of strategy excess returns on MKT-RF, SMB, HML."""
    df = pd.DataFrame({'Date': dates, 'strat_ret': strat_ret_series.values}).dropna()
    m = df.merge(fac[['Date', 'mkt_rf', 'smb', 'hml', 'rf']], on='Date', how='inner')
    if len(m) < 100:
        return dict(alpha=np.nan, t_alpha=np.nan, p_alpha=np.nan)
    m['excess'] = m['strat_ret'] - m['rf']
    X = sm.add_constant(m[['mkt_rf', 'smb', 'hml']])
    res = sm.OLS(m['excess'], X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    return dict(alpha=float(res.params['const']),
                t_alpha=float(res.tvalues['const']),
                p_alpha=float(res.pvalues['const']))


def bootstrap_sharpe_diff(strat, bench, rf, n_boot=1000, seed=42):
    """Block-bootstrap excess-Sharpe difference; returns (mean diff, 95% CI).
    Uses 21-day blocks (Politis-Romano-Wolf style)."""
    rng = np.random.default_rng(seed)
    common_idx = strat.index.intersection(bench.index).intersection(rf.index)
    s_arr = strat.reindex(common_idx).values
    b_arr = bench.reindex(common_idx).values
    rf_arr = rf.reindex(common_idx).values
    n = len(common_idx)
    block = 21
    diffs = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block, size=n // block + 1)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        s = s_arr[idx]; b = b_arr[idx]; r = rf_arr[idx]
        s_ex = s - r; b_ex = b - r
        sh_s = s_ex.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0.0
        sh_b = b_ex.mean() / b.std() * np.sqrt(252) if b.std() > 0 else 0.0
        diffs.append(sh_s - sh_b)
    diffs = np.array(diffs)
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def run_one_index(df, fac, idx):
    col_y = f'{idx}_log_returns'
    sub = df[['Date', col_y, 'PhotoPes_std_lag3', 'TextPes_std_lag3']].dropna()
    sub = sub.rename(columns={col_y: 'r'}).set_index('Date').sort_index()

    rf_series = fac.set_index('Date')['rf']
    bench_metrics = metrics(sub['r'], rf=rf_series)

    rows = []
    grid = list(itertools.product(W_PHOTO_GRID, WINDOW_GRID, REBAL_GRID, QUANTILE_GRID))
    for w_p, W, R, (q_lo, q_hi) in grid:
        for tc in TC_GRID:
            net_ret, _ = backtest(sub, w_p, W, R, q_lo, q_hi, tc)
            m = metrics(net_ret, rf=rf_series)
            ff3 = ff3_alpha(net_ret, net_ret.index, fac) if tc in (0.0, 0.0030) else dict(alpha=np.nan, t_alpha=np.nan, p_alpha=np.nan)
            rows.append(dict(
                index=idx, w_p=w_p, W=W, R=R, q_lo=q_lo, q_hi=q_hi, tc=tc,
                ann_ret=m['ann_ret'], ann_vol=m['ann_vol'],
                sharpe=m['sharpe'], max_dd=m['max_dd'],
                alpha=ff3['alpha'], t_alpha=ff3['t_alpha'], p_alpha=ff3['p_alpha'],
                beats_sharpe=int(m['sharpe'] > bench_metrics['sharpe']),
                beats_dd=int(m['max_dd'] > bench_metrics['max_dd']),
                positive_alpha=int(ff3['t_alpha'] > 0) if not np.isnan(ff3['t_alpha']) else 0,
                sig_alpha=int(ff3['t_alpha'] > 1.645) if not np.isnan(ff3['t_alpha']) else 0,
            ))

    headline_net, _ = backtest(sub, **{k: HEADLINE[k] for k in ['w_p', 'W', 'R']},
                                q_lo=HEADLINE['Q_LO'], q_hi=HEADLINE['Q_HI'], tc=0.0)
    headline_net30, _ = backtest(sub, **{k: HEADLINE[k] for k in ['w_p', 'W', 'R']},
                                  q_lo=HEADLINE['Q_LO'], q_hi=HEADLINE['Q_HI'], tc=TC_HEADLINE)
    bench_aligned = sub['r'].loc[headline_net.dropna().index]

    boot_mean, boot_lo, boot_hi = bootstrap_sharpe_diff(headline_net.dropna(), bench_aligned, rf_series)
    boot30_mean, boot30_lo, boot30_hi = bootstrap_sharpe_diff(headline_net30.dropna(), bench_aligned, rf_series)

    return bench_metrics, rows, dict(
        boot_diff_tc0=dict(mean=boot_mean, lo=boot_lo, hi=boot_hi),
        boot_diff_tc30=dict(mean=boot30_mean, lo=boot30_lo, hi=boot30_hi),
    )


def summarize(rows, bench, boot):
    arr = pd.DataFrame(rows)

    def at_tc(tc):
        sub = arr[arr['tc'] == tc]
        return dict(
            n              = int(len(sub)),
            sharpe_median  = float(sub['sharpe'].median()),
            sharpe_q25     = float(sub['sharpe'].quantile(0.25)),
            sharpe_q75     = float(sub['sharpe'].quantile(0.75)),
            sharpe_max     = float(sub['sharpe'].max()),
            pct_beats_sharpe = float(100 * sub['beats_sharpe'].mean()),
            pct_beats_dd     = float(100 * sub['beats_dd'].mean()),
            pct_pos_alpha    = float(100 * sub['positive_alpha'].mean()) if 'positive_alpha' in sub else np.nan,
            pct_sig_alpha    = float(100 * sub['sig_alpha'].mean()) if 'sig_alpha' in sub else np.nan,
        )

    headline_row_tc0  = arr[(arr.w_p == HEADLINE['w_p']) & (arr.W == HEADLINE['W']) & (arr.R == HEADLINE['R'])
                            & (arr.q_lo == HEADLINE['Q_LO']) & (arr.q_hi == HEADLINE['Q_HI']) & (arr.tc == 0.0)].iloc[0]
    headline_row_tc30 = arr[(arr.w_p == HEADLINE['w_p']) & (arr.W == HEADLINE['W']) & (arr.R == HEADLINE['R'])
                            & (arr.q_lo == HEADLINE['Q_LO']) & (arr.q_hi == HEADLINE['Q_HI']) & (arr.tc == 0.0030)].iloc[0]

    pctile_tc0  = (arr[arr.tc == 0.0]['sharpe'] < headline_row_tc0['sharpe']).mean() * 100
    pctile_tc30 = (arr[arr.tc == 0.0030]['sharpe'] < headline_row_tc30['sharpe']).mean() * 100

    return dict(
        bench_sharpe = bench['sharpe'],
        bench_max_dd = bench['max_dd'],
        bench_ann_ret = bench['ann_ret'],
        headline_sharpe_tc0  = float(headline_row_tc0['sharpe']),
        headline_sharpe_tc30 = float(headline_row_tc30['sharpe']),
        headline_pctile_tc0  = float(pctile_tc0),
        headline_pctile_tc30 = float(pctile_tc30),
        headline_alpha_tc0   = float(headline_row_tc0['alpha']),
        headline_t_alpha_tc0 = float(headline_row_tc0['t_alpha']),
        headline_alpha_tc30   = float(headline_row_tc30['alpha']),
        headline_t_alpha_tc30 = float(headline_row_tc30['t_alpha']),
        tc0  = at_tc(0.0),
        tc30 = at_tc(0.0030),
        tc50 = at_tc(0.0050),
        boot = boot,
    )


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA, parse_dates=['Date'])
    df = df[df['Date'] >= '2014-01-01'].copy()
    fac = pd.read_csv(FACTOR, parse_dates=['trddy']).rename(columns={'trddy': 'Date'})

    n_cells = len(W_PHOTO_GRID) * len(WINDOW_GRID) * len(REBAL_GRID) * len(QUANTILE_GRID)
    n_total = n_cells * len(TC_GRID) * len(INDICES)
    print("=" * 100)
    print(f"Robustness Grid: {len(W_PHOTO_GRID)} weights x {len(WINDOW_GRID)} windows x "
          f"{len(REBAL_GRID)} rebal x {len(QUANTILE_GRID)} quantile-pairs = {n_cells} cells/index")
    print(f"Each cell evaluated at TC = {TC_GRID} -> {n_cells * len(TC_GRID)} pts/index")
    print(f"Total: {n_total} backtests across {len(INDICES)} indices")
    print("=" * 100)

    summary = {}
    all_rows = []
    for idx in INDICES:
        print(f"\n--- {idx} ---", flush=True)
        bench, rows, boot = run_one_index(df, fac, idx)
        all_rows.extend(rows)
        s = summarize(rows, bench, boot)
        summary[idx] = s

        print(f"  Bench: Sharpe={bench['sharpe']:+.3f}  AnnRet={bench['ann_ret']*100:+.2f}%  "
              f"MaxDD={bench['max_dd']*100:+.2f}%")
        print(f"  Headline (TC=0):  Sharpe={s['headline_sharpe_tc0']:+.3f}  pctile={s['headline_pctile_tc0']:.0f}%  "
              f"FF3 alpha={s['headline_alpha_tc0']*1e4:+.2f}bps/d (t={s['headline_t_alpha_tc0']:+.2f})")
        print(f"  Headline (TC=30): Sharpe={s['headline_sharpe_tc30']:+.3f}  pctile={s['headline_pctile_tc30']:.0f}%  "
              f"FF3 alpha={s['headline_alpha_tc30']*1e4:+.2f}bps/d (t={s['headline_t_alpha_tc30']:+.2f})")
        print(f"  Bootstrap Sharpe diff (TC=0):  {boot['boot_diff_tc0']['mean']:+.3f}  "
              f"95% CI [{boot['boot_diff_tc0']['lo']:+.3f}, {boot['boot_diff_tc0']['hi']:+.3f}]")
        print(f"  Bootstrap Sharpe diff (TC=30): {boot['boot_diff_tc30']['mean']:+.3f}  "
              f"95% CI [{boot['boot_diff_tc30']['lo']:+.3f}, {boot['boot_diff_tc30']['hi']:+.3f}]")
        for tc_name, tc_val in [('TC=0', 'tc0'), ('TC=30', 'tc30'), ('TC=50', 'tc50')]:
            t = s[tc_val]
            print(f"  Grid @ {tc_name:6s}: med Sharpe={t['sharpe_median']:+.3f}  "
                  f"%>B&H Sharpe={t['pct_beats_sharpe']:5.1f}%  %>B&H DD={t['pct_beats_dd']:5.1f}%  "
                  f"%alpha>0={t['pct_pos_alpha']:5.1f}%  %alpha sig+={t['pct_sig_alpha']:5.1f}%")

    pd.DataFrame(all_rows).to_csv(
        os.path.join(OUT_DIR, 'table10_strategy_robustness_full.csv'), index=False)
    with open(os.path.join(OUT_DIR, 'table10_strategy_robustness_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 100)
    print("HEADLINE TABLE — Panel A: Sharpe-based robustness")
    print("=" * 100)
    print(f"{'Index':8s} {'B&H':>7} {'Hdl(TC0)':>9} {'Hdl(TC30)':>10} "
          f"{'Pct@TC0':>8} {'Pct@TC30':>9} {'%>BH@TC0':>9} {'%>BH@TC30':>10} {'%>BH@TC50':>10}")
    for idx in INDICES:
        s = summary[idx]
        print(f"{idx:8s} {s['bench_sharpe']:+7.3f} {s['headline_sharpe_tc0']:+9.3f} "
              f"{s['headline_sharpe_tc30']:+10.3f} {s['headline_pctile_tc0']:>7.0f}% "
              f"{s['headline_pctile_tc30']:>8.0f}% {s['tc0']['pct_beats_sharpe']:>8.1f}% "
              f"{s['tc30']['pct_beats_sharpe']:>9.1f}% {s['tc50']['pct_beats_sharpe']:>9.1f}%")

    print("\n" + "=" * 100)
    print("HEADLINE TABLE — Panel B: FF3-alpha-based robustness")
    print("=" * 100)
    print(f"{'Index':8s} {'Hdl t-alpha(TC0)':>17} {'Hdl t-alpha(TC30)':>18} "
          f"{'%alpha>0@TC0':>14} {'%alpha>0@TC30':>15} {'%sig@TC0':>10} {'%sig@TC30':>11}")
    for idx in INDICES:
        s = summary[idx]
        print(f"{idx:8s} {s['headline_t_alpha_tc0']:+17.3f} {s['headline_t_alpha_tc30']:+18.3f} "
              f"{s['tc0']['pct_pos_alpha']:>13.1f}% {s['tc30']['pct_pos_alpha']:>14.1f}% "
              f"{s['tc0']['pct_sig_alpha']:>9.1f}% {s['tc30']['pct_sig_alpha']:>10.1f}%")

    print(f"\nSaved JSON: {OUT_DIR}/table10_strategy_robustness_summary.json")
    print(f"Saved CSV : {OUT_DIR}/table10_strategy_robustness_full.csv")
