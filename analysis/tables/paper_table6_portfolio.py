"""
paper_table6_portfolio.py
Reproduces Table 6: Portfolio Performance Comparison (2014-2026)
Strategy: Combined PhotoPes+TextPes signal on CSI500
Params: window=90, rebal=5, W_photo=0.80, W_text=0.20, Q_lo=0.20, Q_hi=0.80
Sharpe ratios are computed on excess returns (r - rf), where rf is the daily
Chinese risk-free rate (1-year deposit rate) bundled with the CUFE-distributed
Fama-French three-factor file (sf.cufe.edu.cn), for consistency with the FF3
regression in Table 7 (paper_table7_ff3.py).
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd

DATA   = "results/merged_market_sentiment_data_old.csv"
FACTOR = "three_four_five_factor_daily/fivefactor_daily.csv"
CUM_OUT = "results/cumulative_returns_old.csv"

WINDOW = 90; REBAL = 5; W_PHOTO = 0.80; W_TEXT = 0.20; Q_LO = 0.20; Q_HI = 0.80

def load():
    df = pd.read_csv(DATA, parse_dates=['Date'])
    df = df[df['Date'] >= '2014-01-01'].copy()
    fac = pd.read_csv(FACTOR, parse_dates=['trddy']).rename(columns={'trddy': 'Date'})
    df = df.merge(fac[['Date', 'rf']], on='Date', how='left')
    df['rf'] = df['rf'].ffill().fillna(0.0)
    return df

def run_backtest(df):
    bt = df[['Date','CSI500_log_returns','PhotoPes_std_lag3','TextPes_std_lag3','rf']].dropna().copy()
    bt = bt.set_index('Date').sort_index()

    bt['combined'] = W_PHOTO * bt['PhotoPes_std_lag3'] + W_TEXT * bt['TextPes_std_lag3']
    bt['roll_lo']  = bt['combined'].rolling(WINDOW, min_periods=WINDOW).quantile(Q_LO)
    bt['roll_hi']  = bt['combined'].rolling(WINDOW, min_periods=WINDOW).quantile(Q_HI)
    bt['raw_sig']  = 0.0
    bt.loc[bt['combined'] > bt['roll_hi'], 'raw_sig'] = -1.0
    bt.loc[bt['combined'] < bt['roll_lo'], 'raw_sig'] =  1.0

    cur = 0.0; days = 0; pos = []
    for i in range(len(bt)):
        if days >= REBAL and bt['raw_sig'].iloc[i] != 0:
            cur = bt['raw_sig'].iloc[i]; days = 0
        days += 1; pos.append(cur)

    bt['position']   = pos
    bt['strat_ret']  = bt['position'].shift(1) * bt['CSI500_log_returns']
    bt['strat_ret']  = bt['strat_ret'].fillna(0)
    bt['strat_cum']  = (1 + bt['strat_ret']).cumprod()
    bt['bench_cum']  = (1 + bt['CSI500_log_returns']).cumprod()
    return bt

def calc_metrics(rets, label, rf=None):
    """Sharpe is computed on excess returns: (r - rf) / std(r). Vol uses raw r."""
    excess = rets - rf if rf is not None else rets
    ann_ret_excess = excess.mean() * 252
    ann_ret_total  = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sharpe  = ann_ret_excess / ann_vol if ann_vol > 0 else 0
    cum     = (1 + rets).cumprod()
    max_dd  = (cum / cum.cummax() - 1).min()
    return {'Label': label,
            'Ann Return': f'{ann_ret_total:.2%}',
            'Ann Vol': f'{ann_vol:.2%}',
            'Sharpe (excess)': f'{sharpe:.3f}',
            'Max DD': f'{max_dd:.2%}',
            '_ann_ret': ann_ret_total,
            '_ann_ret_excess': ann_ret_excess,
            '_ann_vol': ann_vol,
            '_sharpe': sharpe,
            '_max_dd': max_dd}

if __name__ == '__main__':
    df = load()
    bt = run_backtest(df)

    rf_aligned = bt['rf'].reindex(bt['strat_ret'].dropna().index)
    strat_m = calc_metrics(bt['strat_ret'].dropna(),         'Sentiment Strategy', rf=rf_aligned)
    bench_m = calc_metrics(bt['CSI500_log_returns'].dropna(), 'CSI 500 Index',
                           rf=bt['rf'].reindex(bt['CSI500_log_returns'].dropna().index))

    print("=" * 60)
    print("Table 6: Performance Comparison (2014-2026)")
    print("=" * 60)
    perf = pd.DataFrame([strat_m, bench_m]).set_index('Label')
    print(perf[['Ann Return','Ann Vol','Sharpe (excess)','Max DD']].to_string())
    print(f"\nFinal cumulative value:")
    print(f"  Strategy:  {bt['strat_cum'].iloc[-1]:.4f}")
    print(f"  Benchmark: {bt['bench_cum'].iloc[-1]:.4f}")

    # Save cumulative returns for figure
    cum_df = bt[['strat_cum','bench_cum']].reset_index()
    cum_df.columns = ['Date','strategy_cum','benchmark_cum']
    cum_df.to_csv(CUM_OUT, index=False)
    print(f"\nSaved cumulative returns: {CUM_OUT}")

    # Print summary; paper values updated to excess-Sharpe definition
    print("\n--- Numerical output ---")
    print(f"Strategy AnnRet={strat_m['Ann Return']}  AnnVol={strat_m['Ann Vol']}  "
          f"Excess Sharpe={strat_m['Sharpe (excess)']}  MaxDD={strat_m['Max DD']}")
    print(f"Bench    AnnRet={bench_m['Ann Return']}  AnnVol={bench_m['Ann Vol']}  "
          f"Excess Sharpe={bench_m['Sharpe (excess)']}  MaxDD={bench_m['Max DD']}")
