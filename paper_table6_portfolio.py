"""
paper_table6_portfolio.py
Reproduces Table 6: Portfolio Performance Comparison (2014-2026)
Strategy: Combined PhotoPes+TextPes signal on CSI500
Params: window=90, rebal=5, W_photo=0.80, W_text=0.20, Q_lo=0.20, Q_hi=0.80
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd

DATA = "results/merged_market_sentiment_data_old.csv"
CUM_OUT = "results/cumulative_returns_old.csv"

WINDOW = 90; REBAL = 5; W_PHOTO = 0.80; W_TEXT = 0.20; Q_LO = 0.20; Q_HI = 0.80

def load():
    df = pd.read_csv(DATA, parse_dates=['Date'])
    return df[df['Date'] >= '2014-01-01'].copy()

def run_backtest(df):
    bt = df[['Date','CSI500_log_returns','PhotoPes_std_lag3','TextPes_std_lag3']].dropna().copy()
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

def calc_metrics(rets, label):
    ann_ret = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
    cum     = (1 + rets).cumprod()
    max_dd  = (cum / cum.cummax() - 1).min()
    return {'Label': label, 'Ann Return': f'{ann_ret:.2%}',
            'Ann Vol': f'{ann_vol:.2%}', 'Sharpe': f'{sharpe:.3f}',
            'Max DD': f'{max_dd:.2%}',
            '_ann_ret': ann_ret, '_ann_vol': ann_vol,
            '_sharpe': sharpe, '_max_dd': max_dd}

if __name__ == '__main__':
    df = load()
    bt = run_backtest(df)

    strat_m = calc_metrics(bt['strat_ret'].dropna(),        'Sentiment Strategy')
    bench_m = calc_metrics(bt['CSI500_log_returns'].dropna(),'CSI 500 Index')

    print("=" * 60)
    print("Table 6: Performance Comparison (2014-2026)")
    print("=" * 60)
    perf = pd.DataFrame([strat_m, bench_m]).set_index('Label')
    print(perf[['Ann Return','Ann Vol','Sharpe','Max DD']].to_string())
    print(f"\nFinal cumulative value:")
    print(f"  Strategy:  {bt['strat_cum'].iloc[-1]:.4f}")
    print(f"  Benchmark: {bt['bench_cum'].iloc[-1]:.4f}")

    # Save cumulative returns for figure
    cum_df = bt[['strat_cum','bench_cum']].reset_index()
    cum_df.columns = ['Date','strategy_cum','benchmark_cum']
    cum_df.to_csv(CUM_OUT, index=False)
    print(f"\nSaved cumulative returns: {CUM_OUT}")

    # Verify paper values
    print("\n--- Paper verification ---")
    print(f"Ann Return: {strat_m['Ann Return']} (paper: 9.03%)")
    print(f"Ann Vol:    {strat_m['Ann Vol']} (paper: 24.86%)")
    print(f"Sharpe:     {strat_m['Sharpe']} (paper: 0.363)")
    print(f"Max DD:     {strat_m['Max DD']} (paper: -62.22%)")
    print(f"Bench Ann:  {bench_m['Ann Return']} (paper: 6.23%)")
    print(f"Bench DD:   {bench_m['Max DD']} (paper: -71.11%)")
