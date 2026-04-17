"""
paper_table7_ff3.py
Reproduces Table 7: Fama-French 3-Factor Regression of Sentiment Strategy (2014-2026)
Factor data: CUFE fivefactor_daily.csv (uses MKT-RF, SMB, HML)
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import statsmodels.api as sm

DATA   = "results/merged_market_sentiment_data_old.csv"
FACTOR = "three_four_five_factor_daily/fivefactor_daily.csv"

WINDOW = 90; REBAL = 5; W_PHOTO = 0.80; W_TEXT = 0.20; Q_LO = 0.20; Q_HI = 0.80

def build_strategy(df):
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
    bt['position']  = pos
    bt['strat_ret'] = bt['position'].shift(1) * bt['CSI500_log_returns']
    bt['strat_ret'] = bt['strat_ret'].fillna(0)
    return bt[['strat_ret']].reset_index()

def run_ff3(bt, fac):
    merged = bt.merge(fac[['Date','mkt_rf','smb','hml','rf']], on='Date', how='inner')
    merged['excess_ret'] = merged['strat_ret'] - merged['rf']
    X = sm.add_constant(merged[['mkt_rf','smb','hml']])
    res = sm.OLS(merged['excess_ret'], X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    return res, merged

def sig(p):
    return '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else ''))

if __name__ == '__main__':
    df  = pd.read_csv(DATA, parse_dates=['Date'])
    df  = df[df['Date'] >= '2014-01-01'].copy()
    fac = pd.read_csv(FACTOR, parse_dates=['trddy']).rename(columns={'trddy':'Date'})

    bt  = build_strategy(df)
    res, merged = run_ff3(bt, fac)

    print("=" * 60)
    print("Table 7: Three-Factor Regression (2014-2026)")
    print("=" * 60)
    print(f"N = {res.nobs:.0f},  R² = {res.rsquared:.4f},  Adj R² = {res.rsquared_adj:.4f}")
    print()
    print(f"{'Factor':<10} {'Coef':>10} {'t-stat':>10} {'p-value':>10} {'Sig':>4}")
    print("-" * 46)
    for f in ['const','mkt_rf','smb','hml']:
        label = 'Alpha' if f == 'const' else f.upper()
        c = res.params[f]; t = res.tvalues[f]; p = res.pvalues[f]
        print(f"{label:<10} {c:>10.6f} {t:>10.4f} {p:>10.4f} {sig(p):>4}")

    print(f"\nAnnualized alpha: {res.params['const']*252:.4f} ({res.params['const']*252*100:.2f}%)")
    print("\n--- Paper verification ---")
    print(f"Alpha: {res.params['const']:.6f} (paper: 0.000236)")
    print(f"t:     {res.tvalues['const']:.4f} (paper: 0.7237)")
    print(f"p:     {res.pvalues['const']:.4f} (paper: 0.4693)")
    print(f"HML:   {res.params['hml']:.4f} (paper: -0.1026)")
    print(f"HML t: {res.tvalues['hml']:.4f} (paper: -1.7976)")
