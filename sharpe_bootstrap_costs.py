"""
sharpe_bootstrap_costs.py
=========================
Two referee-mandated robustness exercises for the trading strategy in §4.5.

(1) Stationary block-bootstrap (Politis-Romano 1994) of the Sharpe-ratio
    difference between the sentiment strategy and the CSI 500 benchmark.
    Reports the bootstrap distribution of  Delta_SR = SR_strat - SR_bench
    plus a 95% percentile CI and a one-sided bootstrap p-value for
    H0: Delta_SR <= 0.

    Block length = 5 trading days (matches the 5-day rebalance).
    B = 10,000 replications.  Random seed = 42.

    Cross-check: parametric Jobson-Korkie (1981) test with Memmel (2003)
    correction.

(2) Transaction-cost grid for round-trip costs in {0, 10, 20, 30, 40} bps.
    For each cost level we recompute the strategy net return on every
    rebalancing day, then recompute the strategy Sharpe and the cumulative
    final value of an investor starting with 1 RMB at 2014-01-06.

    The break-even cost (where strategy excess Sharpe matches the
    benchmark) is reported as the headline interpretation of "is the
    strategy economically alive after frictions?"

Outputs:
   results/sharpe_bootstrap.csv     (bootstrap distribution summary)
   results/sharpe_costs_grid.csv    (Sharpe and cum value at each cost level)
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import norm

np.random.seed(42)

DATA = "results/merged_market_sentiment_data_old.csv"
FACTOR = "three_four_five_factor_daily/fivefactor_daily.csv"
OUTPUT_DIR = "results"

WINDOW = 90; REBAL = 5; W_PHOTO = 0.80; W_TEXT = 0.20; Q_LO = 0.20; Q_HI = 0.80


def load_data():
    df = pd.read_csv(DATA, parse_dates=["Date"])
    df = df[df["Date"] >= "2014-01-01"].copy()
    fac = pd.read_csv(FACTOR, parse_dates=["trddy"]).rename(columns={"trddy": "Date"})
    df = df.merge(fac[["Date", "rf"]], on="Date", how="left")
    df["rf"] = df["rf"].ffill().fillna(0.0)
    return df


def run_backtest(df, cost_round_trip_bps=0.0):
    """Run the long-flat-short strategy. cost_round_trip_bps is charged on every position change."""
    bt = df[["Date", "CSI500_log_returns", "PhotoPes_std_lag3", "TextPes_std_lag3", "rf"]].dropna().copy()
    bt = bt.set_index("Date").sort_index()

    bt["combined"] = W_PHOTO * bt["PhotoPes_std_lag3"] + W_TEXT * bt["TextPes_std_lag3"]
    bt["roll_lo"] = bt["combined"].rolling(WINDOW, min_periods=WINDOW).quantile(Q_LO)
    bt["roll_hi"] = bt["combined"].rolling(WINDOW, min_periods=WINDOW).quantile(Q_HI)
    bt["raw_sig"] = 0.0
    bt.loc[bt["combined"] > bt["roll_hi"], "raw_sig"] = -1.0
    bt.loc[bt["combined"] < bt["roll_lo"], "raw_sig"] = 1.0

    cur, days, pos, prev = 0.0, 0, [], 0.0
    n_position_changes = 0
    for i in range(len(bt)):
        if days >= REBAL and bt["raw_sig"].iloc[i] != 0:
            new_cur = bt["raw_sig"].iloc[i]
            if new_cur != cur:
                n_position_changes += 1
            cur, days = new_cur, 0
        days += 1
        pos.append(cur)

    bt["position"] = pos
    bt["pos_change"] = bt["position"].diff().abs().fillna(0)
    # one-way cost in bps -> daily fractional cost on each change. round-trip = 2 * one-way.
    one_way_cost = (cost_round_trip_bps / 2.0) / 1e4  # convert bps to fraction
    bt["cost_today"] = bt["pos_change"] * one_way_cost
    bt["strat_ret_gross"] = bt["position"].shift(1) * bt["CSI500_log_returns"]
    bt["strat_ret_gross"] = bt["strat_ret_gross"].fillna(0)
    bt["strat_ret_net"] = bt["strat_ret_gross"] - bt["cost_today"]
    bt["strat_cum"] = (1 + bt["strat_ret_net"]).cumprod()
    bt["bench_cum"] = (1 + bt["CSI500_log_returns"]).cumprod()
    return bt


def sharpe_excess(rets, rf):
    excess = rets - rf
    if excess.std() == 0:
        return 0.0
    return excess.mean() / rets.std() * np.sqrt(252)


def stationary_bootstrap_indices(n, block_length, rng):
    """Politis-Romano (1994) stationary bootstrap: block lengths are i.i.d. Geometric(1/L)."""
    p = 1.0 / block_length
    indices = np.empty(n, dtype=int)
    indices[0] = rng.integers(0, n)
    for t in range(1, n):
        if rng.random() < p:
            indices[t] = rng.integers(0, n)
        else:
            indices[t] = (indices[t - 1] + 1) % n
    return indices


def jobson_korkie_memmel(r1, r2, rf):
    """
    Jobson-Korkie (1981) Sharpe difference test with Memmel (2003) correction.
    Tests H0: SR1 = SR2, two-sided, asymptotic normal under null.
    """
    e1 = r1 - rf
    e2 = r2 - rf
    mu1, mu2 = e1.mean(), e2.mean()
    s1, s2 = r1.std(ddof=1), r2.std(ddof=1)
    sr1, sr2 = mu1 / s1, mu2 / s2

    n = len(r1)
    cov = np.cov(r1, r2, ddof=1)
    s12 = cov[0, 1]

    # Memmel (2003) variance of (sr1 - sr2):
    var_diff = (1.0 / n) * (
        2 * (1 - s12 / (s1 * s2))
        + 0.5 * (sr1 ** 2 + sr2 ** 2 - 2 * sr1 * sr2 * (s12 / (s1 * s2)) ** 2)
    )
    z = (sr1 - sr2) / np.sqrt(var_diff)
    p_two = 2 * (1 - norm.cdf(abs(z)))
    return {"SR1_daily": sr1, "SR2_daily": sr2, "diff_daily": sr1 - sr2,
            "z": z, "p_two_sided": p_two}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()

    print("=" * 78)
    print("(1) Stationary block-bootstrap of Delta_SR (strategy minus benchmark)")
    print("=" * 78)

    bt = run_backtest(df, cost_round_trip_bps=0.0)
    strat = bt["strat_ret_net"].dropna().values
    bench = bt["CSI500_log_returns"].reindex(bt["strat_ret_net"].index).fillna(0).values
    rf = bt["rf"].reindex(bt["strat_ret_net"].index).fillna(0).values

    sr_strat = sharpe_excess(strat, rf)
    sr_bench = sharpe_excess(bench, rf)
    delta_obs = sr_strat - sr_bench
    print(f"Observed annualized Sharpe (excess):  strategy = {sr_strat:.4f},  benchmark = {sr_bench:.4f}")
    print(f"Observed Delta_SR = {delta_obs:.4f}")
    print(f"\nBootstrap setup: block length = {REBAL} (matches rebalance freq), B = 10,000")

    rng = np.random.default_rng(seed=42)
    B = 10_000
    boot_deltas = np.empty(B)
    n = len(strat)
    for b in range(B):
        idx = stationary_bootstrap_indices(n, REBAL, rng)
        sb_strat = sharpe_excess(strat[idx], rf[idx])
        sb_bench = sharpe_excess(bench[idx], rf[idx])
        boot_deltas[b] = sb_strat - sb_bench

    centered = boot_deltas - boot_deltas.mean()
    p_one_sided = np.mean(centered + delta_obs <= 0)
    ci_lo, ci_hi = np.percentile(boot_deltas, [2.5, 97.5])

    print(f"\nBootstrap distribution of Delta_SR:")
    print(f"  mean         = {boot_deltas.mean():.4f}")
    print(f"  std          = {boot_deltas.std():.4f}")
    print(f"  95% percentile CI = [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  One-sided p-value (H0: Delta_SR <= 0) = {p_one_sided:.4f}")

    pd.DataFrame({"delta_sr": boot_deltas}).to_csv(
        os.path.join(OUTPUT_DIR, "sharpe_bootstrap.csv"), index=False)

    # Parametric cross-check
    jk = jobson_korkie_memmel(pd.Series(strat), pd.Series(bench), pd.Series(rf))
    print(f"\nJobson-Korkie / Memmel cross-check (parametric, normal-asymptotic):")
    print(f"  z = {jk['z']:.3f},  two-sided p = {jk['p_two_sided']:.4f}")

    print("\n" + "=" * 78)
    print("(2) Transaction-cost grid: round-trip costs in {0, 10, 20, 30, 40} bps")
    print("=" * 78)
    print(f"{'cost (bps r/t)':<18} {'Strat AnnRet':>14} {'Strat AnnVol':>14} {'Strat Sharpe':>14} {'Strat CumVal':>14}")
    print("-" * 78)
    cost_rows = []
    for c in [0, 10, 20, 30, 40, 50]:
        bt_c = run_backtest(df, cost_round_trip_bps=c)
        sr = bt_c["strat_ret_net"].dropna()
        sr_aligned = bt_c["rf"].reindex(sr.index)
        ann_ret = sr.mean() * 252
        ann_vol = sr.std() * np.sqrt(252)
        sharpe = sharpe_excess(sr.values, sr_aligned.values)
        cum_val = bt_c["strat_cum"].iloc[-1]
        bench_sharpe = sharpe_excess(bench, rf)
        cost_rows.append({"cost_bps_round_trip": c, "ann_ret": ann_ret, "ann_vol": ann_vol,
                          "sharpe_excess": sharpe, "cum_value": cum_val,
                          "bench_sharpe": bench_sharpe})
        print(f"{c:<18d} {ann_ret:>13.2%} {ann_vol:>13.2%} {sharpe:>13.4f} {cum_val:>13.4f}")

    print(f"\nBenchmark CSI 500 Sharpe (excess) = {sharpe_excess(bench, rf):.4f}")
    print(f"Benchmark cumulative value         = {bt['bench_cum'].iloc[-1]:.4f}")
    pd.DataFrame(cost_rows).to_csv(os.path.join(OUTPUT_DIR, "sharpe_costs_grid.csv"), index=False)


if __name__ == "__main__":
    main()
