"""
r7_trailing_momentum_control.py
================================
Round-7 robustness: does the headline ChiNext PhotoPes_{t-3} coefficient
survive controls for trailing 5/10/20/60-day cumulative-return signals?

Background:
- The headline regression (Table 4 / tab:PhotoPes_regression_results) controls
  for L5(R) and L5(R^2) -- lags 1 to 5 of daily returns and squared returns.
- Referee 1 (substance) v3 MC5b flagged that ChiNext is the most reversal-prone
  Chinese index, and t-3 is exactly where short-window reversal mechanically
  loads. The L5(R) controls do not absorb trailing 10-day, 20-day, 60-day
  cumulative-return signals.
- Three rounds of the substance referee have flagged this as the single most
  likely sub-tier referee blocker. The fix is a short table that puts trailing
  cumulative ChiNext returns on the right-hand side and shows whether
  PhotoPes_{t-3} survives.

Specification (one column per control set; same dependent variable
ChiNext_log_returns_t throughout):

  Col (1): Headline -- L5(PhotoPes), L5(R), L5(R^2), DOW dummies, constant
  Col (2): Add trailing 5-day cumulative ChiNext return: r_{t-7,t-3}
  Col (3): Add trailing 10-day cumulative: r_{t-12,t-3}
  Col (4): Add trailing 20-day cumulative: r_{t-22,t-3}
  Col (5): Add trailing 60-day cumulative: r_{t-62,t-3}
  Col (6): Add ALL FOUR trailing-cum controls jointly

The trailing window is anchored at t-3 (matched to the lag at which PhotoPes
is loaded in the headline) and extends backward by 5/10/20/60 trading days.
We then ask: does the PhotoPes_{t-3} coefficient (the headline) retain its
sign and significance after adding these controls?

The contemporaneous correlation of PhotoPes_{t-3} with R_{t-3} (R1 MC5a) is
also reported as a sanity check.

Output: results/r7_trailing_momentum_control.csv
        results/r7_trailing_momentum_control.txt
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = "results/merged_market_sentiment_data_old.csv"
OUTPUT_DIR = "results"
NW_LAGS = 5


def load_data():
    df = pd.read_csv(DATA, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df[df["Date"] >= "2014-01-01"].copy()
    return df


def add_trailing_returns(df, ret_col="ChiNext_log_returns"):
    """For each window w in {5, 10, 20, 60}, compute the cumulative ChiNext
    log-return over the window ending at t-3 inclusive, i.e., sum of
    r_{t-3-w+1}, ..., r_{t-3}. We do NOT include r_{t-2}, r_{t-1}, r_t in
    the trailing window because the L5(R) controls already span lags 1-5.
    """
    r = df[ret_col].copy()
    # rolling sum over w days, then shift by 3 so that the window ENDS at t-3
    for w in [5, 10, 20, 60]:
        df[f"trail_cum_{w}d"] = r.rolling(w, min_periods=w).sum().shift(3)
    return df


def build_design(df, idx_ret_col, photopes_lags=range(1, 6),
                 r_lags=range(1, 6), r_sq_lags=range(1, 6),
                 trail_cols=None):
    """Build (y, X) for the regression.
       y = idx_ret_col
       X = constant + PhotoPes lags + R lags + R^2 lags + DOW dummies + trailing
    """
    sub = df.copy()
    cols = []
    for L in photopes_lags:
        cols.append(f"PhotoPes_std_lag{L}")
    for L in r_lags:
        cols.append(f"{idx_ret_col}_lag{L}")
    for L in r_sq_lags:
        cols.append(f"{idx_ret_col}_sq_lag{L}")
    for d in ["weekday_Tue", "weekday_Wed", "weekday_Thu", "weekday_Fri"]:
        cols.append(d)
    if trail_cols:
        cols.extend(trail_cols)

    sub = sub.dropna(subset=cols + [idx_ret_col]).copy()
    y = sub[idx_ret_col].values
    X = sub[cols].values
    X = sm.add_constant(X)
    return y, X, ["const"] + cols, sub


def run_nw(y, X, lags=NW_LAGS):
    """OLS with Newey-West HAC SE."""
    model = sm.OLS(y, X)
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return res


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()
    df = add_trailing_returns(df, ret_col="ChiNext_log_returns")

    # Sanity check: correlation of PhotoPes_{t-3} with R_{t-3} (R1 MC5a)
    sub_chk = df.dropna(subset=["PhotoPes_std_lag3", "ChiNext_log_returns_lag3"])
    rho = np.corrcoef(sub_chk["PhotoPes_std_lag3"], sub_chk["ChiNext_log_returns_lag3"])[0, 1]
    print("=" * 78)
    print("Round-7 trailing-momentum control: ChiNext PhotoPes_{t-3} robustness")
    print("=" * 78)
    print(f"Sanity check (R1 MC5a): contemporaneous correlation of")
    print(f"  PhotoPes_std_lag3 with ChiNext_log_returns_lag3: rho = {rho:+.4f}")
    print(f"  (n = {len(sub_chk)})")
    print(f"  -> {'small (no mechanical loading)' if abs(rho) < 0.10 else 'meaningful (potential mechanical loading)'}")
    print()

    specs = [
        ("(1) Headline", []),
        ("(2) +trail 5d", ["trail_cum_5d"]),
        ("(3) +trail 10d", ["trail_cum_10d"]),
        ("(4) +trail 20d", ["trail_cum_20d"]),
        ("(5) +trail 60d", ["trail_cum_60d"]),
        ("(6) +ALL trail", ["trail_cum_5d", "trail_cum_10d", "trail_cum_20d", "trail_cum_60d"]),
    ]

    rows = []
    print(f"{'Spec':<20} {'N':>5} {'beta(Photo_t-3)':>16} {'NW(5) t':>10} {'p':>8} {'beta(trail)':>14} {'t(trail)':>10}")
    print("-" * 90)

    for name, trail in specs:
        y, X, names, sub = build_design(df, "ChiNext_log_returns", trail_cols=trail)
        res = run_nw(y, X, lags=NW_LAGS)
        # locate PhotoPes_std_lag3 coeff
        idx_p3 = names.index("PhotoPes_std_lag3")
        b = res.params[idx_p3]
        t = res.tvalues[idx_p3]
        p = res.pvalues[idx_p3]
        # locate ALL trailing coeffs (report first if any)
        trail_b, trail_t = "", ""
        if trail:
            cohorts = []
            for tc in trail:
                idx_tc = names.index(tc)
                cohorts.append((tc, res.params[idx_tc], res.tvalues[idx_tc], res.pvalues[idx_tc]))
            # report only the most aggressive (longest-window) for compact display
            trail_b = f"{cohorts[-1][1]:+.5f}"
            trail_t = f"{cohorts[-1][2]:+.3f}"
            row = {
                "spec": name, "N": len(y),
                "beta_photopes_lag3": b, "t_photopes_lag3": t, "p_photopes_lag3": p,
                "trail_cols": "+".join(trail),
                "beta_trail_last": cohorts[-1][1], "t_trail_last": cohorts[-1][2],
                "trail_detail": str([(c[0], round(c[1], 5), round(c[2], 3)) for c in cohorts]),
                "adj_r2": res.rsquared_adj,
            }
        else:
            row = {
                "spec": name, "N": len(y),
                "beta_photopes_lag3": b, "t_photopes_lag3": t, "p_photopes_lag3": p,
                "trail_cols": "", "beta_trail_last": "", "t_trail_last": "",
                "trail_detail": "", "adj_r2": res.rsquared_adj,
            }
        rows.append(row)
        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
        print(f"{name:<20} {len(y):>5d} {b:>+16.6f}{sig:<3} {t:>+10.3f} {p:>8.4f} {trail_b:>14} {trail_t:>10}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUT_DIR, "r7_trailing_momentum_control.csv"), index=False)

    # Detailed report (write into txt)
    print()
    print("Detailed coefficients on each trailing-window control:")
    print("-" * 78)
    detail_lines = []
    for r in rows:
        if r["trail_detail"]:
            detail_lines.append(f"{r['spec']}: {r['trail_detail']}")
            print(f"{r['spec']}: {r['trail_detail']}")

    with open(os.path.join(OUTPUT_DIR, "r7_trailing_momentum_control.txt"), "w") as f:
        f.write("Round-7 trailing-momentum control table\n")
        f.write("=" * 78 + "\n")
        f.write(f"Dependent: ChiNext_log_returns_t\n")
        f.write(f"Newey-West HAC standard errors with {NW_LAGS} lags\n")
        f.write(f"Trailing window anchored at t-3 (sum of r_{{t-3-w+1}}..r_{{t-3}})\n\n")
        f.write(f"R1 MC5a sanity check: corr(PhotoPes_t-3, R_t-3) = {rho:+.4f}\n\n")
        f.write(out.to_string(index=False) + "\n\n")
        f.write("Trailing-coefficient detail:\n")
        for line in detail_lines:
            f.write(line + "\n")

    print()
    print("Interpretation:")
    n_specs = len(rows)
    base_b, base_t, base_p = rows[0]["beta_photopes_lag3"], rows[0]["t_photopes_lag3"], rows[0]["p_photopes_lag3"]
    surv = []
    for r in rows[1:]:
        if r["p_photopes_lag3"] < 0.05:
            surv.append(r["spec"])
    print(f"Headline coef: beta = {base_b:+.5f}, NW(5) t = {base_t:+.3f}, p = {base_p:.4f}")
    print(f"Specs at which PhotoPes_t-3 retains 5%-significance after trailing control: {len(surv)}/{n_specs-1}")
    print(f"Specs that survive: {surv}")


if __name__ == "__main__":
    main()
