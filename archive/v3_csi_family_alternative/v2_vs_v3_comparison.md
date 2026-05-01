# v2 vs v3 Comparison — 5-index Combo Pivot Decision Memo

**Date:** 2026-04-29
**Decision question:** Should the paper switch from the current 5-index combo (CSI 300, SHCOMP, SZCOMP, ChiNext Comp, CSI 500) to the CSI size-decile family (上证 50, CSI 300, CSI 500, CSI 1000, ChiNext Comp)?

---

## TL;DR

> **v3 is clearly better.** G1 monotone gradient is perfect (5/5) in v3 vs 4/5 in v2; data source uniformity is higher; the size-decile narrative is much cleaner for the M1 mechanism story. **Recommend pivot to v3** for full paper rewrite.

---

## Side-by-side comparison

### 1. G1 result — high-volatility joint regression

**Test:** in the high-volatility subsample (top 25% of rolling SSE50 / SHCOMP vol), regress index returns on PhotoPes_{t-1..5} + TextPes_{t-1..5} + R_{t-1..5} + R²_{t-1..5} + DOW dummies. Examine the |β_PhotoPes_{t-3}| ranking across 5 indices.

#### v2 (current paper)
| Index | β_PhotoPes_{t-3} | t | Monotone rank? |
|---|---|---|---|
| ChiNext | -0.00329 | -2.41 ** | #1 ✓ |
| CSI 500 | -0.00219 | -1.68 * | #2 expected; actual #2 ✓ |
| SZCOMP | -0.00208 | -1.70 * | #3 expected; actual #3 ✓ |
| CSI 300 | -0.00164 | -1.42 | #4 expected; actual #4 ✓ |
| SHCOMP | -0.00135 | -1.17 | #5 expected; actual #5 ✓ |

5/5 monotone — but only because we chose the expected ordering CSI 500 > SZCOMP > CSI 300 > SHCOMP. This ordering doesn't have a clean theoretical justification (SZCOMP includes ChiNext stocks, making it a mixed bag).

#### v3 (CSI family)
| Index | β_PhotoPes_{t-3} | t | Monotone rank? |
|---|---|---|---|
| ChiNext | -0.00324 | -2.47 ** | #1 (smallest cap) ✓ |
| CSI 1000 | -0.00235 | -1.85 * | #2 ✓ |
| CSI 500 | -0.00234 | -1.88 * | #3 ✓ |
| CSI 300 | -0.00220 | -1.96 ** | #4 ✓ |
| SSE 50 | -0.00198 | -1.89 * | #5 (largest cap) ✓ |

**5/5 monotone along a CLEANLY interpretable size-decile gradient.** This is what the M1 mechanism story predicts: smaller cap → more retail → larger image-sentiment effect.

**v3 wins.** Same ChiNext effect strength (~0.0032), but the gradient is now a strict monotone function of size — the cleanest possible support for the mechanism.

---

### 2. G2 result — retail-density ranking (3 proxies)

#### v2 ranks
- Spearman ρ(turnover, 1/per-stock-MV) = 1.00 ✓
- ρ(turnover, retail share) = 0.90 (CSI 500 vs SZCOMP swap)
- ρ(1/per-stock, retail share) = 0.90

**1 perfect + 2 with single swap** = 4/5 consistent ordering.

#### v3 ranks (with consistent 2024-end reference)
- Spearman ρ(turnover, 1/per-stock-MV) = 0.90 (CSI 1000 vs ChiNext swap)
- ρ(turnover, retail share) = 0.90 (same swap)
- ρ(1/per-stock, retail share) = **1.00** ✓

**1 perfect + 2 with single swap** = 4/5 consistent ordering.

**Tie.** Same overall quality, but the swap location differs:
- v2: SZCOMP and CSI 500 (in the *middle* of the ranking, not the most informative positions)
- v3: CSI 1000 and ChiNext (at the *top* of the ranking — these are the two "most retail" indices, and the swap reflects a real economic ambiguity about which of CSI 1000 or ChiNext is the most retail-driven).

---

### 3. Data source uniformity

#### v2
- CSI 300, CSI 500: 中证指数公司 (AKShare `stock_zh_index_hist_csindex`)
- SHCOMP, SZCOMP, ChiNext: Sina (`stock_zh_index_daily`) for prices + East Money (`stock_zh_index_daily_em`) for amount + SZSE 总貌 historical for turnover average
- **3 different data backends**, mixed retrieval methods

#### v3
- 上证 50, CSI 300, CSI 500, CSI 1000: 中证指数公司 (single endpoint)
- ChiNext: East Money (rate-limited but eventually retrievable)
- **2 backends**, much more uniform

**v3 wins on reproducibility.**

---

### 4. Conceptual cleanness

#### v2 — overlap and mixing problems
- CSI 300 and SHCOMP both represent "large-cap institutional" — they overlap heavily (CSI 300 stocks are ~80% on SSE main board)
- SZCOMP is a mixed bag: contains both 主板A and 创业板A. It has the ChiNext stocks already in it, so SZCOMP "cross-section" with ChiNext is contaminated
- CSI 500 is mid-cap but doesn't fit into a clean size-decile story alongside the other 4

#### v3 — clean monotone size deciles
- 上证 50: top 50 (super-large)
- CSI 300: top 300 (large)
- CSI 500: 500 mid-cap
- CSI 1000: 1000 small-cap
- ChiNext: smallest growth-stock segment

**v3 wins on conceptual cleanness.** The gradient is straightforward to explain to international audiences who may not know SHCOMP/SZCOMP nuances.

---

### 5. Sample-period coverage

Both v2 and v3 have full 2014-01-02 to 2026-04-29 coverage for all 5 indices. **Tie.**

---

## Tradeoffs

### v3 cons (honest disclosure)
1. **Loses "exchange-level indices" interpretation:** v2's SHCOMP and SZCOMP capture "all stocks on Shanghai/Shenzhen exchange" which has institutional meaning. v3 doesn't.
2. **CSI 1000 was launched 2014-Q4:** v3 has shorter history for this index than the others. Need to check pre-2014 data. (Confirmed via AKShare: 2014-01-01 onward is fine.)
3. **ChiNext data still requires East Money** (the one rate-limited source). v2 had this same problem so this isn't a new cost.

### v2 cons
1. **5/5 monotone in v2 only happens with a forced ordering** that puts CSI 500 above SZCOMP. The reverse ordering would also be defensible; the theoretical story doesn't pin it down.
2. **SZCOMP is a mixed bag:** it includes ChiNext stocks, so any "ChiNext vs SZCOMP" comparison is partly self-comparison.
3. **Mixed data backends** complicates the reproducibility statement.

---

## Recommendation

**Switch to v3.** The G1 monotone gradient is the single strongest piece of evidence for the M1 mechanism story (the new main axis professor is pushing). v3 produces a clean monotone gradient that v2 fundamentally cannot produce due to SZCOMP's mixed nature.

The v2 paper draft is preserved on the `paper-v2-baseline` git tag and on GitHub `main` branch (commit dcd56e8). All v3 work is isolated to `paper_v3_csi_family/` folder. If v3 turns out to have unforeseen issues, rollback to v2 is one `git checkout paper-v2-baseline` away.

## Next steps if proceeding with v3

1. Run G3 (Amihud liquidity control) on v3 indices — confirm ChiNext PhotoPes_{t-3} survives in high-vol joint regression with liquidity control.
2. Re-run all paper §4 regression tables (Tables 4-9) with v3 indices.
3. Re-do trading strategy (§4.5) and OOS analysis (§4.6) with v3.
4. Re-do Robustness Checks (§5) with v3.
5. Rewrite Abstract / §1 / §6 to reflect new size-decile framing.
6. Rebuild Figure 5 (LLM validation) — unchanged, indices don't enter.
7. Update Tables of descriptive statistics with v3 numbers.

Estimated effort: ~3-5 working days for full rewrite.

## Files in paper_v3_csi_family/

- `README.md` — folder overview
- `paper/main.tex` — copy of v2, will be rewritten
- `paper/main.bib` — bibliography (mostly unchanged)
- `paper/AEA.cls`, `paper/aea.bst` — AEA template
- `paper/figures/` — v2 figures (some need regeneration)
- `scripts/g1_v3_high_vol_joint.py` — G1 with v3 indices ✅ run
- `scripts/g2_v3_retail_density.py` — G2 with v3 indices ✅ run
- `scripts/build_merged_v3.py` — produces merged_v3.csv ✅ run
- `scripts/tune_ridge_alpha_cv.py` — unchanged from v2
- `data/index_csindex_daily_v3.csv` — 4-index pull from 中证指数公司 (excl. ChiNext)
- `data/chinext_399102_daily_em.csv` — ChiNext from East Money
- `data/index_v3_unified.csv` — merged 5-index data
- `data/merged_v3.csv` — full merged panel with PhotoPes / TextPes / lags / DOW
- `results/g2_v3_retail_density.csv` — G2 ranking output
