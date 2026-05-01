# Paper v3 — CSI 市值切片家族 (Option A)

**Status:** 🟡 In progress (data pulling phase)
**Started:** 2026-04-29
**Baseline:** Forked from `paper-v2-baseline` git tag (commit dcd56e8)

## What's different from v2 (current published paper)

The 5-index combo changes to a clean monotone CSI市值切片家族:

| Slot | v2 (current) | v3 (CSI family) |
|---|---|---|
| 1 (largest cap, most institutional) | CSI 300 | **上证 50** (000016) |
| 2 | SHCOMP (上证综指) | CSI 300 (000300) |
| 3 | SZCOMP (深证综指) | CSI 500 (000905) |
| 4 | ChiNext (创业板综指) | **中证 1000** (000852) |
| 5 (smallest cap, most retail) | CSI 500 | ChiNext Comp (399102) |

## Why v3 is potentially better

- **Single uniform source:** All 5 indices retrievable via AKShare `stock_zh_index_hist_csindex` (中证指数公司接口). v2 uses mixed sources (中证 + Sina/East Money for 深圳 indices).
- **Monotone retail-density gradient:** No mixed-bag indices like SZCOMP (which combines main board + ChiNext). v3 is pure size-decile slicing.
- **Removes redundancy:** v2 has CSI 300 and SHCOMP both representing "large-cap institutional" — v3 splits this into 上证50 (super-large) + CSI 300 (large).
- **Cleaner cross-sectional gradient for M1 mechanism narrative.**

## Why v3 might be worse

- Loses interpretability of "上证综指" and "深证综指" as exchange-level indices (though these are not commonly used in international finance literature anyway).
- Requires re-running ALL analyses on a new dataset. v2 work doesn't transfer cleanly.

## Files

```
paper_v3_csi_family/
├── README.md                     ← this file
├── paper/
│   ├── main.tex                  ← copy of v2 main.tex (will be modified)
│   ├── main.bib                  ← bibliography (mostly same as v2)
│   ├── AEA.cls + aea.bst         ← AEA template files
│   └── figures/                  ← v2 figures (some need regeneration)
├── scripts/
│   ├── g1_high_vol_joint_v3.py   ← G1 with new 5 indices
│   ├── g2_retail_density_ranking_v3.py  ← G2 with new 5 indices
│   ├── tune_ridge_alpha_cv.py    ← unchanged (just the CV protocol)
│   └── (more scripts to be added: build_dataset_v3, regression_tables_v3, etc.)
├── data/
│   └── index_csindex_daily_v3.csv  ← (to be pulled) all 5 v3 indices full daily history
└── results/
    └── (to be populated: g1_results_v3.csv, g2_ranking_v3.csv, etc.)
```

## Decision tree

After A-4 (re-run G1/G2/G3 on v3), evaluate:

- **If v3 results are CLEARLY better than v2** (cleaner gradient, larger effect sizes, perfect Spearman ρ): proceed to full paper rewrite on v3. Update v2 baseline tag accordingly.
- **If v3 results are SIMILAR to v2:** stick with v2 (keeps existing 5 indices that international audience knows).
- **If v3 results are WORSE than v2:** abandon v3, keep v2.

## To re-run from scratch

```bash
cd paper_v3_csi_family/scripts
python g2_retail_density_ranking_v3.py   # validate v3 ranking
python g1_high_vol_joint_v3.py           # validate v3 joint regression
# ... (more to come)
```
