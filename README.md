# Seeing Sentiment — Codebase

Replication code, data pipeline, and analysis scripts for:

> **"Seeing Sentiment: News Images, Retail Participation, and Chinese A-Share Index Returns"**
> Liu, Wang, Yu, Zeng (2026). Authors listed alphabetically.

**Manuscript repo (LaTeX source):** [Initial-Singularity/seeing-sentiment-manuscript](https://github.com/Initial-Singularity/seeing-sentiment-manuscript) (private)
**Paper-to-script map:** [`docs/REPRODUCE.md`](docs/REPRODUCE.md) lists every table and figure in the paper next to the script that produces it.

---

## What This Paper Does

Two daily market-level sentiment indices are constructed from Sina Finance news (2014–2026):

- **PhotoPes** — image pessimism via a fine-tuned Vision Transformer (ViT-B/16, 88.14% held-out accuracy on PCNN)
- **TextPes** — text pessimism via the Erlangshen-RoBERTa Chinese sentiment classifier

**Headline finding (§4.1, Table 5):** In high-volatility regimes, `PhotoPes_{t-3}` is negative and significant on the ChiNext at the 5% level (NW(5) HAC), with coefficient magnitude rising monotonically across volatility-threshold cutoffs in {50, 60, 70, 75, 80, 90} percentiles.

**Cross-section mechanism (§4.1):** The ranking of `PhotoPes` coefficients across five indices aligns with three independent retail-density proxies (turnover, inverse free-float, retail-account share); ChiNext is on top and CSI 300 / SHCOMP at the bottom across all three.

**Bounded tradability (§4.5):** A long/short rule built from the same signal yields a gross Sharpe gap of 0.118 vs the CSI 500 benchmark, but is **not statistically distinguishable from zero** (bootstrap 95% CI [-0.75, +0.97], one-sided p=0.389) and is fully eroded by ~17 bps of round-trip transaction costs.

---

## Repository Structure

```
.
├── pipeline/                         # Data acquisition + sentiment construction
│   ├── 01_sina_news_category_crawler.py
│   ├── 02_news_scraper.py
│   ├── 03_image_downloader.py
│   ├── 04_image_basic_filter.py
│   ├── 05_clarity_analysis_helper.py
│   ├── 06_text_analysis_helper.py
│   ├── 07_image_quality_processor.py
│   ├── 08_vit_transferlearning_pcnn.py    # ViT fine-tune on PCNN sentiment dataset
│   ├── 09_sentiment_analyzer_pcnn.py
│   ├── 10_calculate_daily_photopes_pcnn.py
│   ├── 11_calculate_daily_textpes.py
│   ├── 12_recalculate_quality_scores.py
│   └── 13_merge_data_and_calculate_returns_pcnn.py  # Writes results/merged_market_sentiment_data_old.csv
│
├── analysis/
│   ├── tables/                       # Paper-table generators (Tables 1-11)
│   ├── figures/                      # Figure generators
│   │   ├── regenerate_figures_bw.py             # All paper figures (B&W, 300 DPI)
│   │   └── plot_extreme_sentiment_images.py     # Appendix figure
│   ├── mechanism/                    # §4.1 cross-sectional mechanism
│   │   ├── g1_high_vol_joint.py
│   │   ├── g2_retail_density_ranking.py
│   │   └── g3_amihud_control.py
│   └── robustness/                   # §4.5 / §6 robustness battery
│       ├── break_test_oos_coefficient.py        # §6.1 PELT + Brown-Durbin-Evans
│       ├── nw_bandwidth_sensitivity.py          # NW(5/10/15/22) + Andrews/Newey plug-in
│       ├── regression_lag_validation.py
│       ├── sharpe_bootstrap_costs.py            # §4.5 bootstrap CI + cost grid
│       ├── tune_ridge_alpha_cv.py               # §4.6 Ridge α CV
│       ├── threshold_sensitivity_sweep.py       # §4.1 supplementary CSV generator
│       ├── threshold_sensitivity.csv            # 6 thresholds × 5 indices × 10 lags
│       ├── r7_block_length_sensitivity.py       # Bootstrap b ∈ {5, 14, 22}
│       ├── r7_block_length_sensitivity.csv
│       ├── r7_trailing_momentum_control.py      # 5/10/20/60-day cumulative controls
│       └── r7_trailing_momentum_control.csv
│
├── scripts/                          # Specialised rounds-of-revision robustness checks
│   ├── tetlock_orthogonalization.py
│   ├── clustered_se_pooled.py
│   ├── oos_rolling_standardization.py
│   ├── regime_conditional_oos.py
│   ├── placebo_image_count.py
│   ├── r2_log_news_count_control.py
│   └── r07_full_sample_auc.py
│
├── ai_image_annotation/              # VLM-ensemble validation pipeline
│   ├── ai_image_sentiment_annotator.py          # Multi-LLM caller (temp=0)
│   ├── ai_image_quality_annotator.py            # Quality LLM annotator
│   ├── validate_old_vit_with_llm_consensus.py   # §6.2 AUC + mean diff
│   ├── retry_failed_rows.py                     # Failed-query retry helper
│   ├── local_auth.example.json                  # API config template
│   └── run_artifacts/
│       └── ai_image_sentiment_annotations.csv   # Deposited per-image labels (1,253 rows)
│
├── docs/
│   ├── REPRODUCE.md                  # Paper element → script map
│   ├── data_sources.md               # External data dependencies
│   ├── phase0_results_summary.md     # Pre-headline gate decisions
│   └── phase1_results_summary.md     # Mechanism-test phase summary
│
├── archive/v3_csi_family_alternative/  # Rejected alternative index family
│
├── requirements.txt
├── LICENSE                           # MIT
└── README.md
```

---

## Reproduce Paper Tables & Figures

### Step 1 — Get the data

The published artifact ships scripts but not the heavy data files. Place
externally:

| File | Where to put it | Source |
|---|---|---|
| `merged_market_sentiment_data_old.csv` | `results/` | rebuild via `pipeline/13_merge_data_and_calculate_returns_pcnn.py`, or request from authors |
| `fivefactor_daily.csv` | `three_four_five_factor_daily/` | CUFE Chinese Fama-French daily file |
| `improved_vit_sentiment_model_old.pth` (329 MB) | repo root | available from authors on request; train locally via `pipeline/08_vit_transferlearning_pcnn.py` on the public You et al. (2015) PCNN dataset |

### Step 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Run analysis scripts (always from repo root)

```bash
# Headline regressions (Tables 1-4)
python analysis/tables/regression_tables_oldmodel.py

# §4.1 cross-sectional mechanism
python analysis/mechanism/g1_high_vol_joint.py
python analysis/mechanism/g2_retail_density_ranking.py
python analysis/mechanism/g3_amihud_control.py
python analysis/robustness/threshold_sensitivity_sweep.py

# §4.5 strategy + transaction cost
python analysis/tables/paper_table6_portfolio.py
python analysis/tables/paper_table7_ff3.py
python analysis/robustness/sharpe_bootstrap_costs.py
python analysis/robustness/r7_block_length_sensitivity.py
python analysis/robustness/r7_trailing_momentum_control.py

# §4.6 out-of-sample
python analysis/tables/paper_table8_oos.py
python analysis/tables/paper_table8b_oos_extensions_old.py
python analysis/robustness/tune_ridge_alpha_cv.py
python scripts/oos_rolling_standardization.py
python scripts/regime_conditional_oos.py
python scripts/clustered_se_pooled.py

# §6 robustness
python analysis/robustness/break_test_oos_coefficient.py
python analysis/robustness/nw_bandwidth_sensitivity.py
python analysis/tables/paper_table9_robustness.py
python analysis/tables/paper_table10_strategy_robustness_old.py
python analysis/tables/paper_table11_vit_quality_robustness.py

# Figures
python analysis/figures/regenerate_figures_bw.py
python analysis/figures/plot_extreme_sentiment_images.py
```

> **Run all commands from the repository root.** Scripts use relative paths
> (e.g., `DATA = "results/merged_market_sentiment_data_old.csv"`), so
> invoke as `python analysis/tables/...` rather than `cd analysis/tables && python ...`.

See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for the full paper-element → script map.

---

## Rebuild the Sentiment Dataset (Optional)

To rebuild PhotoPes / TextPes from raw Sina news:

```bash
# Requires MongoDB and a GPU for steps 08-09
python pipeline/01_sina_news_category_crawler.py
python pipeline/02_news_scraper.py
python pipeline/03_image_downloader.py
python pipeline/04_image_basic_filter.py
python pipeline/05_clarity_analysis_helper.py
python pipeline/06_text_analysis_helper.py
python pipeline/07_image_quality_processor.py
python pipeline/08_vit_transferlearning_pcnn.py     # Trains improved_vit_sentiment_model_old.pth
python pipeline/09_sentiment_analyzer_pcnn.py
python pipeline/10_calculate_daily_photopes_pcnn.py
python pipeline/11_calculate_daily_textpes.py
python pipeline/12_recalculate_quality_scores.py
python pipeline/13_merge_data_and_calculate_returns_pcnn.py
```

---

## §6.2 LLM External Validation Reproducibility

All AUC, mean-difference, and Pearson correlation numbers reported in §6.2
of the paper can be reproduced deterministically from the deposited
per-image labels at
`ai_image_annotation/run_artifacts/ai_image_sentiment_annotations.csv`
(1,253 rows; three model scores + labels + free-form reasons per image)
without re-querying any LLM API.

The three-model VLM ensemble was queried at `temperature=0`; because the
Anthropic API does not currently expose a `seed` parameter and the
deployment uses a third-party aggregator, exact LLM outputs are not
guaranteed to replicate byte-for-byte across re-queries. Downstream
statistics are unaffected because they are computed on the deposited
labels.

---

## Citation

Working paper draft (citation TBA on Zenodo deposit).

---

## License

MIT License — see `LICENSE`.
