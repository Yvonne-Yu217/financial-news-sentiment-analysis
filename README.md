# Quantifying Investor Sentiment with Multimodal Data in the Chinese Stock Market

Replication code and paper source for:

> **"Quantifying Investor Sentiment with Multimodal Data in the Chinese Stock Market"**
> Yi Yu — M.S. Data Science, New York University · yy5919@nyu.edu

---

## What This Paper Does

Two daily market-level sentiment indices are constructed from Sina Finance news (2014–2026):

- **PhotoPes** — image pessimism via a fine-tuned Vision Transformer (ViT-B/16, 84.7% held-out accuracy)
- **TextPes** — text pessimism via a BERT-Chinese sentiment classifier

**Headline finding (Table 4):** PhotoPes_{t-3} negatively predicts ChiNext returns at the 1% level (Newey-West HAC, 5 lags), with weaker effects on SZCOMP and CSI 500.

**Robustness:** the result survives trailing-momentum controls (5/10/20/60-day cumulative-return regressors anchored at t-3), Politis-Romano stationary block-bootstrap at b ∈ {5, 14, 22}, Andrews(1991) plug-in NW bandwidth selection, Bai-Perron break tests, and Bonferroni correction at m = 25.

**Sub-period decomposition:** in-sample significance concentrates in 2014–2017; the 2018–2023 window is null; 2024–2026 shows partial recovery. The paper presents this as honest sub-period drift rather than a uniform alpha.

---

## Repository Structure

```
├── paper/                          # LaTeX manuscript + bibliography
│   ├── main.tex                    # 54-page manuscript
│   ├── main.bib                    # Bibliography
│   ├── AEA.cls / aea.bst           # AEA journal class files
│   ├── main.pdf                    # Compiled PDF
│   └── figures/                    # Paper figures (B&W, 300 DPI)
│
├── 1–14*.py                        # Data pipeline (sequential)
│
├── regression_tables_oldmodel.py   # Tables 1–4: headline regressions
├── paper_table5_conditional.py     # Table 5: vol-conditional sentiment
├── paper_table6_portfolio.py       # Table 6: portfolio sorts
├── paper_table7_ff3.py             # Table 7: Fama-French-3 alphas
├── paper_table8_oos.py             # Table 8: out-of-sample R²
├── paper_table8b_oos_extensions_old.py  # Table 8b: regime-conditional OOS
├── paper_table9_robustness.py      # Table 9: robustness battery
├── paper_table10_strategy_robustness_old.py  # Table 10: strategy robustness
├── paper_table11_vit_quality_robustness.py   # Table 11: ViT quality slices
│
├── Mechanism / cross-sectional analysis
│   ├── g1_high_vol_joint.py        # G1: high-vol joint regression across 5 indices
│   ├── g2_retail_density_ranking.py  # G2: retail-density ordering proxies
│   └── g3_amihud_control.py        # G3: Amihud illiquidity control
│
├── analysis/
│   └── robustness/                 # Round 7 robustness scripts + outputs
│       ├── r7_block_length_sensitivity.py    # Politis-Romano b ∈ {5, 14, 22}
│       ├── r7_block_length_sensitivity.csv   # bootstrap CIs at each b
│       ├── r7_trailing_momentum_control.py   # 5/10/20/60-day controls
│       └── r7_trailing_momentum_control.csv  # 6 specs × β/t/p
│
├── nw_bandwidth_sensitivity.py     # NW lags ∈ {5, 10, 15, 22}
├── break_test_oos_coefficient.py   # Bai-Perron + Brown-Durbin-Evans
│
├── Backtesting
│   ├── sharpe_bootstrap_costs.py   # Strategy bootstrap CI + transaction costs
│   ├── tune_ridge_alpha_cv.py      # Pre-OOS Ridge CV (locked α)
│   └── regenerate_figures_bw.py    # All paper figures (B&W, 300 DPI)
│
├── ai_image_annotation/            # LLM consensus validation pipeline
│
├── docs/                           # Project documentation
│   ├── data_sources.md             # AKShare endpoints + CSMAR coverage memo
│   ├── phase0_results_summary.md   # Phase 0 gate decisions
│   └── phase1_results_summary.md   # Phase 1 mechanism-test summary
│
├── archive/                        # Materials kept for transparency, not published
│   ├── README.md                   # Archive index
│   └── v3_csi_family_alternative/  # Alternative CSI-family index design
│                                     evaluated 2026-04-29 and not adopted
│
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT
└── README.md                       # This file
```

---

## Reproduce Paper Tables & Figures

### Step 1 — Get the data

Download from [Google Drive](https://drive.google.com/drive/folders/1Xzrj9ul8x8Ydis3fNpBx9-FgEJvRFDgx?usp=drive_link) and place in:
- `results/merged_market_sentiment_data_old.csv`
- `three_four_five_factor_daily/fivefactor_daily.csv`

### Step 2 — Install Python deps

```bash
pip install -r requirements.txt
```

### Step 3 — Run paper-table scripts (each writes CSV/JSON/figures into `results/`)

```bash
# Headline regressions
python regression_tables_oldmodel.py        # Tables 1–4

# Conditional / portfolio / factor / OOS
python paper_table5_conditional.py
python paper_table6_portfolio.py
python paper_table7_ff3.py
python paper_table8_oos.py                  # ~5 min
python paper_table8b_oos_extensions_old.py
python paper_table9_robustness.py
python paper_table10_strategy_robustness_old.py
python paper_table11_vit_quality_robustness.py

# Mechanism (cross-sectional)
python g1_high_vol_joint.py
python g2_retail_density_ranking.py
python g3_amihud_control.py

# Methodology robustness (Round 7)
python nw_bandwidth_sensitivity.py
python break_test_oos_coefficient.py
python analysis/robustness/r7_block_length_sensitivity.py
python analysis/robustness/r7_trailing_momentum_control.py

# Strategy + figures
python sharpe_bootstrap_costs.py
python regenerate_figures_bw.py
```

---

## Rebuild the Sentiment Dataset (Optional)

To rebuild PhotoPes / TextPes from raw news:

```bash
# Pipeline scripts (require MongoDB and a GPU for steps 8–9)
python 1sina_news_category_crawler.py
python 2news_scraper.py
python 3image_downlowder.py
python 4image_basic_filter.py
python 5clarity_analysis_helper.py
python 6text_analysis_helper.py
python 7image_quality_processor.py
python 8vit_transferlearning_old.py
python 9sentiment_analyzer_oldmodel.py
python 10calculate_daily_photopes_oldmodel.py
python 11calculate_daily_textpes.py
python 12recalculate_quality_scores.py
python 13merge_data_and_calculate_returns_oldmodel.py
```

Pre-trained ViT checkpoint (`improved_vit_sentiment_model_old.pth`) is not in the repo (>500 MB) — train via step 8 or contact the author.

---

## Branches

| Branch | Contents |
|---|---|
| `main` | Submission-ready paper pipeline + reproduction scripts + paper source |
| `pipeline-backup` | Pre-cleanup snapshot of the full repository state |

---

## Citation

If you use this code or data, please cite the working paper draft (citation TBA on Zenodo deposit).

---

## License

MIT License — see `LICENSE`.
