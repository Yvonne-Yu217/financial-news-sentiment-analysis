# Quantifying Investor Sentiment with Multimodal Data in the Chinese Stock Market

This repository contains the full replication code, data pipeline, and paper source for:

> **"Quantifying Investor Sentiment with Multimodal Data in the Chinese Stock Market"**  
> Yi Yu — M.S. Data Science, New York University (yy5919@nyu.edu)

---

## Repository Structure

```
├── paper/                          # Paper source files
│   ├── main.tex                    # LaTeX manuscript (Article old model copy)
│   ├── main.bib                    # Bibliography
│   ├── AEA.cls / aea.bst           # AEA journal class files
│   └── figures/                    # All paper figures (B&W, 300 DPI)
│       ├── training_accuracy.png
│       ├── monthly_sentiment_trends.png
│       ├── cumulative_returns_old.png
│       ├── extreme_sentiment_images.png
│       └── vit_architecture.png
│
├── Data Pipeline (Scripts 1–13)
│   ├── 1sina_news_category_crawler.py      # Crawl Sina Finance categories
│   ├── 2news_scraper.py                    # Scrape news articles
│   ├── 3image_downlowder.py                # Download news images
│   ├── 4image_basic_filter.py              # Basic image filtering
│   ├── 5clarity_analysis_helper.py         # Image clarity analysis
│   ├── 6text_analysis_helper.py            # OCR text in images
│   ├── 7image_quality_processor.py         # Quality scoring
│   ├── 8vit_transferlearning_old.py        # ViT fine-tuning (paper model)
│   ├── 9sentiment_analyzer_oldmodel.py     # Image sentiment scoring
│   ├── 10calculate_daily_photopes_oldmodel.py  # Daily PhotoPes index
│   ├── 11calculate_daily_textpes.py            # Daily TextPes index
│   ├── 12recalculate_quality_scores.py         # Quality recalculation
│   └── 13merge_data_and_calculate_returns_oldmodel.py  # Merge + returns
│
├── Paper Reproduction Scripts
│   ├── regression_tables_oldmodel.py   # Tables 1–4: descriptive stats + regressions
│   ├── paper_table5_conditional.py     # Table 5: conditional by volatility state
│   ├── paper_table6_portfolio.py       # Table 6: portfolio performance
│   ├── paper_table7_ff3.py             # Table 7: Fama-French 3-factor regression
│   ├── paper_table8_oos.py             # Table 8: out-of-sample R²_OOS
│   ├── paper_table9_robustness.py      # Table 9: robustness checks
│   └── regenerate_figures_bw.py        # All paper figures (B&W, 300 DPI)
│
└── Supplementary
    ├── 14plot_extreme_sentiment_images.py  # Appendix figure (requires MongoDB)
    ├── optimize_oos_alpha.py               # OOS grid search (exploratory)
    └── pipeline.py                         # Full pipeline runner
```

---

## Paper Summary

This paper constructs two daily sentiment indices from Sina Finance news (2014–2026):

- **PhotoPes** — pessimism index from news images, using a fine-tuned Vision Transformer (ViT)
- **TextPes** — pessimism index from news text, using a BERT-based model

Key findings:
1. **PhotoPes_t-3** negatively predicts returns 3 days later; significant for ChiNext (1%), SZCOMP and CSI500 (10%)
2. Sentiment effects are **amplified in high-volatility regimes** — both PhotoPes_t-3 and TextPes_t-3 significant (5%–10%)
3. A rule-based sentiment strategy achieves Sharpe = 0.363 vs. 0.248 for CSI500 benchmark
4. OOS R²_OOS: ChiNext 0.29%** (t=2.02), SZCOMP/CSI500 0.07–0.08%* — predictability confirmed out-of-sample

---

## Quick Start: Reproduce All Paper Tables and Figures

### Prerequisites
```bash
pip install -r requirements.txt
```

Data files (too large for GitHub) are available at:  
[Google Drive](https://drive.google.com/drive/folders/1Xzrj9ul8x8Ydis3fNpBx9-FgEJvRFDgx?usp=drive_link)

Place downloaded files in `results/`:
- `results/merged_market_sentiment_data_old.csv` — main dataset (2014–2026)
- `three_four_five_factor_daily/fivefactor_daily.csv` — CUFE factor data

### Run in order

```bash
# Tables 1–4: descriptive stats + main regressions (PhotoPes, TextPes, joint)
python regression_tables_oldmodel.py

# Table 5: conditional analysis by market volatility state
python paper_table5_conditional.py

# Table 6: portfolio performance (also saves cumulative_returns_old.csv)
python paper_table6_portfolio.py

# Table 7: Fama-French 3-factor regression
python paper_table7_ff3.py

# Table 8: out-of-sample predictability (Ridge, lags 3-5, alpha=500)
python paper_table8_oos.py            # ~5 min runtime

# Table 9: robustness checks (no winsorization / GARCH / no extreme returns)
python paper_table9_robustness.py

# All figures (B&W, 300 DPI, serif fonts)
python regenerate_figures_bw.py
```

---

## Data Pipeline (Scripts 1–13)

Run the full pipeline from data collection to sentiment index construction:

```bash
# Full pipeline
python pipeline.py --start-year 2014 --end-year 2026

# Or individual steps
python 1sina_news_category_crawler.py
python 2news_scraper.py
# ...
python 13merge_data_and_calculate_returns_oldmodel.py
```

**Requirements:**
- MongoDB (for image metadata storage)
- GPU recommended for scripts 8–9

---

## Model

Pre-trained ViT model: `improved_vit_sentiment_model_old.pth`  
Training history: `run_artifacts/8vit_transferlearning/8vit_training_metrics.json`  
Test accuracy: **84.7%** on held-out sentiment classification task

---

## Branches

| Branch | Contents |
|--------|----------|
| `main` | Paper reproduction code + paper source files |
| `pipeline-backup` | Original pipeline-only repository state |

---

## Citation

```
Yu, Yi (2026). Quantifying Investor Sentiment with Multimodal Data in 
the Chinese Stock Market. Working Paper, New York University.
```

---

## License

MIT License
