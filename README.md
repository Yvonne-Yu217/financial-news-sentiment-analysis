# Quantifying Investor Sentiment with Multimodal Data in the Chinese Stock Market

Full replication code and paper source for:

> **"Quantifying Investor Sentiment with Multimodal Data in the Chinese Stock Market"**  
> Yi Yu — M.S. Data Science, New York University (yy5919@nyu.edu)

---

## Repository Structure

```
├── paper/                              # Paper source files
│   ├── main.tex                        # LaTeX manuscript
│   ├── main.bib                        # Bibliography
│   ├── AEA.cls / aea.bst               # AEA journal class files
│   └── figures/                        # All paper figures (B&W, 300 DPI)
│
├── Data Pipeline (Scripts 1–13)
│   ├── 1sina_news_category_crawler.py  # Crawl Sina Finance news categories
│   ├── 2news_scraper.py                # Scrape news articles & images
│   ├── 3image_downlowder.py            # Download news images
│   ├── 4image_basic_filter.py          # Basic image filtering
│   ├── 5clarity_analysis_helper.py     # Image clarity scoring
│   ├── 6text_analysis_helper.py        # OCR text detection in images
│   ├── 7image_quality_processor.py     # Final quality scoring
│   ├── 8vit_transferlearning_old.py    # ViT fine-tuning (paper model)
│   ├── 9sentiment_analyzer_oldmodel.py # Image sentiment inference
│   ├── 10calculate_daily_photopes_oldmodel.py  # Daily PhotoPes index
│   ├── 11calculate_daily_textpes.py    # Daily TextPes index
│   ├── 12recalculate_quality_scores.py # Quality score recalculation
│   └── 13merge_data_and_calculate_returns_oldmodel.py  # Merge + returns
│
├── Paper Reproduction Scripts
│   ├── regression_tables_oldmodel.py   # Tables 1–4
│   ├── paper_table5_conditional.py     # Table 5: conditional by volatility state
│   ├── paper_table6_portfolio.py       # Table 6: portfolio performance
│   ├── paper_table7_ff3.py             # Table 7: Fama-French 3-factor
│   ├── paper_table8_oos.py             # Table 8: OOS R²_OOS
│   ├── paper_table9_robustness.py      # Table 9: robustness checks
│   ├── regenerate_figures_bw.py        # All paper figures (B&W, 300 DPI)
│   └── 14plot_extreme_sentiment_images.py  # Appendix figure (requires MongoDB)
│
└── Supplementary
    ├── optimize_oos_alpha.py           # OOS parameter grid search (exploratory)
    └── regression_lag_validation.py    # Lag structure validation
```

---

## Paper Summary

This paper constructs two daily market-level sentiment indices from Sina Finance news (2014–2026):

- **PhotoPes** — image pessimism index via fine-tuned Vision Transformer (ViT-B/16)
- **TextPes** — text pessimism index via BERT-based model

**Key findings:**
1. **PhotoPes_t-3** negatively predicts returns 3 days later — significant for ChiNext (1%), SZCOMP and CSI500 (10%)
2. Sentiment effects are **amplified in high-volatility regimes**: both PhotoPes_t-3 and TextPes_t-3 significant at 5–10%
3. Rule-based sentiment strategy: Sharpe = 0.363 vs. 0.248 for CSI500 benchmark
4. OOS: ChiNext R²_OOS = 0.29%** (t=2.02), SZCOMP/CSI500 significant at 10%

---

## Reproduce All Paper Tables & Figures

### Data

Download data files from [Google Drive](https://drive.google.com/drive/folders/1Xzrj9ul8x8Ydis3fNpBx9-FgEJvRFDgx?usp=drive_link) and place in:
- `results/merged_market_sentiment_data_old.csv`
- `three_four_five_factor_daily/fivefactor_daily.csv`

### Run in order

```bash
pip install -r requirements.txt

# Tables 1–4: descriptive stats, PhotoPes/TextPes regressions, joint regression
python regression_tables_oldmodel.py

# Table 5: conditional analysis by market volatility state
python paper_table5_conditional.py

# Table 6: portfolio performance (also generates cumulative_returns_old.csv)
python paper_table6_portfolio.py

# Table 7: Fama-French 3-factor regression
python paper_table7_ff3.py

# Table 8: out-of-sample R²_OOS  (~5 min)
python paper_table8_oos.py

# Table 9: robustness checks
python paper_table9_robustness.py

# All figures (B&W, 300 DPI, serif fonts)
python regenerate_figures_bw.py
```

---

## Data Pipeline

Run scripts 1–13 sequentially to rebuild the sentiment dataset from scratch:

```bash
python 1sina_news_category_crawler.py    # requires: internet
python 2news_scraper.py                  # requires: MongoDB
python 3image_downlowder.py
python 4image_basic_filter.py
python 5clarity_analysis_helper.py
python 6text_analysis_helper.py
python 7image_quality_processor.py
python 8vit_transferlearning_old.py      # requires: GPU, training data
python 9sentiment_analyzer_oldmodel.py   # requires: GPU, MongoDB
python 10calculate_daily_photopes_oldmodel.py
python 11calculate_daily_textpes.py
python 12recalculate_quality_scores.py
python 13merge_data_and_calculate_returns_oldmodel.py
```

**Requirements:** MongoDB, GPU (for scripts 8–9), Python 3.8+

Pre-trained model: `improved_vit_sentiment_model_old.pth`  
Test accuracy: **84.7%** | Training history: `run_artifacts/8vit_transferlearning/8vit_training_metrics.json`

---

## Branches

| Branch | Contents |
|--------|----------|
| `main` | Old-model paper pipeline + reproduction scripts + paper source |
| `pipeline-backup` | Original repository state (all model variants) |

---

## License

MIT License
