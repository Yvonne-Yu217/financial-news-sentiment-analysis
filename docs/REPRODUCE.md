# Reproduction Map: Paper Tables and Figures → Scripts

This file maps every numerical claim, table, and figure in the paper *Seeing
Sentiment: News Images, Retail Participation, and Chinese A-Share Index
Returns* to the script that produces it. Run all scripts from the repository
root (e.g., `python pipeline/01_sina_news_category_crawler.py`).

All scripts read from `results/merged_market_sentiment_data_old.csv` unless
otherwise noted; this file is produced by `pipeline/13_merge_data_and_calculate_returns_pcnn.py`.

---

## Pipeline (run once, in order, before any analysis)

Each numbered script depends on artifacts from the previous one. Sequence:

| Step | Script | Produces |
|---|---|---|
| 01 | `pipeline/01_sina_news_category_crawler.py` | Sina news category list (MongoDB) |
| 02 | `pipeline/02_news_scraper.py` | Article text + metadata (MongoDB) |
| 03 | `pipeline/03_image_downloader.py` | Local images for each article |
| 04 | `pipeline/04_image_basic_filter.py` | Filtered image set (size / format) |
| 05 | `pipeline/05_clarity_analysis_helper.py` | Image clarity scores |
| 06 | `pipeline/06_text_analysis_helper.py` | Tokenization / cleaning utilities |
| 07 | `pipeline/07_image_quality_processor.py` | Phase 1 image quality filter |
| 08 | `pipeline/08_vit_transferlearning_pcnn.py` | Fine-tunes ViT on PCNN sentiment data; writes `improved_vit_sentiment_model_old.pth` |
| 09 | `pipeline/09_sentiment_analyzer_pcnn.py` | Applies the fine-tuned ViT to Sina photographs; writes per-image PhotoPes scores |
| 10 | `pipeline/10_calculate_daily_photopes_pcnn.py` | Daily PhotoPes index (aggregates per-image scores) |
| 11 | `pipeline/11_calculate_daily_textpes.py` | Daily TextPes index via Erlangshen-RoBERTa |
| 12 | `pipeline/12_recalculate_quality_scores.py` | Refines image quality scores (Phase 0 cleanup) |
| 13 | `pipeline/13_merge_data_and_calculate_returns_pcnn.py` | Merges sentiment indices with index returns → `merged_market_sentiment_data_old.csv` |

The PCNN training corpus is the public You et al. (2015) dataset at
<https://qzyou.github.io/projects/sa-ds/>. The fine-tuned ViT checkpoint
(`improved_vit_sentiment_model_old.pth`, 329 MB) is available from the
authors on request; it is not committed to the repository due to its size.

---

## Empirical Results (§4)

### §4.1 Cross-Sectional Differential Sensitivity in High-Volatility Regimes

| Paper element | Script |
|---|---|
| Table 5: high/low-volatility panel split (NW(5) HAC) | `analysis/tables/paper_table5_conditional.py` |
| Cross-index ranking + retail-density proxy alignment | `analysis/mechanism/g2_retail_density_ranking.py` |
| Joint regression on high-volatility subsample (95th–75th cutoffs) | `analysis/mechanism/g1_high_vol_joint.py` |
| Amihud illiquidity control on high-vol joint | `analysis/mechanism/g3_amihud_control.py` |
| Tetlock-style orthogonalisation of PhotoPes ⊥ TextPes | `scripts/tetlock_orthogonalization.py` |
| News-volume control (`log(n_articles)_{t-1}`) | `scripts/r2_log_news_count_control.py` |
| Sentiment-free placebo (`log(n_images)_{t-3}`) | `scripts/placebo_image_count.py` |
| Trailing cumulative-return controls (5d/10d/20d/60d) | `analysis/robustness/r7_trailing_momentum_control.py` |
| Threshold-sensitivity sweep at {50, 60, 70, 75, 80, 90} | `analysis/robustness/threshold_sensitivity_sweep.py` |

### §4.2-§4.4 PhotoPes / TextPes / Joint regressions

| Table | Script |
|---|---|
| Tables 1-4: headline univariate + joint regressions | `analysis/tables/regression_tables_oldmodel.py` |
| Photo-Text interaction (`PhotoPes × TextPes`) | (returned via `regression_tables_oldmodel.py`, untabulated) |

### §4.5 In-Sample Long/Short/Neutral PhotoPes Trading Rule

| Paper element | Script |
|---|---|
| Table 6: portfolio cumulative returns + Sharpe | `analysis/tables/paper_table6_portfolio.py` |
| Table 7: Fama-French three-factor regression | `analysis/tables/paper_table7_ff3.py` |
| Stationary block-bootstrap CI + JK/Memmel test + cost grid | `analysis/robustness/sharpe_bootstrap_costs.py` |
| Block-length sensitivity (`b ∈ {5, 14, 22}`) | `analysis/robustness/r7_block_length_sensitivity.py` |

### §4.6 Out-of-Sample Predictive Analysis

| Paper element | Script |
|---|---|
| Table 8: expanding-window OOS R² + Clark-West | `analysis/tables/paper_table8_oos.py` |
| Table 8b: regime-conditional OOS + sub-window stability | `analysis/tables/paper_table8b_oos_extensions_old.py` |
| Rolling-window standardization for OOS predictors | `scripts/oos_rolling_standardization.py` |
| CV-Ridge α grid sensitivity | `analysis/robustness/tune_ridge_alpha_cv.py` |
| Pooled cross-index with cluster-by-date / Driscoll-Kraay | `scripts/clustered_se_pooled.py` |
| Regime-conditional Clark-West subset | `scripts/regime_conditional_oos.py` |
| Lag-specific significance validation | `analysis/robustness/regression_lag_validation.py` |

---

## Robustness Block (§6)

| §6.x | Table / element | Script |
|---|---|---|
| §6.1 Structural stability (PELT + Brown-Durbin-Evans CUSUM) | `analysis/robustness/break_test_oos_coefficient.py` |
| §6.2 LLM external validation (AUC 0.71 strong-consensus, AUC 0.54 full sample) | `ai_image_annotation/validate_old_vit_with_llm_consensus.py` |
| §6.2 Full-sample AUC + Pearson r computation | `scripts/r07_full_sample_auc.py` |
| §6.3 Multiple-testing Bonferroni thresholds | (textual; uses Table 8 inputs) |
| Table 9 (winsorization / GARCH / SHCOMP tail) | `analysis/tables/paper_table9_robustness.py` |
| Table 10 (strategy robustness) | `analysis/tables/paper_table10_strategy_robustness_old.py` |
| Table 11 (ViT quality robustness) | `analysis/tables/paper_table11_vit_quality_robustness.py` |
| Newey-West bandwidth sensitivity (NW(5/10/15/22) + Andrews/Newey plug-in) | `analysis/robustness/nw_bandwidth_sensitivity.py` |

---

## Figures

| Figure | Script |
|---|---|
| Fig. 1 (ViT architecture) | External (reproduced from Dosovitskiy et al. 2020) |
| Fig. 2 (ViT training accuracy) | Output of `pipeline/08_vit_transferlearning_pcnn.py` |
| Fig. 3 (monthly sentiment trends) | `analysis/figures/regenerate_figures_bw.py` |
| Fig. 4 (strategy cumulative returns) | `analysis/figures/regenerate_figures_bw.py` |
| Fig. 5 (LLM-validation distribution) | `analysis/figures/regenerate_figures_bw.py` |
| Appendix Fig. A1 (extreme-sentiment photographs) | `analysis/figures/plot_extreme_sentiment_images.py` |

---

## LLM External Validation Supporting Files (§6.2 + Appendix A.1)

| Artifact | Path |
|---|---|
| Per-image VLM ensemble labels (1,253 rows, 3 models × score + label + reason) | `ai_image_annotation/run_artifacts/ai_image_sentiment_annotations.csv` |
| VLM API caller (temperature=0; deposited labels are authoritative) | `ai_image_annotation/ai_image_sentiment_annotator.py` |
| Failed-query retry helper | `ai_image_annotation/retry_failed_rows.py` |
| AUC / mean-difference computation (strong-consensus 309 subset) | `ai_image_annotation/validate_old_vit_with_llm_consensus.py` |
| Image quality LLM annotator (Table 11 input) | `ai_image_annotation/ai_image_quality_annotator.py` |

All §6.2 AUC, mean-difference, and correlation statistics can be recomputed
deterministically from `ai_image_sentiment_annotations.csv` without
re-querying any LLM API.

---

## Archive

`archive/v3_csi_family_alternative/` contains a parallel pipeline that
substitutes CSI 1000 / SSE 50 for the headline five-index family. The
v3 alternative was evaluated and rejected in favour of the v2 (current)
specification; see `archive/v3_csi_family_alternative/v2_vs_v3_comparison.md`
for the decision memo. Scripts under that directory are not part of the
paper's headline analysis.
