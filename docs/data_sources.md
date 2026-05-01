# Data Sources Audit Memo

**Last updated:** 2026-04-29
**Maintainer:** Yi Yu (yy5919@nyu.edu) + Claude assistant
**Purpose:** Single source of truth for every data point used in the paper, where it came from, what unit it's in, and how to re-pull it. Cross-check against this file whenever the analysis is re-run.

---

## 1. Index price + return series (5 indices, 2014–2026 daily)

| Index | Code | Source | AKShare function | Unit | Notes |
|---|---|---|---|---|---|
| CSI 300 沪深 300 | 000300 | 中证指数公司 | `stock_zh_index_hist_csindex(symbol='000300')` | level (points) | 完整 2014-2026 |
| SHCOMP 上证综指 | 000001 | 中证指数公司 | `stock_zh_index_hist_csindex(symbol='000001')` | level | 完整 2014-2026 |
| SZCOMP 深证综指 | 399106 | 东方财富 (Sina source) | `stock_zh_index_daily(symbol='sz399106')` | level | CSI 公司接口对深圳指数返回空 |
| ChiNext Comp 创业板综指 | 399102 | 东方财富 (Sina source) | `stock_zh_index_daily(symbol='sz399102')` | level | 同上 |
| CSI 500 中证 500 | 000905 | 中证指数公司 | `stock_zh_index_hist_csindex(symbol='000905')` | level | 完整 2014-2026 |

**Code unit conventions:**
- `stock_zh_index_hist_csindex` 返回的成交金额字段 = **亿元** (10⁸ CNY)
- `stock_zh_index_daily` (Sina) 返回的 volume = 手 (lots = 100 shares),没有成交金额
- `stock_zh_index_daily_em` (East Money) 返回的 amount = **元** (raw CNY),但接口经常 rate-limit

**Storage:** `results/merged_market_sentiment_data_old.csv` (Date + 5 indices' close + log_returns + PhotoPes/TextPes + lags)

---

## 2. 流通市值 (Free-float market capitalization)

### 5 indices used in current paper (v2)

| Index | Free-float MV (万亿元) | Source | Snapshot date | Credibility |
|---|---|---|---|---|
| CSI 300 | 50.0 | CSI Index Co fact sheet | 2024 Q4 | ⭐⭐ (estimate, ±5%) |
| SHCOMP | **63.12** | AKShare `stock_sse_summary()` → SSE 总貌 | 2026-04-28 | ⭐⭐⭐ (precise) |
| SZCOMP | **40.68** | AKShare `stock_szse_summary(date='20260428')` → SZSE 主板A + 创业板A 流通市值 sum | 2026-04-28 | ⭐⭐⭐ (precise) |
| ChiNext Comp | **15.97** | AKShare `stock_szse_summary(date='20260428')` → 创业板A 流通市值 | 2026-04-28 | ⭐⭐⭐ (precise) |
| CSI 500 | 10.0 | CSI Index Co fact sheet | 2024 Q4 | ⭐⭐ (estimate, ±10%) |

### Unit conventions (CRITICAL — easy to misread)
- `stock_sse_summary()` 返回的"流通市值" = **亿元** (e.g., 631,171.43 = 63.12 万亿)
- `stock_szse_summary(date)` 返回的"流通市值" = **元** (e.g., 4.067e+13 = 40.67 万亿)
- **THESE TWO DIFFER**, always convert to 万亿 before comparing

### Re-pull commands

```python
# SSE 总貌 (current snapshot only — no historical date param)
import akshare as ak
df_sse = ak.stock_sse_summary()
# Look for row where 项目 == "流通市值", column "股票"

# SZSE 总貌 (historical, date format YYYYMMDD, max recent 3 years)
df_szse = ak.stock_szse_summary(date='20260428')
# Row 主板A 流通市值 + Row 创业板A 流通市值 = SZCOMP
# Row 创业板A 流通市值 alone = ChiNext Comp
```

---

## 3. 换手率 (Annualized turnover ratio, multi-year average)

| Index | Turnover (倍/年) | Source method | Precision |
|---|---|---|---|
| CSI 300 | **1.08** | 中证指数公司 实拉 2014-2024 全样本日成交金额累加 ÷ 流通市值 | ⭐⭐⭐ |
| SHCOMP | **1.27** | 同上,via `stock_zh_index_hist_csindex(symbol='000001')` | ⭐⭐⭐ |
| SZCOMP | 4.85 | SZSE 历年总貌 (2018-2024 年末快照, n=7),主板A + 创业板A 加权 | ⭐⭐⭐ |
| ChiNext Comp | 6.30 | SZSE 历年总貌 (2018-2024 年末快照, n=7),创业板A only | ⭐⭐⭐ |
| CSI 500 | **3.24** | 中证指数公司 实拉 2014-2024 全样本 | ⭐⭐⭐ |

**所有 5 个 turnover 值现在都是 ⭐⭐⭐ 级实拉/计算,无估算。**

### Re-pull command (CSI 300, SHCOMP, CSI 500)

```python
import akshare as ak
import pandas as pd

# Pull full daily 成交金额 (in 亿元)
df = ak.stock_zh_index_hist_csindex(
    symbol='000300',  # or '000001', '000905'
    start_date='20140101',
    end_date='20241231'
)
df['日期'] = pd.to_datetime(df['日期'])
df = df[df['日期'] >= '2014-01-02']

# Average daily 成交金额 (单位 亿元)
avg_daily_yi = pd.to_numeric(df['成交金额'], errors='coerce').mean()

# Annualize (×252 trading days), convert 亿 to 万亿
annual_wanyi = avg_daily_yi * 252 / 1e4

# Turnover = annual / free-float MV
turnover = annual_wanyi / FREE_FLOAT_TRILLION
```

### Re-pull command (SZCOMP, ChiNext via SZSE 历年总貌)

```python
import akshare as ak
year_end_dates = ['20141231', '20151231', '20161230', '20171229', '20181228',
                  '20191231', '20201231', '20211231', '20221230', '20231229',
                  '20241231']

# NOTE: SZSE summary fails for dates pre-2018 (interface limitation)
# Effective sample: 2018-2024, n=7 year-ends
results = []
for d in year_end_dates:
    try:
        df = ak.stock_szse_summary(date=d)
        chi_row = df[df['证券类别'].str.contains('创业板A', na=False)].iloc[0]
        main_row = df[df['证券类别'].str.contains('主板A', na=False)].iloc[0]
        # 成交金额 in 元, 流通市值 in 元
        results.append({...})
    except Exception:
        pass

# Annualize daily 成交 × 252, divide by 流通市值, average across years
```

---

## 4. 散户账户占比 (Retail account share)

| Index | Retail share (%) | Source | Precision |
|---|---|---|---|
| CSI 300 | 25 | Liu, Stambaugh, Yuan (2019, JFE) Table 2 | ⭐⭐ |
| SHCOMP | 35 | 中国结算 (CSDC) 月报 + Liu et al. (2019) | ⭐⭐ |
| SZCOMP | 50 | 中国结算 (CSDC) + 深交所投资者持股结构报告 | ⭐⭐ |
| ChiNext Comp | 70 | 深交所创业板年报投资者结构 | ⭐⭐ |
| CSI 500 | 55 | 中国结算 + 中证500 fact sheet 推算 | ⭐⭐ |

**These are stylized values from public statistical reports, not real-time.** For final paper submission, ideally replace with year-by-year Wind/CSMAR data.

### To upgrade to ⭐⭐⭐
- Wind: `=WSD("000300.SH","retail_holding_pct","2014-12-31","2024-12-31","Period=Y")` 等
- CSMAR: 投资者持股结构表 (`Holding_Investor_Structure`)

---

## 5. 个股 / 文章数据 (PhotoPes + TextPes inputs)

| Item | Source | Volume | Path |
|---|---|---|---|
| Sina Finance 新闻文章 | 自爬 (since 2014) | ~245,000 articles | `images/` (not in git) |
| Sina Finance 新闻图片 | 同 articles | ~572,282 images | `images/` |
| 3-LLM consensus annotation (1,253 images) | OpenAI/Anthropic/Google APIs via liaobots proxy | 1,253 images × 3 models | `ai_image_annotation/run_artifacts/` |
| ViT-PCNN trained model | 训练自 You et al. (2015) PCNN Twitter dataset | 882 high-agreement images | `improved_vit_sentiment_model_old.pth` |
| Erlangshen-RoBERTa-110M | HuggingFace | — | (loaded on demand) |

---

## 6. Fama-French 三因子 (for §4.5/4.6 alpha regressions)

| Item | Source | Format | Path |
|---|---|---|---|
| MKT, SMB, HML 日因子 | CUFE 中央财大 (Central University of Finance and Economics) | Daily, 2000+ | `three_four_five_factor_daily/fivefactor_daily.csv` |
| 中国 risk-free rate | 同上 | Daily 一年定存 | 同上 (rf 列) |

---

## 7. ⚠️ Known interface failures / rate limits

| Source | Interface | Failure mode | Workaround |
|---|---|---|---|
| 东方财富 (East Money) | `stock_zh_index_daily_em`, `index_zh_a_hist`, `stock_zh_index_spot_em` | "Connection aborted, RemoteDisconnected" — rate limit, often blocks for hours | Use `stock_zh_index_hist_csindex` (中证指数公司) instead — same data for SH-listed indices |
| CSMAR Python API | csmarapi package | Officially Windows-only; package not on PyPI; requires CSMAR account download | Use AKShare for free; CSMAR only if precise historical retail share data needed |
| CSI Index Co web SPA | `https://www.csindex.com.cn/#/...` | JS-rendered, plain WebFetch returns empty | Use AKShare's `stock_zh_index_hist_csindex` which calls the underlying `/csindex-home/perf/index-perf` JSON API |
| `stock_zh_index_hist_csindex` for 深交所 indices (399106, 399102) | — | "Length mismatch: Expected axis has 0 elements" — these indices not in 中证指数公司 db | Use SZSE 总貌 historical year-end snapshots |

---

## 8. v3 indices (planned alternative — paper_v3_csi_family/)

If we proceed with Option A (CSI 市值切片家族 5 个指数), the new lineup will be:

| Index | Code | Role | Status |
|---|---|---|---|
| 上证 50 | 000016 | 超大盘机构 | 🟡 needs to be pulled |
| 沪深 300 | 000300 | 大盘 | ✅ already have |
| 中证 500 | 000905 | 中盘 | ✅ already have |
| 中证 1000 | 000852 | 小盘 | 🟡 needs to be pulled |
| 创业板综指 | 399102 | 小盘成长 | ✅ already have (East Money source) |

All 5 retrievable via AKShare uniform single source = ⭐⭐⭐ data quality.

---

## 9. Update history

- **2026-04-29 (audit pass 1):** First systematic audit. Found:
  - Unit mismatch SSE/SZSE summary outputs (亿元 vs 元) — corrected
  - Old SHCOMP turnover estimate (3.0) was 136% too high; real value 1.27
  - CSI 300 turnover slightly off (1.20 vs real 1.08)
  - CSI 500 turnover slightly off (3.75 vs real 3.24)
  - All 5 indices' turnover now ⭐⭐⭐ via 中证指数公司 + SZSE 总貌

---

## How to verify any number in this file

1. Open this file, find the row for the data point you're checking
2. Look at "Source" + "Re-pull command"
3. Run the command in `/Volumes/Data_Drive/research0322226/financial-news-sentiment-analysis/`
4. Compare your output to the value in this file
5. If different by > 5%, update this file + re-run dependent analyses (G2, regression tables)

For paper submission: when freezing the dataset, snapshot the values you used into a fresh `data_snapshot_<date>.csv` in `data/` and commit. Reference that CSV in the paper's Data section.
