"""
Regenerate all paper figures in black & white, 300 DPI, AEA-compatible font sizes.
Saves to the paper's Image/chap03/ folder.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams

# ── AEA-compatible style ──
rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif', 'Palatino'],
    'font.size':        10,
    'axes.titlesize':   10,
    'axes.labelsize':   10,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'legend.fontsize':  9,
    'figure.dpi':       300,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'axes.linewidth':   0.8,
    'lines.linewidth':  1.4,
    'grid.linewidth':   0.5,
    'grid.alpha':       0.35,
    'axes.grid':        True,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
})

IMG_OUT = ("/Volumes/Data_Drive/research0322226/"
           "Quantifying_Investor_Sentiment_with_Multimodal_Data_in_the_Chinese_Stock_Market/"
           "Image/chap03")

DATA_OLD = ("/Volumes/Data_Drive/research0322226/"
            "financial-news-sentiment-analysis/results/"
            "merged_market_sentiment_data_old.csv")

METRICS_JSON = ("/Volumes/Data_Drive/research0322226/"
                "financial-news-sentiment-analysis/run_artifacts/"
                "8vit_transferlearning/8vit_training_metrics.json")

# ═══════════════════════════════════════════════════
# Figure 1: Training & Validation Accuracy (B&W)
# ═══════════════════════════════════════════════════
def fig_training_accuracy():
    with open(METRICS_JSON) as f:
        m = json.load(f)

    train_acc = [v * 100 for v in m['history']['train_accuracies']]
    val_acc   = [v * 100 for v in m['history']['val_accuracies']]
    epochs    = list(range(1, len(train_acc) + 1))

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    ax.plot(epochs, train_acc, color='black',     linestyle='-',
            marker='o', markersize=4, label='Training Accuracy')
    ax.plot(epochs, val_acc,   color='black',     linestyle='--',
            marker='s', markersize=4, label='Validation Accuracy')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(epochs)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    ax.legend(frameon=True, framealpha=0.9, edgecolor='gray')
    ax.set_ylim(60, 102)

    plt.tight_layout()
    out = f"{IMG_OUT}/training_accuracy.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════
# Figure 2: Monthly Sentiment Trends (B&W)
# ═══════════════════════════════════════════════════
def fig_monthly_sentiment():
    df = pd.read_csv(DATA_OLD, parse_dates=['Date'])
    df = df[(df['Date'] >= '2014-01-01')].copy()

    monthly = df[['Date', 'PhotoPes', 'TextPes']].dropna().copy()
    monthly['YM'] = monthly['Date'].dt.to_period('M')
    avg = monthly.groupby('YM')[['PhotoPes', 'TextPes']].mean()
    avg.index = avg.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(6.5, 3.0))

    ax.plot(avg.index, avg['PhotoPes'], color='black', linestyle='-',
            linewidth=1.4, label='PhotoPes (Image Pessimism)')
    ax.plot(avg.index, avg['TextPes'],  color='black', linestyle='--',
            linewidth=1.4, label='TextPes (Text Pessimism)')

    # Annotate major events with subtle vertical shading
    events = [
        ('2015-06-01', '2015-09-01', '2015\nCrash'),
        ('2018-01-01', '2018-12-31', '2018\nTrade War'),
        ('2020-01-01', '2020-06-30', 'COVID-19'),
    ]
    for start, end, label in events:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.10, color='gray', linewidth=0)
        mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
        ymax = ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 0.30
        ax.text(mid, 0.005, label, ha='center', va='bottom',
                fontsize=7, color='gray', style='italic')

    ax.set_xlabel('Date')
    ax.set_ylabel('Monthly Average Pessimism Index')
    ax.legend(frameon=True, framealpha=0.9, edgecolor='gray', loc='upper right')
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

    plt.tight_layout()
    out = f"{IMG_OUT}/monthly_sentiment_trends.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════
# Figure 3: Cumulative Returns (B&W)
# ═══════════════════════════════════════════════════
def fig_cumulative_returns():
    df = pd.read_csv(DATA_OLD, parse_dates=['Date'])
    df = df[(df['Date'] >= '2014-01-01')].copy()

    bt = df[['Date', 'CSI500_log_returns', 'PhotoPes_std_lag3',
             'TextPes_std_lag3']].dropna().copy()
    bt = bt.set_index('Date').sort_index()

    WINDOW = 90; REBAL = 5; W_PHOTO = 0.80; W_TEXT = 0.20
    Q_LO = 0.20; Q_HI = 0.80

    bt['combined'] = W_PHOTO * bt['PhotoPes_std_lag3'] + W_TEXT * bt['TextPes_std_lag3']
    bt['roll_lo']  = bt['combined'].rolling(WINDOW, min_periods=WINDOW).quantile(Q_LO)
    bt['roll_hi']  = bt['combined'].rolling(WINDOW, min_periods=WINDOW).quantile(Q_HI)

    bt['raw_signal'] = 0.0
    bt.loc[bt['combined'] > bt['roll_hi'], 'raw_signal'] = -1.0
    bt.loc[bt['combined'] < bt['roll_lo'], 'raw_signal'] =  1.0

    current_pos = 0.0; days_since_rebal = 0; positions = []
    for i in range(len(bt)):
        if days_since_rebal >= REBAL and bt['raw_signal'].iloc[i] != 0:
            current_pos = bt['raw_signal'].iloc[i]
            days_since_rebal = 0
        days_since_rebal += 1
        positions.append(current_pos)

    bt['position']     = positions
    bt['strategy_ret'] = bt['position'].shift(1) * bt['CSI500_log_returns']
    bt['strategy_ret'] = bt['strategy_ret'].fillna(0)
    bt['strategy_cum'] = (1 + bt['strategy_ret']).cumprod()
    bt['benchmark_cum']= (1 + bt['CSI500_log_returns']).cumprod()

    fig, ax = plt.subplots(figsize=(6.5, 3.0))

    ax.plot(bt.index, bt['strategy_cum'],  color='black', linestyle='-',
            linewidth=1.5, label='Sentiment Strategy')
    ax.plot(bt.index, bt['benchmark_cum'], color='black', linestyle='--',
            linewidth=1.5, label='CSI 500 Index')

    ax.axhline(1.0, color='gray', linewidth=0.7, linestyle=':', alpha=0.7)

    ax.set_xlabel('Date')
    ax.set_ylabel('Growth of 1 RMB')
    ax.legend(frameon=True, framealpha=0.9, edgecolor='gray', loc='upper left')
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

    plt.tight_layout()
    out = f"{IMG_OUT}/cumulative_returns_old.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == '__main__':
    import matplotlib.dates
    fig_training_accuracy()
    fig_monthly_sentiment()
    fig_cumulative_returns()
    print("All figures regenerated in B&W at 300 DPI.")
