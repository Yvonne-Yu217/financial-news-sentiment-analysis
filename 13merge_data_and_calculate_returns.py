#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并市场指数、文本情感和图片情感数据，并计算收益率

该脚本将：
1. 从Stock Market Index.xlsx读取市场指数数据
2. 计算各指数的日收益率
3. 合并weighted_textpes.csv和weighted_photopes.csv的情感指标
4. 按日期对齐所有数据，并处理缺失值
5. 输出合并后的数据集
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import matplotlib
# 自动切换到系统可用的中文字体，确保不报错
import matplotlib.font_manager as fm
zh_fonts = [f.name for f in fm.fontManager.ttflist if any(x in f.name for x in ['SimHei','Microsoft YaHei','STHeiti','Heiti','PingFang','Songti','FangSong','KaiTi'])]
if zh_fonts:
    matplotlib.rcParams['font.sans-serif'] = [zh_fonts[0]]
else:
    matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 兜底方案
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import matplotlib.pyplot as plt
import seaborn as sns

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_market_index_data(file_path):
    """
    加载股票市场指数数据并计算收益率
    
    参数:
        file_path: Excel文件路径
    
    返回:
        包含市场指数和收益率的DataFrame
    """
    try:
        # 读取Excel文件
        logging.info(f"读取市场指数数据: {file_path}")
        df = pd.read_excel(file_path)
        
        # 查看列名
        logging.info(f"市场指数数据列名: {df.columns.tolist()}")
        
        # 确保有日期列
        if 'Date' not in df.columns:
            # 尝试找到日期列
            date_cols = [col for col in df.columns if 'date' in col.lower() or '日期' in col.lower()]
            if date_cols:
                df = df.rename(columns={date_cols[0]: 'Date'})
            else:
                logging.error("无法识别日期列")
                return None
        
        # 确保日期列格式正确
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        
        # 按日期排序
        df = df.sort_values('Date')
        
        # 查找所有可能的指数列
        index_cols = [col for col in df.columns if col != 'Date']
        logging.info(f"识别到的指数列: {index_cols}")
        
        # 先填充所有指数列的空值（前向+后向填充，最后用均值填充）
        for col in index_cols:
            df[col] = df[col].ffill().bfill()
            if df[col].isna().any():
                col_mean = df[col].mean()
                df[col] = df[col].fillna(col_mean)
                logging.info(f"市场指数列 {col} 存在无法填充的NaN，已用均值 {col_mean:.4f} 填充")
        
        # 计算每个指数的收益率
        for col in index_cols:
            # 计算日收益率 (今天/昨天 - 1)
            returns_col = f"{col}_returns"
            df[returns_col] = df[col].pct_change(fill_method=None)
            
            # 计算对数收益率
            log_returns_col = f"{col}_log_returns"
            df[log_returns_col] = np.log(df[col] / df[col].shift(1))
        
        # 删除第一行（因为收益率计算会产生NaN）
        df = df.iloc[1:]
        
        logging.info(f"市场指数数据加载完成，共 {len(df)} 行")
        return df
    
    except Exception as e:
        logging.error(f"加载市场指数数据时出错: {e}")
        return None

def load_sentiment_data(textpes_file, photopes_file):
    """
    加载文本和图片情感数据
    
    参数:
        textpes_file: TextPes CSV文件路径
        photopes_file: PhotoPes CSV文件路径
    
    返回:
        合并后的情感数据DataFrame
    """
    try:
        # 加载TextPes数据
        logging.info(f"读取TextPes数据: {textpes_file}")
        df_text = pd.read_csv(textpes_file)
        df_text['news_date'] = pd.to_datetime(df_text['news_date'], errors='coerce')
        df_text = df_text.dropna(subset=['news_date'])
        
        # 选择需要的列
        text_cols = ['news_date', 'TextPes', 'TextPes_likelihood', 'WeightedTextPes']
        if all(col in df_text.columns for col in text_cols):
            df_text = df_text[text_cols]
        else:
            available_cols = ['news_date'] + [col for col in ['TextPes', 'TextPes_likelihood', 'WeightedTextPes'] 
                                             if col in df_text.columns]
            df_text = df_text[available_cols]
            logging.warning(f"TextPes数据缺少某些列，使用可用列: {available_cols}")
        
        # 加载PhotoPes数据
        logging.info(f"读取PhotoPes数据: {photopes_file}")
        df_photo = pd.read_csv(photopes_file)
        df_photo['news_date'] = pd.to_datetime(df_photo['news_date'], errors='coerce')
        df_photo = df_photo.dropna(subset=['news_date'])
        
        # 选择需要的列
        photo_cols = ['news_date', 'PhotoPes', 'PhotoPes_likelihood', 'WeightedPhotoPes']
        if all(col in df_photo.columns for col in photo_cols):
            df_photo = df_photo[photo_cols]
        else:
            available_cols = ['news_date'] + [col for col in ['PhotoPes', 'PhotoPes_likelihood', 'WeightedPhotoPes'] 
                                             if col in df_photo.columns]
            df_photo = df_photo[available_cols]
            logging.warning(f"PhotoPes数据缺少某些列，使用可用列: {available_cols}")
        
        # 合并两个数据集
        df_sentiment = pd.merge(df_text, df_photo, on='news_date', how='outer')
        df_sentiment = df_sentiment.sort_values('news_date')
        
        # 计算加权的likelihood指标（如果不存在）
        if 'WeightedTextPes_likelihood' not in df_sentiment.columns and 'TextPes_likelihood' in df_sentiment.columns:
            df_sentiment['WeightedTextPes_likelihood'] = df_sentiment['TextPes_likelihood']
            logging.info("添加 WeightedTextPes_likelihood 列")
        
        if 'WeightedPhotoPes_likelihood' not in df_sentiment.columns and 'PhotoPes_likelihood' in df_sentiment.columns:
            df_sentiment['WeightedPhotoPes_likelihood'] = df_sentiment['PhotoPes_likelihood']
            logging.info("添加 WeightedPhotoPes_likelihood 列")
        
        logging.info(f"情感数据加载完成，共 {len(df_sentiment)} 行")
        return df_sentiment
    
    except Exception as e:
        logging.error(f"加载情感数据时出错: {e}")
        return None

def merge_data(market_df, sentiment_df):
    """
    合并市场指数和情感数据
    
    参数:
        market_df: 市场指数DataFrame
        sentiment_df: 情感数据DataFrame
    
    返回:
        合并后的DataFrame
    """
    try:
        # 重命名日期列以便合并
        market_df_copy = market_df.copy()
        sentiment_df_copy = sentiment_df.copy()
        
        sentiment_df_copy = sentiment_df_copy.rename(columns={'news_date': 'Date'})
        
        # 合并数据
        merged_df = pd.merge(market_df_copy, sentiment_df_copy, on='Date', how='outer')
        merged_df = merged_df.sort_values('Date')
        
        # 检查缺失值（仅统计2014-2024年）
        missing_mask = merged_df.isnull().any(axis=1)
        # 只保留2014-01-01到2024-12-31之间的日期
        date_start = pd.Timestamp('2014-01-01')
        date_end = pd.Timestamp('2024-12-31')
        date_mask = (merged_df['Date'] >= date_start) & (merged_df['Date'] <= date_end)
        filtered_missing = merged_df.loc[missing_mask & date_mask, 'Date']
        missing_count = filtered_missing.shape[0]
        if missing_count > 0:
            logging.info(f"2014-2024年间有 {missing_count} 天存在缺失数据，缺失日期如下：")
            for dt in filtered_missing:
                logging.info(f"- {dt.strftime('%Y-%m-%d') if pd.notnull(dt) else str(dt)}")
        else:
            logging.info("2014-2024年间无缺失数据。")
        
        # 处理缺失值
        
        # 1. 对于情感指标，使用前向填充（使用前一天的值）
        sentiment_cols = [col for col in merged_df.columns if any(x in col for x in ['TextPes', 'PhotoPes', 'TextPes_likelihood', 'PhotoPes_likelihood'])]
        merged_df[sentiment_cols] = merged_df[sentiment_cols].ffill()
        # 2. 如果仍有缺失值（如开头部分），使用后向填充
        merged_df[sentiment_cols] = merged_df[sentiment_cols].bfill()
        # 3. 如果仍有缺失值，用均值填充
        for col in sentiment_cols:
            if merged_df[col].isna().any():
                col_mean = merged_df[col].mean()
                merged_df[col] = merged_df[col].fillna(col_mean)
                logging.info(f"列 {col} 使用均值 {col_mean:.4f} 填充缺失值")
        # 4. 市场指数和收益率不进行填充，保留NaN
        
        # 删除所有列都为NaN的行
        merged_df = merged_df.dropna(how='all')
        
        logging.info(f"数据合并完成，共 {len(merged_df)} 行")
        return merged_df
    
    except Exception as e:
        logging.error(f"合并数据时出错: {e}")
        return None

def analyze_merged_data(df):
    """
    分析合并后的数据
    
    参数:
        df: 合并后的DataFrame
    """
    try:
        # 计算基本统计量
        logging.info("计算基本统计量:")
        
        # 找出所有收益率列
        returns_cols = [col for col in df.columns if 'returns' in col.lower()]
        sentiment_cols = [col for col in df.columns if any(x in col for x in ['TextPes', 'PhotoPes', 'TextPes_likelihood', 'PhotoPes_likelihood'])]
        
        # 计算缺失日期
        missing_mask = df.isnull().any(axis=1)
        # 只保留2014-01-01到2024-12-31之间的日期
        date_start = pd.Timestamp('2014-01-01')
        date_end = pd.Timestamp('2024-12-31')
        date_mask = (df['Date'] >= date_start) & (df['Date'] <= date_end)
        filtered_missing = df.loc[missing_mask & date_mask, 'Date']
        missing_count = filtered_missing.shape[0]
        if missing_count > 0:
            logging.info(f"2014-2024年间有 {missing_count} 天存在缺失数据，缺失日期如下：")
            for dt in filtered_missing:
                logging.info(f"- {dt.strftime('%Y-%m-%d') if pd.notnull(dt) else str(dt)}")
        else:
            logging.info("2014-2024年间无缺失数据。")
    
    except Exception as e:
        logging.error(f"分析数据时出错: {e}")

def plot_time_series(df, output_dir=None):
    """
    只绘制市场指数收益率和情感指标的时间序列图。
    """
    try:
        import os
        # 检查输出目录是否存在，不存在则创建
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # 识别收益率和情感指标列
        returns_cols = [col for col in df.columns if col.endswith('_returns') or col.endswith('_log_returns')]
        sentiment_cols = [col for col in df.columns if any(x in col for x in ['TextPes', 'PhotoPes', 'TextPes_likelihood', 'PhotoPes_likelihood'])]

        # 1. 绘制收益率时间序列
        if returns_cols:
            plt.figure(figsize=(12, 6))
            for col in returns_cols:
                plt.plot(df['Date'], df[col], label=col)
            plt.title("市场指数收益率时间序列", fontsize=16)
            plt.xlabel("日期", fontsize=12)
            plt.ylabel("收益率", fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.gcf().autofmt_xdate()
            if output_dir:
                plt.savefig(os.path.join(output_dir, "market_returns.png"), dpi=300, bbox_inches='tight')
            plt.close()

        # 2. 绘制情感指标时间序列
        if sentiment_cols:
            plt.figure(figsize=(12, 6))
            for col in sentiment_cols:
                plt.plot(df['Date'], df[col], label=col)
            plt.title("情感指标时间序列", fontsize=16)
            plt.xlabel("日期", fontsize=12)
            plt.ylabel("情感指标值", fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.gcf().autofmt_xdate()
            if output_dir:
                plt.savefig(os.path.join(output_dir, "sentiment_indicators.png"), dpi=300, bbox_inches='tight')
            plt.close()

        logging.info("时间序列图绘制完成")
    except Exception as e:
        logging.error(f"绘制时间序列图时出错: {e}")

def main():
    """主函数"""
    # 文件路径
    market_file = "Stock Market Index.xlsx"
    textpes_file = "results/weighted_textpes.csv"
    photopes_file = "results/weighted_photopes.csv"
    output_file = "results/merged_market_sentiment_data.csv"
    plots_dir = "plots/merged_analysis"
    
    # 1. 加载市场指数数据并计算收益率
    market_df = load_market_index_data(market_file)
    if market_df is None:
        logging.error("无法加载市场指数数据，程序终止")
        return
    
    # 2. 加载情感数据
    sentiment_df = load_sentiment_data(textpes_file, photopes_file)
    if sentiment_df is None:
        logging.error("无法加载情感数据，程序终止")
        return
    
    # 3. 合并数据
    merged_df = merge_data(market_df, sentiment_df)
    if merged_df is None:
        logging.error("无法合并数据，程序终止")
        return
    
    # 4. 保存合并后的数据
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    merged_df.to_csv(output_file, index=False, encoding='utf-8')
    logging.info(f"合并后的数据已保存到: {output_file}")
    
    # 5. 分析合并后的数据
    analyze_merged_data(merged_df)
    
    # 6. 绘制时间序列图
    plot_time_series(merged_df, plots_dir)
    
    logging.info("处理完成")

if __name__ == "__main__":
    main()
