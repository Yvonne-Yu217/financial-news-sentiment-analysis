#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算每日加权TextPes指标脚本

该脚本从MongoDB数据库中读取新闻文本内容，分析其情感倾向，
根据公式 WeightedTextPes_t = Σ(Textneg_it × W_i) / Σ(W_i) 计算每日加权TextPes指标。
其中:
- Textneg_it 是第i篇新闻在t日的负面情绪概率
- W_i 是该新闻的质量分数(quality_score)，如果不存在则使用1.0
"""

import os
import sys
import logging
import argparse
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from pymongo import MongoClient
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('calculate_daily_textpes.log', encoding='utf-8')
    ]
)

# 全局配置
BATCH_SIZE = 8  # 批处理大小，适应M1内存
MAX_LENGTH = 512  # 最大文本长度
# 检测是否为M1 Mac并设置设备
is_mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
DEVICE = torch.device("mps" if is_mps_available else "cpu")

# 使用Erlangshen-Roberta-110M-Sentiment模型
MODEL_NAME = 'IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment'

# 情感标签映射（二分类）
SENTIMENT_LABELS = ['负面', '正面']

class NewsDataset(Dataset):
    """新闻文本数据集"""
    
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

def connect_to_mongodb():
    """连接到MongoDB数据库"""
    try:
        # 连接到MongoDB
        client = MongoClient('localhost', 27017)
        db = client['sina_news_dataset_test']
        logging.info("成功连接到MongoDB")
        return db
    except Exception as e:
        logging.error(f"连接MongoDB失败: {e}")
        sys.exit(1)

def load_model_and_tokenizer():
    """加载预训练模型和分词器"""
    logging.info(f"加载模型: {MODEL_NAME}，使用设备: {DEVICE}")
    
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
        model = model.to(DEVICE)
        model.eval()
        return model, tokenizer, 2  # 固定为二分类
    except Exception as e:
        logging.error(f"加载模型失败: {e}")
        raise

def predict_sentiment(model, tokenizer, texts, num_labels):
    """预测文本情感"""
    dataset = NewsDataset(texts, tokenizer, MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            batch_predictions = probs.detach().cpu().numpy()
            predictions.extend(batch_predictions)
    
    return np.array(predictions)

def fetch_news_by_date(collection, news_date):
    """从数据库获取指定日期的新闻"""
    # 只获取 has_valid_images 为 true 的记录
    cursor = collection.find({'news_date': news_date, 'has_valid_images': True})
    news_list = list(cursor)
    texts = []
    news_ids = []
    
    for news in news_list:
        # 确保有内容字段
        if 'content' in news and news['content']:
            content = news['content'][:MAX_LENGTH * 4]
            texts.append(content)
            news_ids.append(news['_id'])
    
    return texts, news_ids, news_list

def calculate_daily_textpes(db, years, output_file=None, text_collection_suffix='_filtered'):
    """
    计算每日加权TextPes指标
    
    参数:
        db: MongoDB数据库连接
        years: 要处理的年份列表
        output_file: 输出文件路径，如果为None则不保存到文件
        text_collection_suffix: 文本集合后缀，默认为'_filtered'
    
    返回:
        包含每日加权TextPes指标的DataFrame
    """
    # 加载情感分析模型
    model, tokenizer, num_labels = load_model_and_tokenizer()
    
    all_results = []
    
    for year in years:
        if isinstance(year, int):
            year = str(year)
            
        logging.info(f"处理 {year} 年的数据...")
        
        # 获取相关集合
        text_collection_name = f"{year}{text_collection_suffix}"
        news_sentiment_collection_name = f"{year}_news_sentiment"
        collections = db.list_collection_names()
        
        # 检查集合是否存在
        if text_collection_name not in collections:
            logging.warning(f"{text_collection_name} 集合不存在，跳过处理")
            continue
        
        # 获取文本情感集合（不删除已有集合，而是更新）
        text_collection = db[text_collection_name]
        news_sentiment_collection = db[news_sentiment_collection_name]
        
        # 获取不同的日期
        date_cursor = text_collection.distinct('news_date', {'has_valid_images': True})
        date_list = sorted([d for d in date_cursor if d])  # 过滤掉空日期
        
        total_days = len(date_list)
        if not date_list:
            logging.warning(f"{year}年没有找到有效的日期数据")
            continue
        
        logging.info(f"开始处理 {year} 年的 {total_days} 个日期")
        
        # 用于存储每日的加权指标
        daily_results = {}
        
        progress_bar = tqdm(date_list, desc=f"{year}年进度")
        for news_date in progress_bar:
            texts, news_ids, news_list = fetch_news_by_date(text_collection, news_date)
            
            if not texts:
                continue
            
            # 分析情感
            predictions = predict_sentiment(model, tokenizer, texts, num_labels)
            predicted_classes = np.argmax(predictions, axis=1)
            
            # 计算基础指标
            negative_count = np.sum(predicted_classes == 0)
            positive_count = np.sum(predicted_classes == 1)
            total_count = len(predictions)
            text_pes = negative_count / total_count if total_count > 0 else 0
            
            # 计算负面概率平均值
            daily_textneg_likelihoods = [float(pred[0]) for pred in predictions]
            textpes_likelihood = np.mean(daily_textneg_likelihoods) if daily_textneg_likelihoods else 0.0
            
            # 获取或创建质量分数
            text_qualities = []
            sum_weighted_negative = 0.0
            sum_weights = 0.0
            
            # 处理每篇新闻
            for i, news_id in enumerate(news_ids):
                # 获取情感分析结果
                sentiment_class = int(predicted_classes[i])
                sentiment_probs = predictions[i].tolist()
                sentiment_label = SENTIMENT_LABELS[sentiment_class]
                textneg_likelihood = float(sentiment_probs[0])
                
                # 获取质量分数（如果可用）
                news = news_list[i]
                quality_score = 1.0  # 默认质量分数
                
                if 'avg_quality_score' in news:
                    quality_score = news['avg_quality_score']
                
                # 计算加权分数
                text_qualities.append(quality_score)
                sum_weighted_negative += textneg_likelihood * quality_score
                sum_weights += quality_score
                
                # 保存到文本情感集合
                text_sentiment_record = {
                    'original_id': news.get('original_id'),
                    'news_date': news_date,
                    'title': news.get('title', ''),
                    'sentiment_class': sentiment_class,
                    'sentiment_label': sentiment_label,
                    'textneg_likelihood': textneg_likelihood,
                    'processed_time': datetime.now()
                }
                
                # 检查是否已存在该记录，如果存在则更新，否则插入
                news_sentiment_collection.update_one(
                    {'original_id': news.get('original_id'), 'title': news.get('title', '')},
                    {'$set': text_sentiment_record},
                    upsert=True
                )
            
            # 计算加权TextPes
            weighted_textpes = sum_weighted_negative / sum_weights if sum_weights > 0 else textpes_likelihood
            avg_quality_score = np.mean(text_qualities) if text_qualities else 1.0
            
            # 添加到日期结果中
            daily_result = {
                'news_date': news_date,
                'total_count': total_count,
                'negative_count': negative_count,
                'positive_count': positive_count,
                'TextPes': text_pes,
                'TextPes_likelihood': textpes_likelihood,
                'WeightedTextPes': weighted_textpes,
                'avg_quality_score': avg_quality_score,
                'year': year
            }
            
            all_results.append(daily_result)
            
            # 更新进度条描述
            progress_bar.set_description(f"{year}年: 处理 {news_date} ({total_count}篇新闻)")
        
        logging.info(f"{year} 年共处理了 {len(all_results) - len(daily_results)} 天的数据")
    
    # 转换为DataFrame
    if not all_results:
        logging.warning("没有找到任何有效的文本情感分析数据")
        return None
    
    df = pd.DataFrame(all_results)
    
    # 确保日期格式正确
    df["news_date"] = pd.to_datetime(df["news_date"], errors='coerce')
    # 删除无效日期的行
    df = df.dropna(subset=["news_date"])
    df = df.sort_values("news_date")
    
    # 计算统计信息
    logging.info(f"总共有 {len(df)} 天的数据")
    logging.info(f"WeightedTextPes 均值: {df['WeightedTextPes'].mean():.4f}")
    logging.info(f"WeightedTextPes 标准差: {df['WeightedTextPes'].std():.4f}")
    logging.info(f"WeightedTextPes 最小值: {df['WeightedTextPes'].min():.4f}")
    logging.info(f"WeightedTextPes 最大值: {df['WeightedTextPes'].max():.4f}")
    
    # 保存到文件
    if output_file:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 根据文件扩展名保存
        ext = os.path.splitext(output_file)[1].lower()
        if ext == '.csv':
            df.to_csv(output_file, index=False, encoding='utf-8')
        elif ext == '.xlsx':
            df.to_excel(output_file, index=False, engine='openpyxl')
        elif ext == '.json':
            df.to_json(output_file, orient='records', date_format='iso')
        else:
            # 默认保存为CSV
            df.to_csv(output_file, index=False, encoding='utf-8')
            
        logging.info(f"已将结果保存到文件: {output_file}")
    
    return df

def plot_weighted_textpes(df, output_dir=None):
    """
    绘制WeightedTextPes指标的时间序列图
    
    参数:
        df: 包含WeightedTextPes指标的DataFrame
        output_dir: 输出目录，如果为None则不保存图表
    """
    if df is None or df.empty:
        logging.warning("没有数据可供绘图")
        return
    
    # 设置绘图风格
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    # 绘制WeightedTextPes时间序列
    plt.plot(df["news_date"], df["WeightedTextPes"], marker='o', linestyle='-', markersize=3, alpha=0.7)
    
    # 添加标题和标签
    plt.title("每日加权TextPes指标 (WeightedTextPes)", fontsize=16)
    plt.xlabel("日期", fontsize=12)
    plt.ylabel("WeightedTextPes", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 添加趋势线
    z = np.polyfit(range(len(df)), df["WeightedTextPes"], 1)
    p = np.poly1d(z)
    plt.plot(df["news_date"], p(range(len(df))), "r--", alpha=0.7, label=f"趋势线 (斜率: {z[0]:.6f})")
    
    # 添加图例
    plt.legend()
    
    # 调整x轴日期显示
    plt.gcf().autofmt_xdate()
    
    # 添加均值线
    mean_val = df["WeightedTextPes"].mean()
    plt.axhline(y=mean_val, color='g', linestyle='--', alpha=0.7, label=f"均值: {mean_val:.4f}")
    
    # 保存图表
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"WeightedTextPes_{timestamp}.png")
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logging.info(f"已保存图表到: {filename}")
    
    plt.tight_layout()
    plt.show()

def compare_text_pes_indicators(df, output_dir=None):
    """
    比较不同TextPes指标
    
    参数:
        df: 包含各种TextPes指标的DataFrame
        output_dir: 输出目录，如果为None则不保存图表
    """
    if df is None or df.empty:
        logging.warning("没有数据可供绘图")
        return
    
    # 设置绘图风格
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # 绘制三种TextPes指标
    plt.plot(df["news_date"], df["TextPes"], marker='.', linestyle='-', alpha=0.6, label="TextPes (比例)")
    plt.plot(df["news_date"], df["TextPes_likelihood"], marker='.', linestyle='-', alpha=0.6, label="TextPes_likelihood (平均概率)")
    plt.plot(df["news_date"], df["WeightedTextPes"], marker='.', linestyle='-', alpha=0.6, label="WeightedTextPes (加权概率)")
    
    # 添加标题和标签
    plt.title("TextPes指标比较", fontsize=16)
    plt.xlabel("日期", fontsize=12)
    plt.ylabel("指标值", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 添加图例
    plt.legend()
    
    # 调整x轴日期显示
    plt.gcf().autofmt_xdate()
    
    # 保存图表
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"TextPes_Comparison_{timestamp}.png")
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logging.info(f"已保存图表到: {filename}")
    
    plt.tight_layout()
    plt.show()
    
    # 计算相关性
    correlation_matrix = df[["TextPes", "TextPes_likelihood", "WeightedTextPes"]].corr()
    logging.info("TextPes指标相关性矩阵:")
    logging.info(f"\n{correlation_matrix}")
    
    # 绘制相关性热图
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0)
    plt.title("TextPes指标相关性矩阵", fontsize=14)
    
    # 保存相关性热图
    if output_dir:
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"TextPes_Correlation_{timestamp}.png")
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logging.info(f"已保存相关性热图到: {filename}")
    
    plt.tight_layout()
    plt.show()

def update_excel_with_textpes(df, excel_file):
    """
    更新Excel文件，添加WeightedTextPes指标
    
    参数:
        df: 包含WeightedTextPes指标的DataFrame
        excel_file: Excel文件路径
    """
    if df is None or df.empty:
        logging.warning("没有数据可供更新Excel")
        return
    
    try:
        # 检查Excel文件是否存在
        if not os.path.exists(excel_file):
            logging.error(f"Excel文件不存在: {excel_file}")
            return
        
        # 读取Excel文件
        logging.info(f"正在读取Excel文件: {excel_file}")
        df_excel = pd.read_excel(excel_file)
        
        # 确保Excel中有Date列
        if "Date" not in df_excel.columns:
            logging.error("Excel文件中没有Date列")
            return
        
        # 确保Date列是日期类型
        df_excel["Date"] = pd.to_datetime(df_excel["Date"])
        
        # 重命名df中的news_date列为Date，以便合并
        df_merge = df.copy()
        df_merge = df_merge.rename(columns={"news_date": "Date"})
        
        # 选择要合并的列
        df_merge = df_merge[["Date", "WeightedTextPes", "TextPes", "TextPes_likelihood"]]
        
        # 合并数据
        df_updated = pd.merge(df_excel, df_merge, on="Date", how="left")
        
        # 保存更新后的Excel文件
        output_file = excel_file.replace(".xlsx", "_with_WeightedTextPes.xlsx")
        df_updated.to_excel(output_file, index=False, engine='openpyxl')
        
        logging.info(f"已将WeightedTextPes指标添加到Excel文件: {output_file}")
        
        # 打印更新统计
        total_rows = len(df_excel)
        updated_rows = df_updated["WeightedTextPes"].notna().sum()
        logging.info(f"总行数: {total_rows}")
        logging.info(f"更新行数: {updated_rows}")
        logging.info(f"更新比例: {updated_rows/total_rows*100:.2f}%")
        
    except Exception as e:
        logging.error(f"更新Excel文件时出错: {e}")

def fix_sentiment_class_label_inversion(db, years):
    """
    修正 sentiment_label 与 sentiment_class 的对应关系：
    - sentiment_label 为“正面”的 sentiment_class 改为 0
    - sentiment_label 为“负面”的 sentiment_class 改为 1
    """
    for year in years:
        if isinstance(year, int):
            year = str(year)
        collection_name = f"{year}_news_sentiment"
        if collection_name not in db.list_collection_names():
            continue
        collection = db[collection_name]
        # 正面 -> 0
        result1 = collection.update_many(
            {"sentiment_label": "正面", "sentiment_class": {"$ne": 0}},
            {"$set": {"sentiment_class": 0}}
        )
        # 负面 -> 1
        result2 = collection.update_many(
            {"sentiment_label": "负面", "sentiment_class": {"$ne": 1}},
            {"$set": {"sentiment_class": 1}}
        )
        logging.info(f"{collection_name}: 修正正面为0 {result1.modified_count} 条，负面为1 {result2.modified_count} 条")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="计算每日加权TextPes指标")
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2014, 2025)),
                        help="要处理的年份列表")
    parser.add_argument("--output", type=str, default="results/weighted_textpes.csv",
                        help="输出文件路径")
    parser.add_argument("--plot", action="store_true",
                        help="是否绘制图表")
    parser.add_argument("--plot-dir", type=str, default="plots",
                        help="图表保存目录")
    parser.add_argument("--compare", action="store_true",
                        help="是否比较不同TextPes指标")
    parser.add_argument("--update-excel", type=str, default=None,
                        help="要更新的Excel文件路径")
    parser.add_argument("--collection-suffix", type=str, default="_filtered",
                        help="文本集合后缀，默认为'_filtered'")
    
    args = parser.parse_args()
    
    # 连接数据库
    db = connect_to_mongodb()
    
    # 计算每日加权TextPes指标
    df = calculate_daily_textpes(db, args.years, args.output, args.collection_suffix)
    
    # 修正 sentiment_class 和 sentiment_label 的对应关系
    fix_sentiment_class_label_inversion(db, args.years)
    
    # 绘制图表
    if args.plot and df is not None:
        plot_weighted_textpes(df, args.plot_dir)
    
    # 比较不同TextPes指标
    if args.compare and df is not None:
        compare_text_pes_indicators(df, args.plot_dir)
    
    # 更新Excel文件
    if args.update_excel and df is not None:
        update_excel_with_textpes(df, args.update_excel)
    
    logging.info("处理完成")

if __name__ == "__main__":
    main()
