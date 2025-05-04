#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算每日加权PhotoPes指标脚本

该脚本从MongoDB数据库中读取图片情感分析结果，
根据公式 WeightedPhotoPes_t = Σ(Neg_it × W_i) / Σ(W_i) 计算每日加权PhotoPes指标。
其中:
- Neg_it 是第i张图片在t日的负面情绪概率
- W_i 是该图片的质量分数(quality_score)
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from pymongo import MongoClient
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('calculate_daily_photopes.log', encoding='utf-8')
    ]
)

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

def calculate_daily_photopes(db, years, output_file=None):
    """
    计算每日加权PhotoPes指标
    
    参数:
        db: MongoDB数据库连接
        years: 要处理的年份列表
        output_file: 输出文件路径，如果为None则不保存到文件
    
    返回:
        包含每日加权PhotoPes指标的DataFrame
    """
    all_results = []
    
    for year in years:
        if isinstance(year, int):
            year = str(year)
            
        logging.info(f"处理 {year} 年的数据...")
        
        # 获取情感分析集合和过滤集合
        sentiment_collection_name = f"{year}_sentiment"
        filtered_collection_name = f"{year}_filtered"
        
        # 检查集合是否存在
        collections = db.list_collection_names()
        if sentiment_collection_name not in collections:
            logging.warning(f"{sentiment_collection_name} 集合不存在，跳过处理")
            continue
        
        if filtered_collection_name not in collections:
            logging.warning(f"{filtered_collection_name} 集合不存在，跳过处理")
            continue
        
        sentiment_collection = db[sentiment_collection_name]
        filtered_collection = db[filtered_collection_name]
        
        # 创建日期映射: original_id -> news_date
        date_mapping = {}
        for doc in filtered_collection.find({}, {"original_id": 1, "news_date": 1}):
            original_id = doc.get("original_id")
            news_date = doc.get("news_date")
            
            if original_id and news_date:
                date_mapping[original_id] = news_date
        
        logging.info(f"从 {filtered_collection_name} 获取到 {len(date_mapping)} 条日期映射")
        
        # 按日期分组处理情感数据
        daily_results = {}
        
        # 处理每条情感记录
        for doc in sentiment_collection.find({}):
            original_id = doc.get("original_id")
            
            # 如果没有original_id或者在映射中找不到日期，跳过
            if not original_id or original_id not in date_mapping:
                continue
            
            # 获取日期
            news_date = date_mapping[original_id]
            
            # 获取情感分析结果
            negative_likelihood = doc.get("negative_likelihood")
            quality_score = doc.get("quality_score", 0.5)  # 默认值为0.5
            predicted_class = doc.get("predicted_class", 0)
            
            # 如果没有负面概率，跳过
            if negative_likelihood is None:
                continue
            
            # 初始化日期记录
            if news_date not in daily_results:
                daily_results[news_date] = {
                    "total_images": 0,
                    "negative_count": 0,
                    "sum_negative_likelihood": 0.0,
                    "sum_weighted_negative": 0.0,
                    "sum_weights": 0.0,
                    "sum_quality_score": 0.0
                }
            
            # 更新日期记录
            daily_results[news_date]["total_images"] += 1
            daily_results[news_date]["negative_count"] += 1 if predicted_class == 1 else 0
            daily_results[news_date]["sum_negative_likelihood"] += negative_likelihood
            daily_results[news_date]["sum_weighted_negative"] += negative_likelihood * quality_score
            daily_results[news_date]["sum_weights"] += quality_score
            daily_results[news_date]["sum_quality_score"] += quality_score
        
        # 计算每日指标
        for news_date, data in daily_results.items():
            total_images = data["total_images"]
            
            if total_images == 0:
                continue
            
            # 计算指标
            photopes = data["negative_count"] / total_images
            photopes_likelihood = data["sum_negative_likelihood"] / total_images
            
            # 计算加权PhotoPes
            sum_weights = data["sum_weights"]
            weighted_photopes = 0.0
            if sum_weights > 0:
                weighted_photopes = data["sum_weighted_negative"] / sum_weights
            
            # 计算平均质量分数
            avg_quality_score = data["sum_quality_score"] / total_images
            
            # 计算平均负面概率
            avg_negative_likelihood = data["sum_negative_likelihood"] / total_images
            
            # 添加到结果列表
            result = {
                "news_date": news_date,
                "total_images": total_images,
                "negative_count": data["negative_count"],
                "PhotoPes": photopes,
                "PhotoPes_likelihood": photopes_likelihood,
                "WeightedPhotoPes": weighted_photopes,
                "avg_quality_score": avg_quality_score,
                "avg_negative_likelihood": avg_negative_likelihood,
                "year": year
            }
            
            all_results.append(result)
        
        logging.info(f"{year} 年共有 {len(daily_results)} 天的数据")
    
    if not all_results:
        logging.warning("没有找到任何有效的情感分析数据")
        return None
    
    # 转换为DataFrame
    df = pd.DataFrame(all_results)
    
    # 确保日期格式正确
    df["news_date"] = pd.to_datetime(df["news_date"], errors='coerce')
    # 删除无效日期的行
    df = df.dropna(subset=["news_date"])
    df = df.sort_values("news_date")
    
    # 计算统计信息
    logging.info(f"总共有 {len(df)} 天的数据")
    logging.info(f"WeightedPhotoPes 均值: {df['WeightedPhotoPes'].mean():.4f}")
    logging.info(f"WeightedPhotoPes 标准差: {df['WeightedPhotoPes'].std():.4f}")
    logging.info(f"WeightedPhotoPes 最小值: {df['WeightedPhotoPes'].min():.4f}")
    logging.info(f"WeightedPhotoPes 最大值: {df['WeightedPhotoPes'].max():.4f}")
    
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

def plot_weighted_photopes(df, output_dir=None):
    """
    绘制WeightedPhotoPes指标的时间序列图
    
    参数:
        df: 包含WeightedPhotoPes指标的DataFrame
        output_dir: 输出目录，如果为None则不保存图表
    """
    if df is None or df.empty:
        logging.warning("没有数据可供绘图")
        return
    
    # 设置绘图风格
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    # 绘制WeightedPhotoPes时间序列
    plt.plot(df["news_date"], df["WeightedPhotoPes"], marker='o', linestyle='-', markersize=3, alpha=0.7)
    
    # 添加标题和标签
    plt.title("每日加权PhotoPes指标 (WeightedPhotoPes)", fontsize=16)
    plt.xlabel("日期", fontsize=12)
    plt.ylabel("WeightedPhotoPes", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 添加趋势线
    z = np.polyfit(range(len(df)), df["WeightedPhotoPes"], 1)
    p = np.poly1d(z)
    plt.plot(df["news_date"], p(range(len(df))), "r--", alpha=0.7, label=f"趋势线 (斜率: {z[0]:.6f})")
    
    # 添加图例
    plt.legend()
    
    # 调整x轴日期显示
    plt.gcf().autofmt_xdate()
    
    # 添加均值线
    mean_val = df["WeightedPhotoPes"].mean()
    plt.axhline(y=mean_val, color='g', linestyle='--', alpha=0.7, label=f"均值: {mean_val:.4f}")
    
    # 保存图表
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"WeightedPhotoPes_{timestamp}.png")
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logging.info(f"已保存图表到: {filename}")
    
    plt.tight_layout()
    plt.show()

def compare_photopes_indicators(df, output_dir=None):
    """
    比较不同PhotoPes指标
    
    参数:
        df: 包含各种PhotoPes指标的DataFrame
        output_dir: 输出目录，如果为None则不保存图表
    """
    if df is None or df.empty:
        logging.warning("没有数据可供绘图")
        return
    
    # 设置绘图风格
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # 绘制三种PhotoPes指标
    plt.plot(df["news_date"], df["PhotoPes"], marker='.', linestyle='-', alpha=0.6, label="PhotoPes (比例)")
    plt.plot(df["news_date"], df["PhotoPes_likelihood"], marker='.', linestyle='-', alpha=0.6, label="PhotoPes_likelihood (平均概率)")
    plt.plot(df["news_date"], df["WeightedPhotoPes"], marker='.', linestyle='-', alpha=0.6, label="WeightedPhotoPes (加权概率)")
    
    # 添加标题和标签
    plt.title("PhotoPes指标比较", fontsize=16)
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
        filename = os.path.join(output_dir, f"PhotoPes_Comparison_{timestamp}.png")
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logging.info(f"已保存图表到: {filename}")
    
    plt.tight_layout()
    plt.show()
    
    # 计算相关性
    correlation_matrix = df[["PhotoPes", "PhotoPes_likelihood", "WeightedPhotoPes"]].corr()
    logging.info("PhotoPes指标相关性矩阵:")
    logging.info(f"\n{correlation_matrix}")
    
    # 绘制相关性热图
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0)
    plt.title("PhotoPes指标相关性矩阵", fontsize=14)
    
    # 保存相关性热图
    if output_dir:
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"PhotoPes_Correlation_{timestamp}.png")
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logging.info(f"已保存相关性热图到: {filename}")
    
    plt.tight_layout()
    plt.show()

def update_excel_with_photopes(df, excel_file):
    """
    更新Excel文件，添加WeightedPhotoPes指标
    
    参数:
        df: 包含WeightedPhotoPes指标的DataFrame
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
        df_merge = df_merge[["Date", "WeightedPhotoPes", "PhotoPes", "PhotoPes_likelihood"]]
        
        # 合并数据
        df_updated = pd.merge(df_excel, df_merge, on="Date", how="left")
        
        # 保存更新后的Excel文件
        output_file = excel_file.replace(".xlsx", "_with_WeightedPhotoPes.xlsx")
        df_updated.to_excel(output_file, index=False, engine='openpyxl')
        
        logging.info(f"已将WeightedPhotoPes指标添加到Excel文件: {output_file}")
        
        # 打印更新统计
        total_rows = len(df_excel)
        updated_rows = df_updated["WeightedPhotoPes"].notna().sum()
        logging.info(f"总行数: {total_rows}")
        logging.info(f"更新行数: {updated_rows}")
        logging.info(f"更新比例: {updated_rows/total_rows*100:.2f}%")
        
    except Exception as e:
        logging.error(f"更新Excel文件时出错: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="计算每日加权PhotoPes指标")
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2014, 2025)),
                        help="要处理的年份列表")
    parser.add_argument("--output", type=str, default="results/weighted_photopes.csv",
                        help="输出文件路径")
    parser.add_argument("--plot", action="store_true",
                        help="是否绘制图表")
    parser.add_argument("--plot-dir", type=str, default="plots",
                        help="图表保存目录")
    parser.add_argument("--compare", action="store_true",
                        help="是否比较不同PhotoPes指标")
    parser.add_argument("--update-excel", type=str, default=None,
                        help="要更新的Excel文件路径")
    
    args = parser.parse_args()
    
    # 连接数据库
    db = connect_to_mongodb()
    
    # 计算每日加权PhotoPes指标
    df = calculate_daily_photopes(db, args.years, args.output)
    
    # 绘制图表
    if args.plot and df is not None:
        plot_weighted_photopes(df, args.plot_dir)
    
    # 比较不同PhotoPes指标
    if args.compare and df is not None:
        compare_photopes_indicators(df, args.plot_dir)
    
    # 更新Excel文件
    if args.update_excel and df is not None:
        update_excel_with_photopes(df, args.update_excel)
    
    logging.info("处理完成")

if __name__ == "__main__":
    main()
