#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新计算图片质量分数脚本

该脚本从MongoDB数据库中读取图片的text_weight、clarity_weight和diversity_score，
根据用户定义的新权重比例重新计算final_score，并更新到数据库中。
"""

import os
import sys
import logging
import argparse
import numpy as np
from pymongo import MongoClient
from tqdm import tqdm
import math

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('recalculate_quality_scores.log', encoding='utf-8')
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

def apply_nonlinear_transformation(score, steepness=1.0, midpoint=0.5):
    """
    应用非线性变换到分数
    使用tanh函数将分数映射到0-1范围，并保持平滑过渡
    """
    # 将分数缩放到合适的范围
    scaled_score = steepness * (score - midpoint)
    # 应用tanh函数并缩放结果到0-1
    transformed = (math.tanh(scaled_score) + 1) / 2
    return transformed

def recalculate_quality_scores(db, year, text_weight_ratio, clarity_weight_ratio, diversity_weight_ratio, apply_nonlinear=False):
    """
    重新计算指定年份的图片质量分数
    
    参数:
        db: MongoDB数据库连接
        year: 要处理的年份
        text_weight_ratio: 文字权重在最终分数中的比例
        clarity_weight_ratio: 清晰度权重在最终分数中的比例
        diversity_weight_ratio: 多样性权重在最终分数中的比例
        apply_nonlinear: 是否应用非线性变换
    """
    # 确保权重比例总和为1
    total_ratio = text_weight_ratio + clarity_weight_ratio + diversity_weight_ratio
    if abs(total_ratio - 1.0) > 0.001:
        logging.warning(f"权重比例总和不为1 ({total_ratio})，将进行归一化处理")
        text_weight_ratio /= total_ratio
        clarity_weight_ratio /= total_ratio
        diversity_weight_ratio /= total_ratio
    
    # 获取相关集合
    quality_collection_name = f"{year}_2"
    sentiment_collection_name = f"{year}_sentiment"
    news_sentiment_collection_name = f"{year}_news_sentiment"
    
    # 检查集合是否存在
    collections = db.list_collection_names()
    if quality_collection_name not in collections:
        logging.warning(f"{quality_collection_name} 集合不存在，跳过处理")
        return
    
    quality_collection = db[quality_collection_name]
    
    # 查询需要更新的文档数量
    total_docs = quality_collection.count_documents({})
    if total_docs == 0:
        logging.warning(f"{quality_collection_name} 集合中没有文档")
        return
    
    logging.info(f"开始重新计算 {year} 年的图片质量分数，共 {total_docs} 条记录")
    logging.info(f"权重比例: 文字={text_weight_ratio:.2f}, 清晰度={clarity_weight_ratio:.2f}, 多样性={diversity_weight_ratio:.2f}")
    
    # 使用进度条
    with tqdm(total=total_docs, desc=f"重新计算 {year} 年质量分数") as pbar:
        # 遍历每条记录
        for doc in quality_collection.find({}):
            try:
                original_id = doc.get("original_id")
                processed_images = doc.get("processed_images", [])
                diversity_score = doc.get("diversity_score", 1.0)
                
                # 如果没有处理过的图片，跳过
                if not processed_images:
                    pbar.update(1)
                    continue
                
                # 重新计算每张图片的质量分数
                updated_images = []
                total_quality_score = 0
                
                for img in processed_images:
                    # 获取原始权重
                    text_w = img.get("text_weight", 0.5)
                    clarity_w = img.get("clarity_weight", 0.5)
                    
                    # 计算新的质量分数
                    base_score = (
                        text_w * text_weight_ratio + 
                        clarity_w * clarity_weight_ratio
                    )
                    
                    # 应用多样性因子
                    if diversity_weight_ratio > 0:
                        final_score = base_score * (1 - diversity_weight_ratio + diversity_weight_ratio * diversity_score)
                    else:
                        final_score = base_score
                    
                    # 应用非线性变换（如果启用）
                    if apply_nonlinear:
                        final_score = apply_nonlinear_transformation(final_score)
                    
                    # 更新图片对象
                    img_copy = img.copy()
                    img_copy["quality_score"] = float(final_score)
                    img_copy["is_high_quality"] = bool(final_score >= 0.5)  # 可以根据需要调整阈值
                    updated_images.append(img_copy)
                    total_quality_score += final_score
                
                # 计算平均质量分数
                avg_quality_score = total_quality_score / len(updated_images) if updated_images else 0
                
                # 更新数据库
                quality_collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {
                        "processed_images": updated_images,
                        "avg_quality_score": float(avg_quality_score)
                    }}
                )
            
            except Exception as e:
                logging.error(f"处理记录出错 {original_id if 'original_id' in locals() else '未知ID'}: {e}")
            
            # 更新进度条
            pbar.update(1)
    
    # 同步更新情感分析集合中的quality_score字段
    update_sentiment_collections(db, year, quality_collection_name)
    
    logging.info(f"{year} 年的图片质量分数重新计算完成")


def update_sentiment_collections(db, year, quality_collection_name):
    """
    同步更新情感分析集合中的quality_score字段
    
    参数:
        db: MongoDB数据库连接
        year: 要处理的年份
        quality_collection_name: 质量评分集合名称
    """
    sentiment_collection_name = f"{year}_sentiment"
    news_sentiment_collection_name = f"{year}_news_sentiment"
    
    # 检查情感分析集合是否存在
    collections = db.list_collection_names()
    if sentiment_collection_name not in collections:
        logging.warning(f"{sentiment_collection_name} 集合不存在，跳过更新")
        return
    
    quality_collection = db[quality_collection_name]
    sentiment_collection = db[sentiment_collection_name]
    
    # 1. 更新单图片情感集合 (year_sentiment)
    logging.info(f"开始更新 {sentiment_collection_name} 集合中的quality_score...")
    
    # 创建映射: original_id -> {image_path -> quality_score}
    quality_score_mapping = {}
    
    for doc in quality_collection.find({}, {"original_id": 1, "processed_images": 1}):
        original_id = doc.get("original_id")
        processed_images = doc.get("processed_images", [])
        
        if not original_id or not processed_images:
            continue
        
        # 为每个original_id创建图片路径到质量分数的映射
        if original_id not in quality_score_mapping:
            quality_score_mapping[original_id] = {}
        
        # 遍历处理过的图片
        for i, img in enumerate(processed_images):
            # 使用索引作为标识符，因为可能没有唯一的图片路径
            quality_score_mapping[original_id][i] = img.get("quality_score", 0.5)
    
    # 更新单图片情感集合
    updated_count = 0
    for doc in sentiment_collection.find({}):
        original_id = doc.get("original_id")
        image_index = doc.get("image_index", -1)  # 如果有图片索引字段
        
        if original_id in quality_score_mapping:
            # 如果有image_index字段且有效
            if image_index >= 0 and image_index in quality_score_mapping[original_id]:
                quality_score = quality_score_mapping[original_id][image_index]
                
                # 更新quality_score，但保留情感分析相关字段
                sentiment_collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"quality_score": float(quality_score)}}
                )
                updated_count += 1
            # 如果没有image_index，尝试通过rel_path匹配
            elif "rel_path" in doc:
                rel_path = doc["rel_path"]
                # 在这里需要更复杂的匹配逻辑，简化处理
                # 使用第一个图片的质量分数
                if quality_score_mapping[original_id]:
                    quality_score = list(quality_score_mapping[original_id].values())[0]
                    sentiment_collection.update_one(
                        {"_id": doc["_id"]},
                        {"$set": {"quality_score": float(quality_score)}}
                    )
                    updated_count += 1
    
    logging.info(f"已更新 {updated_count} 条单图片情感记录")
    
    # 2. 更新新闻级别情感集合 (year_news_sentiment)
    if news_sentiment_collection_name in collections:
        logging.info(f"开始更新 {news_sentiment_collection_name} 集合中的avg_quality_score...")
        
        news_sentiment_collection = db[news_sentiment_collection_name]
        
        # 创建映射: original_id -> avg_quality_score
        avg_quality_score_mapping = {}
        
        for doc in quality_collection.find({}, {"original_id": 1, "avg_quality_score": 1}):
            original_id = doc.get("original_id")
            avg_quality_score = doc.get("avg_quality_score")
            
            if original_id and avg_quality_score is not None:
                avg_quality_score_mapping[original_id] = avg_quality_score
        
        # 更新新闻级别情感集合
        updated_count = 0
        for doc in news_sentiment_collection.find({}):
            original_id = doc.get("original_id")
            
            if original_id in avg_quality_score_mapping:
                # 更新avg_quality_score，但保留情感分析相关字段
                news_sentiment_collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"avg_quality_score": float(avg_quality_score_mapping[original_id])}}
                )
                updated_count += 1
        
        logging.info(f"已更新 {updated_count} 条新闻级别情感记录")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="重新计算图片质量分数")
    parser.add_argument("--years", nargs="+", type=str, default=list(range(2014, 2025)),
                        help="要处理的年份列表")
    parser.add_argument("--text-weight", type=float, default=0.7,
                        help="文字权重在最终分数中的比例")
    parser.add_argument("--clarity-weight", type=float, default=0.3,
                        help="清晰度权重在最终分数中的比例")
    parser.add_argument("--diversity-weight", type=float, default=0,
                        help="多样性权重在最终分数中的比例")
    parser.add_argument("--nonlinear", action="store_true",
                        help="是否应用非线性变换")
    parser.add_argument("--only-update-sentiment", action="store_true",
                        help="仅更新情感集合中的quality_score，不重新计算质量分数")
    
    args = parser.parse_args()
    
    # 连接数据库
    db = connect_to_mongodb()
    
    # 处理每年的图片
    for year in args.years:
        if isinstance(year, int):
            year = str(year)
            
        if args.only_update_sentiment:
            # 仅更新情感集合中的quality_score
            update_sentiment_collections(db, year, f"{year}_2")
        else:
            # 重新计算质量分数并更新情感集合
            recalculate_quality_scores(
                db, 
                year, 
                args.text_weight, 
                args.clarity_weight, 
                args.diversity_weight,
                args.nonlinear
            )
    
    logging.info("\n质量分数重新计算完成，结果已保存到 MongoDB")

if __name__ == "__main__":
    main()
