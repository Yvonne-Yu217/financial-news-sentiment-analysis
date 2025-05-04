import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pymongo import MongoClient
import os
import logging
from pathlib import Path
import datetime
import re
from tqdm import tqdm
import timm
import sys
import glob
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 从环境变量读取年份配置
years_env = os.environ.get('YEARS_TO_PROCESS')
if years_env:
    years_to_process = [int(y) for y in years_env.split(',')]
    logging.info(f"从环境变量读取年份配置: {years_to_process}")
else:   
    years_to_process = list(range(2014, 2025))  # 默认处理2014-2024年
    logging.info(f"使用默认年份配置: {years_to_process}")

# 删除所有._开头的隐藏文件
def remove_all_hidden_files():
    """删除所有._开头的隐藏文件"""
    count = 0
    for root, dirs, files in os.walk('images'):
        for file in files:
            if file.startswith('._'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    count += 1
                except Exception as e:
                    logging.error(f"删除隐藏文件失败 {file_path}: {e}")
    
    if count > 0:
        logging.info(f"已删除 {count} 个._开头的隐藏文件")

# 改进的ViT模型
class ImprovedViTModel(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.2):
        super().__init__()
        # 使用预训练的ViT模型
        self.backbone = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True)
        
        # 获取特征维度
        in_features = self.backbone.head.in_features
        
        # 替换分类头为更复杂的结构
        self.backbone.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

def check_environment():
    """检查运行环境"""
    logging.info("\n=== 环境检查 ===")
    logging.info(f"PyTorch 版本: {torch.__version__}")
    logging.info(f"CUDA 可用: {torch.cuda.is_available()}")
    if hasattr(torch.backends, 'mps'):
        logging.info(f"MPS 可用: {torch.backends.mps.is_available()}")
    
    # 检查必要目录
    required_dirs = ['images']
    for d in required_dirs:
        if os.path.exists(d):
            logging.info(f"目录存在: {d}")
        else:
            logging.info(f"目录不存在: {d}")
    logging.info("=== 检查完成 ===\n")

def load_model(model_path):
    """加载训练好的模型"""
    try:
        # 选择合适的设备
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logging.info("使用 CUDA 设备")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.info("使用 MPS 设备")
        else:
            device = torch.device("cpu")
            logging.info("使用 CPU 设备")
        
        # 创建模型实例
        model = ImprovedViTModel()
        
        # 加载模型权重
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        logging.info(f"模型 {model_path} 加载成功")
        return model, device
        
    except Exception as e:
        logging.error(f"模型加载失败: {e}")
        raise

def classify_image(model, image_path, device):
    """对单张图片进行分类"""
    try:
        # 图像预处理 - 与训练时完全相同
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 加载并转换图像
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # 模型推理
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            probs = probabilities.cpu().numpy()[0]
            _, predicted = torch.max(outputs, 1)
            
            # 返回预测类别和概率
            # 注意：positive=0, negative=1
            return predicted.item(), probs[1]  # 返回类别和负面概率
            
    except Exception as e:
        logging.error(f"处理图片失败 {image_path}: {e}")
        return None, None

def connect_to_mongodb():
    """连接到 MongoDB 数据库"""
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['sina_news_dataset_test']  # 使用项目数据库名
        
        # 检查数据库连接
        client.server_info()
        logging.info("MongoDB 连接成功")
        return db
        
    except Exception as e:
        logging.error(f"MongoDB 连接失败: {e}")
        raise

def remove_hidden_files(directory):
    """删除._开头的隐藏文件"""
    directory = Path(directory)
    count = 0
    for file in directory.glob("._*"):
        file.unlink()
        count += 1
    if count > 0:
        logging.info(f"已删除 {count} 个._开头的隐藏文件")

def reset_sentiment_state(db, year):
    """重置情感分析状态"""
    logging.info(f"正在重置 {year} 年的情感分析状态...")
    
    # 清空情感分析结果集合
    sentiment_collection_name = f"{year}_sentiment"
    if sentiment_collection_name in db.list_collection_names():
        db[sentiment_collection_name].drop()
        logging.info(f"已清空 {year} 年的情感分析结果集合 {sentiment_collection_name}")

def calculate_enhanced_weight(clarity_weight, text_weight):
    """
    计算增强版的图片权重，使用加权平均和非线性变换
    
    参数:
    clarity_weight: 清晰度权重
    text_weight: 文字权重
    
    返回:
    增强后的权重值
    """
    # 1. 基础权重计算 - 加权平均
    text_importance = 0.6  # 文字因素更重要
    base_weight = clarity_weight * (1-text_importance) + text_weight * text_importance
    
    # 2. 非线性变换增强对比度
    enhanced_weight = np.tanh(base_weight * 1.2)
    
    return float(enhanced_weight)

def process_images_for_year(model, device, db, year):
    """处理指定年份的图片"""
    logging.info(f"\n开始处理 {year} 年的图片...")
    
    # 使用年份作为集合名称，添加 _sentiment 后缀
    sentiment_collection = db[f"{year}_sentiment"]
    
    # 重置情感分析状态
    reset_sentiment_state(db, year)
    
    # 获取质量评分后的图片集合（包含质量评分信息）
    quality_collection_name = f"{year}_2"
    if quality_collection_name not in db.list_collection_names():
        logging.warning(f"找不到质量评分集合: {quality_collection_name}，请先运行图片质量处理程序")
        return
    
    # 获取基础过滤后的图片集合（包含图片路径信息）
    filtered_collection_name = f"{year}_filtered"
    if filtered_collection_name not in db.list_collection_names():
        logging.warning(f"找不到图片路径集合: {filtered_collection_name}，请先运行图片基础过滤程序")
        return
    
    quality_collection = db[quality_collection_name]
    filtered_collection = db[filtered_collection_name]
    
    # 创建映射：original_id -> valid_images
    logging.info("正在创建 original_id 到 图片路径 的映射...")
    path_mapping = {}
    filtered_cursor = filtered_collection.find({})
    for doc in filtered_cursor:
        original_id = doc.get("original_id")
        if original_id and "valid_images" in doc:
            path_mapping[original_id] = doc["valid_images"]
    
    logging.info(f"创建了 {len(path_mapping)} 条图片路径映射记录")
    
    # 获取所有质量评分记录
    cursor = quality_collection.find({})
    total_docs = quality_collection.count_documents({})
    
    if total_docs == 0:
        logging.warning(f"没有找到 {year} 年的图片记录")
        return
    
    logging.info(f"找到 {total_docs} 条新闻记录")
    
    # 初始化计数器
    negative_count = 0
    positive_count = 0
    total_negative_prob = 0.0
    weighted_negative_prob = 0.0
    total_weight = 0.0
    processed_count = 0
    failed_images = []
    total_images = 0
    
    # 处理每条新闻记录
    for doc in tqdm(cursor, total=total_docs, desc=f"处理 {year} 年图片"):
        news_date = doc.get("publish_date", "")
        title = doc.get("title", "")
        original_id = doc.get("original_id")
        
        # 获取处理后的图片信息（质量评分信息）
        processed_images = doc.get("processed_images", [])
        
        # 获取图片多样性信息
        diversity_score = doc.get("diversity_score", 1.0)
        similar_groups = doc.get("similar_groups", 0)
        unique_ratio = doc.get("unique_ratio", 1.0)
        
        # 如果没有图片或没有original_id，跳过此记录
        if not processed_images or not original_id or original_id not in path_mapping:
            continue
            
        # 获取原始图片路径信息
        valid_images = path_mapping[original_id]
        if not valid_images:
            continue
            
        # 确保processed_images和valid_images的长度匹配
        # 如果不匹配，可能是处理过程中过滤掉了一些图片
        if len(processed_images) != len(valid_images):
            logging.warning(f"图片数量不匹配: {title} - 路径:{len(valid_images)}个, 处理结果:{len(processed_images)}个")
            # 取二者中较小的长度
            min_length = min(len(processed_images), len(valid_images))
            processed_images = processed_images[:min_length]
            valid_images = valid_images[:min_length]
            
        # 处理每张图片
        for i, (proc_img, path_img) in enumerate(zip(processed_images, valid_images)):
            total_images += 1
            
            # 获取图片绝对路径
            abs_path = path_img.get("abs_path")
            if not abs_path:
                failed_images.append((None, "图片路径为空"))
                continue
                
            # 检查路径是否存在
            if not os.path.exists(abs_path):
                # 尝试不同的路径组合方式
                possible_paths = [
                    abs_path,  # 原始路径
                    os.path.join(os.getcwd(), abs_path),  # 当前目录 + 相对路径
                    os.path.join(os.getcwd(), "images", abs_path),  # 当前目录 + images + 相对路径
                    os.path.join(os.getcwd(), "images", f"{year}", abs_path),  # 当前目录 + images + 年份 + 相对路径
                ]
                
                # 检查哪个路径存在
                image_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        image_path = path
                        break
                
                # 如果所有路径都不存在
                if not image_path:
                    failed_images.append((abs_path, "文件不存在"))
                    continue
            else:
                image_path = abs_path
            
            # 获取权重信息
            clarity_weight = proc_img.get("clarity_weight", 0.5)
            text_weight = proc_img.get("text_weight", 0.5)
            quality_score = proc_img.get("quality_score", 0.5)
            
            # 计算综合权重 - 使用增强版权重计算
            weight = calculate_enhanced_weight(clarity_weight, text_weight)
            
            # 对图片进行分类
            try:
                predicted_class, negative_prob = classify_image(model, image_path, device)
                
                if predicted_class is not None and negative_prob is not None:
                    # 保存预测结果
                    result = {
                        "original_id": original_id,  # 保存原始记录ID
                        "image_path": image_path,
                        "rel_path": abs_path,  # 使用原始路径作为相对路径
                        "predicted_class": predicted_class,  # 0=积极, 1=消极
                        "negative_likelihood": float(negative_prob),  # 负面情绪概率
                        "news_date": news_date,
                        "title": title,
                        "clarity_weight": clarity_weight,
                        "text_weight": text_weight,
                        "weight": weight,  # 综合权重
                        "quality_score": quality_score,  # 保存质量分数
                        "diversity_score": diversity_score,  # 新增：多样性分数
                        "unique_ratio": unique_ratio,  # 新增：独特图片比例
                        "weighted_score": float(negative_prob * weight * diversity_score),  # 加权情感分数，考虑多样性
                        "timestamp": datetime.datetime.now()
                    }
                    sentiment_collection.insert_one(result)
                    
                    # 更新统计
                    if predicted_class == 1:  # 消极
                        negative_count += 1
                    else:  # 积极
                        positive_count += 1
                        
                    total_negative_prob += negative_prob
                    weighted_negative_prob += negative_prob * weight * diversity_score  # 考虑多样性
                    total_weight += weight
                    processed_count += 1
                    
                else:
                    failed_images.append((image_path, "模型预测失败"))
                    
            except Exception as e:
                logging.error(f"处理失败 {image_path}: {e}")
                failed_images.append((image_path, str(e)))

    # 计算新闻级别的平均情感分数
    if processed_count > 0:
        logging.info("计算新闻级别的平均情感分数...")
        
        # 创建新闻情感汇总集合
        news_sentiment_collection = db[f"{year}_news_sentiment"]
        news_sentiment_collection.drop()  # 先清空集合
        
        # 聚合查询，按新闻(日期+标题)分组计算平均情感分数
        pipeline = [
            {"$group": {
                "_id": {"news_date": "$news_date", "title": "$title"},
                "avg_negative_likelihood": {"$avg": "$negative_likelihood"},
                "avg_weighted_score": {"$avg": "$weighted_score"},
                "avg_weight": {"$avg": "$weight"},
                "avg_quality_score": {"$avg": "$quality_score"},
                "avg_diversity_score": {"$avg": "$diversity_score"},  # 新增：平均多样性分数
                "avg_unique_ratio": {"$avg": "$unique_ratio"},  # 新增：平均独特比例
                "image_count": {"$sum": 1},
                "negative_count": {"$sum": {"$cond": [{"$eq": ["$predicted_class", 1]}, 1, 0]}},
                "positive_count": {"$sum": {"$cond": [{"$eq": ["$predicted_class", 0]}, 1, 0]}}
            }},
            {"$project": {
                "news_date": "$_id.news_date",
                "title": "$_id.title",
                "avg_negative_likelihood": 1,
                "avg_weighted_score": 1,
                "avg_weight": 1,
                "avg_quality_score": 1,
                "avg_diversity_score": 1,  # 新增：平均多样性分数
                "avg_unique_ratio": 1,  # 新增：平均独特比例
                "image_count": 1,
                "negative_count": 1,
                "positive_count": 1,
                "negative_ratio": {"$divide": ["$negative_count", "$image_count"]}
            }},
            {"$sort": {"news_date": 1}}
        ]
        
        news_results = list(sentiment_collection.aggregate(pipeline))
        
        # 保存新闻级别的情感分数
        if news_results:
            news_sentiment_collection.insert_many(news_results)
            logging.info(f"已计算并保存 {len(news_results)} 条新闻的平均情感分数")
    
    # 打印统计信息
    logging.info("\n=== 处理统计 ===")
    logging.info(f"总图片数: {total_images}")
    logging.info(f"处理成功: {processed_count}")
    logging.info(f"处理失败: {len(failed_images)}")
    
    if processed_count > 0:
        logging.info(f"\n分类结果:")
        logging.info(f"积极图片 (0): {positive_count} ({positive_count/processed_count*100:.1f}%)")
        logging.info(f"消极图片 (1): {negative_count} ({negative_count/processed_count*100:.1f}%)")
        logging.info(f"平均负面概率: {total_negative_prob/processed_count:.3f}")
        logging.info(f"加权平均负面概率: {weighted_negative_prob/total_weight:.3f}")
    
    # 记录失败的图片
    if failed_images:
        logging.info("\n=== 失败记录 ===")
        for img_path, error in failed_images[:10]:  # 只显示前10个
            logging.info(f"- {img_path}: {error}")
        if len(failed_images) > 10:
            logging.info(f"... 还有 {len(failed_images)-10} 个失败记录")

def main():
    """主函数，提供预测功能"""
    try:
        import argparse
        
        # 过滤掉Jupyter相关的参数
        jupyter_args = [arg for arg in sys.argv if arg.startswith('--f=')]
        for arg in jupyter_args:
            sys.argv.remove(arg)
        
        parser = argparse.ArgumentParser(description='ViT情感分析模型预测')
        parser.add_argument('--years', type=int, nargs='+', default=years_to_process,
                            help='要处理的年份列表，例如: --years 2019 2022 2024')
        parser.add_argument('--model', type=str, default='improved_vit_sentiment_model.pth',
                            help='模型路径，默认使用improved_vit_sentiment_model.pth')
        
        args = parser.parse_args()
        
        # 检查环境
        check_environment()
        
        # 删除隐藏文件
        remove_all_hidden_files()
        
        # 获取当前工作目录
        current_dir = os.getcwd()
        logging.info(f"当前工作目录: {current_dir}")
        
        # 检查模型文件
        if not os.path.exists(args.model):
            logging.error(f"模型文件不存在: {args.model}")
            return
        
        # 加载模型
        model, device = load_model(args.model)
        
        # 连接数据库
        db = connect_to_mongodb()
        
        # 处理每年的图片
        for year in args.years:
            process_images_for_year(model, device, db, year)
        
        logging.info("\n情感分析完成，结果已保存到 MongoDB")
        
    except Exception as e:
        logging.error(f"程序运行出错: {e}")
        raise

if __name__ == "__main__":
    main()
