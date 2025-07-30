"""
草图-图片检索实验脚本
用于验证跨模态对齐效果
"""
import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import json

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.SLMDataset import SketchImageDataset
from encoders.unified_crossmodal_model import create_sketch_image_model

class SketchImageRetrieval:
    """草图-图片检索评估器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载测试数据集
        self.setup_dataset()
        
        # 加载训练好的模型
        self.load_models()
        
    def setup_dataset(self):
        """设置测试数据集"""
        print("加载测试数据集...")
        
        self.test_dataset = SketchImageDataset(
            n_skh_points=self.args.n_sketch_points,
            is_train=False
        )
        
        # 创建完全确定性的DataLoader，移除所有随机性
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=False
        )
        
        print(f"测试集大小: {len(self.test_dataset)}")
        
        # 构建类别到索引的映射
        self.build_category_mapping()
        
    def build_category_mapping(self):
        """构建类别映射"""
        # 注意：这里不预先构建categories，因为在extract_features中会重新获取
        # 只预先计算unique_categories用于后续处理
        temp_categories = []
        for _, _, _, category in self.test_dataset:
            temp_categories.append(category)
        
        self.unique_categories = list(set(temp_categories))
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.unique_categories)}
        
        print(f"共有 {len(self.unique_categories)} 个类别")
        
    def load_models(self):
        """加载草图-图像对齐模型"""
        print(f"从 {self.args.checkpoint_path} 加载模型...")
        
        checkpoint = torch.load(self.args.checkpoint_path, map_location=self.device, weights_only=False)
        
        # 从检查点中获取embed_dim参数
        if 'args' in checkpoint:
            saved_args = checkpoint['args']
            embed_dim = getattr(saved_args, 'embed_dim', 512)
        else:
            embed_dim = getattr(self.args, 'embed_dim', 512)
        
        # 使用create_sketch_image_model创建模型（借鉴diagnose_retrieval_anomaly.py的方式）
        print("创建草图-图像对齐模型...")
        self.model = create_sketch_image_model(
            sketch_points=self.args.n_sketch_points,
            embed_dim=embed_dim
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 加载模型参数（使用与diagnose_retrieval_anomaly.py相同的方式）
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 加载state_dict
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device)
        
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        print(f"草图-图像对齐模型加载完成，参数数量: {sum(p.numel() for p in self.model.parameters())}")
        print(f"可训练参数数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        
    def extract_features(self):
        """使用草图-图像对齐模型提取所有测试样本的特征"""
        print("提取特征...")
        
        sketch_features = []
        image_features = []
        categories = []
        
        with torch.no_grad():
            for sketch_data, sketch_mask, image_data, category_batch in tqdm(self.test_loader):
                sketch_data = sketch_data.to(self.device)
                sketch_mask = sketch_mask.to(self.device)
                image_data = image_data.to(self.device)
                
                # 使用草图-图像对齐模型提取特征（与diagnose_retrieval_anomaly.py保持一致）
                # 提取草图特征
                sketch_embed = self.model.encode_sketch(sketch_data, sketch_mask)
                
                # 提取图片特征
                image_embed = self.model.encode_image(image_data)
                
                # 归一化特征
                sketch_embed = F.normalize(sketch_embed, dim=-1, p=2)
                image_embed = F.normalize(image_embed, dim=-1, p=2)
                
                sketch_features.append(sketch_embed.cpu().numpy())
                image_features.append(image_embed.cpu().numpy())
                categories.extend(category_batch)
        
        # 合并所有特征
        self.sketch_features = np.vstack(sketch_features)
        self.image_features = np.vstack(image_features)
        self.test_categories = categories
        
        print(f"提取特征完成: sketch {self.sketch_features.shape}, image {self.image_features.shape}")
        
    def evaluate_retrieval(self):
        """评估检索性能"""
        print("评估检索性能...")
        
        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(self.sketch_features, self.image_features)
        print(f"相似度矩阵形状: {similarity_matrix.shape}")
        print(f"相似度矩阵统计: min={similarity_matrix.min():.4f}, max={similarity_matrix.max():.4f}, mean={similarity_matrix.mean():.4f}")
        
        # 计算各种评估指标
        results = {}
        
        # Top-K准确率
        for k in [1, 5, 10]:
            top_k_acc = self.compute_top_k_accuracy(similarity_matrix, k)
            results[f'top_{k}_accuracy'] = top_k_acc
            print(f"Top-{k} 准确率: {top_k_acc:.4f}")
        
        # mAP (mean Average Precision)
        map_score = self.compute_map(similarity_matrix)
        results['mAP'] = map_score
        print(f"mAP: {map_score:.4f}")
        
        # 类别级别的评估
        category_results = self.evaluate_by_category(similarity_matrix)
        results['category_results'] = category_results
        
        return results
    
    def compute_top_k_accuracy(self, similarity_matrix, k):
        """计算Top-K准确率"""
        n_queries = similarity_matrix.shape[0]
        correct = 0
        
        for i in range(n_queries):
            # 获取第i个草图的相似度排序
            sorted_indices = np.argsort(similarity_matrix[i])[::-1]
            top_k_indices = sorted_indices[:k]
            
            # 检查Top-K中是否有同类别的图片
            query_category = self.test_categories[i]
            for idx in top_k_indices:
                if self.test_categories[idx] == query_category:
                    correct += 1
                    break
        
        return correct / n_queries
    
    def compute_map(self, similarity_matrix):
        """计算mAP"""
        n_queries = similarity_matrix.shape[0]
        aps = []
        
        for i in range(n_queries):
            # 获取排序后的索引
            sorted_indices = np.argsort(similarity_matrix[i])[::-1]
            
            # 找到所有同类别的图片
            query_category = self.test_categories[i]
            relevant_indices = [j for j, cat in enumerate(self.test_categories) if cat == query_category]
            
            if len(relevant_indices) == 0:
                continue
            
            # 计算AP
            precision_at_k = []
            num_relevant = 0
            
            for k, idx in enumerate(sorted_indices):
                if idx in relevant_indices:
                    num_relevant += 1
                    precision_at_k.append(num_relevant / (k + 1))
            
            if len(precision_at_k) > 0:
                ap = np.mean(precision_at_k)
                aps.append(ap)
        
        return np.mean(aps) if aps else 0.0
    
    def evaluate_by_category(self, similarity_matrix):
        """按类别评估"""
        category_results = {}
        
        print(f"评估 {len(self.unique_categories)} 个类别的性能...")
        
        for category in self.unique_categories:
            # 找到该类别的所有草图和图片索引
            sketch_indices = [i for i, cat in enumerate(self.test_categories) if cat == category]
            image_indices = [i for i, cat in enumerate(self.test_categories) if cat == category]
            
            if len(sketch_indices) == 0 or len(image_indices) == 0:
                print(f"警告: 类别 {category} 没有找到样本")
                continue
            
            # 计算该类别的Top-1准确率
            correct = 0
            for sketch_idx in sketch_indices:
                top_1_idx = np.argmax(similarity_matrix[sketch_idx])
                if self.test_categories[top_1_idx] == category:
                    correct += 1
            
            top_1_acc = correct / len(sketch_indices)
            category_results[category] = {
                'top_1_accuracy': top_1_acc,
                'num_samples': len(sketch_indices)
            }
            
            # 打印调试信息（仅对非零准确率或特殊类别）
            if top_1_acc > 0 or category == 'pickup_truck':
                print(f"  {category}: {correct}/{len(sketch_indices)} = {top_1_acc:.4f}")
        
        return category_results
    
    def visualize_results(self, results, save_path):
        """可视化结果"""
        print("生成可视化结果...")
        
        # 1. Top-K准确率柱状图
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        top_k_accs = [results[f'top_{k}_accuracy'] for k in [1, 5, 10]]
        plt.bar(['Top-1', 'Top-5', 'Top-10'], top_k_accs)
        plt.title('Top-K Accuracy')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        # 2. 类别级别准确率
        plt.subplot(1, 3, 2)
        category_results = results['category_results']
        categories = list(category_results.keys())[:20]  # 只显示前20个类别
        category_accs = [category_results[cat]['top_1_accuracy'] for cat in categories]
        
        plt.bar(range(len(categories)), category_accs)
        plt.title('Top-1 Accuracy by Category')
        plt.xlabel('Category')
        plt.ylabel('Accuracy')
        plt.xticks(range(len(categories)), categories, rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # 3. 整体性能汇总
        plt.subplot(1, 3, 3)
        metrics = ['Top-1', 'Top-5', 'Top-10', 'mAP']
        values = [results['top_1_accuracy'], results['top_5_accuracy'], 
                 results['top_10_accuracy'], results['mAP']]
        
        plt.bar(metrics, values)
        plt.title('Overall Performance')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可视化结果已保存到: {save_path}")
    
    def run_evaluation(self):
        """运行完整的评估流程"""
        # 提取特征
        self.extract_features()
        
        # 评估检索性能
        results = self.evaluate_retrieval()
        
        # 保存结果
        output_dir = os.path.dirname(self.args.checkpoint_path)
        results_path = os.path.join(output_dir, 'retrieval_results.json')
        
        with open(results_path, 'w') as f:
            # 处理不能序列化的numpy类型
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                               for k, v in value.items()}
                else:
                    serializable_results[key] = float(value) if isinstance(value, (np.float32, np.float64)) else value
            
            json.dump(serializable_results, f, indent=2)
        
        print(f"结果已保存到: {results_path}")
        
        # 生成可视化
        viz_path = os.path.join(output_dir, 'retrieval_visualization.png')
        self.visualize_results(results, viz_path)
        
        return results


def main():
    parser = argparse.ArgumentParser(description='草图-图片检索评估')
    
    # 数据相关参数
    parser.add_argument('--sketch_root', type=str,
                       default=r'E:\Master\Experiment\data\stroke-normal',
                       help='草图数据根目录')
    parser.add_argument('--image_root', type=str,
                       default=r'E:\Master\Experiment\data\photo',
                       help='图片数据根目录')
    parser.add_argument('--n_sketch_points', type=int, default=256,
                       help='草图点数')
    
    # 模型相关参数
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='模型检查点路径')
    
    # 评估相关参数
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载线程数')
    
    args = parser.parse_args()
    
    # 检查检查点文件是否存在
    if not os.path.exists(args.checkpoint_path):
        print(f"错误: 检查点文件不存在: {args.checkpoint_path}")
        return
    
    # 运行评估
    evaluator = SketchImageRetrieval(args)
    results = evaluator.run_evaluation()
    
    # 打印最终结果
    print("\n=== 最终评估结果 ===")
    print(f"Top-1 准确率: {results['top_1_accuracy']:.4f}")
    print(f"Top-5 准确率: {results['top_5_accuracy']:.4f}")
    print(f"Top-10 准确率: {results['top_10_accuracy']:.4f}")
    print(f"mAP: {results['mAP']:.4f}")
    
    # 打印最佳和最差的类别
    category_results = results['category_results']
    if category_results:
        # 找到准确率最高和最低的类别
        best_category = max(category_results.keys(), 
                           key=lambda x: category_results[x]['top_1_accuracy'])
        worst_category = min(category_results.keys(),
                            key=lambda x: category_results[x]['top_1_accuracy'])
        
        print(f"最佳类别: {best_category} ({category_results[best_category]['top_1_accuracy']:.4f})")
        print(f"最差类别: {worst_category} ({category_results[worst_category]['top_1_accuracy']:.4f})")
        
        # 统计准确率分布
        accuracies = [result['top_1_accuracy'] for result in category_results.values()]
        non_zero_count = sum(1 for acc in accuracies if acc > 0)
        
        print(f"类别统计: 总数={len(accuracies)}, 准确率>0={non_zero_count}, 准确率=0={len(accuracies) - non_zero_count}")
        
        # 特别检查pickup_truck
        if 'pickup_truck' in category_results:
            pickup_acc = category_results['pickup_truck']['top_1_accuracy']
            print(f"pickup_truck准确率: {pickup_acc:.4f}")
    else:
        print("警告: 没有类别评估结果")


if __name__ == '__main__':
    main()
