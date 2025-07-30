"""
诊断草图-图片检索异常的脚本
分析为什么只有少数类别被频繁检索
"""
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import json
from collections import Counter, defaultdict

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.SLMDataset import SketchImageDataset
from encoders.unified_crossmodal_model import create_sketch_image_model


class RetrievalAnomalyDiagnosis:
    """检索异常诊断器"""
    
    def __init__(self, checkpoint_path, n_sketch_points=256, batch_size=32):
        self.checkpoint_path = checkpoint_path
        self.n_sketch_points = n_sketch_points
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置确定性
        torch.manual_seed(42)
        np.random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        print(f"使用设备: {self.device}")
        
    def load_data_and_model(self):
        """加载数据集和模型"""
        print("加载测试数据集...")
        
        # 加载测试数据集
        self.test_dataset = SketchImageDataset(
            n_skh_points=self.n_sketch_points,
            is_train=False
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False
        )
        
        print(f"测试集大小: {len(self.test_dataset)}")
        
        # 加载模型
        print(f"加载模型: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model = create_sketch_image_model(
            sketch_points=self.n_sketch_points,
            embed_dim=512
        )
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"模型加载完成，参数数量: {sum(p.numel() for p in self.model.parameters())}")
        
    def extract_features_with_analysis(self):
        """提取特征并进行分析"""
        print("提取特征并分析...")
        
        sketch_features = []
        image_features = []
        categories = []
        sketch_norms = []
        image_norms = []
        
        with torch.no_grad():
            for sketch_data, sketch_mask, image_data, category_batch in tqdm(self.test_loader):
                sketch_data = sketch_data.to(self.device)
                sketch_mask = sketch_mask.to(self.device)
                image_data = image_data.to(self.device)
                
                # 提取特征（归一化前）
                sketch_embed = self.model.encode_sketch(sketch_data, sketch_mask)
                image_embed = self.model.encode_image(image_data)
                
                # 记录归一化前的范数
                sketch_norms.extend(torch.norm(sketch_embed, dim=-1).cpu().numpy())
                image_norms.extend(torch.norm(image_embed, dim=-1).cpu().numpy())
                
                # 归一化特征
                sketch_embed = F.normalize(sketch_embed, dim=-1, p=2)
                image_embed = F.normalize(image_embed, dim=-1, p=2)
                
                sketch_features.append(sketch_embed.cpu().numpy())
                image_features.append(image_embed.cpu().numpy())
                categories.extend(category_batch)
        
        self.sketch_features = np.vstack(sketch_features)
        self.image_features = np.vstack(image_features)
        self.categories = categories
        self.sketch_norms = np.array(sketch_norms)
        self.image_norms = np.array(image_norms)
        
        print(f"特征提取完成: sketch {self.sketch_features.shape}, image {self.image_features.shape}")
        
    def analyze_feature_distribution(self):
        """分析特征分布"""
        print("\\n=== 特征分布分析 ===")
        
        # 1. 特征范数分析
        print(f"草图特征范数: min={self.sketch_norms.min():.6f}, max={self.sketch_norms.max():.6f}, "
              f"mean={self.sketch_norms.mean():.6f}, std={self.sketch_norms.std():.6f}")
        print(f"图像特征范数: min={self.image_norms.min():.6f}, max={self.image_norms.max():.6f}, "
              f"mean={self.image_norms.mean():.6f}, std={self.image_norms.std():.6f}")
        
        # 2. 归一化后特征的统计
        sketch_mean = self.sketch_features.mean(axis=0)
        image_mean = self.image_features.mean(axis=0)
        
        print(f"草图特征均值范数: {np.linalg.norm(sketch_mean):.6f}")
        print(f"图像特征均值范数: {np.linalg.norm(image_mean):.6f}")
        
        # 3. 特征维度方差分析
        sketch_var = self.sketch_features.var(axis=0)
        image_var = self.image_features.var(axis=0)
        
        print(f"草图特征方差统计: min={sketch_var.min():.6f}, max={sketch_var.max():.6f}, "
              f"mean={sketch_var.mean():.6f}")
        print(f"图像特征方差统计: min={image_var.min():.6f}, max={image_var.max():.6f}, "
              f"mean={image_var.mean():.6f}")
        
        # 4. 检查是否有维度塌陷
        zero_var_dims_sketch = np.sum(sketch_var < 1e-6)
        zero_var_dims_image = np.sum(image_var < 1e-6)
        
        print(f"草图特征零方差维度数: {zero_var_dims_sketch}/{len(sketch_var)}")
        print(f"图像特征零方差维度数: {zero_var_dims_image}/{len(image_var)}")
        
    def analyze_similarity_patterns(self):
        """分析相似度模式"""
        print("\\n=== 相似度模式分析 ===")
        
        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(self.sketch_features, self.image_features)
        
        print(f"相似度矩阵统计: min={similarity_matrix.min():.6f}, max={similarity_matrix.max():.6f}, "
              f"mean={similarity_matrix.mean():.6f}, std={similarity_matrix.std():.6f}")
        
        # 分析每个草图的检索结果
        retrieval_counter = Counter()
        category_retrieval = defaultdict(list)
        
        for i in range(len(self.categories)):
            # 找到最相似的图像
            most_similar_idx = np.argmax(similarity_matrix[i])
            retrieved_category = self.categories[most_similar_idx]
            retrieval_counter[retrieved_category] += 1
            category_retrieval[self.categories[i]].append(retrieved_category)
        
        # 统计检索频次
        print(f"\\n检索频次最高的前10个类别:")
        for category, count in retrieval_counter.most_common(10):
            print(f"  {category}: {count} 次被检索 ({count/len(self.categories)*100:.2f}%)")
        
        # 统计有多少类别从未被检索
        all_categories = set(self.categories)
        retrieved_categories = set(retrieval_counter.keys())
        never_retrieved = all_categories - retrieved_categories
        
        print(f"\\n从未被检索的类别数: {len(never_retrieved)}/{len(all_categories)}")
        print(f"被检索过的类别数: {len(retrieved_categories)}/{len(all_categories)}")
        
        # 分析类别内检索准确性
        correct_retrievals = 0
        for i, query_category in enumerate(self.categories):
            most_similar_idx = np.argmax(similarity_matrix[i])
            if self.categories[most_similar_idx] == query_category:
                correct_retrievals += 1
        
        print(f"\\n类别内正确检索数: {correct_retrievals}/{len(self.categories)} "
              f"({correct_retrievals/len(self.categories)*100:.2f}%)")
        
        return similarity_matrix, retrieval_counter, category_retrieval
    
    def analyze_problematic_categories(self, similarity_matrix, retrieval_counter):
        """分析问题类别"""
        print("\\n=== 问题类别分析 ===")
        
        # 1. 分析被过度检索的类别
        most_retrieved = retrieval_counter.most_common(3)
        print("\\n被过度检索的类别分析:")
        
        for category, count in most_retrieved:
            # 找到该类别的所有图像索引
            category_indices = [i for i, cat in enumerate(self.categories) if cat == category]
            
            if len(category_indices) == 0:
                continue
                
            # 分析该类别图像的相似度特征
            category_similarities = similarity_matrix[:, category_indices]
            avg_similarity = category_similarities.mean()
            max_similarity = category_similarities.max()
            
            print(f"  {category} (被检索{count}次):")
            print(f"    样本数: {len(category_indices)}")
            print(f"    平均相似度: {avg_similarity:.6f}")
            print(f"    最大相似度: {max_similarity:.6f}")
            
            # 分析该类别图像特征的统计特性
            category_image_features = self.image_features[category_indices]
            feature_mean = category_image_features.mean(axis=0)
            feature_std = category_image_features.std(axis=0)
            
            print(f"    特征均值范数: {np.linalg.norm(feature_mean):.6f}")
            print(f"    特征标准差均值: {feature_std.mean():.6f}")
        
        # 2. 分析从未被检索的类别
        all_categories = set(self.categories)
        retrieved_categories = set(retrieval_counter.keys())
        never_retrieved = all_categories - retrieved_categories
        
        if never_retrieved:
            print(f"\\n从未被检索的类别示例 (共{len(never_retrieved)}个):")
            for i, category in enumerate(list(never_retrieved)[:5]):  # 只显示前5个
                category_indices = [i for i, cat in enumerate(self.categories) if cat == category]
                if len(category_indices) > 0:
                    category_similarities = similarity_matrix[:, category_indices]
                    avg_similarity = category_similarities.mean()
                    print(f"  {category}: 样本数={len(category_indices)}, 平均相似度={avg_similarity:.6f}")
    
    def check_model_components(self):
        """检查模型组件"""
        print("\\n=== 模型组件检查 ===")
        
        # 检查logit_scale
        logit_scale = self.model.logit_scale.exp().item()
        print(f"logit_scale: {logit_scale:.6f}")
        
        # 检查投影矩阵
        if hasattr(self.model, 'sketch_projection'):
            sketch_proj_norm = torch.norm(self.model.sketch_projection).item()
            print(f"草图投影矩阵范数: {sketch_proj_norm:.6f}")
        
        if hasattr(self.model, 'image_projection'):
            image_proj_norm = torch.norm(self.model.image_projection).item()
            print(f"图像投影矩阵范数: {image_proj_norm:.6f}")
        
        # 检查草图编码器的线性层
        if hasattr(self.model, 'sketch_encoder') and hasattr(self.model.sketch_encoder, 'linear'):
            for i, layer in enumerate(self.model.sketch_encoder.linear):
                if hasattr(layer, 'weight'):
                    weight_norm = torch.norm(layer.weight).item()
                    print(f"草图编码器线性层{i}权重范数: {weight_norm:.6f}")
    
    def generate_diagnostic_report(self, output_dir):
        """生成诊断报告"""
        print("\\n=== 生成诊断报告 ===")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 运行所有分析
        self.analyze_feature_distribution()
        similarity_matrix, retrieval_counter, category_retrieval = self.analyze_similarity_patterns()
        self.analyze_problematic_categories(similarity_matrix, retrieval_counter)
        self.check_model_components()
        
        # 2. 保存统计结果
        diagnosis_results = {
            'feature_stats': {
                'sketch_norm_stats': {
                    'min': float(self.sketch_norms.min()),
                    'max': float(self.sketch_norms.max()),
                    'mean': float(self.sketch_norms.mean()),
                    'std': float(self.sketch_norms.std())
                },
                'image_norm_stats': {
                    'min': float(self.image_norms.min()),
                    'max': float(self.image_norms.max()),
                    'mean': float(self.image_norms.mean()),
                    'std': float(self.image_norms.std())
                }
            },
            'similarity_stats': {
                'min': float(similarity_matrix.min()),
                'max': float(similarity_matrix.max()),
                'mean': float(similarity_matrix.mean()),
                'std': float(similarity_matrix.std())
            },
            'retrieval_distribution': dict(retrieval_counter),
            'model_params': {
                'logit_scale': float(self.model.logit_scale.exp().item())
            }
        }
        
        # 保存结果
        report_path = os.path.join(output_dir, 'diagnosis_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(diagnosis_results, f, indent=2, ensure_ascii=False)
        
        print(f"诊断报告已保存到: {report_path}")
        
        # 3. 生成可视化
        self.visualize_diagnosis(similarity_matrix, retrieval_counter, output_dir)
        
    def visualize_diagnosis(self, similarity_matrix, retrieval_counter, output_dir):
        """生成诊断可视化"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 相似度分布直方图
        axes[0, 0].hist(similarity_matrix.flatten(), bins=100, alpha=0.7)
        axes[0, 0].set_title('Similarity Distribution')
        axes[0, 0].set_xlabel('Cosine Similarity')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. 检索频次分布
        most_common = retrieval_counter.most_common(20)
        categories, counts = zip(*most_common)
        axes[0, 1].bar(range(len(categories)), counts)
        axes[0, 1].set_title('Top 20 Most Retrieved Categories')
        axes[0, 1].set_xlabel('Category')
        axes[0, 1].set_ylabel('Retrieval Count')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 特征范数分布
        axes[1, 0].hist(self.sketch_norms, bins=50, alpha=0.7, label='Sketch', density=True)
        axes[1, 0].hist(self.image_norms, bins=50, alpha=0.7, label='Image', density=True)
        axes[1, 0].set_title('Feature Norm Distribution')
        axes[1, 0].set_xlabel('L2 Norm')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        
        # 4. 相似度矩阵热图（采样显示）
        sample_size = min(200, len(self.categories))
        sample_indices = np.random.choice(len(self.categories), sample_size, replace=False)
        sample_matrix = similarity_matrix[np.ix_(sample_indices, sample_indices)]
        
        im = axes[1, 1].imshow(sample_matrix, cmap='viridis', aspect='auto')
        axes[1, 1].set_title(f'Similarity Matrix Sample ({sample_size}x{sample_size})')
        axes[1, 1].set_xlabel('Image Index')
        axes[1, 1].set_ylabel('Sketch Index')
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        
        # 保存图像
        viz_path = os.path.join(output_dir, 'diagnosis_visualization.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"诊断可视化已保存到: {viz_path}")


def main():
    # 配置路径
    checkpoint_path = "outputs/sketch_image_alignment_20250727_172806/best_checkpoint.pth"
    output_dir = "outputs/diagnosis_results"
    
    # 创建诊断器
    diagnosis = RetrievalAnomalyDiagnosis(
        checkpoint_path=checkpoint_path,
        n_sketch_points=256,
        batch_size=32
    )
    
    # 加载数据和模型
    diagnosis.load_data_and_model()
    
    # 提取特征
    diagnosis.extract_features_with_analysis()
    
    # 生成诊断报告
    diagnosis.generate_diagnostic_report(output_dir)


if __name__ == "__main__":
    main()
