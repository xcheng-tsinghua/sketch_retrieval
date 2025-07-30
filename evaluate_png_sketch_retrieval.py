"""
PNG草图-图像检索评估脚本
评估训练好的PNG草图-图像对齐模型的检索性能
"""

import os
import sys
import torch
import numpy as np
import argparse
from datetime import datetime
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import average_precision_score

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入数据集和模型
from data.PNGSketchImageDataset import create_png_sketch_dataloaders
from encoders.png_sketch_image_model import create_png_sketch_image_model


def compute_retrieval_metrics(similarity_matrix, labels):
    """
    计算检索指标
    
    Args:
        similarity_matrix: 相似度矩阵 [N_sketch, N_image]
        labels: 标签数组，相同类别的样本有相同标签
        
    Returns:
        metrics: 检索指标字典
    """
    N_sketch, N_image = similarity_matrix.shape
    
    # 计算Top-K准确率
    top1_correct = 0
    top5_correct = 0 
    top10_correct = 0
    
    # 计算mAP
    all_aps = []
    
    for i in range(N_sketch):
        # 获取第i个草图的相似度
        similarities = similarity_matrix[i]
        
        # 排序获取检索结果
        sorted_indices = torch.argsort(similarities, descending=True)
        
        # 获取真实标签
        sketch_label = labels[i]
        
        # 计算Top-K准确率
        if labels[sorted_indices[0]] == sketch_label:
            top1_correct += 1
            
        if any(labels[sorted_indices[:5]] == sketch_label):
            top5_correct += 1
            
        if any(labels[sorted_indices[:10]] == sketch_label):
            top10_correct += 1
        
        # 计算AP (Average Precision)
        relevant_mask = (labels == sketch_label).float()
        if relevant_mask.sum() > 0:
            ap = average_precision_score(
                relevant_mask.cpu().numpy(),
                similarities.cpu().numpy()
            )
            all_aps.append(ap)
    
    # 计算指标
    metrics = {
        'top1_accuracy': top1_correct / N_sketch,
        'top5_accuracy': top5_correct / N_sketch,
        'top10_accuracy': top10_correct / N_sketch,
        'mAP': np.mean(all_aps) if all_aps else 0.0
    }
    
    return metrics


def evaluate_by_category(similarity_matrix, labels, category_names, categories):
    """
    按类别评估检索性能
    
    Args:
        similarity_matrix: 相似度矩阵
        labels: 标签数组
        category_names: 类别名称列表
        categories: 所有类别列表
        
    Returns:
        category_metrics: 各类别的检索指标
    """
    category_metrics = {}
    
    for cat_idx, category in enumerate(categories):
        # 找到该类别的草图索引
        cat_sketch_indices = [i for i, cat in enumerate(category_names) if cat == category]
        
        if len(cat_sketch_indices) == 0:
            continue
        
        # 计算该类别的Top-1准确率
        cat_correct = 0
        for sketch_idx in cat_sketch_indices:
            similarities = similarity_matrix[sketch_idx]
            best_match_idx = torch.argmax(similarities).item()
            
            if labels[best_match_idx] == labels[sketch_idx]:
                cat_correct += 1
        
        cat_accuracy = cat_correct / len(cat_sketch_indices)
        category_metrics[category] = {
            'accuracy': cat_accuracy,
            'num_samples': len(cat_sketch_indices),
            'correct': cat_correct
        }
    
    return category_metrics


def visualize_retrieval_results(similarity_matrix, sketch_loader, image_loader, 
                              output_dir, num_examples=5):
    """
    可视化检索结果
    
    Args:
        similarity_matrix: 相似度矩阵
        sketch_loader: 草图数据加载器
        image_loader: 图像数据加载器
        output_dir: 输出目录
        num_examples: 可视化示例数量
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    import torchvision.transforms as transforms
    
    # 反归一化变换
    denormalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    # 收集图像数据
    sketch_images = []
    image_images = []
    sketch_categories = []
    image_categories = []
    
    # 从数据加载器收集数据
    for sketches, _, _, cat_names in sketch_loader:
        for i in range(len(sketches)):
            sketch_img = denormalize(sketches[i]).clamp(0, 1)
            sketch_images.append(sketch_img)
            sketch_categories.append(cat_names[i])
        if len(sketch_images) >= num_examples:
            break
    
    for images, _, _, cat_names in image_loader:
        for i in range(len(images)):
            image_img = denormalize(images[i]).clamp(0, 1)
            image_images.append(image_img)
            image_categories.append(cat_names[i])
    
    # 创建可视化
    fig, axes = plt.subplots(num_examples, 6, figsize=(18, 3 * num_examples))
    fig.suptitle('PNG Sketch-Image Retrieval Results', fontsize=16)
    
    for i in range(min(num_examples, len(sketch_images))):
        # 显示草图
        axes[i, 0].imshow(sketch_images[i].permute(1, 2, 0))
        axes[i, 0].set_title(f'Sketch\\n{sketch_categories[i]}')
        axes[i, 0].axis('off')
        
        # 获取Top-5检索结果
        similarities = similarity_matrix[i]
        top5_indices = torch.argsort(similarities, descending=True)[:5]
        
        for j, img_idx in enumerate(top5_indices):
            if img_idx < len(image_images):
                axes[i, j+1].imshow(image_images[img_idx].permute(1, 2, 0))
                score = similarities[img_idx].item()
                correct = "✓" if image_categories[img_idx] == sketch_categories[i] else "✗"
                axes[i, j+1].set_title(f'{correct} {image_categories[img_idx]}\\n{score:.3f}')
                axes[i, j+1].axis('off')
    
    plt.tight_layout()
    viz_path = os.path.join(output_dir, 'png_retrieval_visualization.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化结果已保存到: {viz_path}")


def main():
    parser = argparse.ArgumentParser(description='评估PNG草图-图像检索性能')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='训练好的模型检查点路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载进程数')
    parser.add_argument('--output_dir', type=str, default=None, help='结果输出目录')
    parser.add_argument('--visualize', action='store_true', help='生成可视化结果')
    parser.add_argument('--num_viz_examples', type=int, default=10, help='可视化示例数量')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置输出目录
    if args.output_dir is None:
        checkpoint_dir = os.path.dirname(args.checkpoint_path)
        args.output_dir = checkpoint_dir
    
    # 创建数据加载器
    print("加载测试数据集...")
    split_file = './data/fixed_splits/png_sketch_image_dataset_splits.pkl'
    
    if not os.path.exists(split_file):
        print("PNG草图数据集划分文件不存在，请先运行 create_png_sketch_dataset.py")
        return
    
    train_loader, test_loader, dataset_info = create_png_sketch_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fixed_split_path=split_file
    )
    
    print(f"测试集大小: {dataset_info['test_info']['total_pairs']}")
    print(f"共有 {dataset_info['category_info']['num_categories']} 个类别")
    
    # 创建模型
    print(f"从 {args.checkpoint_path} 加载模型...")
    model = create_png_sketch_image_model(embed_dim=512)
    
    # 加载检查点
    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"成功加载检查点 (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        print(f"检查点文件不存在: {args.checkpoint_path}")
        return
    
    model.to(device)
    model.eval()
    
    # 参数统计
    param_counts = model.get_parameter_count()
    print(f"模型参数数量: {param_counts['total']:,}")
    print(f"可训练参数数量: {param_counts['trainable']:,}")
    
    # 提取特征
    print("提取特征...")
    sketch_features = []
    image_features = []
    sketch_labels = []
    image_labels = []
    sketch_categories = []
    image_categories = []
    
    with torch.no_grad():
        # 提取草图特征
        for sketches, images, category_indices, category_names in tqdm(test_loader):
            sketches = sketches.to(device)
            images = images.to(device)
            
            # 编码草图和图像
            sketch_feat = model.encode_sketch(sketches)
            image_feat = model.encode_image(images)
            
            sketch_features.append(sketch_feat.cpu())
            image_features.append(image_feat.cpu())
            sketch_labels.extend(category_indices.cpu().numpy())
            image_labels.extend(category_indices.cpu().numpy())
            sketch_categories.extend(category_names)
            image_categories.extend(category_names)
    
    # 合并特征
    sketch_features = torch.cat(sketch_features, dim=0)
    image_features = torch.cat(image_features, dim=0)
    sketch_labels = torch.tensor(sketch_labels)
    image_labels = torch.tensor(image_labels)
    
    print(f"提取特征完成: sketch {sketch_features.shape}, image {image_features.shape}")
    
    # 计算相似度矩阵
    print("评估检索性能...")
    similarity_matrix = torch.matmul(sketch_features, image_features.t())
    
    print(f"相似度矩阵形状: {similarity_matrix.shape}")
    print(f"相似度矩阵统计: min={similarity_matrix.min():.4f}, "
          f"max={similarity_matrix.max():.4f}, mean={similarity_matrix.mean():.4f}")
    
    # 计算检索指标
    metrics = compute_retrieval_metrics(similarity_matrix, image_labels)
    
    print(f"\\n=== 检索性能评估结果 ===")
    print(f"Top-1 准确率: {metrics['top1_accuracy']:.4f}")
    print(f"Top-5 准确率: {metrics['top5_accuracy']:.4f}")
    print(f"Top-10 准确率: {metrics['top10_accuracy']:.4f}")
    print(f"mAP: {metrics['mAP']:.4f}")
    
    # 按类别评估
    categories = dataset_info['category_info']['categories']
    category_metrics = evaluate_by_category(
        similarity_matrix, image_labels, sketch_categories, categories
    )
    
    print(f"\\n评估 {len(categories)} 个类别的性能...")
    
    # 显示部分类别结果
    sorted_categories = sorted(category_metrics.items(), 
                             key=lambda x: x[1]['accuracy'], reverse=True)
    
    for i, (category, cat_metrics) in enumerate(sorted_categories[:10]):
        accuracy = cat_metrics['accuracy']
        num_samples = cat_metrics['num_samples']
        correct = cat_metrics['correct']
        print(f"  {category}: {correct}/{num_samples} = {accuracy:.4f}")
    
    # 保存结果
    results = {
        'overall_metrics': metrics,
        'category_metrics': category_metrics,
        'similarity_stats': {
            'min': similarity_matrix.min().item(),
            'max': similarity_matrix.max().item(),
            'mean': similarity_matrix.mean().item(),
            'std': similarity_matrix.std().item()
        },
        'dataset_info': dataset_info,
        'model_info': {
            'checkpoint_path': args.checkpoint_path,
            'total_params': param_counts['total'],
            'trainable_params': param_counts['trainable']
        }
    }
    
    results_file = os.path.join(args.output_dir, 'png_retrieval_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"结果已保存到: {results_file}")
    
    # 生成可视化
    if args.visualize:
        print("生成可视化结果...")
        visualize_retrieval_results(
            similarity_matrix, test_loader, test_loader,
            args.output_dir, args.num_viz_examples
        )
    
    # 最终总结
    print(f"\\n=== 最终评估结果 ===")
    print(f"Top-1 准确率: {metrics['top1_accuracy']:.4f}")
    print(f"Top-5 准确率: {metrics['top5_accuracy']:.4f}")
    print(f"Top-10 准确率: {metrics['top10_accuracy']:.4f}")
    print(f"mAP: {metrics['mAP']:.4f}")
    
    # 找到最佳和最差类别
    if category_metrics:
        best_category = max(category_metrics.items(), key=lambda x: x[1]['accuracy'])
        worst_category = min(category_metrics.items(), key=lambda x: x[1]['accuracy'])
        
        print(f"最佳类别: {best_category[0]} ({best_category[1]['accuracy']:.4f})")
        print(f"最差类别: {worst_category[0]} ({worst_category[1]['accuracy']:.4f})")
        
        # 统计类别分布
        num_good = sum(1 for cat_metrics in category_metrics.values() 
                      if cat_metrics['accuracy'] > 0)
        num_zero = sum(1 for cat_metrics in category_metrics.values() 
                      if cat_metrics['accuracy'] == 0)
        
        print(f"类别统计: 总数={len(category_metrics)}, 准确率>0={num_good}, 准确率=0={num_zero}")


if __name__ == '__main__':
    main()
