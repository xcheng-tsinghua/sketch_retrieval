"""
可视化前5好类别的PNG草图-图像检索效果
专门用于展示模型在最佳类别上的检索表现
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms as transforms
import json
import argparse

# 导入数据集和模型
from data.retrieval_datasets import create_png_sketch_dataloaders
from encoders import sbir_model_wrapper
from utils import utils


def parse_args():
    parser = argparse.ArgumentParser(description='训练PNG草图-图像对齐模型')
    parser.add_argument('--bs', type=int, default=100, help='批次大小')
    parser.add_argument('--epoch', type=int, default=1000, help='最大训练轮数')

    parser.add_argument('--embed_dim', type=int, default=512, help='嵌入维度')
    parser.add_argument('--is_freeze_image_encoder', type=str, choices=['True', 'False'], default='True', help='冻结图像编码器')
    parser.add_argument('--is_freeze_sketch_backbone', type=str, choices=['True', 'False'], default='False', help='冻结草图编码器主干网络')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载进程数')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--weight_dir', type=str, default='model_trained', help='输出目录')
    parser.add_argument('--sketch_format', type=str, default='vector', choices=['vector', 'image'], help='使用矢量草图还是图片草图')
    parser.add_argument('--vec_sketch_type', type=str, default='STK_11_32', choices=['STK_11_32', 'S5'],help='矢量草图格式')
    parser.add_argument('--is_create_fix_data_file', type=str, choices=['True', 'False'], default='False', help='是否创建固定数据集划分文件')
    parser.add_argument('--is_load_ckpt', type=str, choices=['True', 'False'], default='False', help='是否加载检查点')
    parser.add_argument('--sketch_image_subdirs', type=tuple, default=('sketch_s3_352', 'sketch_png', 'photo'), help='[0]: vector_sketch, [1]: image_sketch, [2]: photo')
    parser.add_argument('--save_str', type=str, default='lstm_vit', help='保存名')
    parser.add_argument('--output_dir', type=str, default='vis_results', help='可视化存储目录')

    parser.add_argument('--local', default='False', choices=['True', 'False'], type=str, help='是否本地运行')
    parser.add_argument('--root_sever', type=str, default=r'/opt/data/private/data_set/sketch_retrieval')
    parser.add_argument('--root_local', type=str, default=r'D:\document\DeepLearning\DataSet\sketch_retrieval\sketchy')

    args = parser.parse_args()
    return args


def compute_category_metrics(similarity_matrix, sketch_labels, image_labels, sketch_categories):
    """计算每个类别的检索指标"""
    category_metrics = {}
    
    # 获取唯一类别
    unique_categories = list(set(sketch_categories))
    
    for category in unique_categories:
        # 找到该类别的草图索引
        cat_sketch_indices = [i for i, cat in enumerate(sketch_categories) if cat == category]
        
        if len(cat_sketch_indices) == 0:
            continue
        
        # 计算该类别的Top-1, Top-5准确率
        top1_correct = 0
        top5_correct = 0
        
        for sketch_idx in cat_sketch_indices:
            similarities = similarity_matrix[sketch_idx]
            sorted_indices = torch.argsort(similarities, descending=True)
            
            # Top-1准确率
            if image_labels[sorted_indices[0]] == sketch_labels[sketch_idx]:
                top1_correct += 1
            
            # Top-5准确率
            if any(image_labels[sorted_indices[:5]] == sketch_labels[sketch_idx]):
                top5_correct += 1
        
        category_metrics[category] = {
            'top1_accuracy': top1_correct / len(cat_sketch_indices),
            'top5_accuracy': top5_correct / len(cat_sketch_indices),
            'num_samples': len(cat_sketch_indices),
            'top1_correct': top1_correct,
            'top5_correct': top5_correct
        }
    
    return category_metrics


def visualize_category_retrieval(similarity_matrix, sketch_features, image_features, 
                               sketch_labels, image_labels, sketch_categories, image_categories,
                               sketch_loader, image_loader, category, output_dir, num_examples=8):
    """
    可视化特定类别的检索结果
    
    Args:
        similarity_matrix: 相似度矩阵
        sketch_features: 草图特征
        image_features: 图像特征  
        sketch_labels: 草图标签
        image_labels: 图像标签
        sketch_categories: 草图类别名
        image_categories: 图像类别名
        sketch_loader: 草图数据加载器
        image_loader: 图像数据加载器
        category: 要可视化的类别
        output_dir: 输出目录
        num_examples: 可视化示例数量
    """
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 找到该类别的草图索引
    cat_sketch_indices = [i for i, cat in enumerate(sketch_categories) if cat == category]
    
    if len(cat_sketch_indices) == 0:
        print(f"类别 {category} 没有找到草图样本")
        return
    
    # 随机选择几个示例
    selected_indices = np.random.choice(cat_sketch_indices, 
                                      min(num_examples, len(cat_sketch_indices)), 
                                      replace=False)
    
    # 收集所有图像数据用于显示
    all_sketch_data = []
    all_image_data = []
    
    # 从数据加载器中收集原始图像数据
    denormalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    # 收集草图数据
    sketch_count = 0
    for sketches, images, _, cat_names in sketch_loader:
        for i in range(len(sketches)):

            sketch_img = denormalize(utils.s5_to_tensor_img(sketches[i])).clamp(0, 1)
            # sketch_img = denormalize(sketches[i]).clamp(0, 1)
            all_sketch_data.append(sketch_img)
            sketch_count += 1
            
    # 收集图像数据
    image_count = 0
    for sketches, images, _, cat_names in image_loader:
        for i in range(len(images)):
            image_img = denormalize(images[i]).clamp(0, 1)
            all_image_data.append(image_img)
            image_count += 1
    
    print(f"收集到 {len(all_sketch_data)} 个草图, {len(all_image_data)} 个图像")
    
    # 创建大的可视化画布
    fig = plt.figure(figsize=(20, 4 * len(selected_indices)))
    
    for row, sketch_idx in enumerate(selected_indices):
        # 获取该草图的Top-6检索结果
        similarities = similarity_matrix[sketch_idx]
        top_indices = torch.argsort(similarities, descending=True)[:6]
        
        # 显示草图
        ax_sketch = plt.subplot(len(selected_indices), 7, row * 7 + 1)
        if sketch_idx < len(all_sketch_data):
            ax_sketch.imshow(all_sketch_data[sketch_idx].permute(1, 2, 0))
        ax_sketch.set_title(f'草图\\n{category}', fontsize=12, fontweight='bold')
        ax_sketch.axis('off')
        
        # 显示检索结果
        for col, img_idx in enumerate(top_indices):
            ax_result = plt.subplot(len(selected_indices), 7, row * 7 + col + 2)
            
            if img_idx < len(all_image_data):
                ax_result.imshow(all_image_data[img_idx].permute(1, 2, 0))
                
                # 获取检索结果信息
                score = similarities[img_idx].item()
                retrieved_category = image_categories[img_idx] if img_idx < len(image_categories) else "Unknown"
                is_correct = retrieved_category == category
                
                # 设置标题颜色
                title_color = 'green' if is_correct else 'red'
                mark = "✓" if is_correct else "✗"
                
                ax_result.set_title(f'{mark} {retrieved_category}\\n{score:.3f}', 
                                  fontsize=10, color=title_color)
            
            ax_result.axis('off')
    
    plt.suptitle(f'类别 "{category}" 的检索结果可视化', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存结果
    safe_category = category.replace('/', '_').replace('\\', '_')
    viz_path = os.path.join(output_dir, f'category_{safe_category}_retrieval.png')
    plt.savefig(viz_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"类别 {category} 的可视化结果已保存到: {viz_path}")


def create_category_summary_plot(category_metrics, output_dir):
    """创建类别性能总结图"""
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'] 
    plt.rcParams['axes.unicode_minus'] = False
    
    # 按Top-1准确率排序
    sorted_categories = sorted(category_metrics.items(), 
                             key=lambda x: x[1]['top1_accuracy'], reverse=True)
    
    # 取前15个类别用于显示
    top_categories = sorted_categories[:15]
    
    categories = [item[0] for item in top_categories]
    top1_accs = [item[1]['top1_accuracy'] for item in top_categories]
    top5_accs = [item[1]['top5_accuracy'] for item in top_categories]
    sample_counts = [item[1]['num_samples'] for item in top_categories]
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # 第一个子图：Top-1和Top-5准确率对比
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, top1_accs, width, label='Top-1 准确率', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x + width/2, top5_accs, width, label='Top-5 准确率', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('类别')
    ax1.set_ylabel('准确率')
    ax1.set_title('各类别检索准确率对比 (前15名)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 在柱状图上显示数值
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 第二个子图：样本数量分布
    bars3 = ax2.bar(categories, sample_counts, alpha=0.7, color='lightgreen')
    ax2.set_xlabel('类别')
    ax2.set_ylabel('样本数量')
    ax2.set_title('各类别样本数量分布')
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # 在柱状图上显示数值
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    summary_path = os.path.join(output_dir, 'category_performance_summary.png')
    plt.savefig(summary_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"类别性能总结图已保存到: {summary_path}")
    
    return top_categories


def main(args):
    print("开始可视化前5好类别的PNG草图-图像检索效果...")
    
    # 设置路径
    checkpoint_path = os.path.join(args.weight_dir, args.save_str + '.pth')

    if args.sketch_format == 'vector':
        split_file = './data/fixed_splits/vec_sketch_image_dataset_splits.pkl'
    else:
        split_file = './data/fixed_splits/png_sketch_image_dataset_splits.pkl'
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"检查点文件不存在: {checkpoint_path}")
        return
    
    if not os.path.exists(split_file):
        print(f"数据集划分文件不存在: {split_file}")
        return
    
    # 创建数据加载器
    print("加载测试数据集...")
    root = args.root_local if eval(args.local) else args.root_sever
    train_loader, test_loader, dataset_info = create_png_sketch_dataloaders(
        batch_size=32,
        num_workers=4,
        fixed_split_path=split_file,
        root=root,
        sketch_format=args.sketch_format,
        vec_sketch_type=args.vec_sketch_type,
        sketch_image_subdirs=args.sketch_image_subdirs
    )
    
    print(f"测试集大小: {dataset_info['test_info']['total_pairs']}")
    print(f"共有 {dataset_info['category_info']['num_categories']} 个类别")
    
    # 创建并加载模型
    print(f"从 {checkpoint_path} 加载模型...")
    model = sbir_model_wrapper.create_sbir_model_wrapper(
        embed_dim=args.embed_dim,
        freeze_image_encoder=eval(args.is_freeze_image_encoder),
        freeze_sketch_backbone=eval(args.is_freeze_sketch_backbone),
        sketch_format=args.sketch_format
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"成功加载检查点 (epoch {checkpoint.get('epoch', 'unknown')})")
    
    model.to(device)
    model.eval()
    
    # 提取特征
    print("提取特征...")
    sketch_features = []
    image_features = []
    sketch_labels = []
    image_labels = []
    sketch_categories = []
    image_categories = []
    
    with torch.no_grad():
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
    print("计算相似度矩阵...")
    similarity_matrix = torch.matmul(sketch_features, image_features.t())
    
    # 计算每个类别的性能指标
    print("计算类别性能指标...")
    category_metrics = compute_category_metrics(
        similarity_matrix, sketch_labels, image_labels, sketch_categories
    )
    
    # 创建类别性能总结图
    print("创建类别性能总结图...")
    top_categories = create_category_summary_plot(category_metrics, args.output_dir)
    
    # 显示前5好类别的统计信息
    print(f"\n=== 前5好类别统计 ===")
    for i, (category, metrics) in enumerate(top_categories[:5]):
        print(f"{i+1}. {category}:")
        print(f"   Top-1准确率: {metrics['top1_accuracy']:.4f}")
        print(f"   Top-5准确率: {metrics['top5_accuracy']:.4f}")
        print(f"   样本数量: {metrics['num_samples']}")
        print(f"   Top-1正确数: {metrics['top1_correct']}/{metrics['num_samples']}")
        print(f"   Top-5正确数: {metrics['top5_correct']}/{metrics['num_samples']}")
        print()
    
    # 为前5好类别创建详细的可视化
    print("为前5好类别创建详细可视化...")
    for i, (category, metrics) in enumerate(top_categories[:5]):
        print(f"正在处理类别 {i+1}: {category}")
        visualize_category_retrieval(
            similarity_matrix, sketch_features, image_features,
            sketch_labels, image_labels, sketch_categories, image_categories,
            test_loader, test_loader, category, args.output_dir, num_examples=6
        )
    
    # 保存详细结果
    results = {
        'top_5_categories': {
            category: metrics for category, metrics in top_categories[:5]
        },
        'all_category_metrics': category_metrics,
        'dataset_info': dataset_info
    }
    
    results_file = os.path.join(args.output_dir, 'top5_categories_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果已保存到: {results_file}")
    print(f"所有可视化文件已保存到目录: {args.output_dir}")
    
    # 最终总结
    print(f"\n=== 可视化完成总结 ===")
    print(f"处理了 {len(category_metrics)} 个类别")
    print(f"为前5好类别创建了详细的检索效果可视化")
    print(f"平均Top-1准确率: {np.mean([m['top1_accuracy'] for m in category_metrics.values()]):.4f}")
    print(f"平均Top-5准确率: {np.mean([m['top5_accuracy'] for m in category_metrics.values()]):.4f}")
    
    print(f"\n输出文件:")
    print(f"  - 类别性能总结图: category_performance_summary.png")
    print(f"  - 各类别详细可视化: category_*_retrieval.png")
    print(f"  - 详细数据结果: top5_categories_results.json")


if __name__ == '__main__':
    main(parse_args())

