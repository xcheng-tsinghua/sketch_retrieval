"""
数据集预处理和固定化脚本
将SketchImageDataset的数据对保存为固定的pickle文件，确保每次加载完全一致
"""
import os
import pickle
import numpy as np
from pathlib import Path
import random
from data.SLMDataset import get_subdirs, get_allfiles

def create_fixed_dataset_splits(
    sketch_root=r'E:\Master\Experiment\data\stroke-normal',
    image_root=r'E:\Master\Experiment\data\photo',
    output_dir='data/fixed_splits',
    train_split=0.8,
    random_seed=42
):
    """
    创建固定的数据集划分并保存到文件
    
    Args:
        sketch_root: 草图数据根目录
        image_root: 图片数据根目录  
        output_dir: 输出目录
        train_split: 训练集比例
        random_seed: 随机种子
    """
    
    print("开始创建固定的数据集划分...")
    print(f"草图路径: {sketch_root}")
    print(f"图片路径: {image_root}")
    print(f"训练集比例: {train_split}")
    print(f"随机种子: {random_seed}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # 获取所有类别
    sketch_categories = get_subdirs(sketch_root)
    image_categories = get_subdirs(image_root)
    
    # 找到草图和图片都有的类别
    common_categories = list(set(sketch_categories) & set(image_categories))
    common_categories.sort()
    
    print(f'找到 {len(common_categories)} 个共同类别')
    
    all_data_pairs = []
    category_stats = {}
    
    for category in common_categories:
        sketch_category_path = os.path.join(sketch_root, category)
        image_category_path = os.path.join(image_root, category)
        
        # 获取该类别下的所有草图文件和图片文件
        sketch_files = get_allfiles(sketch_category_path, 'npy', filename_only=True)
        image_files = get_allfiles(image_category_path, 'jpg', filename_only=True)
        
        # 构建图片实例字典和对应的草图列表
        instance_sketches = {}  # {实例id: [草图文件列表]}
        image_dict = {}  # {实例id: 图片路径}
        
        # 构建图片实例字典
        for image_file in image_files:
            instance_id = image_file.replace('.jpg', '')
            image_path = os.path.join(image_category_path, image_file)
            image_dict[instance_id] = image_path
            instance_sketches[instance_id] = []
        
        # 收集每个实例对应的所有草图
        for sketch_file in sketch_files:
            if '_stroke5_normalized.npy' in sketch_file:
                base_name = sketch_file.replace('_stroke5_normalized.npy', '')
                if '-' in base_name:
                    instance_id = base_name.rsplit('-', 1)[0]
                else:
                    instance_id = base_name
                
                if instance_id in instance_sketches:
                    instance_sketches[instance_id].append(sketch_file)
        
        # 为每个实例选择一张草图与图片配对（使用确定性方法）
        category_pairs = []
        for instance_id, sketch_list in instance_sketches.items():
            if len(sketch_list) > 0 and instance_id in image_dict:
                # 使用确定性的选择方式（基于instance_id hash来选择）
                sketch_idx = hash(category + instance_id + str(random_seed)) % len(sketch_list)
                selected_sketch = sketch_list[sketch_idx]
                sketch_path = os.path.join(sketch_category_path, selected_sketch)
                image_path = image_dict[instance_id]
                category_pairs.append((sketch_path, image_path, category))
        
        all_data_pairs.extend(category_pairs)
        category_stats[category] = len(category_pairs)
        print(f"  {category}: {len(category_pairs)} 对")
    
    print(f"总共创建了 {len(all_data_pairs)} 个数据对")
    
    # 按类别进行分层采样划分训练集和测试集
    category_pairs_dict = {}
    for pair in all_data_pairs:
        category = pair[2]
        if category not in category_pairs_dict:
            category_pairs_dict[category] = []
        category_pairs_dict[category].append(pair)
    
    # 对每个类别进行训练/测试划分
    train_pairs = []
    test_pairs = []
    
    train_stats = {}
    test_stats = {}
    
    for category, pairs in category_pairs_dict.items():
        # 使用固定种子打乱该类别的样本
        random.Random(random_seed + hash(category)).shuffle(pairs)
        
        split_idx = int(len(pairs) * train_split)
        
        # 确保每个类别在训练集和测试集中都有至少1个样本
        if split_idx == 0:
            split_idx = 1
        if split_idx == len(pairs):
            split_idx = len(pairs) - 1
            
        category_train = pairs[:split_idx]
        category_test = pairs[split_idx:]
        
        train_pairs.extend(category_train)
        test_pairs.extend(category_test)
        
        train_stats[category] = len(category_train)
        test_stats[category] = len(category_test)
    
    # 最终打乱
    random.Random(random_seed).shuffle(train_pairs) 
    random.Random(random_seed + 1).shuffle(test_pairs)
    
    print(f"训练集: {len(train_pairs)} 对")
    print(f"测试集: {len(test_pairs)} 对")
    
    # 保存数据集划分
    dataset_info = {
        'train_pairs': train_pairs,
        'test_pairs': test_pairs,
        'category_stats': category_stats,
        'train_stats': train_stats,
        'test_stats': test_stats,
        'total_categories': len(common_categories),
        'train_split': train_split,
        'random_seed': random_seed,
        'common_categories': common_categories
    }
    
    # 保存为pickle文件
    dataset_file = os.path.join(output_dir, 'sketch_image_dataset_splits.pkl')
    with open(dataset_file, 'wb') as f:
        pickle.dump(dataset_info, f)
    
    print(f"数据集划分已保存到: {dataset_file}")
    
    # 保存统计信息为文本文件
    stats_file = os.path.join(output_dir, 'dataset_statistics.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("数据集统计信息\\n")
        f.write("=" * 50 + "\\n")
        f.write(f"总类别数: {len(common_categories)}\\n")
        f.write(f"总数据对数: {len(all_data_pairs)}\\n")
        f.write(f"训练集: {len(train_pairs)} 对\\n")
        f.write(f"测试集: {len(test_pairs)} 对\\n")
        f.write(f"训练集比例: {train_split}\\n")
        f.write(f"随机种子: {random_seed}\\n\\n")
        
        f.write("各类别统计:\\n")
        f.write("-" * 30 + "\\n")
        for category in sorted(common_categories):
            total = category_stats.get(category, 0)
            train = train_stats.get(category, 0)
            test = test_stats.get(category, 0)
            f.write(f"{category:20s}: 总计={total:3d}, 训练={train:3d}, 测试={test:3d}\\n")
    
    print(f"统计信息已保存到: {stats_file}")
    
    return dataset_info

if __name__ == '__main__':
    # 创建固定的数据集划分
    dataset_info = create_fixed_dataset_splits()
    
    print("\\n数据集划分创建完成！")
    print("现在可以在SketchImageDataset中使用固定的数据划分了。")
