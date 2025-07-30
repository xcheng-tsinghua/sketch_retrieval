"""
PNG草图-图像数据集加载器
用于加载PNG格式的草图和对应的图片进行训练
"""
import os
import pickle
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random


class PNGSketchImageDataset(Dataset):
    """
    PNG草图-图像配对数据集
    """
    
    def __init__(self,
                 root=r'D:\document\DeepLearning\DataSet\sketch_retrieval\sketchy',
                 mode='train',
                 fixed_split_path='./data/fixed_splits/png_sketch_image_dataset_splits.pkl',
                 sketch_transform=None,
                 image_transform=None):
        """
        初始化数据集
        
        Args:
            mode: 'train' 或 'test'
            fixed_split_path: 固定数据集划分文件路径
            sketch_transform: 草图变换
            image_transform: 图像变换
        """
        
        print(f"PNGSketchImageDataset initialized with:")
        print(f"  Mode: {mode}")
        print(f"  Fixed split path: {fixed_split_path}")
        
        self.mode = mode
        self.fixed_split_path = fixed_split_path
        self.root = root
        
        # 默认变换
        self.sketch_transform = sketch_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.image_transform = image_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # 加载固定的数据集划分
        self._load_fixed_split()
        
    def _load_fixed_split(self):
        """加载固定的数据集划分"""
        print(f"Loading fixed dataset split from: {self.fixed_split_path}")
        
        if not os.path.exists(self.fixed_split_path):
            raise FileNotFoundError(f"固定数据集划分文件不存在: {self.fixed_split_path}")
            
        with open(self.fixed_split_path, 'rb') as f:
            dataset_info = pickle.load(f)
        
        # 根据模式选择数据
        if self.mode == 'train':
            self.data_pairs = dataset_info['train_pairs']
        elif self.mode == 'test':
            self.data_pairs = dataset_info['test_pairs']
        else:
            raise ValueError(f"不支持的模式: {self.mode}")
        
        self.categories = dataset_info['common_categories']
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        
        print(f"Loaded fixed split: {len(self.data_pairs)} pairs")
        print(f"Total categories: {len(self.categories)}")
        
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        """
        获取一个数据样本
        
        Returns:
            sketch: 预处理后的草图张量 [3, 224, 224]
            image: 预处理后的图像张量 [3, 224, 224]  
            category_idx: 类别索引
            category_name: 类别名称
        """
        sketch_path, image_path, category = self.data_pairs[idx]
        # sketch_path: 'E:\\Master\\Experiment\\data\\sketch\\teddy_bear\\n04399382_22297-5.png'
        # image_path: 'E:\\Master\\Experiment\\data\\photo\\teddy_bear\\n04399382_22297.jpg'
        # category: 'teddy_bear'

        sketch_path = sketch_path.replace('E:\\Master\\Experiment\\data\\sketch', self.root + '\\sketch_s3_352')
        sketch_path = os.path.splitext(sketch_path)[0]
        sketch_path = sketch_path + '.txt'
        image_path = image_path.replace('E:\\Master\\Experiment\\data', self.root)
        # sketch_path: 'D:\\document\\DeepLearning\\DataSet\\sketch_retrieval\\sketchy\\sketch_s3_352\\strawberry\\n07745940_1188-4.png'
        # image_path: 'D:\\document\\DeepLearning\\DataSet\\sketch_retrieval\\sketchy\\photo\\strawberry\\n07745940_1188.jpg'

        try:
            # 加载PNG草图
            # sketch_pil = Image.open(sketch_path).convert('RGB')
            # sketch = self.sketch_transform(sketch_pil)

            # 加载 S3 草图
            sketch, mask = s3_file_to_s5(sketch_path)
            
            # 加载JPG图像
            image_pil = Image.open(image_path).convert('RGB')
            image = self.image_transform(image_pil)
            
            # 获取类别索引
            category_idx = self.category_to_idx[category]
            
            return sketch, image, category_idx, category
            
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            print(f"Sketch path: {sketch_path}")
            print(f"Image path: {image_path}")
            raise e
    
    def get_category_info(self):
        """获取类别信息"""
        return {
            'categories': self.categories,
            'category_to_idx': self.category_to_idx,
            'num_categories': len(self.categories)
        }
    
    def get_data_info(self):
        """获取数据集信息"""
        category_counts = {}
        for _, _, category in self.data_pairs:
            category_counts[category] = category_counts.get(category, 0) + 1
            
        return {
            'mode': self.mode,
            'total_pairs': len(self.data_pairs),
            'num_categories': len(self.categories),
            'category_counts': category_counts
        }


def s3_file_to_s5(root, max_length=11*32, coor_mode='REL', is_shuffle_stroke=False):
    """
    将草S3图转换为 S5 格式，(x, y, s1, s2, s3)
    默认存储绝对坐标
    :param root:
    :param max_length:
    :param coor_mode: ['ABS', 'REL'], 'ABS': absolute coordinate. 'REL': relative coordinate [(x,y), (△x, △y), (△x, △y), ...].
    :param is_shuffle_stroke: 是否打乱笔划
    :return:
    """
    data_raw = np.loadtxt(root, delimiter=',')

    # 打乱笔划
    if is_shuffle_stroke:
        stroke_list = np.split(data_raw, np.where(data_raw[:, 2] == 0)[0] + 1)[:-1]
        random.shuffle(stroke_list)
        data_raw = np.vstack(stroke_list)

    # 多于指定点数则进行截断
    n_point_raw = len(data_raw)
    if n_point_raw > max_length:
        data_raw = data_raw[:max_length, :]

    # 相对坐标
    if coor_mode == 'REL':
        coordinate = data_raw[:, :2]
        coordinate[1:] = coordinate[1:] - coordinate[:-1]
        data_raw[:, :2] = coordinate

    elif coor_mode == 'ABS':
        # 无需处理
        pass

    else:
        raise TypeError('error coor mode')

    c_sketch_len = len(data_raw)
    data_raw = torch.from_numpy(data_raw)

    data_cube = torch.zeros(max_length, 5, dtype=torch.float)
    mask = torch.zeros(max_length, dtype=torch.float)

    data_cube[:c_sketch_len, :2] = data_raw[:, :2]
    data_cube[:c_sketch_len, 2] = data_raw[:, 2]
    data_cube[:c_sketch_len, 3] = 1 - data_raw[:, 2]
    data_cube[-1, 4] = 1

    mask[:c_sketch_len] = 1

    return data_cube, mask


def create_png_sketch_dataloaders(batch_size=32, 
                                num_workers=4,
                                fixed_split_path='./data/fixed_splits/png_sketch_image_dataset_splits.pkl'):
    """
    创建训练和测试数据加载器
    
    Args:
        batch_size: 批次大小
        num_workers: 数据加载进程数
        fixed_split_path: 固定数据集划分文件路径
        
    Returns:
        train_loader, test_loader, dataset_info
    """
    
    # 数据增强变换
    train_sketch_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = PNGSketchImageDataset(
        mode='train',
        fixed_split_path=fixed_split_path,
        sketch_transform=train_sketch_transform,
        image_transform=train_image_transform
    )
    
    test_dataset = PNGSketchImageDataset(
        mode='test', 
        fixed_split_path=fixed_split_path,
        sketch_transform=test_transform,
        image_transform=test_transform
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 获取数据集信息
    dataset_info = {
        'train_info': train_dataset.get_data_info(),
        'test_info': test_dataset.get_data_info(),
        'category_info': train_dataset.get_category_info()
    }
    
    return train_loader, test_loader, dataset_info


if __name__ == '__main__':
    # 测试数据集加载
    print("测试PNG草图数据集加载...")
    
    try:
        train_loader, test_loader, dataset_info = create_png_sketch_dataloaders(batch_size=4)
        
        print(f"训练集: {dataset_info['train_info']['total_pairs']} 对")
        print(f"测试集: {dataset_info['test_info']['total_pairs']} 对")
        print(f"类别数: {dataset_info['category_info']['num_categories']}")
        
        # 测试加载一个批次
        for batch_idx, (sketches, images, category_indices, category_names) in enumerate(train_loader):
            print(f"\\n第 {batch_idx + 1} 个批次:")
            print(f"  草图形状: {sketches.shape}")
            print(f"  图像形状: {images.shape}")
            print(f"  类别索引: {category_indices}")
            print(f"  类别名称: {category_names}")
            break
            
    except Exception as e:
        print(f"数据集加载失败: {e}")
        print("请先运行 create_png_sketch_dataset.py 创建数据集划分文件")
