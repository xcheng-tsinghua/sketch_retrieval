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
import math
from pathlib import Path
from functools import partial

from utils import utils


def get_subdirs(dir_path):
    """
    获取 dir_path 的所有一级子文件夹
    仅仅是文件夹名，不是完整路径
    """
    path_allclasses = Path(dir_path)
    directories = [str(x) for x in path_allclasses.iterdir() if x.is_dir()]
    dir_names = [item.split(os.sep)[-1] for item in directories]

    return dir_names


def get_allfiles(dir_path, suffix='txt', filename_only=False):
    """
    获取dir_path下的全部文件路径
    :param dir_path:
    :param suffix: 文件后缀，不需要 "."，如果是 None 则返回全部文件，不筛选类型
    :param filename_only:
    :return: [file_path0, file_path1, ...]
    """
    filepath_all = []

    for root, dirs, files in os.walk(dir_path):
        for file in files:

            if suffix is not None:
                if file.split('.')[-1] == suffix:
                    if filename_only:
                        current_filepath = file
                    else:
                        current_filepath = str(os.path.join(root, file))
                    filepath_all.append(current_filepath)

            else:
                if filename_only:
                    current_filepath = file
                else:
                    current_filepath = str(os.path.join(root, file))
                filepath_all.append(current_filepath)

    return filepath_all


class RetrievalDataset(Dataset):
    def __init__(self,
                 root,
                 mode='train',
                 max_seq_length=11*32,
                 test_ratio=0.2,
                 sketch_image_subdirs=('sketch_s3_352', 'sketch_png', 'photo'),  # [0]: vector_sketch, [1]: image_sketch, [2]: photo
                 sketch_format='vector',  # ['vector', 'image']
                 sketch_transform=None,
                 image_transform=None
                 ):

        print(f'Retrieval dataset from: {root}')
        self.max_seq_length = max_seq_length
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

        self.mode = mode
        is_train = True if mode == 'train' else False
        self.sketch_format = sketch_format

        if self.sketch_format == 'vector':
            suffix = 'txt'
            sketch_subdir = sketch_image_subdirs[0]
        else:
            suffix = 'png'
            sketch_subdir = sketch_image_subdirs[1]
        photo_subdir = sketch_image_subdirs[2]

        # 草图根目录
        sketch_root = os.path.join(root, sketch_subdir)

        # 草图类别
        classes = get_subdirs(sketch_root)
        self.categories = classes

        self.sketch_image = []  # (sketch_path, image_path, category_name)
        self.images = set()  # 记录图片路径

        for c_class in classes:
            c_class_root = os.path.join(sketch_root, c_class)

            # 获取全部草图 txt 文件
            c_sketch_all = get_allfiles(c_class_root, suffix)

            n_c_sketch = len(c_sketch_all)
            test_idx = math.ceil(n_c_sketch * test_ratio)

            for idx, c_sketch in enumerate(c_sketch_all):
                # 获取图片文件名
                img_name = os.path.basename(c_sketch).split('-')[0] + '.jpg'
                img_path = os.path.join(root, photo_subdir, c_class, img_name)

                # 训练时需要将 idx >= test_idx 的数据
                if is_train and idx >= test_idx:
                    self.images.add(img_path)
                    self.sketch_image.append((c_sketch, img_path, c_class))

                # 测试时取 idx < test_idx 的数据
                elif not is_train and idx < test_idx:
                    self.images.add(img_path)
                    self.sketch_image.append((c_sketch, img_path, c_class))

        self.images = list(self.images)
        self.category_to_idx = dict(zip(sorted(classes), range(len(classes))))  # 类别名到int数据的映射
        print(f'instance all: {len(self.sketch_image)}')

    def __getitem__(self, index):
        sketch_path, image_path, category = self.sketch_image[index]

        if self.sketch_format == 'vector':
            sketch, mask = utils.s3_file_to_s5(sketch_path, self.max_seq_length)
        else:
            sketch_pil = Image.open(sketch_path).convert('RGB')
            sketch = self.sketch_transform(sketch_pil)

        # 加载JPG图像
        image_pil = Image.open(image_path).convert('RGB')
        image = self.image_transform(image_pil)

        # 获取类别索引
        category_idx = self.category_to_idx[category]
        return sketch, image, category_idx, category

    def __len__(self):
        return len(self.sketch_image)

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
        for _, _, category in self.sketch_image:
            category_counts[category] = category_counts.get(category, 0) + 1

        return {
            'mode': self.mode,
            'total_pairs': len(self.sketch_image),
            'num_categories': len(self.categories),
            'category_counts': category_counts
        }


class PNGSketchImageDataset(Dataset):
    """
    PNG草图-图像配对数据集
    """
    
    def __init__(self,
                 root,
                 mode,
                 fixed_split_path,
                 sketch_transform=None,
                 image_transform=None,
                 sketch_format='vector',  # ['vector', 'image']
                 sketch_image_subdirs=('sketch_s3_352', 'sketch_png', 'photo'),  # [0]: vector_sketch, [1]: image_sketch, [2]: photo
                 vec_seq_length=11*32
                 ):
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
        self.sketch_format = sketch_format

        # 默认变换
        self.sketch_transform = sketch_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.image_transform = image_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.image_subdir = sketch_image_subdirs[2]
        if self.sketch_format == 'vector':
            self.sketch_subdir = sketch_image_subdirs[0]
            self.sketch_loader = partial(
                utils.s3_file_to_s5,
                max_length=vec_seq_length,
                coor_mode='REL',
                is_shuffle_stroke=False,
                is_back_mask=False
            )
        else:
            self.sketch_subdir = sketch_image_subdirs[1]
            self.sketch_loader = partial(
                image_loader,
                image_transform=self.sketch_transform
            )
        
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
        sketch_file, image_file, category = self.data_pairs[idx]

        sketch_path = os.path.join(self.root, self.sketch_subdir, category, sketch_file)
        image_path = os.path.join(self.root, self.image_subdir, category, image_file)

        try:
            # 加载草图
            sketch = self.sketch_loader(sketch_path)

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

        # max_iter = 100
        # c_iter = 0
        # while True:
        #     if c_iter > max_iter:
        #         raise ValueError('having searched too much files, no valid file is available')
        #
        #     sketch_path, image_path, category = self.get_path(idx)
        #
        #     if os.path.exists(sketch_path) and os.path.exists(image_path):
        #         break
        #     else:
        #         idx = self.next(idx)
        #         c_iter += 1
        #
        # try:
        #     # 加载PNG草图
        #     # sketch_pil = Image.open(sketch_path).convert('RGB')
        #     # sketch = self.sketch_transform(sketch_pil)
        #
        #     # 加载 S3 草图
        #     sketch, mask = s3_file_to_s5(sketch_path)
        #
        #     # 加载JPG图像
        #     image_pil = Image.open(image_path).convert('RGB')
        #     image = self.image_transform(image_pil)
        #
        #     # 获取类别索引
        #     category_idx = self.category_to_idx[category]
        #
        #     return sketch, image, category_idx, category
        #
        # except Exception as e:
        #     print(f"Error loading data at index {idx}: {e}")
        #     print(f"Sketch path: {sketch_path}")
        #     print(f"Image path: {image_path}")
        #     raise e

    def get_category_info(self):
        """获取类别信息"""
        return {
            'categories': self.categories,
            'category_to_idx': self.category_to_idx,
            'num_categories': len(self.categories)
        }

    def next(self, idx):
        if idx == len(self) - 1:
            idx = 0
        else:
            idx += 1
        return idx
    
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


def image_loader(image_path, image_transform):
    image_pil = Image.open(image_path).convert('RGB')
    image = image_transform(image_pil)
    return image


def create_png_sketch_dataloaders(batch_size,
                                  num_workers,
                                  fixed_split_path,
                                  root,
                                  sketch_format
                                  ):
    """
    创建训练和测试数据加载器
    
    Args:
        batch_size: 批次大小
        num_workers: 数据加载进程数
        fixed_split_path: 固定数据集划分文件路径
        root:
        sketch_format:
        
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = PNGSketchImageDataset(
        mode='train',
        fixed_split_path=fixed_split_path,
        sketch_transform=train_sketch_transform,
        image_transform=train_image_transform,
        root=root,
        sketch_format=sketch_format
    )

    test_dataset = PNGSketchImageDataset(
        mode='test',
        fixed_split_path=fixed_split_path,
        sketch_transform=test_transform,
        image_transform=test_transform,
        root=root,
        sketch_format=sketch_format
    )

    # train_dataset = RetrievalDataset(
    #     mode='train',
    #     sketch_transform=train_sketch_transform,
    #     image_transform=train_image_transform,
    #     root=root,
    #     sketch_format=sketch_format,
    #     sketch_image_subdirs=sketch_image_subdirs
    # )
    #
    # test_dataset = RetrievalDataset(
    #     mode='test',
    #     sketch_transform=test_transform,
    #     image_transform=test_transform,
    #     root=root,
    #     sketch_format=sketch_format,
    #     sketch_image_subdirs=sketch_image_subdirs
    # )
    
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
        print("请先运行 dataset_split.py 创建数据集划分文件")
