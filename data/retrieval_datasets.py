"""
PNG草图-图像数据集加载器
用于加载PNG格式的草图和对应的图片进行训练
"""
import os
import pickle
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pathlib import Path
from functools import partial
import numpy as np
import random

from utils import utils
from data import sketchy_configs as scfg


class SketchImageDataset(Dataset):
    """
    PNG草图-图像配对数据集
    """
    def __init__(self,
                 root,
                 mode,
                 fixed_split_path,
                 vec_sketch_rep,  # [S5, STK_11_32]
                 sketch_image_subdirs,  # [0]: vector_sketch, [1]: image_sketch, [2]: photo
                 sketch_transform=None,
                 image_transform=None,
                 sketch_format='vector',  # ['vector', 'image']
                 vec_seq_length=11*32,
                 is_back_image_only=False
                 ):
        """
        初始化数据集
        
        Args:
            mode: 'train' 或 'test'
            fixed_split_path: 固定数据集划分文件路径
            sketch_transform: 草图变换
            image_transform: 图像变换
            is_back_image_only: 是否仅返回图像，用于一张图片对应多个草图时，进行测试时不返回重复的图片

        """
        assert mode in ('train', 'test')

        print(f"PNGSketchImageDataset initialized with:")
        print(f"  Mode: {mode}")
        print(f"  Fixed split path: {fixed_split_path}")
        
        self.mode = mode
        self.fixed_split_path = fixed_split_path
        self.root = root
        self.sketch_format = sketch_format
        self.is_back_image_only = is_back_image_only

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

            if vec_sketch_rep == 'S5':
                self.sketch_loader = partial(
                    utils.s3_file_to_s5,
                    max_length=vec_seq_length,
                    coor_mode='REL',
                    is_shuffle_stroke=False,
                    is_back_mask=False
                )
            elif 'STK' in vec_sketch_rep:
                self.sketch_loader = partial(
                    utils.load_stk_sketch,
                    stk_name=vec_sketch_rep
                )
            else:
                raise TypeError('error vector sketch type')

        else:
            self.sketch_subdir = sketch_image_subdirs[1]
            self.sketch_loader = partial(
                utils.image_loader,
                image_transform=self.sketch_transform
            )
        
        # 加载固定的数据集划分
        self._load_fixed_split()
        # print(self.mode + f' pairs: {len(self)}')
        
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
        self.images_set = dataset_info['images_set']
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        
        print(f"Loaded fixed split: {len(self.data_pairs)} pairs")
        print(f"Total categories: {len(self.categories)}")
        
    def __len__(self):
        if self.is_back_image_only:
            return len(self.images_set)
        else:
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

        if self.is_back_image_only:
            image_file, category = self.images_set[idx]
            image_path = self.get_image_path(category, image_file)

            # 加载JPG图像
            image = utils.image_loader(image_path, self.image_transform)

            # 获取类别索引
            category_idx = self.category_to_idx[category]

            return idx, image, category_idx, category

        else:
            sketch_file, image_file, category = self.data_pairs[idx]

            sketch_path = self.get_sketch_path(category, sketch_file)
            image_path = self.get_image_path(category, image_file)

            try:
                # 加载草图
                sketch = self.sketch_loader(sketch_path)

                # 加载JPG图像
                image = utils.image_loader(image_path, self.image_transform)

                # 获取类别索引
                category_idx = self.category_to_idx[category]

                return sketch, image, category_idx, category
                # return idx, sketch, image

            except Exception as e:
                print(f"Error loading data at index {idx}: {e}")
                print(f"Sketch path: {sketch_path}")
                print(f"Image path: {image_path}")
                raise e

    def get_sketch_path(self, category, sketch_file):
        """
        sketch_file: 例如 aaa.png
        """
        sketch_path = os.path.join(self.root, self.sketch_subdir, category, sketch_file)
        return sketch_path

    def get_image_path(self, category, image_file):
        """
        image_file: 例如 aaa.jpg
        """
        image_path = os.path.join(self.root, self.image_subdir, category, image_file)
        return image_path

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

    def find_image_idx(self, category, filename):
        filename = filename.split('-')[0] + '.jpg'
        image_path = self.get_image_path(category, filename)
        img_index = self.images_set.index((image_path, category))

        return img_index

    def get_file_pair_by_index(self, idx):
        sketch_file, image_file, category = self.data_pairs[idx]

        sketch_path = self.get_sketch_path(category, sketch_file)
        image_path = self.get_image_path(category, image_file)

        return sketch_path, image_path

    def eval(self):
        self.is_back_image_only = True

    def train(self):
        self.is_back_image_only = False


class DatasetPreload(object):
    """
    定位文件夹层级如下：

    sketch_root
    ├─ Bushes
    │   ├─0.png
    │   ├─1.png
    │   ...
    │
    ├─ Clamps
    │   ├─0.png
    │   ├─1.png
    │   ...
    │
    ├─ Bearing
    │   ├─0.png
    │   ├─1.png
    │   ...
    │
    ...

    image_root
    ├─ Bushes
    │   ├─0.jpg
    │   ├─1.jpg
    │   ...
    │
    ├─ Clamps
    │   ├─0.jpg
    │   ├─1.jpg
    │   ...
    │
    ├─ Bearing
    │   ├─0.jpg
    │   ├─1.jpg
    │   ...
    │
    ...

    """
    def __init__(self,
                 sketch_root,
                 image_root,
                 sketch_image_suffix,  # ('txt', 'jpg') or ('png', 'jpg')
                 train_split=0.8,
                 random_seed=42,
                 is_multi_pair=False,
                 split_mode='ZS-SBIR',  # ['SBIR', 'ZS-SBIR'],
                 is_full_train=False
                 # 'SBIR': 使用所有类别，每个类别内取出一定数量用作测试。
                 # 'ZS-SBIR': 一部分类别全部用于训练，另一部分类别全部用于测试，即训练类别和测试类别不重合
                 ):

        self.train_pairs = []
        self.test_pairs = []
        self.images_set = []
        self.category_stats = {}
        self.train_stats = {}
        self.test_stats = {}
        self.total_categories = 0
        self.train_split = train_split
        self.random_seed = random_seed
        self.common_categories = []

        self.load_data(sketch_root,
                       image_root,
                       sketch_image_suffix,
                       train_split,
                       random_seed,
                       is_multi_pair,
                       split_mode,  # ['SBIR', 'ZS-SBIR'],
                       is_full_train
                       )

    def load_data(self,
                  sketch_root,
                  image_root,
                  sketch_image_suffix,
                  train_split=0.8,
                  random_seed=42,
                  is_multi_pair=False,
                  split_mode='ZS-SBIR',
                  full_train=False
                  ):
        assert split_mode in ['SBIR', 'ZS-SBIR']

        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)

        # 获取所有类别
        sketch_categories = get_subdirs(sketch_root)
        image_categories = get_subdirs(image_root)

        # 找到草图和图片都有的类别
        self.common_categories = list(set(sketch_categories) & set(image_categories))
        self.common_categories.sort()
        self.total_categories = len(self.common_categories)

        all_data_pairs = []

        for category in self.common_categories:
            sketch_category_path = os.path.join(sketch_root, category)
            image_category_path = os.path.join(image_root, category)

            # 获取该类别下的所有草图文件和图片文件
            sketch_files = get_allfiles(sketch_category_path, sketch_image_suffix[0], filename_only=True)
            image_files = get_allfiles(image_category_path, sketch_image_suffix[1], filename_only=True)

            # 构建图片实例字典和对应的草图列表
            # id 即不带路径也后缀的图片文件名，每个id和一个具体图片文件一一对应
            # 每个 id 可能对应多个草图，因为一张图片可能画了多张草图
            # 在 sketchy 数据集中，图片文件名和草图文件名的对应关系为 photo: aaa.jpg, sketch: [aaa_1.jpg, aaa_2.jpg, ...]
            #    即用下划线加序号表示同一张图片绘制的多个草图
            skhid_name = {}  # 草图 id 和文件名的对应字典, {实例id: [草图文件列表]}
            imgid_name = {}  # 图片 id 和文件名的对应字典, {实例id: 图片路径},

            # 构建图片实例字典
            for image_file in image_files:
                # 获取图片文件名，不带路径与后缀
                instance_id = os.path.splitext(image_file)[0]

                relative_image_path = os.path.join(image_category_path, image_file)  # 仅带类别和文件名的路径，例如 cat/aaa.jpg
                imgid_name[instance_id] = relative_image_path
                skhid_name[instance_id] = []

            # 收集每个实例对应的所有草图
            for sketch_file in sketch_files:
                # 去掉扩展名获取基础名称
                sketch_base_name = os.path.splitext(sketch_file)[0]

                # 尝试匹配实例ID（处理可能的草图变体）
                # 首先尝试直接匹配
                if sketch_base_name in imgid_name:
                    skhid_name[sketch_base_name].append(sketch_file)
                elif '-' in sketch_base_name:
                    # 如果有'-'分隔符，尝试取前面部分作为实例ID
                    instance_id = sketch_base_name.rsplit('-', 1)[0]
                    if instance_id in imgid_name:
                        skhid_name[instance_id].append(sketch_file)

            # 为每个草图选择一张图片配对（使用确定性方法）
            category_pairs = []
            for instance_id, sketch_list in skhid_name.items():
                if len(sketch_list) > 0 and instance_id in imgid_name:
                    image_name = imgid_name[instance_id]
                    self.images_set.append((image_name, category))

                    # 一张图片对应多张草图
                    if is_multi_pair:
                        for sketch_name in sketch_list:
                            category_pairs.append((sketch_name, image_name, category))

                    # 一张图片对应一张草图
                    else:
                        # 使用确定性的选择方式（基于instance_id hash来选择）
                        sketch_idx = hash(category + instance_id + str(random_seed)) % len(sketch_list)
                        sketch_name = sketch_list[sketch_idx]
                        category_pairs.append((sketch_name, image_name, category))

            all_data_pairs.extend(category_pairs)
            self.category_stats[category] = len(category_pairs)

        # 按类别进行分层采样划分训练集和测试集
        category_pairs_dict = {}  # {category: (sketch_name, image_name, category)}
        for pair in all_data_pairs:
            category = pair[2]
            if category not in category_pairs_dict:
                category_pairs_dict[category] = []
            category_pairs_dict[category].append(pair)

        # 对每个类别进行训练/测试划分
        # zero-shot 检索直接将类别划分为训练类别和测试类别
        if split_mode == 'ZS-SBIR':
            for category, pairs in category_pairs_dict.items():

                if full_train:
                    self.train_pairs.extend(pairs)
                    self.train_stats[category] = len(pairs)

                    if category in scfg.sketchy_test_classes:
                        self.test_pairs.extend(pairs)
                        self.test_stats[category] = len(pairs)

                    else:
                        self.test_stats[category] = 0

                else:

                    if category in scfg.sketchy_test_classes:
                        self.test_pairs.extend(pairs)
                        self.test_stats[category] = len(pairs)

                    else:
                        self.train_pairs.extend(pairs)
                        self.train_stats[category] = len(pairs)

        else:
            for category, pairs in category_pairs_dict.items():
                # 使用固定种子打乱该类别的样本
                random.Random(random_seed + hash(category)).shuffle(pairs)

                if full_train:
                    category_train = pairs
                    category_test = pairs if category in scfg.sketchy_test_classes else []

                    # category_test = pairs[:10]

                else:
                    split_idx = int(len(pairs) * train_split)

                    # 确保每个类别在训练集和测试集中都有至少1个样本
                    if split_idx == 0:
                        split_idx = 1
                    if split_idx == len(pairs):
                        split_idx = len(pairs) - 1

                    category_train = pairs[:split_idx]
                    category_test = pairs[split_idx:]

                self.train_pairs.extend(category_train)
                self.test_pairs.extend(category_test)

                self.train_stats[category] = len(category_train)
                self.test_stats[category] = len(category_test)

        # 最终打乱
        random.Random(random_seed).shuffle(self.train_pairs)
        random.Random(random_seed + 1).shuffle(self.test_pairs)

        print(f"训练集: {len(self.train_pairs)} 对")
        print(f"测试集: {len(self.test_pairs)} 对")


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


def create_sketch_image_dataloaders(batch_size,
                                    num_workers,
                                    fixed_split_path,
                                    root,
                                    sketch_format,
                                    vec_sketch_rep,
                                    sketch_image_subdirs,
                                    is_back_dataset=False
                                    ):
    """
    创建训练和测试数据加载器
    
    Args:
        batch_size: 批次大小
        num_workers: 数据加载进程数
        fixed_split_path: 固定数据集划分文件路径
        root:
        sketch_format:
        vec_sketch_rep: 矢量草图格式 [S5, STK_11_32]
        sketch_image_subdirs:
        is_back_dataset: 是否返回数据集
        
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
    train_dataset = SketchImageDataset(
        mode='train',
        fixed_split_path=fixed_split_path,
        sketch_transform=train_sketch_transform,
        image_transform=train_image_transform,
        root=root,
        sketch_format=sketch_format,
        vec_sketch_rep=vec_sketch_rep,
        sketch_image_subdirs=sketch_image_subdirs
    )

    test_dataset = SketchImageDataset(
        mode='test',
        fixed_split_path=fixed_split_path,
        sketch_transform=test_transform,
        image_transform=test_transform,
        root=root,
        sketch_format=sketch_format,
        vec_sketch_rep=vec_sketch_rep,
        sketch_image_subdirs=sketch_image_subdirs
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

    if is_back_dataset:
        return train_dataset, test_dataset, train_loader, test_loader, dataset_info
    else:
        return train_loader, test_loader, dataset_info


def create_dataset_split_file(
        save_root,
        sketch_root,
        image_root,
        sketch_image_suffix,  # ('txt', 'jpg') or ('png', 'jpg')
        train_split=0.8,
        random_seed=42,
        is_multi_pair=False,
        split_mode='ZS-SBIR',  # ['SBIR', 'ZS-SBIR'],
        full_train=False
        # 'SBIR': 使用所有类别，每个类别内取出一定数量用作测试。
        # 'ZS-SBIR': 一部分类别全部用于训练，另一部分类别全部用于测试，即训练类别和测试类别不重合
):
    """
    创建PNG草图的固定数据集划分并保存到文件

    Args:
        save_root: 划分文件存储路径及其文件名
        sketch_root: PNG草图数据根目录
        image_root: 图片数据根目录
        sketch_image_suffix: 草图和图片的文件后缀
        train_split: 训练集比例
        random_seed: 随机种子
        is_multi_pair: 是否一张图片对应多个草图
        split_mode: 检索任务类别
    """
    assert split_mode in ['SBIR', 'ZS-SBIR']
    print("开始创建PNG草图的固定数据集划分...")
    print(f"草图路径: {sketch_root}")
    print(f"图片路径: {image_root}")
    print(f"训练集比例: {train_split}")
    print(f"随机种子: {random_seed}")

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
    images_set = []
    category_stats = {}

    for category in common_categories:
        sketch_category_path = os.path.join(sketch_root, category)
        image_category_path = os.path.join(image_root, category)

        # 获取该类别下的所有草图文件和图片文件
        sketch_files = get_allfiles(sketch_category_path, sketch_image_suffix[0], filename_only=True)
        image_files = get_allfiles(image_category_path, sketch_image_suffix[1], filename_only=True)

        print(f"{category}: 找到 {len(sketch_files)} 个{sketch_image_suffix[0].upper()}草图, {len(image_files)} 个{sketch_image_suffix[1].upper()}图片")

        # 构建图片实例字典和对应的草图列表
        # id 即不带路径也后缀的图片文件名，每个id和一个具体图片文件一一对应
        # 每个 id 可能对应多个草图，因为一张图片可能画了多张草图
        # 在 sketchy 数据集中，图片文件名和草图文件名的对应关系为 photo: aaa.jpg, sketch: [aaa_1.jpg, aaa_2.jpg, ...]
        #    即用下划线加序号表示同一张图片绘制的多个草图
        skhid_name = {}  # 草图 id 和文件名的对应字典, {实例id: [草图文件列表]}
        imgid_name = {}  # 图片 id 和文件名的对应字典, {实例id: 图片路径},

        # 构建图片实例字典
        for image_file in image_files:
            # 获取图片文件名，不带路径与后缀
            instance_id = os.path.splitext(image_file)[0]

            relative_image_path = os.path.join(image_category_path, image_file)  # 仅带类别和文件名的路径，例如 cat/aaa.jpg
            imgid_name[instance_id] = relative_image_path
            skhid_name[instance_id] = []

        # 收集每个实例对应的所有草图
        for sketch_file in sketch_files:
            # 去掉扩展名获取基础名称
            sketch_base_name = os.path.splitext(sketch_file)[0]

            # 尝试匹配实例ID（处理可能的草图变体）
            # 首先尝试直接匹配
            if sketch_base_name in imgid_name:
                skhid_name[sketch_base_name].append(sketch_file)
            elif '-' in sketch_base_name:
                # 如果有'-'分隔符，尝试取前面部分作为实例ID
                instance_id = sketch_base_name.rsplit('-', 1)[0]
                if instance_id in imgid_name:
                    skhid_name[instance_id].append(sketch_file)

        # 为每个草图选择一张图片配对（使用确定性方法）
        category_pairs = []
        for instance_id, sketch_list in skhid_name.items():
            if len(sketch_list) > 0 and instance_id in imgid_name:
                image_name = imgid_name[instance_id]
                images_set.append((image_name, category))

                # 一张图片对应多张草图
                if is_multi_pair:
                    for sketch_name in sketch_list:
                        category_pairs.append((sketch_name, image_name, category))

                # 一张图片对应一张草图
                else:
                    # 使用确定性的选择方式（基于instance_id hash来选择）
                    sketch_idx = hash(category + instance_id + str(random_seed)) % len(sketch_list)
                    sketch_name = sketch_list[sketch_idx]
                    category_pairs.append((sketch_name, image_name, category))

        all_data_pairs.extend(category_pairs)
        category_stats[category] = len(category_pairs)
        print(f"    成功配对: {len(category_pairs)} 对")

    print(f"总共创建了 {len(all_data_pairs)} 个数据对")

    # 按类别进行分层采样划分训练集和测试集
    category_pairs_dict = {}  # {category: (sketch_name, image_name, category)}
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

    # zero-shot 检索直接将类别划分为训练类别和测试类别
    if split_mode == 'ZS-SBIR':
        for category, pairs in category_pairs_dict.items():

            if full_train:
                train_pairs.extend(pairs)
                train_stats[category] = len(pairs)

                if category in scfg.sketchy_test_classes:
                    test_pairs.extend(pairs)
                    test_stats[category] = len(pairs)

                else:
                    test_stats[category] = 0

            else:

                if category in scfg.sketchy_test_classes:
                    test_pairs.extend(pairs)
                    test_stats[category] = len(pairs)

                else:
                    train_pairs.extend(pairs)
                    train_stats[category] = len(pairs)

    else:
        for category, pairs in category_pairs_dict.items():
            # 使用固定种子打乱该类别的样本
            random.Random(random_seed + hash(category)).shuffle(pairs)

            if full_train:
                category_train = pairs
                category_test = pairs if category in scfg.sketchy_test_classes else []

                # category_test = pairs[:10]

            else:
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
        'images_set': images_set,
        'category_stats': category_stats,
        'train_stats': train_stats,
        'test_stats': test_stats,
        'total_categories': len(common_categories),
        'train_split': train_split,
        'random_seed': random_seed,
        'common_categories': common_categories,
        'data_type': 'png_sketch'  # 标识这是PNG草图数据集
    }

    # 保存为pickle文件
    with open(save_root, 'wb') as f:
        pickle.dump(dataset_info, f)

    print(f"PNG草图数据集划分已保存到: {save_root}")
    return dataset_info


def get_split_file_name(sketch_format: str, pair_mode: str, task: str):
    """
    统一的数据集分割文件名获取方式

    sketch_format: ('vector', 'image')
    pair_mode: ('multi_pair', 'single_pair')
    task: ('sbir', 'zs_sbir')
    """
    assert sketch_format in ('vector', 'image')
    assert pair_mode in ('multi_pair', 'single_pair')
    assert task in ('sbir', 'zs_sbir')

    split_file = f'./data/fixed_splits/dataset_split_{pair_mode}_{task}_{sketch_format}_sketch.pkl'
    return split_file


if __name__ == '__main__':
    pass

