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
from pathlib import Path
from functools import partial
import numpy as np
import random

from utils import utils


# sketchy 数据集的测试类别，在 zero-shot 任务中
sketchy_evaluate = [
    'bat',
    'cabin',
    'cow',
    'dolphin',
    'door',
    'giraffe',
    'helicopter',
    'mouse',
    'pear',
    'raccoon',
    'rhinoceros',
    'saw',
    'scissors',
    'seagull',
    'skyscraper',
    'songbird',
    'sword',
    'tree',
    'wheelchair',
    'windmill',
    'window'
]


class PNGSketchImageDataset(Dataset):
    """
    PNG草图-图像配对数据集
    """
    
    def __init__(self,
                 root,
                 mode,
                 fixed_split_path,
                 vec_sketch_type,  # [S5, STK_11_32]
                 sketch_image_subdirs,  # [0]: vector_sketch, [1]: image_sketch, [2]: photo
                 sketch_transform=None,
                 image_transform=None,
                 sketch_format='vector',  # ['vector', 'image']
                 vec_seq_length=11*32,
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

            if vec_sketch_type == 'S5':
                self.sketch_loader = partial(
                    utils.s3_file_to_s5,
                    max_length=vec_seq_length,
                    coor_mode='REL',
                    is_shuffle_stroke=False,
                    is_back_mask=False
                )
            elif 'STK' in vec_sketch_type:
                self.sketch_loader = partial(
                    utils.load_stk_sketch,
                    stk_name=vec_sketch_type
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


def create_png_sketch_dataloaders(batch_size,
                                  num_workers,
                                  fixed_split_path,
                                  root,
                                  sketch_format,
                                  vec_sketch_type,
                                  sketch_image_subdirs
                                  ):
    """
    创建训练和测试数据加载器
    
    Args:
        batch_size: 批次大小
        num_workers: 数据加载进程数
        fixed_split_path: 固定数据集划分文件路径
        root:
        sketch_format:
        vec_sketch_type: 矢量草图格式 [S5, STK_11_32]
        sketch_image_subdirs
        
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
        sketch_format=sketch_format,
        vec_sketch_type=vec_sketch_type,
        sketch_image_subdirs=sketch_image_subdirs
    )

    test_dataset = PNGSketchImageDataset(
        mode='test',
        fixed_split_path=fixed_split_path,
        sketch_transform=test_transform,
        image_transform=test_transform,
        root=root,
        sketch_format=sketch_format,
        vec_sketch_type=vec_sketch_type,
        sketch_image_subdirs=sketch_image_subdirs
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


# def create_dataset_split_file(
#         save_root,
#         sketch_root,
#         image_root,
#         sketch_image_suffix,  # ('txt', 'jpg') or ('png', 'jpg')
#         train_split=0.8,
#         random_seed=42,
#         split_mode='FG-SBIR'  # ['SBIR', 'ZS-SBIR'],
#         # 'SBIR': 使用所有类别，每个类别内取出一定数量用作测试。
#         # 'ZS-SBIR': 一部分类别全部用于训练，另一部分类别全部用于测试，即训练类别和测试类别不重合
# ):
#     """
#     创建PNG草图的固定数据集划分并保存到文件
#
#     Args:
#         save_root: 划分文件存储路径及其文件名
#         sketch_root: PNG草图数据根目录
#         image_root: 图片数据根目录
#         sketch_image_suffix: 草图和图片的文件后缀
#         train_split: 训练集比例
#         random_seed: 随机种子
#         split_mode: 检索任务类别
#     """
#
#     print("开始创建PNG草图的固定数据集划分...")
#     print(f"草图路径: {sketch_root}")
#     print(f"图片路径: {image_root}")
#     print(f"训练集比例: {train_split}")
#     print(f"随机种子: {random_seed}")
#
#     # 设置随机种子
#     random.seed(random_seed)
#     np.random.seed(random_seed)
#
#     # 获取所有类别
#     sketch_categories = get_subdirs(sketch_root)
#     image_categories = get_subdirs(image_root)
#
#     # 找到草图和图片都有的类别
#     common_categories = list(set(sketch_categories) & set(image_categories))
#     common_categories.sort()
#
#     print(f'找到 {len(common_categories)} 个共同类别')
#
#     all_data_pairs = []
#     category_stats = {}
#
#     for category in common_categories:
#         sketch_category_path = os.path.join(sketch_root, category)
#         image_category_path = os.path.join(image_root, category)
#
#         # 获取该类别下的所有草图文件和图片文件
#         sketch_files = get_allfiles(sketch_category_path, sketch_image_suffix[0], filename_only=True)
#         image_files = get_allfiles(image_category_path, sketch_image_suffix[1], filename_only=True)
#
#         print(f"{category}: 找到 {len(sketch_files)} 个{sketch_image_suffix[0].upper()}草图, {len(image_files)} 个{sketch_image_suffix[1].upper()}图片")
#
#         # 构建图片实例字典和对应的草图列表
#         # id 即不带路径也后缀的图片文件名，每个id和一个具体图片文件一一对应
#         # 每个 id 可能对应多个草图，因为一张图片可能画了多张草图
#         # 在 sketchy 数据集中，图片文件名和草图文件名的对应关系为 photo: aaa.jpg, sketch: [aaa_1.jpg, aaa_2.jpg, ...]
#         #    即用下划线加序号表示同一张图片绘制的多个草图
#         skhid_relpth = {}  # 草图 id 和文件相对路径的对应字典, {实例id: [草图文件列表]}
#         imgid_relpth = {}  # 图片 id 和文件相对路径的对应字典, {实例id: 图片路径},
#
#         # 构建图片实例字典
#         for image_file in image_files:
#             # 获取图片文件名，不带路径与后缀
#             instance_id = os.path.splitext(image_file)[0]
#
#             relative_image_path = os.path.join(image_category_path, image_file)  # 仅带类别和文件名的路径，例如 cat/aaa.jpg
#             imgid_relpth[instance_id] = relative_image_path
#             skhid_relpth[instance_id] = []
#
#         # 收集每个实例对应的所有草图
#         for sketch_file in sketch_files:
#             # 去掉扩展名获取基础名称
#             base_name = os.path.splitext(sketch_file)[0]
#
#             # 尝试匹配实例ID（处理可能的草图变体）
#             # 首先尝试直接匹配
#             if base_name in skhid_relpth:
#                 skhid_relpth[base_name].append(sketch_file)
#             elif '-' in base_name:
#                 # 如果有'-'分隔符，尝试取前面部分作为实例ID
#                 instance_id = base_name.rsplit('-', 1)[0]
#                 if instance_id in skhid_relpth:
#                     skhid_relpth[instance_id].append(sketch_file)
#
#         # 为每个实例选择一张草图与图片配对（使用确定性方法）
#         category_pairs = []
#         for instance_id, sketch_list in skhid_relpth.items():
#             if len(sketch_list) > 0 and instance_id in imgid_relpth:
#                 # 使用确定性的选择方式（基于instance_id hash来选择）
#                 sketch_idx = hash(category + instance_id + str(random_seed)) % len(sketch_list)
#                 selected_sketch = sketch_list[sketch_idx]
#                 sketch_path = os.path.join(sketch_category_path, selected_sketch)
#                 image_path = imgid_relpth[instance_id]
#
#                 # 获取不带路径的文件名
#                 sketch_path = os.path.basename(sketch_path)
#                 image_path = os.path.basename(image_path)
#
#                 category_pairs.append((sketch_path, image_path, category))
#
#         all_data_pairs.extend(category_pairs)
#         category_stats[category] = len(category_pairs)
#         print(f"    成功配对: {len(category_pairs)} 对")
#
#     print(f"总共创建了 {len(all_data_pairs)} 个数据对")
#
#     # 按类别进行分层采样划分训练集和测试集
#     category_pairs_dict = {}
#     for pair in all_data_pairs:
#         category = pair[2]
#         if category not in category_pairs_dict:
#             category_pairs_dict[category] = []
#         category_pairs_dict[category].append(pair)
#
#     # 对每个类别进行训练/测试划分
#     train_pairs = []
#     test_pairs = []
#
#     train_stats = {}
#     test_stats = {}
#
#     for category, pairs in category_pairs_dict.items():
#         # 使用固定种子打乱该类别的样本
#         random.Random(random_seed + hash(category)).shuffle(pairs)
#
#         split_idx = int(len(pairs) * train_split)
#
#         # 确保每个类别在训练集和测试集中都有至少1个样本
#         if split_idx == 0:
#             split_idx = 1
#         if split_idx == len(pairs):
#             split_idx = len(pairs) - 1
#
#         category_train = pairs[:split_idx]
#         category_test = pairs[split_idx:]
#
#         train_pairs.extend(category_train)
#         test_pairs.extend(category_test)
#
#         train_stats[category] = len(category_train)
#         test_stats[category] = len(category_test)
#
#     # 最终打乱
#     random.Random(random_seed).shuffle(train_pairs)
#     random.Random(random_seed + 1).shuffle(test_pairs)
#
#     print(f"训练集: {len(train_pairs)} 对")
#     print(f"测试集: {len(test_pairs)} 对")
#
#     # 保存数据集划分
#     dataset_info = {
#         'train_pairs': train_pairs,
#         'test_pairs': test_pairs,
#         'category_stats': category_stats,
#         'train_stats': train_stats,
#         'test_stats': test_stats,
#         'total_categories': len(common_categories),
#         'train_split': train_split,
#         'random_seed': random_seed,
#         'common_categories': common_categories,
#         'data_type': 'png_sketch'  # 标识这是PNG草图数据集
#     }
#
#     # 保存为pickle文件
#     with open(save_root, 'wb') as f:
#         pickle.dump(dataset_info, f)
#
#     print(f"PNG草图数据集划分已保存到: {save_root}")
#     return dataset_info


def create_dataset_split_file(
        save_root,
        sketch_root,
        image_root,
        sketch_image_suffix,  # ('txt', 'jpg') or ('png', 'jpg')
        train_split=0.8,
        random_seed=42,
        is_multi_pair=False,
        split_mode='ZS-SBIR'  # ['SBIR', 'ZS-SBIR'],
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

    # zero-shot 检索直接将类别划分为训练类别和测试类别
    if split_mode == 'ZS-SBIR':
        for category, pairs in category_pairs_dict.items():
            if category in sketchy_evaluate:
                test_pairs.extend(pairs)
                train_stats[category] = len(pairs)
            else:
                test_pairs.extend(pairs)
                test_stats[category] = len(pairs)

    else:
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
        'common_categories': common_categories,
        'data_type': 'png_sketch'  # 标识这是PNG草图数据集
    }

    # 保存为pickle文件
    with open(save_root, 'wb') as f:
        pickle.dump(dataset_info, f)

    print(f"PNG草图数据集划分已保存到: {save_root}")
    return dataset_info


if __name__ == '__main__':
    pass

