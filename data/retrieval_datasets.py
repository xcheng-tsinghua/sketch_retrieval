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
                 mode,
                 pre_load,
                 vec_sketch_rep,  # [S5, STK_11_32]
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
            pre_load: 固定数据集划分
            sketch_transform: 草图变换
            image_transform: 图像变换
            is_back_image_only: 是否仅返回图像，用于一张图片对应多个草图时，进行测试时不返回重复的图片

        """
        assert mode in ('train', 'test')

        print(f"SketchImageDataset initialized with:")
        print(f"Mode: {mode}")
        # print(f"  Fixed split path: {fixed_split_path}")
        
        self.mode = mode
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

        if self.sketch_format == 'vector':
            if vec_sketch_rep == 'S5':
                self.sketch_loader = partial(
                    utils.s3_file_to_s5,
                    max_length=vec_seq_length,
                    coor_mode='REL',
                    is_shuffle_stroke=False,
                    is_back_mask=False
                )
            elif vec_sketch_rep == 'IMG':
                self.sketch_loader = partial(
                    utils.s3_to_tensor_img,
                    image_size=(224, 224),
                    line_thickness=2,
                    pen_up=1,
                    coor_mode='ABS',
                    save_path=None
                )
            elif 'stk' in vec_sketch_rep and 'stkpnt' in vec_sketch_rep:
                self.sketch_loader = partial(
                    utils.load_stk_sketch,
                    stk_name=vec_sketch_rep
                )
            else:
                raise TypeError('error vector sketch type')

        else:
            self.sketch_loader = partial(
                utils.image_loader,
                image_transform=self.sketch_transform
            )

        self._load_fixed_split(pre_load)
        
    def _load_fixed_split(self, pre_load):
        """
        加载固定的数据集划分
        """
        # 根据模式选择数据
        if self.mode == 'train':
            self.data_pairs = pre_load.train_pairs

        elif self.mode == 'test':
            self.data_pairs = pre_load.test_pairs

        else:
            raise ValueError(f"不支持的模式: {self.mode}")
        
        self.categories = pre_load.common_categories
        self.images_set = pre_load.images_set
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
            (sketch_tensor, image_tensor, category_idx, category_name)
            sketch: 预处理后的草图张量 [3, 224, 224]
            image: 预处理后的图像张量 [3, 224, 224]  
            category_idx: 类别索引
            category_name: 类别名称
        """

        if self.is_back_image_only:
            image_path, category = self.images_set[idx]

            # 加载JPG图像
            image = utils.image_loader(image_path, self.image_transform)

            # 获取类别索引
            category_idx = self.category_to_idx[category]

            return idx, image, category_idx, category

        else:
            sketch_path, image_path, category = self.data_pairs[idx]
            # try:
            # 加载草图
            sketch = self.sketch_loader(sketch_path)

            # 加载JPG图像
            image = utils.image_loader(image_path, self.image_transform)

            # 获取类别索引
            category_idx = self.category_to_idx[category]

            return sketch, image, category_idx, category
            # return idx, sketch, image

            # except Exception as e:
            #     print(f"Error loading data at index {idx}: {e}")
            #     print(f"Sketch path: {sketch_path}")
            #     print(f"Image path: {image_path}")
            #     raise e
    
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
    读取文件并分割为训练集测试集，如果文件夹内的训练集测试集已划分好，则检索类别由文件夹内划分的训练集测试集决定
    要求：
    1. sketch_root 和 image_root 是两个不同的文件夹
    2. 草图和图片文件夹下的类别文件夹要一致
    3. 实例级配对的图片和草图文件名一致（扩展名可不一致），若某张图片对应多个草图，草图末尾加 _1 等区分，即下划线加序号

    定位文件夹层级示例如下：
    -> 不区分训练集测试集，自动划分的情况
    sketch_root
    ├─ Class1
    │   ├─0.png
    │   ├─1.png
    │   ...
    │
    ├─ Class2
    │   ├─0.png
    │   ├─1.png
    │   ...
    │
    ...

    image_root
    ├─ Class1
    │   ├─0.jpg
    │   ├─1.jpg
    │   ...
    │
    ├─ Class2
    │   ├─0.jpg
    │   ├─1.jpg
    │   ...
    │
    ...

    -> 训练集测试集已区分好的情况
    sketch_root
    ├─ train
    │   ├─ Class1
    │   │   ├─0.png
    │   │   ├─1.png
    │   │   │   ...
    │   │
    │   ├─ Class2
    │   │   ├─0.png
    │   │   ├─1.png
    │   │   ...
    │   │
    │   ...
    │
    ├─ test
    │   ├─ Class1
    │   │   ├─0.png
    │   │   ├─1.png
    │   │   ...
    │   │
    │   ├─ Class2
    │   │   ├─0.png
    │   │   ├─1.png
    │   │   ...
    │   │
    │   ...

    image_root
    ├─ train
    │   ├─ Class1
    │   │   ├─0.jpg
    │   │   ├─1.jpg
    │   │   │   ...
    │   │
    │   ├─ Class2
    │   │   ├─0.jpg
    │   │   ├─1.jpg
    │   │   ...
    │   │
    │   ...
    │
    ├─ test
    │   ├─ Class1
    │   │   ├─0.jpg
    │   │   ├─1.jpg
    │   │   ...
    │   │
    │   ├─ Class2
    │   │   ├─0.jpg
    │   │   ├─1.jpg
    │   │   ...
    │   │
    │   ...

    """
    def __init__(self,
                 sketch_root,
                 image_root,
                 sketch_suffix,
                 image_suffix,
                 train_split=0.8,
                 random_seed=42,
                 is_multi_pair=False,
                 split_mode='zs-sbir',  # ['sbir', 'zs-sbir']
                 is_full_train=False,
                 multi_sketch_split='_'  # 一张照片对应多个草图，草图命名应为 "图片名(不带后缀)+multi_sketch_split+草图后缀"
                 ):

        self.train_pairs = []  # (sketch_root, image_root, class_name)
        self.test_pairs = []
        self.images_set = []
        self.category_stats = {}
        self.train_stats = {}
        self.test_stats = {}
        self.train_split = train_split
        self.random_seed = random_seed
        self.common_categories = []

        self.load_data(sketch_root,
                       image_root,
                       sketch_suffix,
                       image_suffix,
                       train_split,
                       random_seed,
                       is_multi_pair,
                       split_mode,  # ['sbir', 'zs-sbir']
                       is_full_train,
                       multi_sketch_split
                       )

    def load_data(self,
                  sketch_root,
                  image_root,
                  sketch_suffix,
                  image_suffix,
                  train_split,
                  random_seed,
                  is_multi_pair,
                  split_mode,
                  full_train,
                  multi_sketch_split
                  ):
        assert split_mode in ['sbir', 'zs-sbir'], TypeError(f'error solit mode: {split_mode}')

        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)

        # 获取所有类别
        sketch_categories = get_subdirs(sketch_root)
        image_categories = get_subdirs(image_root)

        # 判断是否存在 'train', 'test' 文件夹
        is_train_test_divided = self.check_is_divided(sketch_categories, image_categories)

        # 已划分好训练集测试集后需要重新获取类别
        if is_train_test_divided:
            print('Training set and testing set are already divided.')

            # 重新获取类别
            sketch_categories_train = get_subdirs(os.path.join(sketch_root, 'train'))
            sketch_categories_test = get_subdirs(os.path.join(sketch_root, 'test'))

            image_categories_train = get_subdirs(os.path.join(image_root, 'train'))
            image_categories_test = get_subdirs(os.path.join(image_root, 'test'))

            common_categories_train = list(set(sketch_categories_train) & set(image_categories_train))
            common_categories_test = list(set(sketch_categories_test) & set(image_categories_test))

            self.common_categories = list(set(common_categories_train) & set(common_categories_test))

            # 获取训练集下的配对列表
            for category in common_categories_train:
                category_pairs = self.get_category_pairs(
                    os.path.join(sketch_root, 'train', category),
                    os.path.join(image_root, 'train', category),
                    sketch_suffix,
                    image_suffix,
                    category,
                    multi_sketch_split,
                    is_multi_pair,
                    random_seed
                )
                self.train_pairs.extend(category_pairs)

            # 获取测试集下的配对列表
            for category in common_categories_test:
                category_pairs = self.get_category_pairs(
                    os.path.join(sketch_root, 'test', category),
                    os.path.join(image_root, 'test', category),
                    sketch_suffix,
                    image_suffix,
                    category,
                    multi_sketch_split,
                    is_multi_pair,
                    random_seed
                )
                self.test_pairs.extend(category_pairs)

        else:
            print('Auto divided the training set and testing set.')

            # 找到草图和图片都有的类别
            common_categories = list(set(sketch_categories) & set(image_categories))
            common_categories.sort()
            self.common_categories = common_categories

            # 找到全部草图和图片的配对
            all_category_pairs = {}  # {'class_name': [skh_path, img_path, 'class_name'], ...}

            for category in common_categories:
                category_pairs = self.get_category_pairs(
                    os.path.join(sketch_root, category),
                    os.path.join(image_root, category),
                    sketch_suffix,
                    image_suffix,
                    category,
                    multi_sketch_split,
                    is_multi_pair,
                    random_seed
                )

                all_category_pairs[category] = category_pairs

            # 划分训练集和测试集
            for class_name, class_pair_list in all_category_pairs.items():
                if split_mode == 'zs-sbir':  # zero-shot 检索直接将类别划分为训练类别和测试类别

                    if class_name in scfg.sketchy_test_classes:
                        self.test_pairs.extend(class_pair_list)
                        if full_train:
                            self.train_pairs.extend(class_pair_list)
                    else:
                        self.train_pairs.extend(class_pair_list)

                else:
                    # 使用固定种子打乱该类别的样本
                    random.Random(random_seed + hash(class_name)).shuffle(class_pair_list)

                    split_idx = int(len(class_pair_list) * train_split)
                    # 确保每个类别在训练集和测试集中都有至少1个样本
                    if split_idx == 0:
                        split_idx = 1
                    if split_idx == len(class_pair_list):
                        split_idx = len(class_pair_list) - 1

                    category_train = class_pair_list if full_train else class_pair_list[:split_idx]
                    category_test = class_pair_list[split_idx:]

                    self.train_pairs.extend(category_train)
                    self.test_pairs.extend(category_test)

        # 最终打乱
        random.Random(random_seed).shuffle(self.train_pairs)
        random.Random(random_seed + 1).shuffle(self.test_pairs)

        print(f'-> 预加载数据信息')
        print(f"训练集: {len(self.train_pairs)} 对")
        print(f"测试集: {len(self.test_pairs)} 对")

    @staticmethod
    def check_is_divided(sketch_categories: list, image_categories: list):
        is_train_in = 'train' in sketch_categories and 'train' in image_categories
        is_test_in = 'test' in sketch_categories and 'test' in image_categories
        is_len_2 = len(sketch_categories) == 2 and len(image_categories) == 2

        return is_train_in and is_test_in and is_len_2

    @staticmethod
    def get_category_pairs(skh_root, img_root, skh_suffix, img_suffix, class_name, multi_sketch_split, is_multi_pair,
                           random_seed):
        """
        将对应文件夹下的草图和图片文件进行配对
        配对依据为文件名 skh_file_name = img_file_name + multi_sketch_split + num
        """
        # 获取该类别下的所有草图文件和图片文件
        sketch_files = get_allfiles(skh_root, skh_suffix, filename_only=True)
        image_files = get_allfiles(img_root, img_suffix, filename_only=True)

        # 构建图片实例字典和对应的草图列表
        # id 即不带路径也后缀的图片文件名，每个id和一个具体图片文件一一对应
        # 每个 id 可能对应多个草图，因为一张图片可能画了多张草图
        # 在 sketchy 数据集中，图片文件名和草图文件名的对应关系为 photo: aaa.jpg, sketch: [aaa_1.jpg, aaa_2.jpg, ...]
        #    即用下划线加序号表示同一张图片绘制的多个草图
        skhid_name = {}  # 草图 id 和文件名的对应字典, {实例id: [草图文件路径列表]}
        imgid_name = {}  # 图片 id 和文件名的对应字典, {实例id: 图片路径},

        # 构建图片实例字典
        for image_file in image_files:
            # 获取图片文件名，不带路径与后缀
            instance_id = os.path.splitext(image_file)[0]

            skhid_name[instance_id] = []
            imgid_name[instance_id] = os.path.join(img_root, image_file)

        # 收集每个实例对应的所有草图
        for sketch_file in sketch_files:
            # 去掉扩展名获取基础名称
            sketch_base_name = os.path.splitext(sketch_file)[0]

            # 尝试匹配实例ID（处理可能的草图变体）
            # 首先尝试直接匹配
            if sketch_base_name in imgid_name.keys():
                skhid_name[sketch_base_name].append(sketch_file)

            elif multi_sketch_split in sketch_base_name:
                # 如果有'-'分隔符，尝试取前面部分作为实例ID
                instance_id = sketch_base_name.rsplit(multi_sketch_split, 1)[0]
                if instance_id in imgid_name.keys():
                    skhid_name[instance_id].append(os.path.join(skh_root, sketch_file))

        # 为每个草图选择一张图片配对
        category_pairs = []
        for instance_id, skh_path_list in skhid_name.items():
            if len(skh_path_list) > 0 and instance_id in imgid_name.keys():
                img_path = imgid_name[instance_id]

                if is_multi_pair:  # 一张图片对应多张草图
                    for skh_path in skh_path_list:
                        category_pairs.append((skh_path, img_path, class_name))

                else:  # 一张图片对应一张草图
                    # 使用确定性的选择方式（基于instance_id hash来选择）
                    sketch_idx = hash(class_name + instance_id + str(random_seed)) % len(skh_path_list)
                    skh_path = skh_path_list[sketch_idx]
                    category_pairs.append((skh_path, img_path, class_name))

        return category_pairs


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
                                    pre_load,
                                    sketch_format,
                                    vec_sketch_rep,
                                    is_back_dataset=False
                                    ):
    """
    创建训练和测试数据加载器
    
    Args:
        batch_size: 批次大小
        num_workers: 数据加载进程数
        pre_load: 固定数据集划分信息
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
        pre_load=pre_load,
        sketch_transform=train_sketch_transform,
        image_transform=train_image_transform,
        sketch_format=sketch_format,
        vec_sketch_rep=vec_sketch_rep,
    )

    test_dataset = SketchImageDataset(
        mode='test',
        pre_load=pre_load,
        sketch_transform=test_transform,
        image_transform=test_transform,
        sketch_format=sketch_format,
        vec_sketch_rep=vec_sketch_rep,
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

    if is_back_dataset:
        return train_dataset, test_dataset, train_loader, test_loader
    else:
        return train_loader, test_loader


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


# def get_split_file_name(sketch_format: str, pair_mode: str, task: str):
#     """
#     统一的数据集分割文件名获取方式
#
#     sketch_format: ('vector', 'image')
#     pair_mode: ('multi_pair', 'single_pair')
#     task: ('sbir', 'zs_sbir')
#     """
#     assert sketch_format in ('vector', 'image')
#     assert pair_mode in ('multi_pair', 'single_pair')
#     assert task in ('sbir', 'zs_sbir')
#
#     split_file = f'./data/fixed_splits/dataset_split_{pair_mode}_{task}_{sketch_format}_sketch.pkl'
#     return split_file


if __name__ == '__main__':
    pass

