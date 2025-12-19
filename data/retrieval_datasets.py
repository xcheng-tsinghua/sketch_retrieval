"""
PNG草图-图像数据集加载器
用于加载PNG格式的草图和对应的图片进行训练
"""
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pathlib import Path
from functools import partial
import numpy as np
import random
from torchvision.transforms.functional import to_pil_image

from utils import utils
from data import sketchy_configs as scfg
import options


class SketchImageDataset(Dataset):
    """
    PNG草图-图像配对数据集
    TODO: 未完善 category-level 数据加载
    """
    def __init__(self,
                 is_train,
                 pre_load,
                 sketch_transform,
                 image_transform,
                 sketch_format,
                 is_full_train,
                 n_neg=8,  # 每次返回的负样本数
                 ):
        """
        初始化数据集
        
        Args:
            train_data: 'train' 或 'test'
            pre_load: 固定数据集划分
            sketch_transform: 草图变换
            image_transform: 图像变换

        """
        self.is_train = is_train
        self.n_neg = n_neg
        self.back_mode = 'train_data'  # ['train_data', 'sketch', 'image']
        self.neg_instance = None  # 负样本

        # 默认变换
        self.sketch_transform = sketch_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.image_transform = image_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 创建草图加载器
        sketch_format = options.parse_sketch_format(sketch_format)
        if sketch_format['fmt'] == 's5':
            self.sketch_loader = partial(
                utils.s3_file_to_s5,
                max_length=sketch_format['max_length'],
                coor_mode='REL',
                is_shuffle_stroke=False,
                is_back_mask=False
            )
        elif sketch_format['fmt'] == 'stk':
            self.sketch_loader = partial(
                utils.load_stk_sketch,
                n_stk=sketch_format['n_stk'],
                n_stk_pnt=sketch_format['n_stk_pnt'],
            )
        elif sketch_format['fmt'] == 's3 -> img':
            self.sketch_loader = partial(
                utils.s3_to_tensor_img,
                image_size=(224, 224),
                line_thickness=1,
                pen_up=1,
                coor_mode='ABS',
                save_path=None
            )
        elif sketch_format['fmt'] == 'img':
            self.sketch_loader = partial(
                utils.image_loader,
                image_transform=self.sketch_transform
            )
        else:
            raise TypeError('unsupported sketch format')

        # 创建图片加载器
        self.image_loader = partial(utils.image_loader,
                                    image_transform=self.image_transform
                                    )

        self._load_fixed_split(pre_load, is_full_train)
        print(f'-> SketchImageDataset initialized with: {"Training" if self.is_train else "Testing"}.')
        print(f'   sketch: {len(self.sketch_list_with_id)}, image: {len(self.image_list)}, categories: {len(self.categories)}')
        
    def _load_fixed_split(self, pre_load, is_full_train):
        """
        加载固定的数据集划分
        """
        # 根据模式选择数据
        if self.is_train:
            self.data_pairs = pre_load.train_pairs
            if is_full_train:
                self.data_pairs.extend(pre_load.test_pairs)

        else:
            self.data_pairs = pre_load.test_pairs

        # 获取草图列表、图片列表、id 映射
        self.image_list = []
        for skh_path, img_path, class_name in self.data_pairs:
            if img_path not in self.image_list:
                self.image_list.append(img_path)

        # 防止列表被修改
        self.image_list = tuple(self.image_list)

        # 将 sketch_list 中的每个样本加上 id，以表明其匹配的图片
        self.sketch_list_with_id = []
        self.sketch_paired_id = []
        for skh_path, img_path, _ in self.data_pairs:
            c_paired_img_idx = self.image_list.index(img_path)

            self.sketch_paired_id.append(c_paired_img_idx)
            self.sketch_list_with_id.append((skh_path, c_paired_img_idx))

        self.sketch_paired_id = tuple(self.sketch_paired_id)
        self.sketch_list_with_id = tuple(self.sketch_list_with_id)

        self.categories = tuple(pre_load.common_categories)
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        
    def __len__(self):
        # return len(self.data_pairs)
        if self.back_mode == 'train_data':
            return len(self.sketch_list_with_id)

        elif self.back_mode == 'image':
            return len(self.image_list)

        elif self.back_mode == 'sketch':
            return len(self.sketch_list_with_id)

        else:
            raise ValueError('unsupported back mode')
    
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

        if self.back_mode == 'train_data':
            # 选出草图样本
            skh_path, ins_id = self.sketch_list_with_id[idx]
            skh_tensor = self.sketch_loader(skh_path)

            # 选出草图正样本
            pos_img = self.image_list[ins_id]
            pos_img_tensor = self.image_loader(pos_img)

            # 选出指定个数负样本：
            neg_img_list = self.sample_exclude(self.image_list, self.n_neg, (ins_id,))

            # 如果有精选的负样本，使用一半精选负样本
            if self.neg_instance is not None:
                neg_img_sel = self.neg_instance[idx]
                neg_img_sel = [self.image_list[i] for i in neg_img_sel]

                half = self.n_neg // 2
                neg_img_rand = random.sample(neg_img_list, half)
                neg_img_sel = random.sample(neg_img_sel, self.n_neg - half)

                neg_img_list = neg_img_sel + neg_img_rand

            # if self.neg_instance is None:  # 随机选取
            #     neg_img_list = self.sample_exclude(self.image_list, self.n_neg, (ins_id, ))
            # else:  # 根据指定的负样本选取
            #     neg_img_list = self.neg_instance[idx]
            #     neg_img_list = [self.image_list[i] for i in neg_img_list]

            neg_img_tensor_list = []
            for c_neg in neg_img_list:
                c_neg_tensor = self.image_loader(c_neg).unsqueeze(0)
                neg_img_tensor_list.append(c_neg_tensor)
            neg_img_tensor_all = torch.cat(neg_img_tensor_list, dim=0)

            return skh_tensor, pos_img_tensor, neg_img_tensor_all

        elif self.back_mode == 'image':
            img_path = self.image_list[idx]
            img_tensor = self.image_loader(img_path)
            return img_tensor

        elif self.back_mode == 'sketch':
            skh_path, _ = self.sketch_list_with_id[idx]
            skh_tensor = self.sketch_loader(skh_path)
            return skh_tensor

        else:
            raise ValueError('unsupported back mode')

    @staticmethod
    def sample_exclude(lst, k, forbidden_idx) -> list:
        # 把允许抽的索引先列出来
        allowed_idx = [i for i in range(len(lst)) if i not in forbidden_idx]

        if len(allowed_idx) < k:
            raise ValueError('可选元素不足')

        # 从 allowed_idx 里随机选 k 个索引
        picked_idx = random.sample(allowed_idx, k)

        # 返回对应的元素
        return [lst[i] for i in picked_idx]

    def back_image(self):
        self.back_mode = 'image'

    def back_sketch(self):
        self.back_mode = 'sketch'

    def back_train_data(self):
        self.back_mode = 'train_data'


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
                 multi_sketch_split='_'  # 一张照片对应多个草图，草图命名应为 "图片名(不带后缀)+multi_sketch_split+草图后缀"
                 ):

        self.train_pairs = []  # (sketch_root, image_root, class_name)
        self.test_pairs = []
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
                       multi_sketch_split
                       )

        print(f'-> preload data info: ')
        print(f'   preload sketch from: {sketch_root}')
        print(f'   preload image from: {image_root}')
        print(f'   training set: {len(self.train_pairs)} pairs')
        print(f'   testing set: {len(self.test_pairs)} pairs')

    def load_data(self,
                  sketch_root,
                  image_root,
                  sketch_suffix,
                  image_suffix,
                  train_split,
                  random_seed,
                  is_multi_pair,
                  split_mode,
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

                    category_train = class_pair_list[:split_idx]
                    category_test = class_pair_list[split_idx:]

                    self.train_pairs.extend(category_train)
                    self.test_pairs.extend(category_test)

        # 最终打乱
        random.Random(random_seed).shuffle(self.train_pairs)
        random.Random(random_seed + 1).shuffle(self.test_pairs)

    @staticmethod
    def check_is_divided(sketch_categories: list, image_categories: list):
        is_train_in = 'train' in sketch_categories and 'train' in image_categories
        is_test_in = 'test' in sketch_categories and 'test' in image_categories
        is_len_2 = len(sketch_categories) == 2 and len(image_categories) == 2

        return is_train_in and is_test_in and is_len_2

    @staticmethod
    def get_category_pairs(skh_root, img_root, skh_suffix, img_suffix, class_name, multi_sketch_split, is_multi_pair, random_seed):
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
                                    is_full_train,
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
        back_mode:
        is_full_train:
        
    Returns:
        train_loader, test_loader, dataset_info
    """

    # 数据增强变换
    train_sketch_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomRotation(degrees=10),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomRotation(degrees=10),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = SketchImageDataset(
        is_train=True,
        pre_load=pre_load,
        sketch_transform=train_sketch_transform,
        image_transform=train_image_transform,
        sketch_format=sketch_format,
        is_full_train=is_full_train,
    )

    test_dataset = SketchImageDataset(
        is_train=False,
        pre_load=pre_load,
        sketch_transform=test_transform,
        image_transform=test_transform,
        sketch_format=sketch_format,
        is_full_train=False,
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, test_loader


if __name__ == '__main__':
    pass

