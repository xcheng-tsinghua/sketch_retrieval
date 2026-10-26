"""
PNG草图-图像数据集加载器
用于加载PNG格式的草图和对应的图片进行训练
"""
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial
import numpy as np
import random
import hashlib
from pathlib import Path
from torchvision.transforms.functional import to_pil_image

from utils import utils
from data import sketchy_configs as scfg
import options


class SketchImageDataset(Dataset):
    """
    PNG草图-图像配对数据集
    """
    def __init__(self,
                 is_train: bool,
                 pre_load,
                 sketch_transform,
                 image_transform,
                 sketch_format,
                 is_full_train,
                 task,
                 n_pos: int = 1,  # 每次返回的正样本数（细粒度检索仅返回一个正样本，无视该值）
                 n_neg: int = 8,  # 每次返回的负样本数
                 ):
        """
        初始化数据集
        
        Args:
            is_train:
            pre_load: 固定数据集划分
            sketch_transform: 草图变换
            image_transform: 图像变换

        """
        self.is_train = is_train
        self.n_pos = n_pos
        self.n_neg = n_neg
        if n_pos < 1 or n_neg < 1:
            raise ValueError('n_pos and n_neg must be positive')
        self.back_mode = 'train_data'  # ['train_data', 'sketch', 'image']
        self.neg_instance = None  # 负样本
        self.task = task
        assert self.task in ('fg', 'cl')

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
        if sketch_format['fmt'] == 's3 -> s5':
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
            self.data_pairs = list(pre_load.train_pairs)
            if is_full_train:
                self.data_pairs.extend(pre_load.test_pairs)

        else:
            self.data_pairs = pre_load.test_pairs

        # 建立类别名映射
        self.categories = tuple(pre_load.common_categories)
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}

        # 获取草图列表、图片列表、id 映射
        gallery = list(pre_load.train_images if self.is_train else pre_load.test_images)
        if self.is_train and is_full_train:
            gallery.extend(pre_load.test_images)
        gallery = sorted(set(gallery))
        self.image_list = [path for path, _ in gallery]
        self.image_cat = [self.category_to_idx[cat] for _, cat in gallery]
        image_to_idx = {path: i for i, path in enumerate(self.image_list)}

        # id: 某张草图实例级匹配的图片，在 image_list 中的索引
        self.sketch_list_with_id = []  # 将 sketch_list 中的每个样本路径加上 id，以表明其匹配的图片
        self.sketch_cat = []
        for skh_path, img_path, class_name in self.data_pairs:
            c_paired_img_idx = image_to_idx.get(img_path, -1)
            if self.task == 'fg' and c_paired_img_idx < 0:
                raise ValueError(f'Missing FG target for {skh_path}')

            self.sketch_list_with_id.append((skh_path, c_paired_img_idx))
            self.sketch_cat.append(self.category_to_idx[class_name])

        # 防止列表被修改
        self.image_list = tuple(self.image_list)
        self.image_cat = tuple(self.image_cat)
        self.sketch_list_with_id = tuple(self.sketch_list_with_id)
        self.sketch_cat = tuple(self.sketch_cat)

        self.cat_to_img_list = {}
        if self.task == 'cl':
            for cat_idx, img_path in zip(self.image_cat, self.image_list):
                if cat_idx in self.cat_to_img_list.keys():
                    self.cat_to_img_list[cat_idx].append(img_path)
                else:
                    self.cat_to_img_list[cat_idx] = [img_path]
        if not self.image_list or not self.sketch_list_with_id:
            raise ValueError('Empty retrieval split: check paths, suffixes and held-out classes')
        if self.is_train and self.task == 'cl' and len(set(self.image_cat)) < 2:
            raise ValueError('CL training requires at least two image categories')
        if self.is_train and self.task == 'fg' and len(self.image_list) < 2:
            raise ValueError('FG training requires at least two images')

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
            if self.task == 'cl':
                # 找到草图所属类别索引
                pos_cat = self.sketch_cat[idx]
                pos_img_all = self.cat_to_img_list[pos_cat]

                # 随机选取指定个数正样本
                pos_imgs = self.sample_pool(pos_img_all, self.n_pos)

                pos_img_tensor_list = []
                for c_pos in pos_imgs:
                    c_neg_tensor = self.image_loader(c_pos).unsqueeze(0)
                    pos_img_tensor_list.append(c_neg_tensor)
                pos_img_tensor = torch.cat(pos_img_tensor_list, dim=0)

                # 选出指定个数负样本
                neg_img_all_list = []
                for cat_key in self.cat_to_img_list.keys():
                    if cat_key != pos_cat:
                        neg_img_all_list.extend(self.cat_to_img_list[cat_key])
                neg_img_list = self.sample_pool(neg_img_all_list, self.n_neg)

            elif self.task == 'fg':
                pos_img = self.image_list[ins_id]
                pos_img_tensor = self.image_loader(pos_img)

                # 选出指定个数负样本：
                neg_img_list = self.sample_exclude(self.image_list, self.n_neg, (ins_id,))

                # 如果有精选的负样本 (上次检索失败获得的样本)，使用一半精选负样本
                if self.neg_instance is not None:
                    neg_img_sel = self.neg_instance[idx]
                    neg_img_sel = [self.image_list[i] for i in neg_img_sel]

                    half = self.n_neg // 2
                    neg_img_rand = random.sample(neg_img_list, half)
                    neg_img_sel = neg_img_sel[:self.n_neg - half]  # 选出最相近的精选负样本

                    neg_img_list = neg_img_sel + neg_img_rand

            else:
                raise ValueError

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
    def sample_pool(pool, k):
        if not pool:
            raise ValueError('No eligible samples in the requested pool')
        return random.sample(pool, k) if len(pool) >= k else random.choices(pool, k=k)

    @staticmethod
    def sample_exclude(lst, k, forbidden_idx) -> list:
        # 把允许抽的索引先列出来
        allowed_idx = [i for i in range(len(lst)) if i not in forbidden_idx]

        picked_idx = SketchImageDataset.sample_pool(allowed_idx, k)

        # 返回对应的元素
        return [lst[i] for i in picked_idx]

    def back_image(self):
        self.back_mode = 'image'

    def back_sketch(self):
        self.back_mode = 'sketch'

    def back_train_data(self):
        self.back_mode = 'train_data'


class DatasetPreload:
    """Build query records and independent galleries; retain unpaired gallery images.

    FG targets are filename-matched original photos. CL relevance is the category.
    Predefined train/test folders take precedence, and zero-shot disjointness is checked.
    """
    def __init__(self, sketch_root, image_root, sketch_suffix, image_suffix,
                 train_split=0.8, random_seed=42, is_multi_pair=False,
                 is_zero_shot=True, multi_sketch_split='_',
                 zs_test_classes=scfg.sketchy_test_classes, task='fg',
                 extended_image_root=None, gallery_mode='auto'):
        if task not in ('cl', 'fg'):
            raise ValueError('task must be cl or fg')
        self.task = task
        if gallery_mode not in ('auto', 'all', 'paired'):
            raise ValueError('gallery_mode must be auto, all or paired')
        if gallery_mode == 'auto':
            gallery_mode = 'paired' if task == 'fg' and not is_zero_shot else 'all'
        self.is_zero_shot = is_zero_shot
        self.train_pairs, self.test_pairs = [], []
        self.train_images, self.test_images = [], []
        self.random_seed = random_seed
        self.train_split = train_split
        self.category_stats, self.train_stats, self.test_stats = {}, {}, {}
        roots = [Path(sketch_root), Path(image_root)]
        if extended_image_root:
            roots.append(Path(extended_image_root))
        for root in roots:
            if not root.is_dir():
                raise FileNotFoundError(f'Dataset directory not found: {root}')
        sk_root, im_root = roots[:2]
        ext_root = roots[2] if len(roots) > 2 else None
        divided = all((root / split).is_dir() for root in roots[:2] for split in ('train', 'test'))
        if not divided and any((root / split).is_dir() for root in roots[:2] for split in ('train', 'test')):
            raise ValueError('Both sketch and photo roots must contain train AND test folders')
        held_out = set(zs_test_classes)

        def files(folder, suffix, photos=False):
            suffixes = {'.' + suffix.lstrip('.').lower()}
            if photos:
                suffixes.update({'.jpg', '.jpeg', '.png'})
            return sorted(str(p) for p in folder.iterdir() if p.is_file() and p.suffix.lower() in suffixes)

        def scan(sk_dir, im_dir, ex_dir):
            categories = sorted({p.name for p in sk_dir.iterdir() if p.is_dir()} &
                                {p.name for p in im_dir.iterdir() if p.is_dir()})
            records, gallery = [], []
            for cat in categories:
                photos = files(im_dir / cat, image_suffix, photos=True)
                if not photos:
                    raise ValueError(f'No jpg/jpeg/png or {image_suffix} photos in {im_dir / cat}')
                by_stem = {Path(p).stem: p for p in photos}
                if len(by_stem) != len(photos):
                    raise ValueError(f'Duplicate photo stems in {im_dir / cat}')
                gallery.extend((p, cat) for p in photos)
                if ex_dir is not None and (ex_dir / cat).is_dir():
                    gallery.extend((p, cat) for p in files(ex_dir / cat, image_suffix, photos=True))
                matched = {}
                for sk in files(sk_dir / cat, sketch_suffix):
                    stem = Path(sk).stem
                    target = by_stem.get(stem)
                    if target is None and multi_sketch_split in stem:
                        target = by_stem.get(stem.rsplit(multi_sketch_split, 1)[0])
                    if task == 'fg' and target is None:
                        raise ValueError(f'No paired photo for FG query: {sk}; check suffix/separator')
                    key = target if target is not None else sk
                    matched.setdefault(key, []).append((sk, target, cat))
                for key, rows in sorted(matched.items()):
                    if is_multi_pair or task == 'cl':
                        records.extend(rows)
                    else:
                        digest = hashlib.sha256(f'{cat}/{Path(key).name}/{random_seed}'.encode()).digest()
                        records.append(rows[int.from_bytes(digest[:8], 'big') % len(rows)])
            return records, sorted(set(gallery))

        if divided:
            if ext_root and not all((ext_root / split).is_dir() for split in ('train', 'test')):
                raise ValueError('Extended gallery must also have train/test folders for predefined splits')
            for split in ('train', 'test'):
                records, gallery = scan(sk_root / split, im_root / split,
                                        ext_root / split if ext_root else None)
                setattr(self, split + '_pairs', records)
                setattr(self, split + '_images', gallery)
        else:
            records, gallery = scan(sk_root, im_root, ext_root)
            if is_zero_shot:
                for split, test in (('train', False), ('test', True)):
                    setattr(self, split + '_pairs', [r for r in records if (r[2] in held_out) == test])
                    setattr(self, split + '_images', [r for r in gallery if (r[1] in held_out) == test])
            else:
                if not 0 < train_split < 1:
                    raise ValueError('train_split must lie between zero and one')
                # Group all sketches of one photo together to avoid instance leakage.
                for cat in sorted({r[2] for r in records}):
                    groups = {}
                    for row in records:
                        if row[2] == cat:
                            groups.setdefault(row[1] or row[0], []).append(row)
                    keys = sorted(groups)
                    if len(keys) < 2:
                        raise ValueError(f'Need at least two instances to split category {cat}')
                    digest = hashlib.sha256(cat.encode()).digest()
                    rng = random.Random(random_seed + int.from_bytes(digest[:8], 'big'))
                    rng.shuffle(keys)
                    cut = min(len(keys) - 1, max(1, int(len(keys) * train_split)))
                    train_keys = set(keys[:cut])
                    for key in keys:
                        (self.train_pairs if key in train_keys else self.test_pairs).extend(groups[key])
                    for photo, category in gallery:
                        if category == cat:
                            in_train = photo in train_keys if photo in groups else rng.random() < train_split
                            (self.train_images if in_train else self.test_images).append((photo, category))
        if gallery_mode == 'paired':
            for split in ('train', 'test'):
                targets = {r[1] for r in getattr(self, split + '_pairs')}
                setattr(self, split + '_images', [r for r in getattr(self, split + '_images') if r[0] in targets])
        self.common_categories = sorted({c for _, c in self.train_images + self.test_images})
        train_cats = {r[2] for r in self.train_pairs}
        test_cats = {r[2] for r in self.test_pairs}
        if is_zero_shot and train_cats & test_cats:
            raise ValueError(f'Zero-shot train/test categories overlap: {sorted(train_cats & test_cats)}')
        for split in ('train', 'test'):
            rows = getattr(self, split + '_pairs')
            gallery = getattr(self, split + '_images')
            if not rows or not gallery:
                raise ValueError(f'Empty {split} split: check suffixes and zero-shot class list')
            rows.sort()
            gallery.sort()
            print(f'{split}: {len(rows)} sketches, {len(gallery)} gallery images, '
                  f'{len({r[2] for r in rows})} query categories')


def preload_from_args(args):
    info = options.get_encoder_info(args.sketch_model)
    root = args.root_local if eval(args.local) else args.root_sever
    test_classes = scfg.sketchy_test_classes
    if args.zs_classes_file:
        test_classes = [line.strip() for line in Path(args.zs_classes_file).read_text(encoding='utf-8').splitlines()
                        if line.strip() and not line.lstrip().startswith('#')]
    return DatasetPreload(
        sketch_root=os.path.join(root, args.sketch_subdir or info['sketch_subdir']),
        image_root=os.path.join(root, info['image_subdir']),
        sketch_suffix=args.sketch_suffix, image_suffix=args.image_suffix,
        is_multi_pair=args.pair_mode == 'multi_pair', is_zero_shot=eval(args.is_zero_shot),
        multi_sketch_split=args.multi_sketch_split, task=args.task,
        extended_image_root=args.extended_image_root, zs_test_classes=test_classes,
        random_seed=args.seed, gallery_mode=args.gallery_mode)


def create_sketch_image_dataloaders(batch_size,
                                    num_workers,
                                    pre_load,
                                    sketch_format,
                                    is_full_train,
                                    task,  # ['cl', 'fg']
                                    n_pos=1, n_neg=8,
                                    ):
    """
    创建训练和测试数据加载器
    
    Args:
        batch_size: 批次大小
        num_workers: 数据加载进程数
        pre_load: 固定数据集划分信息
        sketch_format:
        is_full_train:
        task:
        
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
    
    if is_full_train and pre_load.is_zero_shot:
        raise ValueError('is_full_train is incompatible with zero-shot evaluation')

    # 创建数据集
    train_dataset = SketchImageDataset(
        is_train=True,
        pre_load=pre_load,
        sketch_transform=train_sketch_transform,
        image_transform=train_image_transform,
        sketch_format=sketch_format,
        is_full_train=is_full_train,
        task=task, n_pos=n_pos, n_neg=n_neg
    )

    test_dataset = SketchImageDataset(
        is_train=False,
        pre_load=pre_load,
        sketch_transform=test_transform,
        image_transform=test_transform,
        sketch_format=sketch_format,
        is_full_train=False,
        task=task, n_pos=n_pos, n_neg=n_neg
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

