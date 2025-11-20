"""
可视化前5好类别的PNG草图-图像检索效果
专门用于展示模型在最佳类别上的检索表现
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# 导入数据集和模型
from data import retrieval_datasets
from encoders import sbir_model_wrapper
from utils import utils
import options
from encoders import create_sketch_encoder
from safetensors.torch import save_file, load_file


DATASET_IMG_ROOT = r'D:\document\DeepLearning\DataSet\草图项目\retrieval_cad\sketch_png'
DATASET_SKH_ROOT = r'D:\document\DeepLearning\DataSet\草图项目\retrieval_cad\sketch_ai'

# ========== 永久解决中文显示问题 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei',
                                  'Arial Unicode MS', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False
# ===========================================


def sketch_file_list_to_tensor(sketch_file_list, sketch_dataset: retrieval_datasets.SketchImageDataset):
    """
    将文件列表转化为 tensor
    注意每个文件表达方式为 类别@文件名，例如 'pear@n12651611_7402-2'
    """
    sketch_format = sketch_dataset.sketch_format
    sketch_suffix = '.txt' if sketch_format == 'vector' else '.png'

    sketch_cat = []
    sketch_tensor_list = []
    pair_image_idx = []
    for c_sketch_file in sketch_file_list:
        category, base_name = c_sketch_file.split('@')
        sketch_cat.append(category)
        base_name = base_name + sketch_suffix

        # c_path = sketch_dataset.get_sketch_path(category, base_name)
        c_path = os.path.join(DATASET_SKH_ROOT, category, base_name)
        # c_pair_image_idx = sketch_dataset.find_image_idx(category, base_name)

        c_img_path = os.path.join(DATASET_IMG_ROOT, category, base_name)
        c_pair_image_idx = sketch_dataset.imgs_all.index((c_img_path, sketch_dataset.classes_idx[category]))

        c_sketch_tensor = sketch_dataset.sketch_loader(c_path)

        pair_image_idx.append(c_pair_image_idx)
        sketch_tensor_list.append(c_sketch_tensor)

    sketch_tensor = torch.stack(sketch_tensor_list, dim=0)
    pair_image_idx = torch.tensor(pair_image_idx)
    return sketch_tensor, pair_image_idx, sketch_cat


def find_topk_matching_images(
    sketch_features: torch.Tensor,      # [m, c]
    image_features: torch.Tensor,       # [n, c]
    image_file_tensor: torch.Tensor,    # [n, c, h, w]
    k: int = 5,
    image_cat=None  # 全部图片的类别索引 [n, ]
) -> [torch.Tensor, torch.Tensor]:
    """
    使用余弦相似度，在 image_file_tensor 中找出与 sketch_features 最相似的 k 张图像。

    Returns:
        matched_images: [m, k, c, h, w]
    """
    # 计算余弦相似度（假设已经归一化过）
    sim_matrix = torch.matmul(sketch_features, image_features.T)  # [m, n]

    # 取每个 sketch 对应 top-k 最相似的 image 索引
    topk_indices = torch.topk(sim_matrix, k=k, dim=1, largest=True).indices  # [m, k]

    # 按索引收集图像
    matched_images = []
    matched_cat = []
    for i in range(sketch_features.size(0)):
        matched = image_file_tensor[topk_indices[i]]  # [k, c, h, w]
        matched_images.append(matched)

        matched_cat.append(image_cat[topk_indices[i]])

    # [m, k, c, h, w]
    matched_images = torch.stack(matched_images, dim=0)  # [m, k, c, h, w]

    matched_cat = torch.stack(matched_cat, dim=0)  # [m, k]
    return matched_images, topk_indices, matched_cat


def visualize_sketch_retrieval_results(
        sketch_file_tensor: torch.Tensor,        # [m, c, h, w]
        topk_images: torch.Tensor,               # [m, k, c, h, w]
        sketch_cat,  # [m]，草图对应的类别索引
        image_cat,  # [m, k]，检索到的图片的类别索引
        save_dir,
        cat_name_idx_dict: dict
):
    def get_key_by_value(_target_value):
        for _k, _v in cat_name_idx_dict.items():
            if _v == _target_value:
                return _k
        return None  # 没找到返回 None

    m, k, c, h, w = topk_images.shape
    fig, axes = plt.subplots(m, k + 1, figsize=(1.5 * (k + 2), 1.5 * m))

    if m == 1:
        axes = [axes]  # 保证统一二维结构

    for i in range(m):
        # 第 0 列显示草图
        sketch_np = tensor_to_image(sketch_file_tensor[i])
        axes[i][0].imshow(sketch_np, cmap='gray' if sketch_np.ndim == 2 else None)
        axes[i][0].axis("off")
        axes[i][0].set_title(get_key_by_value(sketch_cat[i]))

        for j in range(k):
            img = topk_images[i][j]
            img_np = tensor_to_image(img)

            skh_class = sketch_cat[i]
            img_class = image_cat[i, j].item()

            if skh_class != img_class:
                img_np = add_red_border(img_np)

            axes[i][j + 1].imshow(img_np, cmap='gray' if img_np.ndim == 2 else None)
            axes[i][j + 1].axis("off")
            axes[i][j + 1].set_title(get_key_by_value(img_class))

            # c_image_path = os.path.join(save_dir, f'image_{i}_{j}.png')
            # plt.imsave(c_image_path, img_np, cmap='gray' if img_np.ndim == 2 else None)

    plt.tight_layout()
    plt.savefig(save_dir)
    plt.close()
    print(f"Saved visualization to: {save_dir}")


def tensor_to_image(tensor):
    """
    将 [c, h, w] 的 tensor 转为 numpy 图像 (h, w, c)，范围 [0, 1]，可用于 plt.imshow。
    """
    tensor = tensor.detach().cpu()
    if tensor.size(0) == 1:  # 单通道
        img = tensor.squeeze(0).numpy()
        return img
    else:  # 多通道
        img = tensor.permute(1, 2, 0).numpy()  # [h, w, c]
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)  # normalize to [0, 1]
        return img


def add_red_border(img, border=5):
    """
    img   : ndarray, shape (H, W, 3), dtype float or uint8, range [0,1] or [0,255]
    border: int, 边框厚度（像素）
    return: 带红色边框的图像
    """
    img = img.copy()          # 不破坏原图
    if img.dtype != np.float32 and img.dtype != np.float64:
        # 如果是 uint8，先把 255 归一化到 1
        img = img.astype(np.float32) / 255.0

    H, W = img.shape[:2]
    # 上、下
    img[:border, :, :] = [1, 0, 0]
    img[H-border:, :, :] = [1, 0, 0]
    # 左、右（注意避开已涂过的角）
    img[:, :border, :] = [1, 0, 0]
    img[:, W-border:, :] = [1, 0, 0]

    return img


def get_ground_truth_images(image_file_tensor, pair_image_idx, image_idx):
    """
    保存每个草图对应的 Ground Truth 图像。

    参数：
        - sketch_file_tensor: (m, C, H, W) 草图图像张量
        - image_file_tensor: (n, C, H, W) 图像张量
        - pair_image_idx: (m,) 对应草图的匹配图像索引（基于 image_idx 的原始值）
        - image_idx: (n,) 表示 image_file_tensor 中每个图像的索引编号

    """
    # 从 image_idx 中找出 pair_image_idx 的位置
    image_idx_map = {idx: i for i, idx in enumerate(image_idx.tolist())}

    gt_images = []
    for i in range(len(pair_image_idx)):
        gt_idx = pair_image_idx[i].item()
        gt_pos = image_idx_map.get(gt_idx, None)
        if gt_pos is None:
            print(f"[Warning] Cannot find GT image for sketch {i}, index {gt_idx} not found.")
            continue

        gt_img = tensor_to_image(image_file_tensor[gt_pos])
        gt_images.append(gt_img)

    return gt_images


def sketch_tensor_to_pixel_image(sketch_tensor, sketch_rep):
    """
    sketch_tensor: 直接输入到模型的 tensor
    sketch_rep: 草图表达形式
    """
    if sketch_rep == 'S5':
        trans_func = utils.s5_to_tensor_img

    elif 'STK' in sketch_rep:
        trans_func = utils.stk_to_tensor_image

    elif sketch_rep == 'IMG':
        return sketch_tensor

    else:
        raise TypeError('unsupported sketch_rep')

    transed_tensor = []
    for i in range(sketch_tensor.size(0)):
        c_transed = trans_func(sketch_tensor[i])
        transed_tensor.append(c_transed)

    transed_tensor = torch.stack(transed_tensor, dim=0)
    return transed_tensor


def main(args, eval_sketches):
    print("开始可视化前5好类别的PNG草图-图像检索效果...")
    
    # 设置路径
    save_str = utils.get_save_str(args)
    checkpoint_path = utils.get_check_point(args.weight_dir, save_str)
    sketch_info = create_sketch_encoder.get_sketch_info(args.sketch_model)
    # split_file = retrieval_datasets.get_split_file_name(sketch_info['format'], args.pair_mode, args.task)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    current_vis_file = os.path.join(args.output_dir, save_str + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.png')
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"检查点文件不存在: {checkpoint_path}")
        return

    # 创建数据加载器
    print("加载测试数据集...")
    # root = args.root_local if eval(args.local) else args.root_sever

    root = args.root_local if eval(args.local) else args.root_sever

    if sketch_info['format'] == 'vector':
        sketch_subdir = sketch_info['subdirs'][0]
        sketch_image_suffix = ('txt', 'jpg')
    else:
        sketch_subdir = sketch_info['subdirs'][1]
        sketch_image_suffix = ('png', 'jpg')

    if eval(args.local):
        sketch_root = r'D:\document\DeepLearning\DataSet\草图项目\retrieval_cad\sketch_ai'
        image_root = r'D:\document\DeepLearning\DataSet\草图项目\retrieval_cad\sketch_png'
    else:
        sketch_root = r'/opt/data/private/data_set/sketch_retrieval/retrieval_cad/sketch_ai'
        image_root = r'/opt/data/private/data_set/sketch_retrieval/retrieval_cad/sketch_png'

    pre_load = retrieval_datasets.DatasetPreload(
        sketch_root=sketch_root,
        image_root=image_root,
        sketch_image_suffix=sketch_image_suffix,
        is_multi_pair=True if args.pair_mode == 'multi_pair' else False,
        split_mode='ZS-SBIR' if args.task == 'zs_sbir' else 'SBIR',
        is_full_train=eval(args.is_full_train),
        sketch_choice=300
    )

    _, test_set, _, test_loader, dataset_info = retrieval_datasets.create_sketch_image_dataloaders(
        batch_size=args.bs,
        num_workers=args.num_workers,
        pre_load=pre_load,
        root=root,
        sketch_format=sketch_info['format'],
        vec_sketch_rep=sketch_info['rep'],
        sketch_image_subdirs=sketch_info['subdirs'],
        is_back_dataset=True
    )
    test_set.set_back_mode('image')

    # 创建并加载模型
    print(f"从 {checkpoint_path} 加载模型...")
    model = sbir_model_wrapper.create_sbir_model_wrapper(
        embed_dim=args.embed_dim,
        freeze_image_encoder=True,
        freeze_sketch_backbone=True,
        sketch_model_name=args.sketch_model,
        image_model_name=args.image_model
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"成功加载检查点 (epoch {checkpoint.get('epoch', 'unknown')})")

    model.to(device)
    model.eval()
    
    # 提取特征
    print("提取特征...")
    sketch_file_tensor, pair_image_idx, sketch_cat = sketch_file_list_to_tensor(eval_sketches, test_set)
    sketch_features = model.encode_sketch(sketch_file_tensor.to(device)).cpu()

    # 将草图的名字转化为索引
    sketch_cat = [test_set.classes_idx[c_name] for c_name in sketch_cat]

    # tensor_save_path = 'model_trained/img_feas_inferred.safetensors'
    if eval(args.is_load_features):
        print('load data from: ' + args.fea_save_path)
        data_loaded = load_file(args.fea_save_path)
        image_cat = data_loaded['image_cat']
        image_features = data_loaded['image_features']
        image_file_tensor = data_loaded['image_file_tensor']

        print('image_cat size: ', image_cat.size())
        print('image_features size: ', image_features.size())
        print('image_file_tensor size: ', image_file_tensor.size())

    else:
        image_file_tensor = []
        image_features = []
        image_cat = []
        with torch.no_grad():
            for idx, images, category_indices in tqdm(test_loader):
                images = images.to(device)
                image_file_tensor.append(images.cpu())
                # image_idx.append(idx.cpu())
                image_cat.append(category_indices.cpu())

                # 编码草图和图像
                image_feat = model.encode_image(images)
                image_features.append(image_feat.cpu())

        # 合并特征
        image_cat = torch.cat(image_cat)
        image_features = torch.cat(image_features, dim=0)
        image_file_tensor = torch.cat(image_file_tensor, dim=0)
        print(f"提取特征完成: sketch {sketch_features.shape}, image {image_features.shape}")

        img_feas_inferred = {
            'image_features': image_features,  # 图片的特征 [n, c]
            'image_cat': image_cat,  # 图片的类别编号 [n, ]
            'image_file_tensor': image_file_tensor,  # 用于图片显示 [n, 3, w, h]
        }

        # torch.save(img_feas_inferred, 'model_trained/img_feas_inferred.pt')
        save_file(img_feas_inferred, args.fea_save_path)
        print('属性保存完成，保存到：' + args.fea_save_path)

    # 找到最近的图片索引
    topk_images, topk_indices, matched_cat = find_topk_matching_images(sketch_features, image_features, image_file_tensor, args.n_vis_images, image_cat)

    # 可视化图片 tensor
    sketch_pixel_tensor = sketch_tensor_to_pixel_image(sketch_file_tensor, sketch_info['rep'])
    visualize_sketch_retrieval_results(sketch_pixel_tensor, topk_images, sketch_cat, matched_cat, current_vis_file, test_set.classes_idx)


if __name__ == '__main__':
    # 用于设置需要进行可视化的草图
    # 以 类别 @ 文件名 为格式

    setting_sketches = [
        # 'tree@n11759853_24427-2',
        # 'seagull@n02041246_20823-3',
        '衬套@0ece4f05e97dcfc6ea9750dac8aa4988_1',
        # 'helicopter@n03512147_1414-3',
        '挡圈@0af3e3cdd1fee208b4b62dffd485f96d_2',
        '脚轮@34dc393db3e9fd6be4b336de0b3a22cb_1',
        # 'songbird@n01532829_2619-1',
        # 'sword@n04373894_59060-3',
        # 'door@n03222318_7012-1',
        # 'scissors@n04148054_3393-3',
        # 'saw@n02770585_4664-1',
        # 'songbird@n01527347_17110-3',
        '涡轮@2e383fb609c43c89198bc96cd7efb18a_1',

        # 'bat@n02139199_10978-1',
        # 'bat@n02139199_11031-2',
        # 'bat@n02139199_11187-1',

        # 'cabin@n02932400_10327-2',
        # 'cow@n01887787_10513-2',
        # 'dolphin@n02068974_10564-1',
        # 'dolphin@n02068974_1146-2',
        # 'door@n03222176_6155-1',
        # 'giraffe@n02439033_10491-3',
        # 'helicopter@n03512147_1191-5',
        # 'mouse@n02330245_1044-3',
        '风扇@4cb9ab1cc91d75644fda9b8aac56bb35_4',
        # 'raccoon@n02508021_15891-10',
        # 'wheelchair@n04576002_10456-2',

        # 'tree@n12523475_9706-4',
        # 'pear@n07768230_1040-4',
        # 'sword@n04373894_31746-2',
        # 'songbird@n01530575_10073-3',
        # 'skyscraper@n04233124_14086-4',
        # 'helicopter@n03512147_1318-6',
        '键@7c938d4073f9e54754aabc972c48ff90_4'

    ]

    main(options.parse_args(), setting_sketches)


    # atensor = torch.rand(3, )
    # btansor = torch.rand(5, )
    #
    # alist = [atensor, btansor]
    # alist = torch.cat(alist)
    # print(alist.size())



