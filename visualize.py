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


def find_topk_matching_images(skh_fea, img_fea, img_tensor, k) -> [torch.Tensor, torch.Tensor]:
    """
    使用余弦相似度，在 img_tensor 中找出与 skh_fea 最相似的 k 张图像。
    Args:
        skh_fea: [m, c]
        img_fea: [n, c]
        img_tensor: [n, c, h, w]
        k:

    Returns:
        matched_images: [m, k, c, h, w]
        matched_indices: [m, k]
    """
    # 计算余弦相似度（假设已经归一化过）
    sim_matrix = torch.matmul(skh_fea, img_fea.T)  # [m, n]

    # 取每个 sketch 对应 top-k 最相似的 image 索引
    topk_indices = torch.topk(sim_matrix, k=k, dim=1, largest=True).indices  # [m, k]

    # 按索引收集图像
    matched_images = []
    for i in range(skh_fea.size(0)):
        matched = img_tensor[topk_indices[i]]  # [k, c, h, w]
        matched_images.append(matched)

    # [m, k, c, h, w]
    matched_images = torch.stack(matched_images, dim=0)  # [m, k, c, h, w]
    return matched_images, topk_indices


def visualize_sketch_retrieval_results(skh_pixel, gt_imgs,gt_img_idx, topk_imgs, topk_img_idx, save_dir):
    """
    将检索结果按规则排列
    Args:
        skh_pixel: [m, c, h, w]
        gt_imgs: [m, c, h, w]
        gt_img_idx: [m]
        topk_imgs: [m, k, c, h, w]
        topk_img_idx: [m, k]
        save_dir:

    """
    m, k, c, h, w = topk_imgs.shape
    fig, axes = plt.subplots(m, k + 2, figsize=(1.5 * (k + 2), 1.5 * m))

    if m == 1:
        axes = [axes]  # 保证统一二维结构

    for i in range(m):
        # 第 0 列显示草图
        sketch_np = tensor_to_image(skh_pixel[i])
        axes[i][0].imshow(sketch_np, cmap='gray' if sketch_np.ndim == 2 else None)
        axes[i][0].axis("off")

        c_sketch_path = os.path.join(save_dir, f'sketch_{i}.png')
        plt.imsave(c_sketch_path, sketch_np, cmap='gray' if sketch_np.ndim == 2 else None)

        sketch_gt = tensor_to_image(gt_imgs[i])
        axes[i][1].imshow(sketch_gt, cmap='gray' if sketch_np.ndim == 2 else None)
        axes[i][1].axis("off")

        c_sketch_path = os.path.join(save_dir, f'gt_{i}.png')
        plt.imsave(c_sketch_path, sketch_gt, cmap='gray' if sketch_gt.ndim == 2 else None)

        gt_idx = gt_img_idx[i]

        for j in range(k):
            img = topk_imgs[i][j]
            img_np = tensor_to_image(img)

            # topk_idx = topk_img_idx[i][j]
            # if topk_idx != gt_idx:
            #     img_np = add_red_border(img_np)

            axes[i][j + 2].imshow(img_np, cmap='gray' if img_np.ndim == 2 else None)
            axes[i][j + 2].axis("off")

            c_image_path = os.path.join(save_dir, f'image_{i}_{j}.png')
            plt.imsave(c_image_path, img_np, cmap='gray' if img_np.ndim == 2 else None)

    summary_path = os.path.join(save_dir, 'overall.png')
    plt.tight_layout()
    plt.savefig(summary_path)
    plt.close()
    print(f"Saved visualization to: {summary_path}")


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


def add_red_border(img, border_width=5):
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
    img[:border_width, :, :] = [1, 0, 0]
    img[H-border_width:, :, :] = [1, 0, 0]
    # 左、右（注意避开已涂过的角）
    img[:, :border_width, :] = [1, 0, 0]
    img[:, W-border_width:, :] = [1, 0, 0]

    return img


def sketch_tensor_to_pixel_image(sketch_tensor, sketch_rep):
    """
    sketch_tensor: 直接输入到模型的 tensor
    sketch_rep: 草图表达形式
    """
    if sketch_rep == 's5':
        trans_func = utils.s5_to_tensor_img

    elif 'stk' in sketch_rep:
        trans_func = utils.stk_to_tensor_image

    elif '-> img' in sketch_rep:
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
    encoder_info = options.get_encoder_info(args.sketch_model)

    # 创建输出目录
    current_vis_dir = os.path.join(args.output_dir, save_str + '_' + datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    os.makedirs(current_vis_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"检查点文件不存在: {checkpoint_path}")
        return

    # 创建数据加载器
    root = args.root_local if eval(args.local) else args.root_sever
    sketch_root = os.path.join(root, encoder_info['sketch_subdir'])
    image_root = os.path.join(root, encoder_info['image_subdir'])
    pre_load = retrieval_datasets.DatasetPreload(
        sketch_root=sketch_root,
        image_root=image_root,
        sketch_suffix=encoder_info['sketch_suffix'],
        image_suffix=encoder_info['image_suffix'],
        is_multi_pair=True if args.pair_mode == 'multi_pair' else False,
        split_mode=args.task,
        is_full_train=eval(args.is_full_train),
        multi_sketch_split=args.multi_sketch_split
    )

    vis_loader = retrieval_datasets.create_sketch_image_dataloaders(
        batch_size=args.bs,
        num_workers=args.num_workers,
        pre_load=pre_load,
        sketch_format=encoder_info['sketch_format'],
        back_mode='vis'
    )

    # 创建并加载模型
    model = sbir_model_wrapper.create_sbir_model_wrapper(
        embed_dim=args.embed_dim,
        freeze_image_encoder=True,
        freeze_sketch_backbone=True,
        sketch_model_name=args.sketch_model,
        image_model_name=args.image_model,
        sketch_format=encoder_info['sketch_format'],
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"成功加载检查点 (epoch {checkpoint.get('epoch', 'unknown')})")
    
    model.to(device)
    model.eval()

    # 将输入的草图信息转化为绝对路径
    eval_sketch_path = []
    for c_eval_skh in eval_sketches:
        c_class, c_base_name = c_eval_skh.split('/')
        c_real_path = os.path.join(sketch_root, 'test', c_class, c_base_name + '.' + encoder_info['sketch_suffix'])
        eval_sketch_path.append(c_real_path)

    # 获取全部的图片路径列表
    img_path_list = []
    for c_pair in vis_loader.dataset.data_pairs:
        img_path_list.append(c_pair[1])

    # 找到查询草图对应的图片在图片列表中的索引
    gt_img_idx = []
    for c_skh_path in eval_sketch_path:
        # 根据草图路径找到对应的图片路径
        c_img_path = c_skh_path.replace(sketch_root, image_root)
        c_img_path = c_img_path.rsplit(args.multi_sketch_split, 1)[0]
        c_img_path = c_img_path + '.' + encoder_info['image_suffix']

        # 找到草图对应的图片在图片列表中的索引
        c_paired_img_idx = img_path_list.index(c_img_path)
        gt_img_idx.append(c_paired_img_idx)

    # 提取草图特征
    sketch_tensor_list = []
    for c_skh_path in eval_sketch_path:
        c_skh_tensor = vis_loader.dataset.sketch_loader(c_skh_path)
        sketch_tensor_list.append(c_skh_tensor)
    skh_tensor = torch.stack(sketch_tensor_list, dim=0)
    skh_fea = model.encode_sketch(skh_tensor.to(device)).cpu()

    image_tensor_list = []
    image_fea_list = []
    with torch.no_grad():
        for _, img_tensor, category_indices in tqdm(vis_loader):
            # 由于 shuffle=False，无需担心加载过程中图片顺序被打乱
            image_tensor_list.append(img_tensor.cpu())

            # 编码图像
            img_tensor = img_tensor.to(device)
            img_fea = model.encode_image(img_tensor)
            image_fea_list.append(img_fea.cpu())

    # 合并特征
    img_fea = torch.cat(image_fea_list, dim=0)
    img_tensor = torch.cat(image_tensor_list, dim=0)
    print(f"提取特征完成: sketch {skh_fea.shape}, image {img_fea.shape}")

    # 找到最近的图片及索引，索引用于确认是否检索准确
    topk_imgs, topk_img_idx = find_topk_matching_images(skh_fea, img_fea, img_tensor, args.n_vis_images)

    # 获取gt 图片
    gt_img_tensor = img_tensor[gt_img_idx]

    # 将草图转化为可视化图片，有些草图是矢量形式，需要转化为图片
    skh_info_dict = options.parse_sketch_format(encoder_info['sketch_format'])
    skh_pixel_tensor = sketch_tensor_to_pixel_image(skh_tensor, skh_info_dict['fmt'])

    # 将查询草图、gt图片、检索图片（gt有红框）进行排列
    visualize_sketch_retrieval_results(skh_pixel_tensor, gt_img_tensor, gt_img_idx, topk_imgs, topk_img_idx, current_vis_dir)


if __name__ == '__main__':
    # 因为不同的草图Encoder的草图文件不同，因此直接以绝对路径wi输入需要多次调整，改为 类别/文件名（不带后缀）
    setting_sketches = [

        # "D:\\document\\DeepLearning\\DataSet\\sketch_retrieval\\qmul_v2_fit\\chair\\sketch_stk12_stkpnt32\\test\\class\\SOFHDX002BRO-UK_v1_SaddleBrownPremiumLeather_1.txt",
        # "D:\\document\\DeepLearning\\DataSet\\sketch_retrieval\\qmul_v2_fit\\chair\\sketch_stk12_stkpnt32\\test\\class\\cm87101-542_2.txt",
        # "D:\\document\\DeepLearning\\DataSet\\sketch_retrieval\\qmul_v2_fit\\chair\\sketch_stk12_stkpnt32\\test\\class\\nin0-3a-noir_1.txt",
        # "D:\\document\\DeepLearning\\DataSet\\sketch_retrieval\\qmul_v2_fit\\chair\\sketch_stk12_stkpnt32\\test\\class\\ge_11.txt",
        # "D:\\document\\DeepLearning\\DataSet\\sketch_retrieval\\qmul_v2_fit\\chair\\sketch_stk12_stkpnt32\\test\\class\\cm87101-542_1.txt",
        # "D:\\document\\DeepLearning\\DataSet\\sketch_retrieval\\qmul_v2_fit\\chair\\sketch_stk12_stkpnt32\\test\\class\\mgup-_1.txt",
        # "D:\\document\\DeepLearning\\DataSet\\sketch_retrieval\\qmul_v2_fit\\chair\\sketch_stk12_stkpnt32\\test\\class\\mosmkedb_1.txt",
        # "D:\\document\\DeepLearning\\DataSet\\sketch_retrieval\\qmul_v2_fit\\chair\\sketch_stk12_stkpnt32\\test\\class\\mgup-_12.txt",
        # "D:\\document\\DeepLearning\\DataSet\\sketch_retrieval\\qmul_v2_fit\\chair\\sketch_stk12_stkpnt32\\test\\class\\CHABRA003GRY-UK_v1_PearlGrey_1.txt",
        # "D:\\document\\DeepLearning\\DataSet\\sketch_retrieval\\qmul_v2_fit\\chair\\sketch_stk12_stkpnt32\\test\\class\\gubi9-blkhirek-chrome_2.txt",
        # "D:\\document\\DeepLearning\\DataSet\\sketch_retrieval\\qmul_v2_fit\\chair\\sketch_stk12_stkpnt32\\test\\class\\CHAPIC004GRY-UK_v1_ShadowSlateGrey_1.txt",
        # "D:\\document\\DeepLearning\\DataSet\\sketch_retrieval\\qmul_v2_fit\\chair\\sketch_stk12_stkpnt32\\test\\class\\gubi9-blkhirek-chrome_1.txt",
        # "D:\\document\\DeepLearning\\DataSet\\sketch_retrieval\\qmul_v2_fit\\chair\\sketch_stk12_stkpnt32\\test\\class\\ocs04gr_10.txt",
        # "D:\\document\\DeepLearning\\DataSet\\sketch_retrieval\\qmul_v2_fit\\chair\\sketch_stk12_stkpnt32\\test\\class\\fau_2.txt"

        "class/SOFHDX002BRO-UK_v1_SaddleBrownPremiumLeather_1",
        'class/gubi9-blkhirek-chrome_2',
        'class/ge_11',
        'class/mgup-_1',
        'class/mgup-_12',
        'class/cm87101-542_1',
        "class/mc3-smalls_12",
        "class/cm87101-542_2",
        "class/mgankara_1",
        "class/saku231-vi30_2",
        "class/SOFMLI002GRY-UK_v1_Gra_2",
        "class/nin0-3a-noir_1",
        "class/mosmkedb_2",
        "class/mgankara_3",
        "class/mosmkedb_1",
        "class/saku231-vi30_10",
        "class/CHABRA003GRY-UK_v1_PearlGrey_1",
        "class/fau_2",
        "class/cm87201-2445_1",
        'class/CHAPIC004GRY-UK_v1_ShadowSlateGrey_1',
        'class/gubi9-blkhirek-chrome_1',
        'class/ocs04gr_10'

    ]

    main(options.parse_args(), setting_sketches)


