import numpy as np
import torch
import cv2
import random
import matplotlib.pyplot as plt
from PIL import Image
import logging
import torch.nn as nn
from functools import partial
import os
import torchvision.transforms as transforms
import einops


class MLP(nn.Module):
    def __init__(self, dimension: int, channels: tuple, bias: bool = True, dropout: float = 0.4, final_proc=False):
        """
        :param dimension: 输入维度数，[0, 1, 2, 3]
            输入数据维度: [bs, c], dimension = 0
            输入数据维度: [bs, c, d], dimension = 1
            输入数据维度: [bs, c, d, e], dimension = 2
            输入数据维度: [bs, c, d, e, f], dimension = 3
        :param channels: 输入层到输出层的维度，[in, hid1, hid2, ..., out]
        :param bias:
        :param dropout: dropout 概率
        :param final_proc:
        """
        super().__init__()

        self.linear_layers = nn.ModuleList()
        self.batch_normals = nn.ModuleList()
        self.activates = nn.ModuleList()
        self.drop_outs = nn.ModuleList()

        self.n_layers = len(channels)
        self.final_proc = final_proc
        if dropout == 0:
            self.is_drop = False
        else:
            self.is_drop = True

        if dimension == 0:
            fc = nn.Linear
            bn = nn.BatchNorm1d
            dp = nn.Dropout

        elif dimension == 1:
            fc = partial(nn.Conv1d, kernel_size=1)
            bn = nn.BatchNorm1d
            dp = nn.Dropout1d

        elif dimension == 2:
            fc = partial(nn.Conv2d, kernel_size=1)
            bn = nn.BatchNorm2d
            dp = nn.Dropout2d

        elif dimension == 3:
            fc = partial(nn.Conv3d, kernel_size=1)
            bn = nn.BatchNorm3d
            dp = nn.Dropout3d

        else:
            raise ValueError('error dimension value, [0, 1, 2, 3] is supported')

        for i in range(self.n_layers - 2):
            self.linear_layers.append(fc(channels[i], channels[i + 1], bias=bias))
            self.batch_normals.append(bn(channels[i + 1]))
            self.activates.append(nn.LeakyReLU(negative_slope=0.2))
            self.drop_outs.append(dp(dropout))

        self.outlayer = fc(channels[-2], channels[-1], bias=bias)

        self.outbn = bn(channels[-1])
        self.outat = nn.LeakyReLU(negative_slope=0.2)
        self.outdp = dp(dropout)

    def forward(self, fea):
        """
        :param fea:
        :return:
        """

        for i in range(self.n_layers - 2):
            fc = self.linear_layers[i]
            bn = self.batch_normals[i]
            at = self.activates[i]
            dp = self.drop_outs[i]

            if self.is_drop:
                fea = dp(at(bn(fc(fea))))
            else:
                fea = at(bn(fc(fea)))

        fea = self.outlayer(fea)

        if self.final_proc:
            fea = self.outbn(fea)
            fea = self.outat(fea)

            if self.is_drop:
                fea = self.outdp(fea)

        return fea


def s3_to_tensor_img(sketch, image_size=(224, 224), line_thickness=2, pen_up=1, coor_mode='ABS', save_path=None):
    """
    将 S3 草图转化为 Tensor 图片
    sketch: np.ndarray

    x1, y1, s1
    x2, y2, s2
    ...
    xn, yn, sn

    x, y 为绝对坐标
    s = 1: 下一个点属于当前笔划
    s = 0: 下一个点不属于当前笔划
    注意 Quickdraw 中存储相对坐标，不能直接使用

    :param sketch: 文件路径或者加载好的 [n, 3] 草图
    :param image_size:
    :param line_thickness:
    :param pen_up: 下一个点抬笔的标志位，对于 S3 而言应该是1，但之前弄成了 0，需要注意
    :param coor_mode: 输入坐标是相对坐标还是绝对坐标
    :return: [3, h, w]
    """
    assert coor_mode in ['REL', 'ABS']
    width, height = image_size

    if isinstance(sketch, str):
        points_with_state = np.loadtxt(sketch)
    else:
        points_with_state = sketch

    if coor_mode == 'REL':
        points_with_state[:, :2] = np.cumsum(points_with_state[:, :2], axis=0)

    # 1. 坐标归一化
    pts = np.array(points_with_state[:, :2], dtype=np.float32)
    states = np.array(points_with_state[:, 2], dtype=np.int32)

    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    diff_xy = max_xy - min_xy

    if np.allclose(diff_xy, 0):
        scale_x = scale_y = 1.0
    else:
        scale_x = (width - 1) / diff_xy[0] if diff_xy[0] > 0 else 1.0
        scale_y = (height - 1) / diff_xy[1] if diff_xy[1] > 0 else 1.0
    scale = min(scale_x, scale_y)

    pts_scaled = (pts - min_xy) * scale
    pts_int = np.round(pts_scaled).astype(np.int32)

    offset_x = (width - (diff_xy[0] * scale)) / 2 if diff_xy[0] > 0 else 0
    offset_y = (height - (diff_xy[1] * scale)) / 2 if diff_xy[1] > 0 else 0
    pts_int[:, 0] += int(round(offset_x))
    pts_int[:, 1] += int(round(offset_y))

    # 2. 创建白色画布
    img = np.ones((height, width), dtype=np.uint8) * 255

    # 3. 笔划切分
    split_indices = np.where(states == pen_up)[0] + 1  # 下一个点是新笔划，所以+1
    strokes = np.split(pts_int, split_indices)

    # 4. 绘制每条笔划
    for stroke in strokes:
        if len(stroke) >= 2:  # 至少2个点才能画线
            stroke = stroke.reshape(-1, 1, 2)
            cv2.polylines(img, [stroke], isClosed=False, color=0, thickness=line_thickness, lineType=cv2.LINE_AA)

    # 5. 转为归一化float32 Tensor
    tensor_img = torch.from_numpy(img).float() / 255.0

    if save_path is not None:
        cv2.imwrite(save_path, img)

    # 将 [h, w] 的图片转化为 [3, h, w] 的
    tensor_img = einops.repeat(tensor_img, '... -> 3 ...')
    return tensor_img


def s5_to_tensor_img(s5_tensor, coor_mode='REL', save_path=None):
    """
    s5_tensor: [n_point, 5]
    return: [3, h, w]
    """
    s5_tensor = s5_tensor.cpu().numpy()
    recov_s3 = s5_tensor[:, [0, 1, 2]]

    if coor_mode == 'REL':
        recov_s3[:, :2] = np.cumsum(recov_s3[:, :2], 0)

    tensor_img = s3_to_tensor_img(recov_s3, save_path=save_path)
    return tensor_img


def stk_to_tensor_image(stk_tensor, save_path=None):
    """
    将 stk_tensor 转化为可视化图片
    stk_tensor: [n_stk, n_stk_pnt, 2]
    """
    # 先将stk_tensor转化为s3
    n_stk, n_stk_pnt, _ = stk_tensor.size()
    new_x = torch.cat([stk_tensor, torch.zeros(n_stk, n_stk_pnt, 1)], dim=2)

    for i in range(n_stk):
        new_x[i, -1, -1] = 1

    new_x = new_x.view(-1, 3)
    tensor_img = s3_to_tensor_img(new_x, save_path=save_path)
    return tensor_img


def s3_file_to_s5(root, max_length, pen_up=1, coor_mode='REL', is_shuffle_stroke=False, is_back_mask=True):
    """
    将草S3图转换为 S5 格式，(x, y, s1, s2, s3)
    默认存储绝对坐标
    :param root:
    :param max_length:
    :param pen_up: S3 格式中，表示笔划结束的标志位，
    :param coor_mode: 返回的坐标格式，输入默认绝对坐标 ['ABS', 'REL'], 'ABS': absolute coordinate. 'REL': relative coordinate [(x,y), (△x, △y), (△x, △y), ...].
    :param is_shuffle_stroke: 是否打乱笔划
    :param is_back_mask:
    :return:
    """
    data_raw = np.loadtxt(root)

    # 打乱笔划
    if is_shuffle_stroke:
        stroke_list = np.split(data_raw, np.where(data_raw[:, 2] == pen_up)[0] + 1)[:-1]
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

    if is_back_mask:
        return data_cube, mask
    else:
        return data_cube


def load_stk_sketch(s3_file, n_stk, n_stk_pnt):
    s3_data = np.loadtxt(s3_file, dtype=np.float32)
    # n_stk = re.findall(r'stk(\d+)', stk_name)[0]
    # n_stk_pnt = re.findall(r'stkpnt(\d+)', stk_name)[0]
    stk_data = s3_data.reshape(int(n_stk), int(n_stk_pnt), 2)
    stk_data = torch.from_numpy(stk_data)

    return stk_data


def vis_s3(s3_file, delimiter=' '):
    if isinstance(s3_file, str):
        data = np.loadtxt(s3_file, delimiter=delimiter)
    else:
        data = s3_file

    # 分割笔划
    data[-1, 2] = 1
    sketch = np.split(data[:, :2], np.where(data[:, 2] == 0)[0] + 1)

    for s in sketch:
        plt.plot(s[:, 0], -s[:, 1])

    plt.axis("equal")
    plt.show()


def image_loader(image_path,
                 image_transform=transforms.Compose([transforms.Resize((224, 224)),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                                                    ),
                 empty_fill=(255, 255, 255)
                 ):
    """
    读取图片到 torch.tensor，且png图片的空白区域填充 empty_fill 指定颜色
    """
    image_pil = Image.open(image_path)

    if image_pil.mode == 'RGBA':
        background = Image.new('RGBA', image_pil.size, (*empty_fill, 255))
        image_pil = Image.alpha_composite(background, image_pil)

    image_pil = image_pil.convert('RGB')
    image = image_transform(image_pil)
    return image

    # image_pil = Image.open(image_path).convert('RGBA')
    #
    # # 创建一个白色背景图像
    # background = Image.new('RGBA', image_pil.size, (*empty_fill, 255))
    #
    # # 将原图粘贴到背景上，用 alpha 通道作为 mask
    # image_pil = Image.alpha_composite(background, image_pil)
    #
    # # 去掉 alpha 通道
    # image_pil = image_pil.convert('RGB')
    #
    # image = image_transform(image_pil)
    # return image


def get_log(log_root: str):
    """
    :param log_root: log文件路径，例如 ./opt/private/log.txt
    :return:
    """
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_root)  # 日志文件路径
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_check_point(weight_dir, save_str):
    """
    统一的通过权重目录名以及保存字符串获取检查点文件路径的方法
    """
    check_point = os.path.join(weight_dir, save_str + '.pth')
    return check_point


def get_save_str(args):
    """
    统一的获取保存名的方式
    """
    is_full_train = 'full_' if eval(args.is_full_train) else ''
    pair_mode = 'single_pair' if args.pair_mode == 'single_pair' else ''

    save_str = (args.sketch_model + '_' +
                args.image_model + '_' +
                args.retrieval_mode + '_' +
                args.task + '_' +
                is_full_train +
                pair_mode +
                args.add_str)

    return save_str


def basename_without_ext(file_name):
    """
    将带路径和后缀的文件名
    去掉路径和后缀
    :param file_name:
    :return:
    """
    name = os.path.splitext(os.path.basename(file_name))[0]
    return name


if __name__ == '__main__':
    # as3_file = r'D:\document\DeepLearning\DataSet\sketch_retrieval\sketchy\sketch_s3_352\airplane\n02691156_196-5.txt'
    # stk_file = r'D:\document\DeepLearning\DataSet\sketch_retrieval\sketchy\sketch_stk11_stkpnt32\airplane\n02691156_58-1.txt'
    # trans_save = r'C:\Users\ChengXi\Desktop\60mm20250708\rel_skh.png'
    #
    # # raw_s3 = np.loadtxt(as3_file, delimiter=',')
    # # s5_tensor = s3_file_to_s5(as3_file, is_back_mask=False)
    # # s5_to_tensor_img(s5_tensor, save_path=trans_save)
    #
    # raw_stk = np.loadtxt(stk_file, delimiter=',')
    # raw_stk = raw_stk.reshape(11, 32, 2)
    #
    # stk_to_tensor_image(torch.from_numpy(raw_stk), trans_save)

    # image_transform_ = transforms.Compose([
    #     transforms.Resize((256, 256)),
    #     transforms.ToTensor()
    # ])
    #
    # tensor_img = image_loader(r'D:\document\DeepLearning\DataSet\草图项目\retrieval_cad\sketch_ai\衬套\0ece4f05e97dcfc6ea9750dac8aa4988_1.png', image_transform_)
    #
    # # Tensor 格式通常是 (C, H, W)，要转为 (H, W, C)
    # plt.imshow(tensor_img.permute(1, 2, 0))
    # plt.axis('off')  # 去掉坐标轴
    # plt.show()

    # res = s3_to_tensor_img(r'D:\document\DeepLearning\DataSet\sketch_retrieval\qmul_v2_fit\chair\sketch_s3_352\test\class\9910-02-carbon_1.txt', pen_up=1, save_path=r'C:\Users\ChengXi\Desktop\cstnet2\gen2.png')
    #
    # res2 = image_loader(r'D:\document\DeepLearning\DataSet\sketch_retrieval\qmul_v2_fit\chair\photo\test\class\123-din-s.png')
    # stk_name = 'sketch_stk12_stkpnt32_autospace'
    # n_stk = re.findall(r'stk(\d+)', stk_name)
    # n_stk_pnt = re.findall(r'stkpnt(\d+)', stk_name)
    # print(n_stk, n_stk_pnt)

    file_nale = r'D:\document\DeepLearning\DataSet\sketch_retrieval\qmul_v2_fit\chair\sketch_s3_352\train\class\2kn308a2ca10_1.txt'
    data = np.loadtxt(file_nale)
    print(data.shape)

    pass



