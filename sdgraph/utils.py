import torch
import torch.nn as nn
import os
import numpy as np
import re
import matplotlib.pyplot as plt
import logging
from functools import partial


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
            self.activates.append(activate_func())
            self.drop_outs.append(dp(dropout))

        self.outlayer = fc(channels[-2], channels[-1], bias=bias)

        self.outbn = bn(channels[-1])
        self.outat = activate_func()
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


def activate_func():
    """
    控制激活函数
    :return:
    """
    return nn.LeakyReLU(negative_slope=0.2)
    # return nn.ReLU()
    # return nn.GELU()
    # return nn.SiLU()


def knn(vertices, neighbor_num):
    """
    找到最近的点的索引，不包含自身
    :param vertices: [bs, n_point, 2]
    :param neighbor_num:
    :return: [bs, n_point, k]
    """
    bs, n_point, _ = vertices.size()

    inner = torch.bmm(vertices, vertices.transpose(1, 2))  # (bs, v, v)
    quadratic = torch.sum(vertices**2, dim=2)  # (bs, v)
    distance = inner * (-2) + quadratic.unsqueeze(1) + quadratic.unsqueeze(2)

    neighbor_index = torch.topk(distance, k=neighbor_num + 1, dim=-1, largest=False)[1]
    neighbor_index = neighbor_index[:, :, 1:]

    return neighbor_index


def square_distance(src, dst):
    """
    计算两个批量矩阵之间的平方距离
    :param src: 矩阵1，[B, N, C]
    :param dst: 矩阵2，[B, M, C]
    :return: [B, N, M]

    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def fps(xyz, n_samples):
    """
    最远采样法进行采样，返回采样点的索引
    :param xyz: [bs, n_point, channel]
    :param n_samples: 采样点数
    :return : [bs, n_samples]
    """
    device = xyz.device

    # xyz: [24, 1024, 3], B: batch_size, N: number of points, C: channels
    B, N, C = xyz.shape

    # 生成 B 行，n_samples 列的全为零的矩阵
    centroids = torch.zeros(B, n_samples, dtype=torch.long).to(device)

    # 生成 B 行，N 列的矩阵，每个元素为 1e10
    distance = torch.ones(B, N).to(device) * 1e10

    # 生成随机整数tensor，整数范围在[0，N)之间，包含0不包含N，矩阵各维度长度必须用元组传入，因此写成(B,)
    # 即生成初始点的随机索引
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)

    # 生成 [0, B) 整数序列
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(n_samples):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask].float()
        farthest = torch.max(distance, -1)[1]

    return centroids


def index_points(points, idx, is_reverse_nc=False):
    """
    将索引值替换为对应的数值
    :param points: [B, N, C] (维度数必须为3，最后一个为特征维度)
    :param idx: [B, D, E, F, ..., X]
    :param is_reverse_nc: 是否将 points 表示为 [B, C, N]
    :return: [B, D, E, F, ..., X, C], 如果 is_reverse_nc，[B, C, D, E, F, ..., X]
    """
    if is_reverse_nc:
        points = points.permute(0, 2, 1)

    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    new_points = points[batch_indices, idx, :]

    if is_reverse_nc:
        new_points = new_points.movedim(-1, 1)

    return new_points


def index_vals(vals, inds):
    '''
    将索引替换为对应的值
    :param vals: size([bs, n_item, n_channel])
    :param inds: size([bs, n_item, n_vals])(int， 索引矩阵，从vals里找到对应的数据填进去)
    :return: size([bs, n_item, n_vals])
    '''
    bs, n_item, n_vals = inds.size()

    # 生成0维度索引
    sequence = torch.arange(bs)
    sequence_expanded = sequence.unsqueeze(1)
    sequence_3d = sequence_expanded.tile((1, n_item))
    sequence_4d = sequence_3d.unsqueeze(-1)
    batch_indices = sequence_4d.repeat(1, 1, n_vals)

    # 生成1维度索引
    view_shape = [n_item, n_vals]
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = [bs, n_item, n_vals]
    repeat_shape[1] = 1
    channel_indices = torch.arange(n_item, dtype=torch.long).view(view_shape).repeat(repeat_shape)

    return vals[batch_indices, channel_indices, inds]


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def clear_log(folder_path, k=5):
    """
    遍历文件夹内的所有 .txt 文件，删除行数小于 k 的文件。

    :param folder_path: 要处理的文件夹路径
    :param k: 行数阈值，小于 k 的文件会被删除
    """
    os.makedirs(folder_path, exist_ok=True)

    for filename in os.listdir(folder_path):
        # 构造文件的完整路径
        file_path = os.path.join(folder_path, filename)

        # 检查是否为 .txt 文件
        if os.path.isfile(file_path) and filename.endswith('.txt'):
            try:
                # 统计文件的行数
                with open(file_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    num_lines = len(lines)

                # 如果行数小于 k，则删除文件
                if num_lines < k:
                    print(f"Deleting file: {file_path} (contains {num_lines} lines)")
                    os.remove(file_path)
            except Exception as e:
                # 捕获读取文件时的错误（如编码问题等）
                print(f"Error reading file {file_path}: {e}")


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


def get_log_floats(log_file: str) -> np.ndarray:
    # 定义正则表达式，匹配浮点数
    float_pattern = r'[-+]?\d*\.\d+|\d+\.\d*e[-+]?\d+'

    # 用于存储提取的浮点数
    floats = []

    # 打开文件并逐行读取
    with open(log_file, 'r') as file:
        for line in file:
            c_line = []
            # 查找所有匹配的浮点数
            matches = re.findall(float_pattern, line)
            for match in matches:
                # 将匹配的字符串转换为浮点数
                num = float(match)
                # 如果是整数，则跳过
                if num.is_integer():
                    continue
                # 将浮点数添加到列表中
                c_line.append(num)

            floats.append(c_line)

    floats = np.array(floats)
    return floats


def vis_cls_log(log_file: str, floats_idx_1=0, floats_idx_2=1):
    floats = get_log_floats(log_file)

    # 绘图
    plt.figure(figsize=(10, 5))
    n = floats.shape[0]
    x = np.arange(n)

    y1 = floats[:, floats_idx_1]
    y2 = floats[:, floats_idx_2]

    print(max(y2))

    # 绘制第一条折线
    plt.plot(x, y1, label='train ins acc', linestyle='-', color='b')

    # 绘制第二条折线
    plt.plot(x, y2, label='eval ins acc', linestyle='-', color='r')

    # 添加标题和标签
    plt.xlabel('epoch')
    plt.ylabel('accuracy')

    plt.legend()
    plt.grid(True, linestyle='--', color='gray', alpha=0.7)

    # 显示图形
    plt.show()


def vis_log_comp(log1: str, log2: str, comp_idx: int = 1) -> None:
    floats1 = get_log_floats(log1)
    floats2 = get_log_floats(log2)

    # 绘图
    plt.figure(figsize=(10, 5))
    x1 = np.arange(floats1.shape[0])
    y1 = floats1[:, comp_idx]
    x2 = np.arange(floats2.shape[0])
    y2 = floats2[:, comp_idx]

    # 绘制第一条折线
    plt.plot(x1, y1, label='log1', linestyle='-', color='b')

    # 绘制第二条折线
    plt.plot(x2, y2, label='log2', linestyle='-', color='r')

    # 添加标题和标签
    plt.xlabel('epoch')
    plt.ylabel('accuracy')

    plt.legend()
    plt.grid(True, linestyle='--', color='gray', alpha=0.7)

    # 显示图形
    plt.show()


def get_false_instance(all_preds: list, all_labels: list, all_indexes: list, dataset, save_path: str = './log/false_instance.txt'):
    """
    获取全部分类错误的实例路径
    :param all_preds:
    :param all_labels:
    :param all_indexes:
    :param dataset:
    :param save_path:
    :return:
    """
    # 将所有batch的预测和真实标签整合在一起
    all_preds = np.vstack(all_preds)  # 形状为 [n_samples, n_classes]
    all_labels = np.hstack(all_labels)  # 形状为 [n_samples]
    all_indexes = np.hstack(all_indexes)  # 形状为 [n_samples]

    # 确保all_labels, all_indexes中保存的为整形数据
    assert np.issubdtype(all_labels.dtype, np.integer) and np.issubdtype(all_indexes.dtype, np.integer)

    all_preds = np.argmax(all_preds, axis=1)  # -> [n_samples, ]
    incorrect_index = np.where(all_preds != all_labels)[0]
    incorrect_index = all_indexes[incorrect_index]
    incorrect_preds = all_preds[incorrect_index]

    if save_path is not None:
        with open(save_path, 'w', encoding='utf-8') as f:
            for c_idx, c_data_idx in enumerate(incorrect_index):
                # 找到分类错误的类型：
                false_class = ''
                for k, v in dataset.classes.items():
                    if incorrect_preds[c_idx] == v:
                        false_class = k
                        break

                f.write(dataset.datapath[c_data_idx][1] + ' | ' + false_class + '\n')

        print('save incorrect cls instance: ', save_path)


def sequence_extend(seq, side_extend):
    """

    :param seq: [bs, emb, seq_len]
    :param side_extend:
    :return: [bs, emb, len_seq + 2 * n_extend]
    """
    if isinstance(seq, np.ndarray):
        seq = torch.from_numpy(seq)

    former_n = seq[:, :, :side_extend]
    later_n = seq[:, :, seq.size(2) - side_extend:]

    # 反转末端
    later_n = torch.flip(later_n, [2])

    # 数组拼接 -> [bs, emb, len_seq + 2 * n_extend]
    seq_extend = torch.cat([later_n, seq, former_n], dim=2)

    return seq_extend


def count_parameters(model):
    """
    统计模型的参数量
    :param model:
    :return:
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def shuffle_along_dim(x, dim):
    """
    将一个pytorch的Tensor在指定维度打乱
    :param x:
    :param dim: 需要打乱的维度
    :return:
    """
    idx = torch.randperm(x.size(dim)).to(x.device)
    return x.index_select(dim, idx), idx


def recover_along_dim(x_shuffled: torch.Tensor, idx: torch.Tensor, dim: int):
    """
    还原在某一维度 dim 上被打乱的张量。输入的打乱 idx 需要是 shuffle_along_dim 函数输出的 idx

    参数:
        x_shuffled: 被打乱后的张量
        idx: 打乱使用的索引（即 torch.randperm 得到的）
        dim: 被打乱的维度

    返回:
        x_recovered: 还原后的张量
    """
    # 构造反向索引
    inv_idx = torch.empty_like(idx)
    inv_idx[idx] = torch.arange(len(idx), device=idx.device)

    # 使用 index_select 进行还原
    x_recovered = torch.index_select(x_shuffled, dim, inv_idx)
    return x_recovered


if __name__ == '__main__':

    test_tensor = torch.arange(10)
    test_tensor = test_tensor.view(2, 5)
    print(test_tensor)

    shuffle_tensor, idx__ = shuffle_along_dim(test_tensor, 1)
    print(shuffle_tensor)
    print(recover_along_dim(shuffle_tensor, idx__, 1))


    # asdasdas = r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt_all\Bolt\0a016b5f95eae21eaa9b95e7571d5bb3_1.txt'
    # std_to_tensor_img(np.loadtxt(asdasdas, delimiter=','))

    #
    # import global_defs
    # from matplotlib import pyplot as plt
    #
    # def vis_sketch_unified(root, n_stroke=global_defs.n_stk, n_stk_pnt=global_defs.n_stk_pnt, show_dot=False):
    #     """
    #     显示笔划与笔划点归一化后的草图
    #     """
    #     # -> [n, 4] col: 0 -> x, 1 -> y, 2 -> pen state (17: drawing, 16: stroke end), 3 -> None
    #     sketch_data = np.loadtxt(root, delimiter=',')
    #
    #     # 2D coordinates
    #     coordinates = sketch_data[:, :2]
    #
    #     # sketch mass move to (0, 0), x y scale to [-1, 1]
    #     coordinates = coordinates - np.expand_dims(np.mean(coordinates, axis=0), 0)  # 实测是否加expand_dims效果一样
    #     dist = np.max(np.sqrt(np.sum(coordinates ** 2, axis=1)), 0)
    #     coordinates = coordinates / dist
    #
    #     coordinates = torch.from_numpy(coordinates)
    #     coordinates = coordinates.view(n_stroke, n_stk_pnt, 2)
    #
    #     coordinates = coordinates.unsqueeze(0).repeat(5, 1, 1, 1)
    #
    #     coordinates = coordinates.view(5, n_stroke, n_stk_pnt * 2)
    #     idxs = torch.randint(0, n_stroke, (10, )).unsqueeze(0).repeat(5, 1)
    #
    #     print(idxs[0, :])
    #
    #     coordinates = index_points(coordinates, idxs)
    #     coordinates = coordinates.view(5, 10, n_stk_pnt, 2)
    #     coordinates = coordinates[0, :, :, :]
    #
    #
    #     for i in range(10):
    #         plt.plot(coordinates[i, :, 0].numpy(), -coordinates[i, :, 1].numpy())
    #
    #         if show_dot:
    #             plt.scatter(coordinates[i, :, 0].numpy(), -coordinates[i, :, 1].numpy())
    #
    #     # plt.axis('off')
    #     plt.show()
    #
    #
    # vis_sketch_unified(r'D:\document\DeepLearning\DataSet\unified_sketch_from_quickdraw\apple_stk16_stkpnt32\21.txt')
    #

    # vis_cls_log(r'C:\Users\ChengXi\Desktop\cad_dsample-2025-03-28 02-11-56.txt')

    # vis_log_comp(r'C:\Users\ChengXi\Desktop\cad_dsample-2025-03-27 11-46-11.txt', r'C:\Users\ChengXi\Desktop\cad_dsample-2025-03-28 02-11-56.txt')

    # vis_cls_log(r'C:\Users\ChengXi\Desktop\sd_qw_valid2.txt', 0, 1)

    pass
