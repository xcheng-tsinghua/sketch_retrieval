import numpy as np
import torch
import cv2
import random
import matplotlib.pyplot as plt


def s3_to_tensor_img(sketch, image_size=(224, 224), line_thickness=1, pen_up=0, coor_mode='ABS', save_path=None):
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
    :param pen_up:
    :return: list(image_size), 224, 224 为预训练的 vit 的图片大小
    """
    assert coor_mode in ['REL', 'ABS']
    width, height = image_size

    if isinstance(sketch, str):
        points_with_state = np.loadtxt(sketch, delimiter=',')
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

    return tensor_img


def s5_to_tensor_img(s5_tensor, coor_mode='REL', save_path=None):
    s5_tensor = s5_tensor.cpu().numpy()
    recov_s3 = s5_tensor[:, [0, 1, 2]]

    if coor_mode == 'REL':
        recov_s3[:, :2] = np.cumsum(recov_s3[:, :2], 0)

    tensor_img = s3_to_tensor_img(recov_s3, save_path=save_path)
    tensor_img.unsqueeze_(0)
    tensor_img = tensor_img.repeat(3, 1, 1)
    return tensor_img


def s3_file_to_s5(root, max_length=11*32, coor_mode='REL', is_shuffle_stroke=False, is_back_mask=True):
    """
    将草S3图转换为 S5 格式，(x, y, s1, s2, s3)
    默认存储绝对坐标
    :param root:
    :param max_length:
    :param coor_mode: ['ABS', 'REL'], 'ABS': absolute coordinate. 'REL': relative coordinate [(x,y), (△x, △y), (△x, △y), ...].
    :param is_shuffle_stroke: 是否打乱笔划
    :param is_back_mask:
    :return:
    """
    data_raw = np.loadtxt(root, delimiter=',')

    # 打乱笔划
    if is_shuffle_stroke:
        stroke_list = np.split(data_raw, np.where(data_raw[:, 2] == 0)[0] + 1)[:-1]
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


def vis_s3(s3_file, delimiter=','):
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


if __name__ == '__main__':
    as3_file = r'D:\document\DeepLearning\DataSet\sketch_retrieval\sketchy\sketch_s3_352\airplane\n02691156_196-5.txt'
    trans_save = r'C:\Users\ChengXi\Desktop\60mm20250708\rel_skh.png'

    raw_s3 = np.loadtxt(as3_file, delimiter=',')
    s5_tensor = s3_file_to_s5(as3_file, is_back_mask=False)

    # recov_s3 = s5_tensor[:, [0, 1, 2]]
    # recov_s3[:, :2] = np.cumsum(recov_s3[:, :2], 0)
    #
    # vis_s3(recov_s3)
    #
    # s5_tensor = s3_file_to_s5(as3_file, is_back_mask=False)
    #
    # s3_to_tensor_img(recov_s3, save_path=trans_save)

    s5_to_tensor_img(s5_tensor, save_path=trans_save)





