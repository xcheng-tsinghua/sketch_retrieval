import argparse
import re
import json


def parse_args():
    parser = argparse.ArgumentParser()

    # training & visualizing
    parser.add_argument('--bs', type=int, default=20, help='批次大小')  # 200
    parser.add_argument('--embed_dim', type=int, default=512, help='嵌入维度')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载进程数')
    parser.add_argument('--weight_dir', type=str, default='model_trained', help='输出目录')

    parser.add_argument('--sketch_model', type=str, default='bidir_lstm', choices=['vit', 'lstm', 'bidir_lstm', 'sdgraph', 'sdgraph_attn', 'sketch_transformer', 'gru', 'bidir_gru'], help='草图Encoder的名字')
    parser.add_argument('--image_model', type=str, default='vit', choices=['vit', ], help='--')
    parser.add_argument('--retrieval_mode', type=str, default='fg', choices=['cl', 'fg'], help='cl: category-level, fg: fine-grained')
    parser.add_argument('--task', type=str, default='sbir', choices=['sbir', 'zs_sbir'], help='检索任务类型')
    parser.add_argument('--pair_mode', type=str, default='multi_pair', choices=['multi_pair', 'single_pair'], help='图片与草图是一对一还是一对多')
    parser.add_argument('--multi_sketch_split', type=str, default='_', help='一张图片绘制多个草图时，标号分隔符')  # 对于 QMUL 是 '_‘, 对于 sketchy 是 '-'

    parser.add_argument('--local', default='False', choices=['True', 'False'], type=str, help='是否本地运行')
    parser.add_argument('--root_sever', type=str, default=r'/opt/data/private/data_set/sketch_retrieval/qmul_v2_fit/chair')  # r'/opt/data/private/data_set/sketch_retrieval/retrieval_cad'
    parser.add_argument('--root_local', type=str, default=r'D:\document\DeepLearning\DataSet\sketch_retrieval\qmul_v2_fit\chair')  # r'D:\document\DeepLearning\DataSet\sketch_retrieval\sketchy'
    parser.add_argument('--add_str', type=str, default='chair', help='其它描述字符串')

    # training
    parser.add_argument('--epoch', type=int, default=300, help='最大训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--is_freeze_image_encoder', type=str, choices=['True', 'False'], default='True', help='冻结图像编码器')
    parser.add_argument('--is_freeze_sketch_backbone', type=str, choices=['True', 'False'], default='False', help='冻结草图编码器主干网络')
    parser.add_argument('--is_load_ckpt', type=str, choices=['True', 'False'], default='True', help='是否加载检查点')

    parser.add_argument('--is_vis', type=str, choices=['True', 'False'], default='False', help='是否可视化草图特征，可视化后不进行训练')
    parser.add_argument('--is_full_train', type=str, choices=['True', 'False'], default='False', help='使用全部数据训练')

    # visualizing
    parser.add_argument('--output_dir', type=str, default='vis_results', help='可视化存储目录')
    parser.add_argument('--n_vis_images', type=int, default=5, help='每张草图查询的图片数')
    parser.add_argument('--vis_mode', type=str, default='summary', choices=['summary', 'example', 'cluster', ], help='---')

    args = parser.parse_args()
    return args


supported_encoders = {
    'vit': {
        'sketch_format': 'fmt: s3 -> img',
        'sketch_subdir': 'sketch_s3_352',
        'image_subdir': 'photo',
        'sketch_suffix': 'txt',
        'image_suffix': 'png',
    },

    'sdgraph': {
        'sketch_format': 'fmt: stk, n_stk: 12, n_stk_pnt: 32',
        'sketch_subdir': 'sketch_stk12_stkpnt32',
        'image_subdir': 'photo',
        'sketch_suffix': 'txt',
        'image_suffix': 'png',
    },

    'sdgraph_attn': {
        'sketch_format': 'fmt: stk, n_stk: 12, n_stk_pnt: 32',
        'sketch_subdir': 'sketch_stk12_stkpnt32_autospace',
        'image_subdir': 'photo',
        'sketch_suffix': 'txt',
        'image_suffix': 'png',
    },

    'lstm': {
        'sketch_format': 'fmt: s5, max_length: 352',
        'sketch_subdir': 'sketch_s3_352',
        'image_subdir': 'photo',
        'sketch_suffix': 'txt',
        'image_suffix': 'png',
    },

    'bidir_lstm': {
        'sketch_format': 'fmt: s5, max_length: 352',
        'sketch_subdir': 'sketch_s3_352',
        'image_subdir': 'photo',
        'sketch_suffix': 'txt',
        'image_suffix': 'png',
    },

    'gru': {
        'sketch_format': 'fmt: s5, max_length: 352',
        'sketch_subdir': 'sketch_s3_352',
        'image_subdir': 'photo',
        'sketch_suffix': 'txt',
        'image_suffix': 'png',
    },

    'bidir_gru': {
        'sketch_format': 'fmt: s5, max_length: 352',
        'sketch_subdir': 'sketch_s3_352',
        'image_subdir': 'photo',
        'sketch_suffix': 'txt',
        'image_suffix': 'png',
    },

    'sketch_transformer': {
        'sketch_format': 'fmt: s5, max_length: 352',
        'sketch_subdir': 'sketch_s3_352',
        'image_subdir': 'photo',
        'sketch_suffix': 'txt',
        'image_suffix': 'png',
    }

}


def get_encoder_info(sketch_model: str):
    """
    根据草图模型名获取其使用的草图类别及其他信息
    """
    sketch_format = supported_encoders[sketch_model]
    return sketch_format


def parse_sketch_format(format_str):
    def _parse_value(_val_str: str):
        """
        将整形转化为 int，浮点数转化为 float
        """
        _val_str = _val_str.strip()

        # int（必须放在 float 之前）
        if re.fullmatch(r'[+-]?\d+', _val_str):
            return int(_val_str)

        # float
        if re.fullmatch(r'[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?', _val_str):
            return float(_val_str)

        # 其它保持字符串
        return _val_str

    pairs = re.findall(r'(\w+)\s*:\s*([^,]+)', format_str)
    format_dict = {k: _parse_value(v) for k, v in pairs}
    return format_dict


if __name__ == '__main__':

    # f1, f2, fout = r'C:\Users\ChengXi\Desktop\60mm20250708\acc-1.txt', r'C:\Users\ChengXi\Desktop\60mm20250708\acc-5.txt', r'C:\Users\ChengXi\Desktop\60mm20250708\acc-5-filter.txt'
    #
    # with open(f1, 'r', encoding='utf-8') as fp:
    #     set1 = set(line.rstrip('\n') for line in fp)
    #
    # with open(f2, 'r', encoding='utf-8') as fp:
    #     set2 = set(line.rstrip('\n') for line in fp)
    #
    # # 并集 - 交集 ＝ 对称差
    # sym_diff = sorted((set1 | set2) - (set1 & set2))
    #
    # with open(fout, 'w', encoding='utf-8') as fp:
    #     for line in sym_diff:
    #         fp.write(line + '\n')
    #
    # print(f"对称差已写入 {fout}，共 {len(sym_diff)} 行。")

    sdgraph_path = './log/revl_ins_sdgraph_vit_fg_sbir_shoe.json'
    vit_path = './log/revl_ins_vit_vit_fg_sbir_shoe.json'
    lstm_path = './log/revl_ins_bidir_lstm_vit_fg_sbir_shoe.json'

    with open(sdgraph_path, encoding='utf-8') as f:
        sdgraph_dict_chair: dict = json.load(f)

    with open(vit_path, encoding='utf-8') as f:
        vit_dict_chair: dict = json.load(f)

    with open(lstm_path, encoding='utf-8') as f:
        lstm_dict_chair: dict = json.load(f)

    # result = [x for x in arr1 if x not in set(arr2)]

    sdgraph_dict_chair = sdgraph_dict_chair["top_1"]
    vit_dict_chair = vit_dict_chair["top_1"]
    lstm_dict_chair = lstm_dict_chair["top_1"]

    other_all = vit_dict_chair + lstm_dict_chair

    sdgraph_dict_chair_filter = [x for x in sdgraph_dict_chair if x not in other_all]

    for c_val in sdgraph_dict_chair_filter:
        print(f'\'{c_val}\',')

    pass
