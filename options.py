import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # training & visualizing
    parser.add_argument('--bs', type=int, default=20, help='批次大小')  # 200
    parser.add_argument('--embed_dim', type=int, default=512, help='嵌入维度')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载进程数')
    parser.add_argument('--weight_dir', type=str, default='model_trained', help='输出目录')
    parser.add_argument('--log_dir', type=str, default='log', help='日志目录')

    parser.add_argument('--sketch_model', type=str, default='sdgraph', choices=['vit', 'lstm', 'sdgraph', 'sketch_transformer', 'gru'], help='草图Encoder的名字')
    parser.add_argument('--image_model', type=str, default='vit', choices=['vit', ], help='使用矢量草图还是图片草图')
    parser.add_argument('--retrieval_mode', type=str, default='cl', choices=['cl', 'fg'], help='cl: category-level, fg: fine-grained')
    parser.add_argument('--task', type=str, default='sbir', choices=['sbir', 'zs_sbir'], help='检索任务类型')
    parser.add_argument('--pair_mode', type=str, default='single_pair', choices=['multi_pair', 'single_pair'], help='图片与草图是一对一还是一对多')

    parser.add_argument('--local', default='False', choices=['True', 'False'], type=str, help='是否本地运行')
    parser.add_argument('--root_sever', type=str, default=r'/opt/data/private/data_set/sketch_retrieval/SketchX_Shoe_ChairV2')
    parser.add_argument('--root_local', type=str, default=r'D:\document\DeepLearning\DataSet\sketch_retrieval\SketchX_Shoe_ChairV2')  # r'D:\document\DeepLearning\DataSet\sketch_retrieval\sketchy'

    # training
    parser.add_argument('--epoch', type=int, default=1000, help='最大训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--is_freeze_image_encoder', type=str, choices=['True', 'False'], default='True', help='冻结图像编码器')
    parser.add_argument('--is_freeze_sketch_backbone', type=str, choices=['True', 'False'], default='False', help='冻结草图编码器主干网络')
    parser.add_argument('--is_load_ckpt', type=str, choices=['True', 'False'], default='True', help='是否加载检查点')
    parser.add_argument('--class_name', type=str, default='ChairV2', choices=['ChairV2', 'ShoeV2'], help='附带的字符串')

    parser.add_argument('--is_vis', type=str, choices=['True', 'False'], default='False', help='是否可视化草图特征，可视化后不进行训练')
    parser.add_argument('--is_full_train', type=str, choices=['True', 'False'], default='True', help='使用全部数据训练')

    # visualizing
    parser.add_argument('--output_dir', type=str, default='vis_results', help='可视化存储目录')
    parser.add_argument('--n_vis_images', type=int, default=5, help='每张草图查询的图片数')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    f1, f2, fout = r'C:\Users\ChengXi\Desktop\60mm20250708\acc-1.txt', r'C:\Users\ChengXi\Desktop\60mm20250708\acc-5.txt', r'C:\Users\ChengXi\Desktop\60mm20250708\acc-5-filter.txt'

    with open(f1, 'r', encoding='utf-8') as fp:
        set1 = set(line.rstrip('\n') for line in fp)

    with open(f2, 'r', encoding='utf-8') as fp:
        set2 = set(line.rstrip('\n') for line in fp)

    # 并集 - 交集 ＝ 对称差
    sym_diff = sorted((set1 | set2) - (set1 & set2))

    with open(fout, 'w', encoding='utf-8') as fp:
        for line in sym_diff:
            fp.write(line + '\n')

    print(f"对称差已写入 {fout}，共 {len(sym_diff)} 行。")




