"""
PNG草图-图像对齐模型训练脚本
使用PNG格式的草图与图像进行对齐训练
"""

import os
import torch
import argparse
from datetime import datetime

# 导入数据集和模型
from data import retrieval_datasets
from encoders import sbir_model_wrapper
from utils import trainer, utils


def parse_args():
    parser = argparse.ArgumentParser(description='训练PNG草图-图像对齐模型')
    parser.add_argument('--bs', type=int, default=200, help='批次大小')
    parser.add_argument('--epoch', type=int, default=1000, help='最大训练轮数')

    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--embed_dim', type=int, default=512, help='嵌入维度')
    parser.add_argument('--is_freeze_image_encoder', type=str, choices=['True', 'False'], default='True', help='冻结图像编码器')
    parser.add_argument('--is_freeze_sketch_backbone', type=str, choices=['True', 'False'], default='False', help='冻结草图编码器主干网络')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载进程数')
    parser.add_argument('--weight_dir', type=str, default='model_trained', help='输出目录')
    parser.add_argument('--vec_sketch_type', type=str, default='S5', choices=['STK_11_32', 'S5'], help='矢量草图格式')
    parser.add_argument('--is_create_fix_data_file', type=str, choices=['True', 'False'], default='False', help='是否创建固定数据集划分文件')
    parser.add_argument('--pair_mode', type=str, default='multi_pair', choices=['multi_pair', 'single_pair'], help='图片与草图是一对一还是一对多')
    parser.add_argument('--is_load_ckpt', type=str, choices=['True', 'False'], default='True', help='是否加载检查点')
    parser.add_argument('--sketch_image_subdirs', type=tuple, default=('sketch_s3_352', 'sketch_png', 'photo'), help='[0]: vector_sketch, [1]: image_sketch, [2]: photo')  # sketch_stk11_stkpnt32, sketch_s3_352
    parser.add_argument('--task', type=str, default='zs_sbir', choices=['sbir', 'zs_sbir'], help='检索任务类型')
    parser.add_argument('--sketch_format', type=str, default='vector', choices=['vector', 'image'], help='使用矢量草图还是图片草图')
    parser.add_argument('--save_str', type=str, default='lstm_vit_fgzssbir_mulpair', help='保存名')

    parser.add_argument('--local', default='False', choices=['True', 'False'], type=str, help='是否本地运行')
    parser.add_argument('--root_sever', type=str, default=r'/opt/data/private/data_set/sketch_retrieval')
    parser.add_argument('--root_local', type=str, default=r'D:\document\DeepLearning\DataSet\sketch_retrieval\sketchy')

    args = parser.parse_args()
    return args


def main(args):
    print('-----> model save name: ' + args.save_str + ' <-----')

    # 设置日志
    os.makedirs('log', exist_ok=True)
    logger = utils.get_log('./log/' + args.save_str + f'-{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.txt')
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 首先创建数据集划分（如果不存在）
    root = args.root_local if eval(args.local) else args.root_sever
    split_file = retrieval_datasets.get_split_file_name(args.sketch_format, args.pair_mode, args.task)
    if args.sketch_format == 'vector':
        sketch_subdir = args.sketch_image_subdirs[0]
        sketch_image_suffix = ('txt', 'jpg')
    else:
        sketch_subdir = args.sketch_image_subdirs[1]
        sketch_image_suffix = ('png', 'jpg')

    if eval(args.is_create_fix_data_file) or not os.path.exists(split_file):
        logger.info("PNG草图数据集划分文件不存在，正在创建...")
        image_subdir = args.sketch_image_subdirs[2]
        retrieval_datasets.create_dataset_split_file(
            save_root=split_file,
            sketch_root=os.path.join(root, sketch_subdir),
            image_root=os.path.join(root, image_subdir),
            sketch_image_suffix=sketch_image_suffix,
            is_multi_pair=True if args.pair_mode == 'multi_pair' else False,
            split_mode='ZS-SBIR' if args.task == 'zs_sbir' else 'SBIR'
        )
    
    # 创建数据加载器
    # logger.info("创建数据加载器...")
    train_loader, test_loader, dataset_info = retrieval_datasets.create_sketch_image_dataloaders(
        batch_size=args.bs,
        num_workers=args.num_workers,
        fixed_split_path=split_file,
        root=root,
        sketch_format=args.sketch_format,
        vec_sketch_type=args.vec_sketch_type,
        sketch_image_subdirs=args.sketch_image_subdirs
    )
    
    print(f"        数据集信息:")
    print(f"  训练集: {dataset_info['train_info']['total_pairs']} 对")
    print(f"  测试集: {dataset_info['test_info']['total_pairs']} 对")
    print(f"  类别数: {dataset_info['category_info']['num_categories']}")
    
    # 创建模型
    print("         创建草图-图像对齐模型...")
    model = sbir_model_wrapper.create_sbir_model_wrapper(
        embed_dim=args.embed_dim,
        freeze_image_encoder=eval(args.is_freeze_image_encoder),
        freeze_sketch_backbone=eval(args.is_freeze_sketch_backbone),
        sketch_format=args.sketch_format
    )
    model.to(device)
    
    # 参数统计
    param_counts = model.get_parameter_count()
    print(f"        模型参数统计:")
    print(f"  总参数: {param_counts['total']:,}")
    print(f"  可训练参数: {param_counts['trainable']:,}")
    print(f"  冻结参数: {param_counts['frozen']:,}")
    
    # 创建训练器
    check_point = utils.get_check_point(args.weight_dir, args.save_str)
    model_trainer = trainer.SBIRTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        check_point=check_point,
        logger=logger,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epoch,
        dataset_info=dataset_info,
        log_dir='log'
    )
    
    # 恢复训练（如果指定）
    if eval(args.is_load_ckpt):
        model_trainer.load_checkpoint(check_point)
    
    # 开始训练
    model_trainer.train()


if __name__ == '__main__':
    main(parse_args())


