"""
PNG草图-图像对齐模型训练脚本
使用PNG格式的草图与图像进行对齐训练
"""

import os
import torch
import argparse
import logging

# 导入数据集和模型
from data.retrieval_datasets import create_png_sketch_dataloaders
from encoders.png_sketch_image_model import create_png_sketch_image_model
from utils import trainer
from dataset_split import create_dataset_splits_file


def parse_args():
    parser = argparse.ArgumentParser(description='训练PNG草图-图像对齐模型')
    parser.add_argument('--bs', type=int, default=100, help='批次大小')
    parser.add_argument('--epoch', type=int, default=100, help='最大训练轮数')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心')

    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--save_every', type=int, default=5, help='保存间隔')
    parser.add_argument('--embed_dim', type=int, default=512, help='嵌入维度')
    parser.add_argument('--is_freeze_image_encoder', type=str, choices=['True', 'False'], default='True', help='冻结图像编码器')
    parser.add_argument('--is_freeze_sketch_backbone', type=str, choices=['True', 'False'], default='False', help='冻结草图编码器主干网络')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载进程数')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--output_dir', type=str, default='model_trained', help='输出目录')
    parser.add_argument('--sketch_format', type=str, default='vector', choices=['vector', 'image'], help='使用矢量草图还是图片草图')
    parser.add_argument('--is_create_fix_data_file', type=str, choices=['True', 'False'], default='False', help='是否创建固定数据集划分文件')
    parser.add_argument('--is_load_ckpt', type=str, choices=['True', 'False'], default='False', help='是否加载检查点')
    parser.add_argument('--sketch_image_subdirs', type=tuple, default=('sketch_s3_352', 'sketch_png', 'photo'), help='[0]: vector_sketch, [1]: image_sketch, [2]: photo')
    parser.add_argument('--save_str', type=str, default='lstm_vit', help='保存名')

    parser.add_argument('--local', default='False', choices=['True', 'False'], type=str, help='是否本地运行')
    parser.add_argument('--root_sever', type=str, default=r'/opt/data/private/data_set/sketch_retrieval')
    parser.add_argument('--root_local', type=str, default=r'D:\document\DeepLearning\DataSet\sketch_retrieval\sketchy')

    args = parser.parse_args()
    return args


def main(args):

    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 首先创建数据集划分（如果不存在）
    root = args.root_local if eval(args.local) else args.root_sever

    if args.sketch_format == 'vector':
        sketch_subdir = args.sketch_image_subdirs[0]
        split_file = './data/fixed_splits/vec_sketch_image_dataset_splits.pkl'
        sketch_image_suffix = ('txt', 'jpg')
    else:
        sketch_subdir = args.sketch_image_subdirs[1]
        split_file = './data/fixed_splits/png_sketch_image_dataset_splits.pkl'
        sketch_image_suffix = ('png', 'jpg')

    if eval(args.is_create_fix_data_file) or not os.path.exists(split_file):
        logger.info("PNG草图数据集划分文件不存在，正在创建...")
        image_subdir = args.sketch_image_subdirs[2]
        create_dataset_splits_file(split_file, os.path.join(root, sketch_subdir), os.path.join(root, image_subdir), sketch_image_suffix)
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    train_loader, test_loader, dataset_info = create_png_sketch_dataloaders(
        batch_size=args.bs,
        num_workers=args.num_workers,
        fixed_split_path=split_file,
        root=root,
        sketch_format=args.sketch_format
    )
    
    logger.info(f"数据集信息:")
    logger.info(f"  训练集: {dataset_info['train_info']['total_pairs']} 对")
    logger.info(f"  测试集: {dataset_info['test_info']['total_pairs']} 对")
    logger.info(f"  类别数: {dataset_info['category_info']['num_categories']}")
    
    # 创建模型
    logger.info("创建PNG草图-图像对齐模型...")
    model = create_png_sketch_image_model(
        embed_dim=args.embed_dim,
        freeze_image_encoder=eval(args.is_freeze_image_encoder),
        freeze_sketch_backbone=eval(args.is_freeze_sketch_backbone),
        sketch_format=args.sketch_format
    )
    model.to(device)
    
    # 参数统计
    param_counts = model.get_parameter_count()
    logger.info(f"模型参数统计:")
    logger.info(f"  总参数: {param_counts['total']:,}")
    logger.info(f"  可训练参数: {param_counts['trainable']:,}")
    logger.info(f"  冻结参数: {param_counts['frozen']:,}")
    
    # 创建训练器
    # model_trainer = trainer.PNGSketchImageTrainer(
    #     model=model,
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    #     device=device,
    #     output_dir=args.output_dir,
    #     logger=logger,
    #     learning_rate=args.learning_rate,
    #     weight_decay=args.weight_decay,
    #     warmup_epochs=args.warmup_epochs,
    #     max_epochs=args.epoch,
    #     patience=args.patience,
    #     save_every=args.save_every
    # )

    check_point = os.path.join(args.output_dir, args.save_str + '.pth')
    model_trainer = trainer.PNGSketchImageTrainer2(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        check_point=check_point,
        logger=logger,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.epoch,
        save_every=args.save_every
    )
    
    # 恢复训练（如果指定）
    if eval(args.is_load_ckpt):
        model_trainer.load_checkpoint(check_point)
    
    # 开始训练
    model_trainer.train()


if __name__ == '__main__':
    main(parse_args())


