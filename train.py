"""
PNG草图-图像对齐模型训练脚本
使用PNG格式的草图与图像进行对齐训练
"""

import os
import torch
from datetime import datetime

# 导入数据集和模型
from data import retrieval_datasets
from encoders import sbir_model_wrapper
from utils import trainer, utils
import options
from encoders import create_sketch_encoder


def main(args):

    save_str = (args.sketch_model + '_' +
                args.image_model + '_' +
                args.retrieval_mode + '_' +
                args.task + '_' +
                args.pair_mode)

    print('-----> model save name: ' + save_str + ' <-----')
    sketch_info = create_sketch_encoder.get_sketch_info(args.sketch_model)

    # 设置日志
    os.makedirs('log', exist_ok=True)
    logger = utils.get_log('./log/' + save_str + f'-{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.txt')
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 首先创建数据集划分（如果不存在）
    root = args.root_local if eval(args.local) else args.root_sever
    split_file = retrieval_datasets.get_split_file_name(sketch_info['format'], args.pair_mode, args.task)
    if sketch_info['format'] == 'vector':
        sketch_subdir = sketch_info['subdirs'][0]
        sketch_image_suffix = ('txt', 'jpg')
    else:
        sketch_subdir = sketch_info['subdirs'][1]
        sketch_image_suffix = ('png', 'jpg')

    if eval(args.is_create_fix_data_file) or not os.path.exists(split_file):
        logger.info("PNG草图数据集划分文件不存在，正在创建...")
        image_subdir = sketch_info['subdirs'][2]
        retrieval_datasets.create_dataset_split_file(
            save_root=split_file,
            sketch_root=os.path.join(root, sketch_subdir),
            image_root=os.path.join(root, image_subdir),
            sketch_image_suffix=sketch_image_suffix,
            is_multi_pair=True if args.pair_mode == 'multi_pair' else False,
            split_mode='ZS-SBIR' if args.task == 'zs_sbir' else 'SBIR',
            full_train=True
        )
    
    # 创建数据加载器
    train_loader, test_loader, dataset_info = retrieval_datasets.create_sketch_image_dataloaders(
        batch_size=args.bs,
        num_workers=args.num_workers,
        fixed_split_path=split_file,
        root=root,
        sketch_format=sketch_info['format'],
        vec_sketch_rep=sketch_info['rep'],
        sketch_image_subdirs=sketch_info['subdirs']
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
        sketch_model_name=args.sketch_model,
        image_model_name=args.image_model
    )
    model.to(device)
    
    # 参数统计
    param_counts = model.get_parameter_count()
    print(f"        模型参数统计:")
    print(f"  总参数: {param_counts['total']:,}")
    print(f"  可训练参数: {param_counts['trainable']:,}")
    print(f"  冻结参数: {param_counts['frozen']:,}")
    
    # 创建训练器
    check_point = utils.get_check_point(args.weight_dir, save_str)
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
    main(options.parse_args())


