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
    save_str = utils.get_save_str(args)
    print('-----> model save name: ' + save_str + ' <-----')

    sketch_info = create_sketch_encoder.get_sketch_info(args.sketch_model)

    # 设置日志
    os.makedirs('log', exist_ok=True)
    logger = utils.get_log('./log/' + save_str + f'-{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.txt')
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 预加载数据集
    root = args.root_local if eval(args.local) else args.root_sever

    if sketch_info['format'] == 'vector':
        sketch_subdir = sketch_info['subdirs'][0]
        sketch_image_suffix = ('txt', 'jpg')
    else:
        sketch_subdir = sketch_info['subdirs'][1]
        # sketch_image_suffix = ('png', 'jpg')
        sketch_image_suffix = ('png', 'png')

    image_subdir = sketch_info['subdirs'][2]

    if eval(args.local):
        sketch_root = r'D:\document\DeepLearning\DataSet\草图项目\retrieval_cad\sketch_ai'
        image_root = r'D:\document\DeepLearning\DataSet\草图项目\retrieval_cad\sketch_png'
    else:
        sketch_root = r'/opt/data/private/data_set/sketch_retrieval/retrieval_cad/sketch_ai'
        image_root = r'/opt/data/private/data_set/sketch_retrieval/retrieval_cad/sketch_png'

    pre_load = retrieval_datasets.DatasetPreloadSketchProj(
        sketch_root=sketch_root,
        image_root=image_root,
        sketch_image_suffix=sketch_image_suffix,
        is_multi_pair=True if args.pair_mode == 'multi_pair' else False,
        split_mode='ZS-SBIR' if args.task == 'zs_sbir' else 'SBIR',
        is_full_train=eval(args.is_full_train)
    )

    # 创建数据加载器
    train_set, test_set, train_loader, test_loader, dataset_info = retrieval_datasets.create_sketch_image_dataloaders(
        batch_size=args.bs,
        num_workers=args.num_workers,
        pre_load=pre_load,
        root=root,
        sketch_format=sketch_info['format'],
        vec_sketch_rep=sketch_info['rep'],
        sketch_image_subdirs=sketch_info['subdirs'],
        is_back_dataset=True
    )

    # 创建模型
    print(" -> 创建草图-图像对齐模型...")
    model = sbir_model_wrapper.create_sbir_model_wrapper(
        embed_dim=args.embed_dim,
        freeze_image_encoder=eval(args.is_freeze_image_encoder),
        freeze_sketch_backbone=eval(args.is_freeze_sketch_backbone),
        sketch_model_name=args.sketch_model,
        image_model_name=args.image_model
    )
    model.to(device)

    # 创建训练器
    check_point = utils.get_check_point(args.weight_dir, save_str)
    model_trainer = trainer.SBIRTrainer(
        model=model,
        train_set=train_set,
        test_set=test_set,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        check_point=check_point,
        logger=logger,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epoch,
        dataset_info=dataset_info,
        log_dir='log',
        retrieval_mode=args.retrieval_mode,
        # stop_val=stop_val
    )
    
    # 恢复训练（如果指定）
    if eval(args.is_load_ckpt):
        model_trainer.load_checkpoint(check_point)
    else:
        print('不加载权重，从零开始训练模型')
    
    # 开始训练
    if eval(args.is_vis):
        model_trainer.vis_fea_cluster()
    else:
        model_trainer.train()


if __name__ == '__main__':
    main(options.parse_args())


