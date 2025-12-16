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


def main(args):
    save_str = utils.get_save_str(args)
    print('-----> model save name: ' + save_str + ' <-----')
    encoder_info = options.get_encoder_info(args.sketch_model)

    # 设置日志
    os.makedirs('log', exist_ok=True)
    logger = utils.get_log('./log/' + save_str + f'-{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.txt')
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 预加载数据集
    root = args.root_local if eval(args.local) else args.root_sever

    pre_load = retrieval_datasets.DatasetPreload(
        sketch_root=os.path.join(root, encoder_info['sketch_subdir']),
        image_root=os.path.join(root, encoder_info['image_subdir']),
        sketch_suffix=encoder_info['sketch_suffix'],
        image_suffix=encoder_info['image_suffix'],
        is_multi_pair=True if args.pair_mode == 'multi_pair' else False,
        split_mode=args.task,
        is_full_train=eval(args.is_full_train),
        multi_sketch_split=args.multi_sketch_split
    )

    # 创建数据加载器
    train_loader, test_loader = retrieval_datasets.create_sketch_image_dataloaders(
        batch_size=args.bs,
        num_workers=args.num_workers,
        pre_load=pre_load,
        sketch_format=encoder_info['sketch_format'],
        back_mode='train'
    )

    # 创建模型
    model = sbir_model_wrapper.create_sbir_model_wrapper(
        embed_dim=args.embed_dim,
        freeze_image_encoder=eval(args.is_freeze_image_encoder),
        freeze_sketch_backbone=eval(args.is_freeze_sketch_backbone),
        sketch_model_name=args.sketch_model,
        image_model_name=args.image_model,
        sketch_format=encoder_info['sketch_format'],
    )
    model.to(device)

    # 创建训练器
    check_point = utils.get_check_point(args.weight_dir, save_str)
    model_trainer = trainer.SBIRTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        check_point=check_point,
        logger=logger,
        retrieval_mode=args.retrieval_mode,
        save_str=save_str,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epoch,
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

    # acc_1_idxes, acc_5_idxes = model_trainer.get_acc_files_epoch()
    # logger.info('acc_1_sketches:')
    #
    # for c_idx in acc_1_idxes:
    #     c_sketch_file, _ = test_set.get_file_pair_by_index(c_idx)
    #     logger.info(c_sketch_file)
    #
    # logger.info('acc_5_sketches:')
    #
    # for c_idx in acc_5_idxes:
    #     c_sketch_file, _ = test_set.get_file_pair_by_index(c_idx)
    #     logger.info(c_sketch_file)


if __name__ == '__main__':
    main(options.parse_args())


