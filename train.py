"""
PNG草图-图像对齐模型训练脚本
使用PNG格式的草图与图像进行对齐训练
"""
import os
import random
import json
import warnings
import numpy as np
import torch
from datetime import datetime
from colorama import Fore, Back, Style

# 导入数据集和模型
from data import retrieval_datasets
from encoders import sbir_model_wrapper
from utils import trainer, utils
from utils.swanlab_logging import init_swanlab, get_project_name
import options
from utils.training_policy import mix_training_split, early_stop_policy


def main(args):
    check_point = args.checkpoint or utils.get_check_point(args.weight_dir, utils.get_save_str(args))
    resume_requested = not args.eval_only and eval(args.is_load_ckpt)
    if args.checkpoint and (resume_requested or args.eval_only) and not os.path.isfile(check_point):
        raise FileNotFoundError(check_point)
    if resume_requested and os.path.isfile(check_point):
        checkpoint = torch.load(check_point, map_location='cpu', weights_only=True)
        saved_args = checkpoint.get('training_args')
        if saved_args:
            # Rebuild the same model, split and loaders; allow extending the epoch limit
            # and relocating the dataset/checkpoint on another machine.
            runtime_keys = {'epoch', 'checkpoint', 'weight_dir', 'local', 'root_local',
                            'root_sever', 'eval_only', 'is_load_ckpt', 'cache_mode',
                            'cache_dir', 'cache_rebuild',
                            'is_load_config', 'is_save_config', 'configs_dir', 'configs_name'}
            for key, value in saved_args.items():
                if key not in runtime_keys and hasattr(args, key):
                    setattr(args, key, value)
        if not checkpoint.get('swanlab') and not args.swanlab_run_id:
            raise RuntimeError('This checkpoint has no SwanLab run ID. It cannot automatically '
                               'resume the original experiment; specify --swanlab_run_id and '
                               '--swanlab_project for the original experiment.')
        if not saved_args:
            warnings.warn('Legacy checkpoint has no training arguments; use the original training options')
        del checkpoint
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_str = utils.get_save_str(args)
    print(Fore.BLACK + Back.CYAN + '-> model save name: ' + save_str + ' <-' + Style.RESET_ALL)
    encoder_info = options.get_encoder_info(args.sketch_model)

    # 设置日志
    os.makedirs('log', exist_ok=True)
    logger = utils.get_log('./log/' + save_str + f'-{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.txt')
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if eval(args.is_full_train) and not args.eval_only:
        raise ValueError('Fixed-size training requires --is_full_train=False; use --test_mix_ratio instead')

    # 预加载数据集
    pre_load = retrieval_datasets.preload_from_args(args)
    cache_preload = pre_load  # Preserve the full union before seed-dependent mixing.
    if not args.eval_only:
        original_count = len(pre_load.train_pairs)
        pre_load = mix_training_split(pre_load, args.test_mix_ratio, args.seed)
        mixed_count = int(original_count * args.test_mix_ratio)
        logger.info(f'Training samples: {original_count}; test samples: {mixed_count}; '
                    f'actual test ratio: {mixed_count / original_count:.6f}')
    root = args.root_local if eval(args.local) else args.root_sever
    stop_policy = early_stop_policy(root, args.task, eval(args.is_zero_shot), args.sketch_model)
    logger.info(f'Early stopping policy: {stop_policy}')
    logger.info(json.dumps(vars(args), ensure_ascii=False))

    # 创建数据加载器
    train_loader, test_loader = retrieval_datasets.create_sketch_image_dataloaders(
        batch_size=args.bs,
        num_workers=args.num_workers,
        pre_load=pre_load,
        sketch_format=encoder_info['sketch_format'],
        is_full_train=False,
        task=args.task, n_pos=args.n_pos, n_neg=args.n_neg,
        cache_mode=args.cache_mode, cache_dir=args.cache_dir,
        cache_rebuild=args.cache_rebuild, cache_preload=cache_preload
    )

    # 创建模型
    model = sbir_model_wrapper.create_sbir_model_wrapper(
        embed_dim=args.embed_dim,
        freeze_image_encoder=eval(args.is_freeze_image_encoder),
        freeze_sketch_backbone=eval(args.is_freeze_sketch_backbone),
        sketch_model_name=args.sketch_model,
        image_model_name=args.image_model,
        sketch_format=encoder_info['sketch_format'],
        pretrained=eval(args.pretrained) and not args.eval_only,
    )

    # 创建训练器
    model_trainer = trainer.SBIRTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        check_point=check_point,
        logger=logger,
        save_str=save_str,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epoch,
        task=args.task, eval_chunk_size=args.eval_chunk_size,
        early_stop=stop_policy
    )
    
    # 恢复训练（如果指定）
    loaded = model_trainer.load_checkpoint(check_point, args.eval_only or eval(args.is_load_ckpt),
                                          is_load_training_state=not args.eval_only)
    if args.eval_only:
        if not loaded:
            raise RuntimeError('Evaluation requires a valid checkpoint')
        model_trainer.validate_epoch()
        print(json.dumps(model_trainer.last_metrics, indent=2))
        logger.info(json.dumps(model_trainer.last_metrics))
        return model_trainer.last_metrics

    # 开始训练
    if loaded and model_trainer.swanlab_state is None:
        model_trainer.swanlab_state = {'id': args.swanlab_run_id,
                                      'project': get_project_name(args),
                                      'workspace': args.swanlab_workspace}
    run = init_swanlab(args, model_trainer.save_str, model_trainer.swanlab_state)
    try:
        model_trainer.swanlab_run = run
        if model_trainer.swanlab_state is None:
            model_trainer.swanlab_state = {'id': run.id, 'project': get_project_name(args),
                                          'workspace': args.swanlab_workspace}
        model_trainer.training_args = vars(args).copy()
        model_trainer.train()
    except BaseException:
        run.finish(state='crashed')
        raise
    else:
        run.finish()


if __name__ == '__main__':
    main(options.parse_args())


