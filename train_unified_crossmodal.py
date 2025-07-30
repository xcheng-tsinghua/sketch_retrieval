"""
使用统一跨模态模型的草图-图片对齐训练脚本
基于ULIP_models结构优化
"""
import argparse
import numpy as np
import torch
from tqdm import tqdm
from data.SLMDataset import SketchImageDataset
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import json
import math

# 导入统一模型
from encoders.unified_crossmodal_model import create_sketch_image_model, get_loss, get_metric_names


def save_checkpoint(epoch, model, optimizer, scheduler, best_loss, save_dir, is_best=False):
    """保存训练检查点（统一格式，参照ULIP）"""
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),  # 包含logit_scale
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss
    }
    
    # 保存当前检查点
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # 如果是最佳模型，额外保存
    if is_best:
        best_path = os.path.join(save_dir, 'best_checkpoint.pth')
        torch.save(checkpoint, best_path)
    
    print(f'检查点已保存: {checkpoint_path}')
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """加载训练检查点"""
    if not os.path.exists(checkpoint_path):
        print(f'检查点文件不存在: {checkpoint_path}')
        return 0, float('inf')
    
    print(f'加载检查点: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    
    # 加载模型状态（包含logit_scale）
    model.load_state_dict(checkpoint['state_dict'])
    
    # 加载优化器和调度器状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
    best_loss = checkpoint['best_loss']
    
    print(f'成功加载检查点，从epoch {start_epoch}开始，最佳损失: {best_loss:.4f}')
    return start_epoch, best_loss


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    """余弦退火学习率调度器（带预热）"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def parse_args():
    parser = argparse.ArgumentParser('Unified Sketch-Image Alignment Training')
    
    # 数据参数
    parser.add_argument('--sketch_root', type=str, 
                        default=r'E:\Master\Experiment\data\stroke-normal',
                        help='sketch data root path')
    parser.add_argument('--image_root', type=str,
                        default=r'E:\Master\Experiment\data\photo', 
                        help='image data root path')
    parser.add_argument('--n_sketch_points', type=int, default=256, help='number of sketch points')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='minimum learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='warmup epochs')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay')
    
    # 模型参数
    parser.add_argument('--embed_dim', type=int, default=512, help='embedding dimension')
    parser.add_argument('--pretrained_image_path', type=str,
                        default='./weights/weight_image_encoder.pth',
                        help='pretrained image encoder path')
    
    # 断点续训参数
    parser.add_argument('--resume', type=str, default='', help='path to resume checkpoint')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch for resume training')
    
    # 其他参数
    parser.add_argument('--output_dir', type=str, default='./outputs', help='output directory')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
    
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    print(f'训练参数: {vars(args)}')
    
    # 创建保存目录
    if args.resume:
        # 从checkpoint路径推断保存目录
        save_dir = os.path.dirname(args.resume)
        print(f'断点续训模式，使用现有目录: {save_dir}')
    else:
        # 新训练，创建新目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(args.output_dir, f'sketch_image_alignment_{timestamp}')
        os.makedirs(save_dir, exist_ok=True)
        print(f'新训练模式，创建目录: {save_dir}')
    
    # 保存训练参数
    args_dict = vars(args)
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(args_dict, f, indent=2)
    
    # 创建TensorBoard writer
    log_dir = os.path.join(save_dir, 'logs')
    writer = SummaryWriter(log_dir)
    print(f'TensorBoard日志保存在: {log_dir}')
    
    # 创建数据集
    print('创建数据集...')
    train_dataset = SketchImageDataset(
        n_skh_points=args.n_sketch_points,
        is_train=True
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        drop_last=True
    )
    
    print(f'数据集大小: {len(train_dataset)}, 批次数: {len(train_loader)}')
    
    # 创建统一模型
    print('创建统一跨模态模型...')
    model = create_sketch_image_model(
        sketch_points=args.n_sketch_points,
        embed_dim=args.embed_dim,
        pretrained_image_path=args.pretrained_image_path
    ).to(device)
    
    # 只训练草图编码器和logit_scale，冻结图片编码器
    trainable_params = []
    frozen_params = []
    for name, param in model.named_parameters():
        if 'vision_model' in name or 'image_projection' in name:
            param.requires_grad = False
            frozen_params.append(name)
        else:
            trainable_params.append(param)
            print(f'训练参数: {name}')
    
    print(f'冻结参数数量: {len(frozen_params)}')
    print(f'训练参数数量: {len(trainable_params)}')
    print(f'总参数数量: {sum(p.numel() for p in model.parameters())}')
    print(f'可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    
    # 优化器和调度器
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=args.learning_rate, 
        betas=(0.9, 0.98),
        eps=1e-08, 
        weight_decay=args.weight_decay
    )
    
    # 使用余弦退火调度器
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps, args.min_lr
    )
    
    # 损失函数
    criterion = get_loss()
    
    # 断点续训逻辑
    start_epoch = 0
    best_loss = float('inf')
    global_step = 0
    
    if args.resume:
        start_epoch, best_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )
        # 如果用户指定了start_epoch，使用用户指定的值
        if args.start_epoch > 0:
            start_epoch = args.start_epoch
            print(f'用户指定从epoch {start_epoch}开始')
        
        # 计算global_step
        global_step = start_epoch * len(train_loader)
        print(f'恢复训练，global_step从 {global_step} 开始')
    else:
        print('开始新的训练...')
    
    # logit_scale约束（参照ULIP）
    def clamp_logit_scale():
        with torch.no_grad():
            model.logit_scale.data.clamp_(0, 4.6052)  # exp(4.6052) ≈ 100
    
    # 训练循环
    print(f'训练范围: epoch {start_epoch} 到 {args.epochs-1}')
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        
        epoch_loss = []
        epoch_acc = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for batch_idx, (sketch_data, sketch_mask, image_data, categories) in enumerate(pbar):
            # 移动数据到设备
            sketch_data = sketch_data.to(device)
            sketch_mask = sketch_mask.to(device)
            image_data = image_data.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播（使用统一模型）
            outputs = model(sketch_data, sketch_mask, image_data)
            sketch_embed = outputs['sketch_embed']
            image_embed = outputs['image_embed']
            logit_scale = outputs['logit_scale']
            
            # 计算损失
            loss_dict = criterion(sketch_embed, image_embed, logit_scale)
            loss = loss_dict['loss']
            acc = loss_dict['sketch_image_acc']
            
            # 反向传播
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # 约束logit_scale（参照ULIP）
            clamp_logit_scale()
            
            # 记录
            epoch_loss.append(loss.item())
            epoch_acc.append(acc.item())
            
            # TensorBoard记录（每个batch）
            writer.add_scalar('Train/Loss_Batch', loss.item(), global_step)
            writer.add_scalar('Train/Accuracy_Batch', acc.item(), global_step)
            writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('Train/Logit_Scale', logit_scale.item(), global_step)
            global_step += 1
            
            # 更新进度条
            if batch_idx % 100 == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{acc.item():.2f}%',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}',
                    'logit_scale': f'{logit_scale.item():.2f}'
                })
        
        # 计算epoch平均值
        avg_loss = np.mean(epoch_loss)
        avg_acc = np.mean(epoch_acc)
        
        # TensorBoard记录（每个epoch）
        writer.add_scalar('Train/Loss_Epoch', avg_loss, epoch)
        writer.add_scalar('Train/Accuracy_Epoch', avg_acc, epoch)
        writer.add_scalar('Train/Best_Loss', best_loss, epoch)
        
        # 记录学习率变化
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/LR_Epoch', current_lr, epoch)
        
        print(f'Epoch {epoch+1}/{args.epochs}: Loss={avg_loss:.4f}, Acc={avg_acc:.2f}%, LR={current_lr:.6f}')
        
        # 保存最佳模型和检查点
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            writer.add_scalar('Train/Best_Loss_Updated', best_loss, epoch)
            print(f'发现更优模型: {best_loss:.4f}')
        
        # 保存检查点（统一格式）
        save_checkpoint(epoch, model, optimizer, scheduler, best_loss, save_dir, is_best)
        
        # 定期保存（每50个epoch额外保存一次）
        if (epoch + 1) % 50 == 0:
            milestone_path = os.path.join(save_dir, f'milestone_epoch_{epoch+1}.pth')
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss
            }
            torch.save(checkpoint, milestone_path)
            print(f'里程碑检查点已保存: {milestone_path}')
    
    # 关闭TensorBoard writer
    writer.close()
    print(f'训练完成！模型保存在: {save_dir}')
    print(f'TensorBoard日志: tensorboard --logdir={log_dir}')
    
    # 显示最终模型信息
    print(f'\\n最终模型信息:')
    print(f'- 最佳损失: {best_loss:.4f}')
    print(f'- 模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    print(f'- logit_scale: {model.logit_scale.exp().item():.4f}')


if __name__ == '__main__':
    main()
