"""
PNG草图-图像对齐模型训练脚本
使用PNG格式的草图与图像进行对齐训练
"""

import os
import torch
import torch.optim as optim
import argparse
from datetime import datetime
import json
import logging
from tqdm import tqdm

# 导入数据集和模型
from data.PNGSketchImageDataset import create_png_sketch_dataloaders
from encoders.png_sketch_image_model import create_png_sketch_image_model
from encoders.loss_func import ContrastiveLoss


# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='训练PNG草图-图像对齐模型')
    parser.add_argument('--batch_size', type=int, default=400, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--max_epochs', type=int, default=50, help='最大训练轮数')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='预热轮数')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心')
    parser.add_argument('--save_every', type=int, default=5, help='保存间隔')
    parser.add_argument('--embed_dim', type=int, default=512, help='嵌入维度')
    parser.add_argument('--freeze_image_encoder', action='store_true', default=False, help='冻结图像编码器')
    parser.add_argument('--freeze_sketch_backbone', action='store_true', default=False, help='冻结草图编码器主干网络')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载进程数')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    parser.add_argument('--sketch_format', type=str, default='vector', choices=['vector', 'image'], help='使用矢量草图还是图片草图')

    parser.add_argument('--local', default='False', choices=['True', 'False'], type=str)
    parser.add_argument('--root_sever', type=str, default=r'/opt/data/private/data_set/sketch_retrieval')
    parser.add_argument('--root_local', type=str, default=r'D:\document\DeepLearning\DataSet\sketch_retrieval\sketchy')

    args = parser.parse_args()
    return args


class PNGSketchImageTrainer:
    """PNG草图-图像对齐训练器"""
    
    def __init__(self, 
                 model,
                 train_loader,
                 test_loader,
                 device,
                 output_dir,
                 learning_rate=1e-4,
                 weight_decay=1e-4,
                 warmup_epochs=5,
                 max_epochs=50,
                 patience=10,
                 save_every=5):
        
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = output_dir
        self.max_epochs = max_epochs
        self.patience = patience
        self.save_every = save_every
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化优化器和学习率调度器
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
            eps=1e-6
        )
        
        # 预热学习率调度器
        self.warmup_epochs = warmup_epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=max_epochs - warmup_epochs,
            eta_min=learning_rate * 0.01
        )
        
        # 损失函数
        self.criterion = ContrastiveLoss(temperature=0.07)
        
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.test_losses = []
        
        logger.info(f"训练器初始化完成:")
        logger.info(f"  输出目录: {output_dir}")
        logger.info(f"  学习率: {learning_rate}")
        logger.info(f"  权重衰减: {weight_decay}")
        logger.info(f"  预热轮数: {warmup_epochs}")
        logger.info(f"  最大轮数: {max_epochs}")
        logger.info(f"  早停耐心: {patience}")
    
    def warmup_lr(self, epoch, warmup_epochs, base_lr):
        """预热学习率"""
        if epoch < warmup_epochs:
            lr = base_lr * (epoch + 1) / warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        return None
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.max_epochs}")
        
        for batch_idx, (sketches, images, category_indices, category_names) in enumerate(progress_bar):
            # 移动数据到设备
            sketches = sketches.to(self.device)
            images = images.to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            sketch_features, image_features, logit_scale = self.model(sketches, images)
            
            # 计算损失
            loss = self.criterion(sketch_features, image_features, logit_scale)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Avg Loss': f"{total_loss / (batch_idx + 1):.4f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.6f}",
                'Temp': f"{logit_scale.item():.4f}"
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_epoch(self):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.test_loader)
        
        with torch.no_grad():
            for sketches, images, category_indices, category_names in tqdm(self.test_loader, 
                                                                         desc="Validating"):
                # 移动数据到设备
                sketches = sketches.to(self.device)
                images = images.to(self.device)
                
                # 前向传播
                sketch_features, image_features, logit_scale = self.model(sketches, images)
                
                # 计算损失
                loss = self.criterion(sketch_features, image_features, logit_scale)
                
                # 累计损失
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, is_best=False, filename=None):
        """保存模型检查点"""
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch + 1}.pth"
        
        checkpoint = {
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'patience_counter': self.patience_counter
        }
        
        checkpoint_path = os.path.join(self.output_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"保存最佳模型: {best_path}")
        
        logger.info(f"保存检查点: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载模型检查点"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.best_loss = checkpoint['best_loss']
            self.train_losses = checkpoint['train_losses']
            self.test_losses = checkpoint['test_losses']
            self.patience_counter = checkpoint['patience_counter']
            
            logger.info(f"从检查点恢复训练: {checkpoint_path}")
            return True
        return False
    
    def train(self):
        """开始训练"""
        logger.info("开始训练PNG草图-图像对齐模型...")
        
        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch
            
            # 预热学习率
            if epoch < self.warmup_epochs:
                current_lr = self.warmup_lr(epoch, self.warmup_epochs, 
                                          self.optimizer.param_groups[0]['lr'])
                logger.info(f"Warmup LR: {current_lr:.6f}")
            
            # 训练一个epoch
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 验证一个epoch
            test_loss = self.validate_epoch()
            self.test_losses.append(test_loss)
            
            # 更新学习率（预热期后）
            if epoch >= self.warmup_epochs:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            logger.info(f"Epoch {epoch + 1}/{self.max_epochs}:")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Test Loss: {test_loss:.4f}")
            logger.info(f"  Learning Rate: {current_lr:.6f}")
            
            # 检查是否是最佳模型
            is_best = test_loss < self.best_loss
            if is_best:
                self.best_loss = test_loss
                self.patience_counter = 0
                logger.info(f"新的最佳测试损失: {test_loss:.4f}")
            else:
                self.patience_counter += 1
                logger.info(f"测试损失未改善，耐心计数: {self.patience_counter}/{self.patience}")
            
            # 保存检查点
            if (epoch + 1) % self.save_every == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
            
            # 早停检查
            if self.patience_counter >= self.patience:
                logger.info(f"连续{self.patience}个epoch测试损失未改善，早停训练")
                break
        
        # 保存训练历史
        self.save_training_history()
        logger.info("训练完成!")
    
    def save_training_history(self):
        """保存训练历史"""
        history = {
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'best_loss': self.best_loss,
            'epochs_trained': len(self.train_losses),
            'final_lr': self.optimizer.param_groups[0]['lr']
        }
        
        history_path = os.path.join(self.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"训练历史已保存: {history_path}")


def main(args):
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"outputs/png_sketch_image_alignment_{timestamp}"
    
    # 首先创建数据集划分（如果不存在）
    split_file = './data/fixed_splits/png_sketch_image_dataset_splits.pkl'
    if not os.path.exists(split_file):
        logger.info("PNG草图数据集划分文件不存在，正在创建...")
        from create_png_sketch_dataset import create_png_sketch_dataset_splits
        create_png_sketch_dataset_splits()
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    train_loader, test_loader, dataset_info = create_png_sketch_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fixed_split_path=split_file,
        root=args.root_local if eval(args.local) else args.root_sever,
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
        freeze_image_encoder=args.freeze_image_encoder,
        freeze_sketch_backbone=args.freeze_sketch_backbone,
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
    trainer = PNGSketchImageTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        patience=args.patience,
        save_every=args.save_every
    )
    
    # 恢复训练（如果指定）
    if args.resume and os.path.exists(args.resume):
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main(parse_args())


