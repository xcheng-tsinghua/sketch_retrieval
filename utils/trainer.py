import os
import torch
import torch.optim as optim
import json
from tqdm import tqdm
from encoders import loss_func
from sklearn.metrics import average_precision_score
import numpy as np
import torch.nn.functional as F


# class PNGSketchImageTrainer:
#     """PNG草图-图像对齐训练器"""
#
#     def __init__(self,
#                  model,
#                  train_loader,
#                  test_loader,
#                  device,
#                  output_dir,
#                  logger,
#                  learning_rate=1e-4,
#                  weight_decay=1e-4,
#                  warmup_epochs=5,
#                  max_epochs=50,
#                  patience=10,
#                  save_every=5):
#
#         self.model = model
#         self.train_loader = train_loader
#         self.test_loader = test_loader
#         self.device = device
#         self.output_dir = output_dir
#         self.max_epochs = max_epochs
#         self.patience = patience
#         self.save_every = save_every
#         self.logger = logger
#
#         # 创建输出目录
#         os.makedirs(output_dir, exist_ok=True)
#
#         # 初始化优化器和学习率调度器
#         self.optimizer = optim.AdamW(
#             filter(lambda p: p.requires_grad, model.parameters()),
#             lr=learning_rate,
#             weight_decay=weight_decay,
#             betas=(0.9, 0.98),
#             eps=1e-6
#         )
#
#         # 预热学习率调度器
#         self.warmup_epochs = warmup_epochs
#         self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
#             self.optimizer,
#             T_max=max_epochs - warmup_epochs,
#             eta_min=learning_rate * 0.01
#         )
#
#         # 损失函数
#         self.criterion = ContrastiveLoss(temperature=0.07)
#
#         # 训练状态
#         self.current_epoch = 0
#         self.best_loss = float('inf')
#         self.patience_counter = 0
#         self.train_losses = []
#         self.test_losses = []
#
#         self.logger.info(f"训练器初始化完成:")
#         self.logger.info(f"  输出目录: {output_dir}")
#         self.logger.info(f"  学习率: {learning_rate}")
#         self.logger.info(f"  权重衰减: {weight_decay}")
#         self.logger.info(f"  预热轮数: {warmup_epochs}")
#         self.logger.info(f"  最大轮数: {max_epochs}")
#         self.logger.info(f"  早停耐心: {patience}")
#
#     def warmup_lr(self, epoch, warmup_epochs, base_lr):
#         """预热学习率"""
#         if epoch < warmup_epochs:
#             lr = base_lr * (epoch + 1) / warmup_epochs
#             for param_group in self.optimizer.param_groups:
#                 param_group['lr'] = lr
#             return lr
#         return None
#
#     def train_epoch(self):
#         """训练一个epoch"""
#         self.model.train()
#         total_loss = 0.0
#         num_batches = len(self.train_loader)
#         progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.max_epochs}")
#
#         for batch_idx, (sketches, images, category_indices, category_names) in enumerate(progress_bar):
#             # 移动数据到设备
#             sketches = sketches.to(self.device)
#             images = images.to(self.device)
#
#             # 清零梯度
#             self.optimizer.zero_grad()
#
#             # 前向传播
#             sketch_features, image_features, logit_scale = self.model(sketches, images)
#
#             # 计算损失
#             loss = self.criterion(sketch_features, image_features, logit_scale)
#
#             # 反向传播
#             loss.backward()
#
#             # 梯度裁剪
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
#
#             # 更新参数
#             self.optimizer.step()
#
#             # 累计损失
#             total_loss += loss.item()
#
#             # 更新进度条
#             progress_bar.set_postfix({
#                 'Loss': f"{loss.item():.4f}",
#                 'Avg Loss': f"{total_loss / (batch_idx + 1):.4f}",
#                 'LR': f"{self.optimizer.param_groups[0]['lr']:.6f}",
#                 'Temp': f"{logit_scale.item():.4f}"
#             })
#
#         avg_loss = total_loss / num_batches
#         return avg_loss
#
#     def validate_epoch(self):
#         """验证一个epoch"""
#         self.model.eval()
#         total_loss = 0.0
#         num_batches = len(self.test_loader)
#
#         with torch.no_grad():
#             for sketches, images, category_indices, category_names in tqdm(self.test_loader,
#                                                                            desc="Validating"):
#                 # 移动数据到设备
#                 sketches = sketches.to(self.device)
#                 images = images.to(self.device)
#
#                 # 前向传播
#                 sketch_features, image_features, logit_scale = self.model(sketches, images)
#
#                 # 计算损失
#                 loss = self.criterion(sketch_features, image_features, logit_scale)
#
#                 # 累计损失
#                 total_loss += loss.item()
#
#         avg_loss = total_loss / num_batches
#         return avg_loss
#
#     def save_checkpoint(self, is_best=False, filename=None):
#         """保存模型检查点"""
#         if filename is None:
#             filename = f"checkpoint_epoch_{self.current_epoch + 1}.pth"
#
#         checkpoint = {
#             'epoch': self.current_epoch + 1,
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'scheduler_state_dict': self.scheduler.state_dict(),
#             'best_loss': self.best_loss,
#             'train_losses': self.train_losses,
#             'test_losses': self.test_losses,
#             'patience_counter': self.patience_counter
#         }
#
#         checkpoint_path = os.path.join(self.output_dir, filename)
#         torch.save(checkpoint, checkpoint_path)
#
#         if is_best:
#             best_path = os.path.join(self.output_dir, 'best_checkpoint.pth')
#             torch.save(checkpoint, best_path)
#             self.logger.info(f"保存最佳模型: {best_path}")
#
#         self.logger.info(f"保存检查点: {checkpoint_path}")
#
#     def load_checkpoint(self, checkpoint_path):
#         """加载模型检查点"""
#         if os.path.exists(checkpoint_path):
#             checkpoint = torch.load(checkpoint_path, map_location=self.device)
#
#             self.model.load_state_dict(checkpoint['model_state_dict'])
#             self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#             self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#             self.current_epoch = checkpoint['epoch']
#             self.best_loss = checkpoint['best_loss']
#             self.train_losses = checkpoint['train_losses']
#             self.test_losses = checkpoint['test_losses']
#             self.patience_counter = checkpoint['patience_counter']
#
#             self.logger.info(f"从检查点恢复训练: {checkpoint_path}")
#             return True
#         return False
#
#     def train(self):
#         """开始训练"""
#         self.logger.info("开始训练PNG草图-图像对齐模型...")
#
#         for epoch in range(self.current_epoch, self.max_epochs):
#             self.current_epoch = epoch
#
#             # 预热学习率
#             if epoch < self.warmup_epochs:
#                 current_lr = self.warmup_lr(epoch, self.warmup_epochs,
#                                             self.optimizer.param_groups[0]['lr'])
#                 self.logger.info(f"Warmup LR: {current_lr:.6f}")
#
#             # 训练一个epoch
#             train_loss = self.train_epoch()
#             self.train_losses.append(train_loss)
#
#             # 验证一个epoch
#             test_loss = self.validate_epoch()
#             self.test_losses.append(test_loss)
#
#             # 更新学习率（预热期后）
#             if epoch >= self.warmup_epochs:
#                 self.scheduler.step()
#
#             current_lr = self.optimizer.param_groups[0]['lr']
#
#             self.logger.info(f"Epoch {epoch + 1}/{self.max_epochs}:")
#             self.logger.info(f"  Train Loss: {train_loss:.4f}")
#             self.logger.info(f"  Test Loss: {test_loss:.4f}")
#             self.logger.info(f"  Learning Rate: {current_lr:.6f}")
#
#             # 检查是否是最佳模型
#             is_best = test_loss < self.best_loss
#             if is_best:
#                 self.best_loss = test_loss
#                 self.patience_counter = 0
#                 self.logger.info(f"新的最佳测试损失: {test_loss:.4f}")
#             else:
#                 self.patience_counter += 1
#                 self.logger.info(f"测试损失未改善，耐心计数: {self.patience_counter}/{self.patience}")
#
#             # 保存检查点
#             if (epoch + 1) % self.save_every == 0 or is_best:
#                 self.save_checkpoint(is_best=is_best)
#
#             # 早停检查
#             if self.patience_counter >= self.patience:
#                 self.logger.info(f"连续{self.patience}个epoch测试损失未改善，早停训练")
#                 break
#
#         # 保存训练历史
#         self.save_training_history()
#         self.logger.info("训练完成!")
#
#     def save_training_history(self):
#         """保存训练历史"""
#         history = {
#             'train_losses': self.train_losses,
#             'test_losses': self.test_losses,
#             'best_loss': self.best_loss,
#             'epochs_trained': len(self.train_losses),
#             'final_lr': self.optimizer.param_groups[0]['lr']
#         }
#
#         history_path = os.path.join(self.output_dir, 'training_history.json')
#         with open(history_path, 'w') as f:
#             json.dump(history, f, indent=2)
#
#         self.logger.info(f"训练历史已保存: {history_path}")


class SBIRTrainer:
    """PNG草图-图像对齐训练器"""

    def __init__(self,
                 model,
                 train_loader,
                 test_loader,
                 device,
                 check_point,
                 logger,
                 dataset_info,
                 log_dir,
                 learning_rate=1e-4,
                 weight_decay=1e-4,
                 max_epochs=50
                 ):

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.check_point = check_point
        self.max_epochs = max_epochs
        self.logger = logger
        self.dataset_info = dataset_info
        self.log_dir = log_dir

        self.check_point_best = os.path.splitext(check_point)[0] + '_best.pth'

        # 创建输出目录
        os.makedirs(os.path.dirname(self.check_point), exist_ok=True)

        # 初始化优化器和学习率调度器
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
            eps=1e-6
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)

        # 损失函数
        self.criterion = loss_func.ContrastiveLoss(temperature=0.07)
        # self.criterion = loss_func.contrastive_loss_cl_zs_sbir
        # self.criterion = loss_func.contrastive_loss_fg_zs_sbir

        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.test_losses = []

        print(f"        训练器初始化完成:")
        print(f"  检查点保存: {self.check_point}")
        print(f"  学习率: {learning_rate}")
        print(f"  最大轮数: {max_epochs}")

    def train_epoch(self):
        """
        训练一个epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.max_epochs}")

        for batch_idx, (sketches, images, category_indices, category_names) in enumerate(progress_bar):
            # 移动数据到设备
            sketches = sketches.to(self.device)
            images = images.to(self.device)
            category_indices = category_indices.to(self.device)

            # 清零梯度
            self.optimizer.zero_grad()

            # 前向传播
            sketch_features, image_features, logit_scale = self.model(sketches, images)

            # 计算损失
            loss = self.criterion(sketch_features, image_features, category_indices, logit_scale)

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
        self.scheduler.step()

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate_epoch(self):
        """
        验证一个epoch
        """
        self.model.eval()

        # 提取特征
        sketch_features = []
        image_features = []
        class_labels = []
        sketch_categories = []

        total_loss = 0.0
        with torch.no_grad():
            # 提取草图特征
            for sketches, images, category_indices, category_names in tqdm(self.test_loader, desc="Validating"):
                sketches = sketches.to(self.device)
                images = images.to(self.device)
                category_indices = category_indices.to(self.device)

                sketch_feat, image_feat, logit_scale = self.model(sketches, images)
                loss = self.criterion(sketch_feat, image_feat, category_indices, logit_scale)
                total_loss += loss.item()

                sketch_features.append(sketch_feat.cpu())
                image_features.append(image_feat.cpu())
                class_labels.extend(category_indices.cpu().numpy())
                sketch_categories.extend(category_names)

        # 合并特征
        sketch_features = torch.cat(sketch_features, dim=0)
        image_features = torch.cat(image_features, dim=0)
        class_labels = torch.tensor(class_labels)

        # 计算相似度矩阵, 数值越大表示越相似
        # 因为 (a1, b1), (a2, b2), cosθ = a1 * b1 + a2 * b2，余弦值越大表示夹角越小，也就是越相似
        # 注意计算余弦距离需要单位向量
        skh_fea_norm = F.normalize(sketch_features, dim=1)
        img_fea_norm = F.normalize(image_features, dim=1)
        similarity_matrix = torch.matmul(skh_fea_norm, img_fea_norm.t())

        # 计算检索指标
        metrics = compute_retrieval_metrics(similarity_matrix, class_labels)

        print(f"=== 检索性能评估结果 ===")
        print(f"Top-1 准确率: {metrics['top1_accuracy']:.4f}")
        print(f"Top-5 准确率: {metrics['top5_accuracy']:.4f}")
        print(f"Top-10 准确率: {metrics['top10_accuracy']:.4f}")
        print(f"mAP_all: {metrics['mAP_all']:.4f}")

        # 按类别评估
        categories = self.dataset_info['category_info']['categories']
        category_metrics = evaluate_by_category(
            similarity_matrix, class_labels, sketch_categories, categories
        )

        print(f"评估 {len(categories)} 个类别的性能...")

        # 显示部分类别结果
        sorted_categories = sorted(category_metrics.items(),
                                   key=lambda x: x[1]['accuracy'], reverse=True)

        for i, (category, cat_metrics) in enumerate(sorted_categories[:10]):
            accuracy = cat_metrics['accuracy']
            num_samples = cat_metrics['num_samples']
            correct = cat_metrics['correct']
            print(f"  {category}: {correct}/{num_samples} = {accuracy:.4f}")

        # 找到最佳和最差类别
        if category_metrics:
            best_category = max(category_metrics.items(), key=lambda x: x[1]['accuracy'])
            worst_category = min(category_metrics.items(), key=lambda x: x[1]['accuracy'])

            print(f"最佳类别: {best_category[0]} ({best_category[1]['accuracy']:.4f})")
            print(f"最差类别: {worst_category[0]} ({worst_category[1]['accuracy']:.4f})")

            # 统计类别分布
            num_good = sum(1 for cat_metrics in category_metrics.values()
                           if cat_metrics['accuracy'] > 0)
            num_zero = sum(1 for cat_metrics in category_metrics.values()
                           if cat_metrics['accuracy'] == 0)

            print(f"类别统计: 总数={len(category_metrics)}, 准确率>0={num_good}, 准确率=0={num_zero}")

        map_200, prec_200 = map_and_precision_at_k(sketch_features, image_features, class_labels)
        acc_1, acc_5 = compute_topk_accuracy(sketch_features, image_features, class_labels)

        print(f'---mAP@200: {map_200:.4f}, Precision@200: {prec_200:.4f}, {acc_1:.4f}, Acc@5: {acc_5:.4f}')

        test_loss = total_loss / len(self.test_loader)
        return test_loss, map_200, prec_200, acc_1, acc_5

    def save_checkpoint(self, is_best=False):
        """保存模型检查点"""

        checkpoint = {
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses
        }

        torch.save(checkpoint, self.check_point)

        if is_best:
            torch.save(checkpoint, self.check_point_best)
            print(f"保存最佳模型: {self.check_point_best}")

        print(f"保存检查点: {self.check_point}")

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

            print(f"从检查点恢复训练: {checkpoint_path}")
            return True
        return False

    def train(self):
        """开始训练"""
        print("开始训练PNG草图-图像对齐模型...")

        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch

            # 训练一个epoch
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # 验证一个epoch
            test_loss, map_200, prec_200, acc_1, acc_5 = self.validate_epoch()
            self.test_losses.append(test_loss)

            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'epoch {epoch + 1}/{self.max_epochs}: train_loss: {train_loss:.4f}, test_loss: {test_loss:.4f}, lr: {current_lr:.6f}')

            # 检查是否是最佳模型
            is_best = test_loss < self.best_loss
            if is_best:
                self.best_loss = test_loss
                print(f"新的最佳测试损失: {test_loss:.4f}")

            # 保存检查点
            self.save_checkpoint(is_best=is_best)

            log_str = f'epoch {epoch + 1}/{self.max_epochs} train_loss {train_loss} test_loss {test_loss} map_200 {map_200} prec_200 {prec_200} acc_1 {acc_1} acc_5 {acc_5}'
            log_str = log_str.replace(' ', '\t')
            self.logger.info(log_str)

        # 保存训练历史
        self.save_training_history()
        print("训练完成!")

    def save_training_history(self):
        """保存训练历史"""
        history = {
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'best_loss': self.best_loss,
            'epochs_trained': len(self.train_losses),
            'final_lr': self.optimizer.param_groups[0]['lr']
        }

        history_path = os.path.join(os.path.dirname(self.check_point), 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        print(f"训练历史已保存: {history_path}")


def compute_retrieval_metrics(similarity_matrix, labels):
    """
    计算检索指标

    Args:
        similarity_matrix: 相似度矩阵 [N_sketch, N_image]
        labels: 标签数组，相同类别的样本有相同标签

    Returns:
        metrics: 检索指标字典
    """
    N_sketch, N_image = similarity_matrix.shape

    # 计算Top-K准确率
    top1_correct = 0
    top5_correct = 0
    top10_correct = 0

    # 计算mAP
    all_aps = []

    for i in range(N_sketch):
        # 获取第i个草图的相似度
        similarities = similarity_matrix[i]

        # 排序获取检索结果
        sorted_indices = torch.argsort(similarities, descending=True)

        # 获取真实标签
        sketch_label = labels[i]

        # 计算Top-K准确率
        if labels[sorted_indices[0]] == sketch_label:
            top1_correct += 1

        if any(labels[sorted_indices[:5]] == sketch_label):
            top5_correct += 1

        if any(labels[sorted_indices[:10]] == sketch_label):
            top10_correct += 1

        # 计算AP (Average Precision)
        relevant_mask = (labels == sketch_label).float()
        if relevant_mask.sum() > 0:
            ap = average_precision_score(
                relevant_mask.cpu().numpy(),
                similarities.cpu().numpy()
            )
            all_aps.append(ap)

    # 计算指标
    metrics = {
        'top1_accuracy': top1_correct / N_sketch,
        'top5_accuracy': top5_correct / N_sketch,
        'top10_accuracy': top10_correct / N_sketch,
        'mAP_all': np.mean(all_aps) if all_aps else 0.0
    }

    return metrics


def evaluate_by_category(similarity_matrix, labels, category_names, categories):
    """
    按类别评估检索性能

    Args:
        similarity_matrix: 相似度矩阵
        labels: 标签数组
        category_names: 类别名称列表
        categories: 所有类别列表

    Returns:
        category_metrics: 各类别的检索指标
    """
    category_metrics = {}

    for cat_idx, category in enumerate(categories):
        # 找到该类别的草图索引
        cat_sketch_indices = [i for i, cat in enumerate(category_names) if cat == category]

        if len(cat_sketch_indices) == 0:
            continue

        # 计算该类别的Top-1准确率
        cat_correct = 0
        for sketch_idx in cat_sketch_indices:
            similarities = similarity_matrix[sketch_idx]
            best_match_idx = torch.argmax(similarities).item()

            if labels[best_match_idx] == labels[sketch_idx]:
                cat_correct += 1

        cat_accuracy = cat_correct / len(cat_sketch_indices)
        category_metrics[category] = {
            'accuracy': cat_accuracy,
            'num_samples': len(cat_sketch_indices),
            'correct': cat_correct
        }

    return category_metrics


def map_and_precision_at_k(sketch_fea, image_fea, class_id, k=200):
    """
    使用欧氏距离评估 mAP@K 和 Precision@K

    参数:
    - sketch_fea: Tensor [bs, fea]，每一行为一个 sketch 特征
    - image_fea:  Tensor [bs, fea]，每一行为一个 image 特征
    - class_id:   Tensor [bs]，每个样本对应的类别标签（sketch 和 image 一一对应）
    - k: 截断前 K 个检索结果

    返回:
    - mean_precision: 所有查询的平均 Precision@K
    - mean_ap: 所有查询的 mAP@K
    """

    # 计算 sketch 和 image 之间的欧氏距离，结果是 [bs, bs]
    dist_matrix = torch.cdist(sketch_fea, image_fea, p=2)

    bs = sketch_fea.size(0)

    precision_list = []
    ap_list = []

    for i in range(bs):
        dists = dist_matrix[i]  # 第 i 个 sketch 和所有 image 的距离
        gt = class_id[i]        # 第 i 个 sketch 的真实类别

        # 按距离升序排序（距离越小越相似）
        sorted_indices = torch.argsort(dists, descending=False)

        # 取 top-k 个图像索引
        topk_indices = sorted_indices[:k]
        topk_class_ids = class_id[topk_indices]

        # 构造相关图像的 mask（同类别为相关）
        relevant = (topk_class_ids == gt).to(torch.float32)  # [k]

        # Precision@k
        precision_at_k = relevant.sum().item() / k
        precision_list.append(precision_at_k)

        # AP@k（Average Precision）
        num_rel = int(relevant.sum().item())
        if num_rel == 0:
            ap = 0.0
        else:
            precision_accum = 0.0
            rel_count = 0
            for rank in range(k):
                if relevant[rank]:
                    rel_count += 1
                    precision_i = rel_count / (rank + 1)
                    precision_accum += precision_i
            ap = precision_accum / num_rel
        ap_list.append(ap)

    mean_precision = sum(precision_list) / bs
    mean_ap = sum(ap_list) / bs

    return mean_precision, mean_ap


def compute_topk_accuracy(sketch_fea, image_fea, class_id, topk=(1, 5)):
    """
    计算 Acc@1 和 Acc@5，不使用 class label，只匹配同索引位置的配对样本
    sketch_fea: [bs, d]
    image_fea: [bs, d]
    """
    # 计算欧氏距离（越小越相似），也可以换成余弦相似度（越大越相似）
    dist_matrix = torch.cdist(sketch_fea, image_fea)  # [bs, bs]

    # 目标是 diagonal 上的元素是配对图像对应的距离
    batch_size = dist_matrix.size(0)

    # 按距离从小到大排序（因为距离越小越相似）
    sorted_indices = dist_matrix.argsort(dim=1)  # [bs, bs]

    # 构造 ground truth index：每个 sketch 匹配 image 中相同索引位置的图像
    target = torch.arange(batch_size).unsqueeze(1).to(sketch_fea.device)  # [bs, 1]

    correct = (sorted_indices[:, :max(topk)] == target).int()  # [bs, topk]

    accs = []
    for k in topk:
        acc = correct[:, :k].sum().item() / batch_size
        accs.append(acc)

    return accs



