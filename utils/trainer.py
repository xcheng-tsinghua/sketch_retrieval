import os
import torch
import torch.optim as optim
import json
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from utils import loss_func


class SBIRTrainer:
    """PNG草图-图像对齐训练器"""

    def __init__(self,
                 model,
                 train_set,
                 test_set,
                 train_loader,
                 test_loader,
                 device,
                 check_point,
                 logger,
                 dataset_info,
                 log_dir,
                 retrieval_mode,  # ['cl', 'fg']
                 learning_rate=1e-4,
                 weight_decay=1e-4,
                 max_epochs=50,
                 # stop_val=100
                 ):
        assert retrieval_mode in ('cl', 'fg')

        self.model = model
        self.train_set = train_set
        self.test_set = test_set
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.check_point = check_point
        self.max_epochs = max_epochs
        self.logger = logger
        self.dataset_info = dataset_info
        self.log_dir = log_dir
        # self.stop_val = stop_val

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
        if retrieval_mode == 'cl':
            self.criterion = loss_func.contrastive_loss_cl_zs_sbir

        else:
            self.criterion = loss_func.ContrastiveLoss(temperature=0.07)
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
        # skh_fea_norm = F.normalize(sketch_features, dim=1)
        # img_fea_norm = F.normalize(image_features, dim=1)
        similarity_matrix = torch.matmul(sketch_features, image_features.t())

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
        acc_1, acc_5 = compute_topk_accuracy(sketch_features, image_features)

        print(f'---mAP@200: {map_200:.4f}, Precision@200: {prec_200:.4f}, Acc@1: {acc_1:.4f}, Acc@5: {acc_5:.4f}')

        test_loss = total_loss / len(self.test_loader)
        return test_loss, map_200, prec_200, acc_1, acc_5

    def get_acc_files_epoch(self):
        """
        验证一个epoch, 并返回 FG-SBIR 检索成功在 Acc@1 及 Acc@5 的草图路径
        """
        self.model.eval()

        # 提取特征
        sketch_features = []
        image_features = []
        indexes = []

        with torch.no_grad():
            # 提取草图特征
            for idx, sketches, images in tqdm(self.test_loader, desc="Validating"):
                sketches = sketches.to(self.device)
                images = images.to(self.device)
                idx = idx.long().to(self.device)

                sketch_feat, image_feat, logit_scale = self.model(sketches, images)

                sketch_features.append(sketch_feat.cpu())
                image_features.append(image_feat.cpu())
                indexes.append(idx.cpu())

        # 合并特征
        sketch_features = torch.cat(sketch_features, dim=0)
        image_features = torch.cat(image_features, dim=0)
        indexes = torch.cat(indexes, dim=0)

        accs, acc_1_idxes, acc_5_idxes = compute_topk_accuracy_with_file(sketch_features, image_features, indexes)
        print(f'--- Acc@1: {accs[0]:.4f}, Acc@5: {accs[1]:.4f}')

        return acc_1_idxes, acc_5_idxes

    def vis_fea_cluster(self):
        """
        可视化学习出来的草图特征
        """
        self.model.eval()

        # 提取特征
        sketch_features = []
        class_labels = []

        with torch.no_grad():
            # 提取草图特征
            for sketches, images, category_indices, category_names in tqdm(self.test_loader, desc="Validating"):
                sketches = sketches.to(self.device)
                category_indices = category_indices.long()

                sketch_feat = self.model.encode_sketch(sketches)

                sketch_features.append(sketch_feat.cpu())
                class_labels.extend(category_indices)

        # 合并特征
        sketch_features = torch.cat(sketch_features, dim=0)
        class_labels = torch.tensor(class_labels)
        class_name = self.test_set.categories

        vis_class = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        visualize_embeddings(sketch_features, class_labels, vis_class, class_name)

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
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.best_loss = checkpoint['best_loss']
            self.train_losses = checkpoint['train_losses']
            self.test_losses = checkpoint['test_losses']

            print(f"从检查点恢复训练: {checkpoint_path}")
            return True

        except:
            print(f'从如下文件加载检查点失败，从零开始训练：{checkpoint_path}')
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

            # if map_200 < self.stop_val:
            #     break

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


def compute_topk_accuracy(sketch_fea, image_fea, topk=(1, 5)):
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


def compute_topk_accuracy_with_file(sketch_fea, image_fea, data_indices, topk=(1, 5)):
    """
    计算 Acc@1 和 Acc@5，同时返回匹配正确的 image 全局索引列表

    参数:
        sketch_fea:     [bs, fea] 查询 sketch 特征
        image_fea:      [bs, fea] 数据库 image 特征（同一批次）
        image_indices:  [bs]，表示 image_fea 中每个样本在全局数据库中的索引
        topk:           Tuple of int，比如 (1, 5)

    返回:
        accs: List of float，对应每个 k 的 accuracy
        acc1_matched_indices: List[int]，Acc@1 匹配正确的 image 索引
        acc5_matched_indices: List[int]，Acc@5 匹配正确的 image 索引
    """

    # 欧氏距离（也可换成余弦）
    dist_matrix = torch.cdist(sketch_fea, image_fea)  # [bs, bs]
    batch_size = dist_matrix.size(0)
    device = sketch_fea.device

    # 距离升序排序，最近的排前面
    sorted_indices = dist_matrix.argsort(dim=1, descending=False)  # [bs, bs]

    # 正确的 image 匹配是与自身同位置的样本（即第 i 个 sketch 对应第 i 个 image）
    ground_truth = torch.arange(batch_size, device=device).unsqueeze(1)  # [bs, 1]
    correct_matrix = (sorted_indices[:, :max(topk)] == ground_truth)  # [bs, max_k]

    accs = []
    acc1_matched_indices = []
    acc5_matched_indices = []

    for k in topk:
        correct_k = correct_matrix[:, :k].any(dim=1)  # [bs]，是否在 top-k 中命中
        acc = correct_k.float().mean().item()
        accs.append(acc)

        # 提取命中的 sketch 索引
        matched_sketch_indices = torch.nonzero(correct_k, as_tuple=False).squeeze(1)  # [num_matched]
        # 转换为对应匹配到的 image 的“全局索引”
        # matched_data_indices = data_indices[matched_sketch_indices].tolist()
        matched_data_indices = [data_indices[i.item()] for i in matched_sketch_indices]

        if k == 1:
            acc1_matched_indices = matched_data_indices
        elif k == 5:
            acc5_matched_indices = list(set(matched_data_indices) - set(acc1_matched_indices))

    return accs, acc1_matched_indices, acc5_matched_indices


def visualize_embeddings(embeddings: torch.Tensor,
                         labels: torch.Tensor,
                         target_classes,
                         class_names,
                         perplexity=30,
                         random_state=42):
    """
    可视化嵌入特征，支持仅展示指定类别
    :param embeddings: [bs, emb] 的 tensor
    :param labels: [bs] 的 int tensor，表示每个样本的类别
    :param target_classes: 可选，int列表或tensor，指示哪些类别需要可视化
    :param class_names: 可选，类别名称列表，长度等于最大类别id+1
    :param perplexity: t-SNE 参数
    :param random_state: 随机种子
    """
    embeddings_np = embeddings.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # 筛选出目标类别
    # if target_classes is not None:
    #     target_classes = set(target_classes)
    #     mask = np.isin(labels_np, list(target_classes))
    #     embeddings_np = embeddings_np[mask]
    #     labels_np = labels_np[mask]

    # t-SNE降维
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    embeddings_2d = tsne.fit_transform(embeddings_np)

    # 绘图
    plt.figure(figsize=(8, 8))
    plt.rcParams['font.family'] = 'Times New Roman'
    unique_labels = np.unique(labels_np)
    cmap = plt.get_cmap('tab10' if len(unique_labels) <= 10 else 'tab20')

    print('-----------------', unique_labels)

    for i, class_idx in enumerate(unique_labels):
        idx = labels_np == class_idx
        label_name = class_names[class_idx] if class_names and class_idx < len(class_names) else f'Class {class_idx}'
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1],
                    label=label_name, alpha=0.7, s=10, color=cmap(i) if i < 20 else 'black')  # cmap(i)

        # label_name = class_names[class_idx] if class_names and class_idx < len(class_names) else f'Class {class_idx}'
        # plt.scatter([], [], label=label_name, color=cmap(i) if i < 20 else 'black')

    # plt.legend()
    # plt.title("t-SNE Visualization of Embeddings")
    plt.tight_layout()
    plt.axis('off')
    plt.show()


