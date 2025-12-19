import os
import torch
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F
import json
from pathlib import Path
from typing import Tuple, Dict
from colorama import Fore, Back, Style

from utils import loss_func


class SBIRTrainer:
    """PNG草图-图像对齐训练器"""
    def __init__(self,
                 model,
                 train_loader,
                 test_loader,
                 device,
                 check_point,
                 logger,
                 save_str,
                 learning_rate,
                 weight_decay,
                 max_epochs,
                 ckpt_save_interval=20,  # 检查点保存的 epoch 间隔
                 topk=(1, 5),
                 ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.check_point = check_point
        self.max_epochs = max_epochs
        self.logger = logger
        self.ckpt_save_interval = ckpt_save_interval
        self.save_str = save_str

        self.model.to(self.device)

        # 保证升序
        topk = list(topk)
        topk.sort()
        self.topk = topk

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
        self.criterion = loss_func.info_nce_multi_neg

        # 训练状态
        self.current_epoch = 0
        self.best_acc = -1.
        self.train_losses = []
        self.test_acc = []

        print(f'-> initiate trainer successful:')
        print(f'   check point save: {self.check_point}')
        print(f'   learning rate: {learning_rate}')
        print(f'   max epoch: {max_epochs}')

    def train_epoch(self):
        """
        训练一个epoch
        """
        self.model.train()
        self.train_loader.dataset.back_train_data()

        total_loss = 0.0
        num_batches = len(self.train_loader)
        progress_bar = tqdm(self.train_loader, desc=f'{self.current_epoch}/{self.max_epochs}')
        for sketches, images_pos, images_neg in progress_bar:
            # 移动数据到设备
            sketches = sketches.to(self.device)
            images_pos = images_pos.to(self.device)
            images_neg = images_neg.to(self.device)

            # 清零梯度
            self.optimizer.zero_grad()

            # 前向传播
            sketch_features = self.model.encode_sketch(sketches)
            pos_features = self.model.encode_image(images_pos)

            # 整理多个负样本
            c_bs, n_neg, c, h, w = images_neg.size()
            images_neg = images_neg.view(c_bs * n_neg, c, h, w)

            # 获取负样本特征
            neg_features = self.model.encode_image(images_neg)
            neg_features = neg_features.view(c_bs, n_neg, -1)

            # 计算损失
            loss = self.criterion(sketch_features, pos_features, neg_features)

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
                'LR': f"{self.optimizer.param_groups[0]['lr']:.6f}",
            })
        self.scheduler.step()

        avg_loss = total_loss / num_batches
        return avg_loss

    # @staticmethod
    # @torch.no_grad()
    # def get_fea(encoder, data_loader, device):
    #     fea_list = []
    #     for sketch_tensor in tqdm(data_loader, desc="loading data"):
    #         # 由于 shuffle=False，无需担心加载过程中图片顺序被打乱
    #         sketch_tensor = sketch_tensor.to(device)
    #
    #         sketch_fea = encoder(sketch_tensor)
    #         fea_list.append(sketch_fea.cpu())
    #
    #     # 合并特征
    #     fea_list = torch.cat(fea_list, dim=0)
    #     return fea_list

    @torch.no_grad()
    def get_acc_revl_success(self):
        """
        获取测试集准确率以及检索成功的草图样例索引
        """
        # self.model.eval()

        # # 提取草图特征
        # self.test_loader.dataset.back_sketch()
        # sketch_features = self.get_fea(self.model.encode_sketch, self.test_loader, self.device)
        #
        # # 提取图片特征
        # self.test_loader.dataset.back_image()
        # image_features = self.get_fea(self.model.encode_image, self.test_loader, self.device)
        #
        # # 计算准确率及匹配的样例
        # # 1. similarity matrix [m, n]
        # sim_matrix = sketch_features @ image_features.t()

        sim_matrix = self.get_similarity_matrix(False)

        max_k = max(self.topk)

        # 2. top-k retrieval
        _, topk_indices = sim_matrix.topk(k=max_k, dim=1, largest=True, sorted=True)  # [m, max_k]

        # 4. GT index
        gt_idx = torch.tensor(self.test_loader.dataset.sketch_paired_id).view(-1, 1)  # [m, 1]

        accs = []
        indices = []
        for k in self.topk:
            # [m] bool
            correct_k = (topk_indices[:, :k] == gt_idx).any(dim=1)

            # Acc@k
            acc_k = correct_k.float().mean().item()

            # 命中的草图索引
            correct_indices = torch.nonzero(correct_k, as_tuple=False).squeeze(1)

            accs.append(acc_k)
            indices.append(correct_indices)

        return accs, indices

    def save_revl_success_ins(self):
        """
        保存检索正确样例，按指定格式
        """
        save_path = f'./log/revl_ins_{self.save_str}.json'

        acc_topk, revl_idx_topk = self.get_acc_revl_success()
        acc_str = ''
        for k, acc in zip(self.topk, acc_topk):
            acc_str += f'Acc@{k}: {acc:.4f} '
        print(acc_str)

        # 找到对应的字符串并保存到对应的文件
        save_dict = {}
        prev_file = []
        for c_topk, c_idx_list in zip(self.topk, revl_idx_topk):
            c_revl_files = []
            for c_idx in c_idx_list:
                skh_path = self.test_loader.dataset.sketch_list_with_id[c_idx][0]

                # 将路径转化为 class/basename 的格式
                path = Path(skh_path)
                skh_path_fmt = f'{path.parent.name}/{path.stem}'
                c_revl_files.append(skh_path_fmt)

            # 保证后一个精度的结果不包含前一个精度的结果
            c_revl_files = [x for x in c_revl_files if x not in prev_file]

            save_dict[f'top_{c_topk}'] = c_revl_files
            prev_file.extend(c_revl_files)

        # 保存到文件
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_dict, f, ensure_ascii=False, indent=4)  # 中文正常、带缩进
        print(f'file save to: {os.path.abspath(save_path)}')

    @torch.no_grad()
    def get_similarity_matrix(self, is_train):
        """
        获取相似度矩阵
        [n_skh, n_img]

        is_train: True 训练集上的相似度矩阵
        False 测试集上的相似度矩阵
        """

        def _get_fea(_encoder):
            """
            获取特征
            """
            _fea_list = []
            for _data_tensor in tqdm(target_loader, desc='loading data'):
                # 由于 shuffle=False，无需担心加载过程中图片顺序被打乱
                _data_tensor = _data_tensor.to(self.device)

                _fea_tensor = _encoder(_data_tensor)
                _fea_list.append(_fea_tensor.cpu())

            # 合并特征
            _fea_list = torch.cat(_fea_list, dim=0)
            return _fea_list

        target_loader = self.train_loader if is_train else self.test_loader

        self.model.eval()
        # 提取草图特征
        target_loader.dataset.back_sketch()
        sketch_features = _get_fea(self.model.encode_sketch)

        # 提取图片特征
        target_loader.dataset.back_image()
        image_features = _get_fea(self.model.encode_image)

        # 计算准确率及匹配的样例
        # 1. similarity matrix [m, n]
        sim_matrix = sketch_features @ image_features.t()

        return sim_matrix

    def validate_epoch(self):
        """
        验证一个epoch
        """
        acc_topk, _ = self.get_acc_revl_success()

        # # 计算负样本，需要利用训练集
        # self.model.eval()
        #
        # # 提取草图特征
        # self.train_loader.dataset.back_sketch()
        # sketch_features = self.get_fea(self.model.encode_sketch, self.train_loader, self.device)
        #
        # # 提取图片特征
        # self.train_loader.dataset.back_image()
        # image_features = self.get_fea(self.model.encode_image, self.train_loader, self.device)
        #
        # # 计算准确率及匹配的样例
        # # 1. similarity matrix [m, n]
        # sim_matrix = sketch_features @ image_features.t()

        sim_matrix = self.get_similarity_matrix(True)

        max_k = self.train_loader.dataset.n_neg + 1

        # 2. top-k retrieval
        _, topk_indices = sim_matrix.topk(k=max_k, dim=1, largest=True, sorted=True)  # [m, max_k]

        # 4. GT index
        gt_idx = torch.tensor(self.train_loader.dataset.sketch_paired_id).view(-1, 1)  # [m, 1]

        # 获取最相近的负样本
        neg_mask = topk_indices != gt_idx  # [m, n_neg + 1]
        neg_idxes = []
        for i in range(gt_idx.size(0)):
            c_revl = topk_indices[i]
            c_neg_mask = neg_mask[i]

            if c_neg_mask.logical_not().any():  # 存在正样本
                c_neg_idx = c_revl[c_neg_mask]
            else:  # 不存在正样本
                c_neg_idx = c_revl[:-1]

            c_neg_idx = c_neg_idx.tolist()
            assert len(c_neg_idx) == self.train_loader.dataset.n_neg, ValueError('error neg instance number')
            neg_idxes.append(c_neg_idx)

        # 设置负样本
        self.train_loader.dataset.neg_instance = neg_idxes

        return acc_topk

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

    def vis_training_status(self):
        """
        绘制epoch过程中loss和acc的曲线
        """
        # self.train_losses = checkpoint['train_losses']
        # self.test_acc = checkpoint['test_acc']
        x = np.arange(len(self.test_acc))

        # 2. 建立画布和主轴（左侧）
        fig, ax1 = plt.subplots(figsize=(6, 4))

        # 3. 画第一条线，并自动绑定在左侧 y 轴
        color1 = 'tab:blue'
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('train loss', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)

        # 4. 生成右侧 y 轴
        ax2 = ax1.twinx()  # ★ 关键：共享 x 轴

        # 5. 在右侧 y 轴画第二条线
        color2 = 'tab:red'
        ax2.set_ylabel('test acc', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)

        acc_np = np.array(self.test_acc)  # [n, len(topk)]
        for i, c_top in enumerate(self.topk):
            ax2.plot(x, acc_np[:, i], label=f'Acc@{c_top}')

        # 6. 可选：把两条线的图例合并到同一框
        plt.legend()

        # 7. 标题 & 布局
        fig.suptitle('training status')
        fig.tight_layout()
        plt.show()

    def save_checkpoint(self, is_best=False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'train_losses': self.train_losses,
            'test_acc': self.test_acc,
        }
        torch.save(checkpoint, self.check_point)

        if is_best:
            torch.save(checkpoint, self.check_point_best)
            print(Fore.BLUE + Back.GREEN + f'save best checkpoint: {self.check_point_best}' + Style.RESET_ALL)

        print(f'save checkpoint: {self.check_point}')

    def load_checkpoint(self, checkpoint_path, is_load, is_load_training_state=True):
        """加载模型检查点"""
        if is_load:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(Fore.GREEN + f"checkpoint load finished: {checkpoint_path}, (epoch {checkpoint.get('epoch', 'unknown')})" + Style.RESET_ALL)

                if is_load_training_state:
                    try:
                        self.train_losses = checkpoint['train_losses']
                        self.test_acc = checkpoint['test_acc']
                        self.best_acc = checkpoint['best_acc']

                        # self.current_epoch = checkpoint['epoch']
                        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

                    except Exception as e:
                        print(Fore.RED + f'load training status failed, error: {e}.' + Style.RESET_ALL)

                return True

            except Exception as e:
                print(Fore.RED + f'checkpoint load failed, training from scratch: {checkpoint_path}, error: {e}.' + Style.RESET_ALL)
                return False
        else:
            print('does not load weight, training from scratch.')
            return False

    def train(self):
        """开始训练"""
        print(Fore.CYAN + 'start training SBIR model' + Style.RESET_ALL)

        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch

            # 训练一个epoch
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # 验证一个epoch
            acc_topk = self.validate_epoch()
            self.test_acc.append(acc_topk)

            # 保存检查点
            if (epoch + 1) % self.ckpt_save_interval == 0 or epoch == self.max_epochs - 1:
                # 检查是否是最佳模型
                is_best = acc_topk[0] > self.best_acc
                if is_best:
                    self.best_acc = acc_topk[0]
                    print(Fore.CYAN + f'new best retrieval accuracy: {acc_topk[0]:.4f}' + Style.RESET_ALL)

                # 保存检查点
                self.save_checkpoint(is_best=is_best)

            current_lr = self.optimizer.param_groups[0]['lr']
            log_str = f'train_loss: {train_loss:.4f}, lr: {current_lr:.6f} '
            for k, acc in zip(self.topk, acc_topk):
                log_str += f'Acc@{k}: {acc:.4f} '

            print(self.save_str + ': ' + log_str)
            log_str = 'epoch {epoch}/{self.max_epochs}: ' + log_str
            log_str = log_str.replace(' ', '\t')
            self.logger.info(log_str)

        print(Fore.BLACK + Back.GREEN + 'training finished!' + Style.RESET_ALL)


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


def compute_topk_accuracy_fg(sketch_fea, image_fea, topk=(1, 5)):
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


def compute_topk_accuracy_cl(sketch_feat, sketch_label, image_feat, image_label, topk_list=(5, 10)):
    """
    计算草图检索图片的 Acc@K 指标
    Args:
        sketch_feat: (m, f) 草图特征
        sketch_label: (m,) 草图类别
        image_feat: (n, f) 图片特征
        image_label: (n,) 图片类别
        topk_list: int 或 list[int], 指定要计算的 K 值，如 [1, 5, 10]
    Returns:
        dict, 例如 {'Acc@5': 0.48, 'Acc@10': 0.65}
    """
    if isinstance(topk_list, int):
        topk_list = [topk_list]
    max_k = max(topk_list)

    # 归一化特征向量（余弦相似度）
    sketch_norm = F.normalize(sketch_feat, dim=1)
    image_norm = F.normalize(image_feat, dim=1)

    # 相似度矩阵
    sim = sketch_norm @ image_norm.T  # (m, n)

    # 获取前 max_k 个最相似图片索引
    topk = sim.topk(k=max_k, dim=1).indices  # (m, max_k)

    # 获取对应的图片类别
    retrieved_labels = image_label[topk]  # (m, max_k)

    # 对每个 K 计算准确率
    results = {}
    for k in topk_list:
        correct = (retrieved_labels[:, :k] == sketch_label.unsqueeze(1)).any(dim=1)
        acc = correct.float().mean().item()
        results[f'Acc@{k}'] = acc

    return results


def compute_topk_accuracy_with_file(sketch_fea, image_fea, topk=(1, 5)):
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
    # dist_matrix = torch.cdist(sketch_fea, image_fea)  # [bs, bs]
    # n_skh = dist_matrix.size(0)
    # device = sketch_fea.device
    #
    # # 距离升序排序，最近的排前面
    # sorted_indices = dist_matrix.argsort(dim=1, descending=False)  # [bs, bs]

    n_skh = sketch_fea.size(0)
    device = sketch_fea.device

    # 1. L2 归一化
    sketch_fea = F.normalize(sketch_fea, p=2, dim=1)
    image_fea = F.normalize(image_fea,  p=2, dim=1)

    # 2. 余弦相似度矩阵 [bs, bs]
    sim_matrix = torch.matmul(sketch_fea, image_fea.t())

    # 距离升序排序
    sorted_indices = sim_matrix.argsort(dim=1, descending=True)

    # 正确的 image 匹配是与自身同位置的样本（即第 i 个 sketch 对应第 i 个 image）
    ground_truth = torch.arange(n_skh, device=device).unsqueeze(1)  # [bs, 1]
    correct_matrix = (sorted_indices[:, :max(topk)] == ground_truth)  # [bs, max_k]

    accs = []
    idxes = []
    for i, k in enumerate(topk):
        # 计算准确率
        correct_k = correct_matrix[:, :k].any(dim=1)  # [bs]，是否在 top-k 中命中
        acc = correct_k.float().mean().item()
        accs.append(acc)

        # 提取命中的 sketch 索引
        matched_sketch_indices = torch.nonzero(correct_k, as_tuple=False).squeeze(1)  # [num_matched]
        idxes.append(matched_sketch_indices)

    return accs, idxes


def compute_topk_accuracy(
    sketch_fea: torch.Tensor,
    image_fea: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5),
):
    """
    Sketch → Image retrieval Acc@k with image feature deduplication.

    Args:
        sketch_fea: Tensor [N, C]
        image_fea:  Tensor [N, C], may contain duplicates
        topk:       tuple of int, e.g. (1, 5)

    Returns:
        acc_dict: {k: float}
        success_indices: {k: Tensor of sketch indices}
    """

    assert sketch_fea.shape == image_fea.shape
    device = sketch_fea.device
    N = sketch_fea.shape[0]

    # --------------------------------------------------
    # 1. 去重 image_fea，并建立 GT 映射
    # --------------------------------------------------
    unique_image_fea, inverse_idx = unique_float_tensor(
        image_fea, eps=1e-6
    )
    # inverse_idx[i] = GT image index for sketch i

    # --------------------------------------------------
    # 2. 归一化（余弦相似度）
    # --------------------------------------------------
    sketch_fea = F.normalize(sketch_fea, dim=1)
    unique_image_fea = F.normalize(unique_image_fea, dim=1)

    # --------------------------------------------------
    # 3. 相似度矩阵 [N, N_unique]
    # --------------------------------------------------
    sim_matrix = sketch_fea @ unique_image_fea.t()

    # --------------------------------------------------
    # 4. 排序（相似度降序）
    # --------------------------------------------------
    sorted_indices = sim_matrix.argsort(dim=1, descending=True)

    # --------------------------------------------------
    # 5. 计算 Acc@k
    # --------------------------------------------------
    acc_dict = []
    success_indices= []

    for k in topk:
        topk_indices = sorted_indices[:, :k]           # [N, k]
        gt = inverse_idx.view(-1, 1)                    # [N, 1]

        correct = (topk_indices == gt).any(dim=1)       # [N]
        acc_dict.append(correct.float().mean().item())
        success_indices.append(torch.nonzero(correct, as_tuple=False).squeeze(1))

    return acc_dict, success_indices


def unique_float_tensor(
    x: torch.Tensor,
    eps: float = 1e-6
):
    """
    Float-safe unique for [N, C] tensor.

    Returns:
        unique_x: Tensor [M, C]
        inverse_idx: Tensor [N]
    """
    # 1. 量化到整数空间
    x_q = torch.round(x / eps).to(torch.int64)

    # 2. unique on quantized tensor
    _, inverse_idx = torch.unique(
        x_q, dim=0, return_inverse=True
    )

    # 3. 用 inverse_idx 聚合真实 float 值
    unique_indices = torch.unique(inverse_idx, sorted=True)
    unique_x = torch.zeros(
        (len(unique_indices), x.shape[1]),
        device=x.device,
        dtype=x.dtype
    )

    for new_i, old_i in enumerate(unique_indices):
        unique_x[new_i] = x[inverse_idx == old_i][0]

    return unique_x, inverse_idx


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


@torch.no_grad()
def compute_acc_at_k_with_indices(
    sketch_features: torch.Tensor,   # [m, fea]
    image_features: torch.Tensor,    # [n, fea]
    paired_idx: torch.Tensor,        # [m]
    topk_tuple=(1, 5)
):
    """
    Returns:
        dict[k] = {
            'acc': float,
            'correct_indices': LongTensor
        }
    """

    # 2. similarity matrix [m, n]
    sim_matrix = sketch_features @ image_features.t()

    max_k = max(topk_tuple)

    # 3. top-k retrieval
    _, topk_indices = sim_matrix.topk(
        k=max_k,
        dim=1,
        largest=True,
        sorted=True
    )  # [m, max_k]

    # 4. GT index
    gt_idx = torch.tensor(paired_idx).view(-1, 1)  # [m, 1]

    accs = []
    indices = []
    for k in topk_tuple:
        # [m] bool
        correct_k = (topk_indices[:, :k] == gt_idx).any(dim=1)

        # Acc@k
        acc_k = correct_k.float().mean().item()

        # 命中的草图索引
        correct_indices = torch.nonzero(correct_k, as_tuple=False).squeeze(1)

        accs.append(acc_k)
        indices.append(correct_indices)

    return accs, indices

