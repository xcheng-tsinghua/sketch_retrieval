import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(
            tensor_all,
            tensor,
            async_op=False  # performance opt
        )

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


class ULIPWithImageLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, pc_embed, text_embed, image_embed, sketch_embed, logit_scale):
        # pc_embed = outputs['pc_embed']
        # text_embed = outputs['text_embed']
        # image_embed = outputs['image_embed']
        # logit_scale = outputs['logit_scale']
        local_batch_size = pc_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * get_rank() + torch.arange(
                local_batch_size, device=pc_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # normalized features
        pc_embed = F.normalize(pc_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        sketch_embed = F.normalize(sketch_embed, dim=-1, p=2)

        # gather features from all GPUs
        pc_embed_all, text_embed_all, image_embed_all, sketch_embed_all = \
            all_gather_batch([pc_embed, text_embed, image_embed, sketch_embed])

        # cosine similarity as logits
        # logits_per_pc_text = logit_scale * pc_embed @ text_embed_all.t()
        # logits_per_text_pc = logit_scale * text_embed @ pc_embed_all.t()
        #
        # logits_per_pc_image = logit_scale * pc_embed @ image_embed_all.t()
        # logits_per_image_pc = logit_scale * image_embed @ pc_embed_all.t()

        logits_per_sketch_image = logit_scale * sketch_embed @ image_embed_all.t()
        logits_per_image_sketch = logit_scale * image_embed @ sketch_embed_all.t()

        logits_per_sketch_pc = logit_scale * sketch_embed @ pc_embed_all.t()
        logits_per_pc_sketch = logit_scale * pc_embed @ sketch_embed_all.t()

        logits_per_sketch_text = logit_scale * sketch_embed @ text_embed_all.t()
        logits_per_text_sketch = logit_scale * text_embed @ sketch_embed_all.t()

        # loss = (F.cross_entropy(logits_per_pc_text, self.labels) + \
        #         F.cross_entropy(logits_per_text_pc, self.labels)) / 2 + \
        #         (F.cross_entropy(logits_per_pc_image, self.labels) +
        #          F.cross_entropy(logits_per_image_pc, self.labels)) / 2

        loss = (F.cross_entropy(logits_per_sketch_image, self.labels) + \
                F.cross_entropy(logits_per_image_sketch, self.labels)) / 2 + \
                (F.cross_entropy(logits_per_sketch_pc, self.labels) +
                F.cross_entropy(logits_per_pc_sketch, self.labels)) / 2 + \
                (F.cross_entropy(logits_per_sketch_text, self.labels) +
                F.cross_entropy(logits_per_text_sketch, self.labels)) / 2

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_sketch_text, dim=-1)
            correct = pred.eq(self.labels).sum()
            sketch_text_acc = 100 * correct / local_batch_size

            pred = torch.argmax(logits_per_sketch_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            sketch_image_acc = 100 * correct / local_batch_size

            pred = torch.argmax(logits_per_sketch_pc, dim=-1)
            correct = pred.eq(self.labels).sum()
            sketch_pc_acc = 100 * correct / local_batch_size

        return {'loss': loss, 'ulip_loss': loss, 'sketch_image_acc': sketch_image_acc, 'sketch_text_acc': sketch_text_acc, 'sketch_pc_acc': sketch_pc_acc}


class SketchImageAlignmentLoss(nn.Module):
    """
    基于ULIP原始方法的草图-图像对齐损失函数
    专门用于草图检索图片的任务，仅训练草图encoder
    """
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, sketch_embed, image_embed, logit_scale):
        """
        Args:
            sketch_embed: 草图特征 [batch_size, embed_dim]
            image_embed: 图像特征 [batch_size, embed_dim]
            logit_scale: 温度参数
        """
        
        local_batch_size = sketch_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * get_rank() + torch.arange(
                local_batch_size, device=sketch_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # normalized features - 按照ULIP原始方法进行L2归一化
        sketch_embed = F.normalize(sketch_embed, dim=-1, p=2)
        image_embed = F.normalize(image_embed, dim=-1, p=2)

        # gather features from all GPUs
        sketch_embed_all, image_embed_all = all_gather_batch([sketch_embed, image_embed])

        # cosine similarity as logits - 按照ULIP原始方法计算相似度
        logits_per_sketch_image = logit_scale * sketch_embed @ image_embed_all.t()
        logits_per_image_sketch = logit_scale * image_embed @ sketch_embed_all.t()

        # 对称的对比学习损失 - 按照ULIP原始方法
        loss = (F.cross_entropy(logits_per_sketch_image, self.labels) + \
                F.cross_entropy(logits_per_image_sketch, self.labels)) / 2

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_sketch_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            sketch_image_acc = 100 * correct / local_batch_size

        return {'loss': loss, 'sketch_image_acc': sketch_image_acc}


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pcd_embed, text_embed, image_embed, sketch_embed, logit_scale):
        loss_skh_txt = F.mse_loss(sketch_embed, text_embed)
        loss_skh_pcd = F.mse_loss(sketch_embed, pcd_embed)
        loss_skh_img = F.mse_loss(sketch_embed, image_embed)
        loss = loss_skh_txt + loss_skh_pcd + loss_skh_img

        return {'loss': loss}


def constructive_loss(x, y, margin=1.0, lambda_=1.0):
    """
    对比损失
    :param x: [bs ,emb]
    :param y: [bs ,emb]
    :param margin:
    :param lambda_:
    :return:
    """
    # x, y: tensors of shape (N, D)
    N = x.size(0)

    # 计算对应行之间的距离
    pos_dist = F.pairwise_distance(x, y, p=2)
    pos_loss = torch.mean(pos_dist ** 2)

    # 计算 x 与 y 中所有不同行之间的距离
    x_exp = x.unsqueeze(1)  # (N, 1, D)
    y_exp = y.unsqueeze(0)  # (1, N, D)
    dist_matrix = torch.norm(x_exp - y_exp, dim=2, p=2)  # (N, N)

    # 创建掩码，排除对角线（即对应行）
    mask = ~torch.eye(N, dtype=torch.bool, device=x.device)
    neg_dist = dist_matrix[mask]

    # 计算不同行之间的损失
    neg_loss = torch.mean(F.relu(margin - neg_dist) ** 2)

    # 总损失
    loss = pos_loss + lambda_ * neg_loss
    return loss


def contrastive_loss_cl_zs_sbir(sketch_tensor, image_tensor, class_tensor, logit_scale=0.0, margin=1.0):
    bs = sketch_tensor.size(0)

    # 计算 pairwise 欧氏距离: [bs, bs]
    dist_matrix = torch.cdist(sketch_tensor, image_tensor, p=2)  # [bs, bs]

    # 构造 label matrix: [bs, bs]，同类为1，异类为0
    label_matrix = (class_tensor.unsqueeze(1) == class_tensor.unsqueeze(0)).float()  # [bs, bs]

    # 正样本: label = 1 → loss = distance^2
    pos_loss = label_matrix * dist_matrix.pow(2)

    # 负样本: label = 0 → loss = max(0, margin - distance)^2
    neg_loss = (1 - label_matrix) * F.relu(margin - dist_matrix).pow(2)

    # 总 loss 归一化
    loss = (pos_loss.sum() + neg_loss.sum()) / (bs * bs)
    return loss


def contrastive_loss_fg_zs_sbir(sketch_tensor, image_tensor, class_tensor, logit_scale=0.0, margin=1.0):
    bs = sketch_tensor.size(0)

    # 计算配对距离（正样本）
    pos_dist = F.pairwise_distance(sketch_tensor, image_tensor, p=2)  # [bs]

    # 构造负样本距离矩阵
    dist_matrix = torch.cdist(sketch_tensor, image_tensor, p=2)  # [bs, bs]
    neg_mask = ~torch.eye(bs, dtype=torch.bool, device=sketch_tensor.device)

    neg_dist = dist_matrix[neg_mask].view(bs, bs - 1)  # [bs, bs-1]

    # 正样本 loss
    pos_loss = pos_dist.pow(2).mean()

    # 负样本 loss：max(0, margin - distance)^2
    neg_loss = F.relu(margin - neg_dist).pow(2).mean()

    return pos_loss + neg_loss


class ContrastiveLoss(nn.Module):
    """
    对比学习损失函数，用于草图-图像对齐训练
    基于InfoNCE损失
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, sketch_features, image_features, class_tensor=None, logit_scale=None):
        """
        计算对比学习损失
        
        Args:
            sketch_features: 草图特征 [batch_size, embed_dim]
            image_features: 图像特征 [batch_size, embed_dim] 
            logit_scale: 温度参数缩放因子
            
        Returns:
            loss: 对比学习损失
        """
        local_batch_size = sketch_features.size(0)
        device = sketch_features.device
        
        # 准备标签
        if local_batch_size != self.last_local_batch_size:
            self.labels = torch.arange(local_batch_size, device=device)
            self.last_local_batch_size = local_batch_size
        
        # L2归一化特征
        sketch_features = F.normalize(sketch_features, dim=-1, p=2)
        image_features = F.normalize(image_features, dim=-1, p=2)
        
        # 使用提供的温度参数或默认值
        if logit_scale is not None:
            temperature = logit_scale
        else:
            temperature = 1.0 / self.temperature
        
        # 计算相似度矩阵
        logits_sketch_to_image = temperature * torch.matmul(sketch_features, image_features.t())
        logits_image_to_sketch = temperature * torch.matmul(image_features, sketch_features.t())
        
        # 计算对称的交叉熵损失
        loss_sketch_to_image = F.cross_entropy(logits_sketch_to_image, self.labels)
        loss_image_to_sketch = F.cross_entropy(logits_image_to_sketch, self.labels)
        
        # 平均损失
        loss = (loss_sketch_to_image + loss_image_to_sketch) / 2.0
        
        return loss


def triplet_loss(x, args):
    """

    :param x: 4*batch -> sk_p, sk_n, im_p, im_n
    :param args:
    :return:
    """
    triplet = nn.TripletMarginLoss(margin=1.0, p=2).cuda()
    sk_p = x[0:args.batch]
    im_p = x[2 * args.batch:3 * args.batch]
    im_n = x[3 * args.batch:]
    loss = triplet(sk_p, im_p, im_n)

    return loss


def rn_loss(predict, target):
    mse_loss = nn.MSELoss().cuda()
    loss = mse_loss(predict, target)

    return loss


def info_nce_loss(
    sketch_fea,
    pos_img_fea,
    neg_img_fea,
    temperature=0.07
):
    """
    sketch_fea:   [B, D]
    pos_img_fea:  [B, D]
    neg_img_fea:  [B, D]
    """

    # 1. L2 normalize（极其重要）
    sketch_fea  = F.normalize(sketch_fea, dim=1)
    pos_img_fea = F.normalize(pos_img_fea, dim=1)
    neg_img_fea = F.normalize(neg_img_fea, dim=1)

    # 2. cosine similarity
    pos_sim = torch.sum(sketch_fea * pos_img_fea, dim=1) / temperature
    neg_sim = torch.sum(sketch_fea * neg_img_fea, dim=1) / temperature

    # 3. logits: [B, 2]，第 0 维是正样本
    logits = torch.stack([pos_sim, neg_sim], dim=1)

    # 4. label 永远是 0（正样本在第 0 位）
    labels = torch.zeros(sketch_fea.size(0), dtype=torch.long, device=sketch_fea.device)

    loss = F.cross_entropy(logits, labels)
    return loss


def info_nce_multi_neg(
    sketch_fea,
    pos_img_fea,
    neg_img_fea,
    temperature=0.07
):
    """
    多负样本

    sketch_fea:   [B, D]
    pos_img_fea:  [B, D]
    neg_img_fea:  [B, K, D]
    """
    B, K, D = neg_img_fea.shape

    sketch_fea = F.normalize(sketch_fea, dim=1)
    pos_img_fea = F.normalize(pos_img_fea, dim=1)
    neg_img_fea = F.normalize(neg_img_fea, dim=2)

    # 正样本相似度 [B, 1]
    pos_sim = torch.sum(sketch_fea * pos_img_fea, dim=1, keepdim=True)

    # 负样本相似度 [B, K]
    neg_sim = torch.bmm(
        neg_img_fea,                 # [B, K, D]
        sketch_fea.unsqueeze(2)      # [B, D, 1]
    ).squeeze(2)

    logits = torch.cat([pos_sim, neg_sim], dim=1) / temperature

    labels = torch.zeros(B, dtype=torch.long, device=sketch_fea.device)

    loss = F.cross_entropy(logits, labels)
    return loss


