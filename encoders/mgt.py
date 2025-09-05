"""
基于MGT (Multi-scale Graph Transformer) 的草图编码器
适配当前项目的草图-图像对齐任务

MGT输入格式: [Cn, fn, p] ∈ RS×4
- Cn = {(xsn, ysn)} ∈ RS×2 坐标序列
- fn: flag_bits (pen state)
- p: position encoding

当前项目输入格式: Stroke-5 [x, y, p1, p2, p3] ∈ RS×5
需要进行格式转换和适配
"""
import torch
import torch.nn as nn
import os
import math
import torch.nn.functional as F


class SkipConnection(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, input, mask=None):
        return input + self.module(input, mask=mask)


class Normalization(nn.Module):
    def __init__(self, embed_dim, normalization='batch'):
        super().__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters
        # with bias 0 and weight unif(0,1) which is too large!
        self.init_parameters()

    def init_parameters(self):
        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input, mask=None):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, input_dim, embed_dim=None,
                 val_dim=None, key_dim=None, dropout=0.1):
        super().__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
        Args:
            q: Input queries (batch_size, n_query, input_dim)
            h: Input data (batch_size, graph_size, input_dim)
            mask: Input attention mask (batch_size, n_query, graph_size)
                  or viewable as that (i.e. can be 2 dim if n_query == 1);
                  Mask should contain -inf if attention is not possible
                  (i.e. mask is a negative adjacency matrix)

        Returns:
            out: Updated data after attention (batch_size, graph_size, input_dim)
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        dropt1_qflat = self.dropout_1(qflat)
        Q = torch.matmul(dropt1_qflat, self.W_query).view(shp_q)

        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        dropt2_hflat = self.dropout_2(hflat)
        K = torch.matmul(dropt2_hflat, self.W_key).view(shp)

        dropt3_hflat = self.dropout_3(hflat)
        V = torch.matmul(dropt3_hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility = compatibility + mask.type_as(compatibility)

        attn = F.softmax(compatibility, dim=-1)

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        # out = self.drop(out)

        return out


class PositionWiseFeedforward(nn.Module):
    def __init__(self, embed_dim, feedforward_dim=512, dropout=0.1):
        super().__init__()
        # modified on 2019 10 23
        self.sub_layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim, bias=True),
            nn.ReLU()
        )

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input, mask=None):
        return self.sub_layers(input)


class MultiGraphTransformerLayer(nn.Module):
    def __init__(self, n_heads, embed_dim, feedforward_dim,
                 normalization='batch', dropout=0.1):
        super().__init__()

        self.self_attention1 = SkipConnection(
            MultiHeadAttention(
                n_heads=n_heads,
                input_dim=embed_dim,
                embed_dim=embed_dim,
                dropout=dropout
            )
        )
        self.self_attention2 = SkipConnection(
            MultiHeadAttention(
                n_heads=n_heads,
                input_dim=embed_dim,
                embed_dim=embed_dim,
                dropout=dropout
            )
        )

        self.self_attention3 = SkipConnection(
            MultiHeadAttention(
                n_heads=n_heads,
                input_dim=embed_dim,
                embed_dim=embed_dim,
                dropout=dropout
            )
        )
        # modified on 2019 10 26.
        self.tmp_linear_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 3, embed_dim, bias=True),
            nn.ReLU(),
        )

        self.norm1 = Normalization(embed_dim, normalization)

        self.positionwise_ff = SkipConnection(
            PositionWiseFeedforward(
                embed_dim=embed_dim,
                feedforward_dim=feedforward_dim,
                dropout=dropout
            )
        )
        self.norm2 = Normalization(embed_dim, normalization)

    def forward(self, input, mask1, mask2, mask3):
        # ipdb.set_trace()
        h1 = self.self_attention1(input, mask=mask1)
        h2 = self.self_attention2(input, mask=mask2)
        h3 = self.self_attention3(input, mask=mask3)
        hh = torch.cat((h1, h2, h3), dim=2)
        hh = self.tmp_linear_layer(hh)
        # ipdb.set_trace()
        hh = self.norm1(hh, mask=mask1)
        hh = self.positionwise_ff(hh, mask=mask1)
        hh = self.norm2(hh, mask=mask1)
        return hh


class GraphTransformerEncoder(nn.Module):
    def __init__(self, coord_input_dim, feat_input_dim, feat_dict_size, n_layers=6, n_heads=8,
                 embed_dim=512, feedforward_dim=2048, normalization='batch', dropout=0.1):
        super().__init__()

        # Embedding/Input layers
        self.coord_embed = nn.Linear(coord_input_dim, embed_dim, bias=False)
        self.feat_embed = nn.Embedding(feat_dict_size, embed_dim)
        # self.in_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.transformer_layers = nn.ModuleList([
            MultiGraphTransformerLayer(n_heads, embed_dim * 3, feedforward_dim, normalization, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, coord, flag, pos, attention_mask1=None, attention_mask2=None, attention_mask3=None):
        # Embed inputs to embed_dim
        # h = self.coord_embed(coord) + self.feat_embed(flag) + self.feat_embed(pos)
        h = torch.cat((self.coord_embed(coord), self.feat_embed(flag)), dim=2)
        h = torch.cat((h, self.feat_embed(pos)), dim=2)
        # h = self.in_drop(h)

        # Perform n_layers of Graph Transformer blocks
        for layer in self.transformer_layers:
            h = layer(h, mask1=attention_mask1, mask2=attention_mask2, mask3=attention_mask3)

        return h


def produce_adjacent_matrix_2_neighbors(flag_bits, stroke_lengths):
    """生成2邻居邻接矩阵（从MGT复制）"""
    batch_size, seq_len = flag_bits.shape[0], flag_bits.shape[1]
    device = flag_bits.device
    adja_matr = torch.full((batch_size, seq_len, seq_len), -1e10, device=device, dtype=torch.float32)

    for b in range(batch_size):
        current_len = min(stroke_lengths[b].item(), seq_len)
        flag_b = flag_bits[b, :current_len]  # [current_len]

        # 自连接
        for i in range(current_len):
            adja_matr[b, i, i] = 0

        # 相邻连接
        for i in range(current_len - 1):
            if flag_b[i] == 100:  # pen down连接
                adja_matr[b, i, i + 1] = 0
                adja_matr[b, i + 1, i] = 0

    return adja_matr


def produce_adjacent_matrix_4_neighbors(flag_bits, stroke_lengths):
    """生成4邻居邻接矩阵（从MGT复制并适配）"""
    batch_size, seq_len = flag_bits.shape[0], flag_bits.shape[1]
    device = flag_bits.device
    adja_matr = torch.full((batch_size, seq_len, seq_len), -1e10, device=device, dtype=torch.float32)

    for b in range(batch_size):
        current_len = min(stroke_lengths[b].item(), seq_len)
        flag_b = flag_bits[b, :current_len]

        # 自连接
        for i in range(current_len):
            adja_matr[b, i, i] = 0

        # 扩展邻接连接
        for i in range(current_len):
            # 前1个和前2个邻居
            if i >= 1 and flag_b[i - 1] == 100:
                adja_matr[b, i, i - 1] = 0
                adja_matr[b, i - 1, i] = 0
            if i >= 2 and flag_b[i - 2] == 100 and flag_b[i - 1] == 100:
                adja_matr[b, i, i - 2] = 0
                adja_matr[b, i - 2, i] = 0

            # 后1个和后2个邻居
            if i < current_len - 1 and flag_b[i] == 100:
                adja_matr[b, i, i + 1] = 0
                adja_matr[b, i + 1, i] = 0
            if i < current_len - 2 and flag_b[i] == 100 and flag_b[i + 1] == 100:
                adja_matr[b, i, i + 2] = 0
                adja_matr[b, i + 2, i] = 0

    return adja_matr


def produce_adjacent_matrix_joint_neighbors(flag_bits, stroke_lengths):
    """生成joint邻居邻接矩阵（从MGT复制并适配）"""
    batch_size, seq_len = flag_bits.shape[0], flag_bits.shape[1]
    device = flag_bits.device
    adja_matr = torch.full((batch_size, seq_len, seq_len), -1e10, device=device, dtype=torch.float32)

    for b in range(batch_size):
        current_len = min(stroke_lengths[b].item(), seq_len)
        flag_b = flag_bits[b, :current_len]

        # 自连接
        for i in range(current_len):
            adja_matr[b, i, i] = 0

        # 首尾连接
        if current_len > 1:
            adja_matr[b, 0, current_len - 1] = 0
            adja_matr[b, current_len - 1, 0] = 0

        # 顺序连接
        for i in range(current_len - 1):
            if flag_b[i] == 101:  # pen down to next
                adja_matr[b, i, i + 1] = 0
                adja_matr[b, i + 1, i] = 0

    return adja_matr


def generate_padding_mask(stroke_lengths, max_len):
    """生成padding mask"""
    batch_size = len(stroke_lengths)
    device = stroke_lengths.device
    padding_mask = torch.ones(batch_size, max_len, 1, dtype=torch.float32, device=device)

    for b in range(batch_size):
        actual_len = min(stroke_lengths[b].item(), max_len)
        if actual_len < max_len:
            padding_mask[b, actual_len:, :] = 0

    return padding_mask


def convert_stroke5_to_mgt_format(stroke5_data, stroke5_mask):
    """
    将Stroke-5格式转换为MGT格式

    Args:
        stroke5_data: [batch_size, seq_len, 5] - [x, y, p1, p2, p3]
        stroke5_mask: [batch_size, seq_len] - 有效点掩码

    Returns:
        coordinate: [batch_size, seq_len, 2] - (x, y) 坐标
        flag_bits: [batch_size, seq_len, 1] - pen state标志
        stroke_lengths: [batch_size] - 实际序列长度
        position_encoding: [batch_size, seq_len, 1] - 位置编码
    """
    batch_size, seq_len, _ = stroke5_data.shape
    device = stroke5_data.device

    # 提取坐标 (x, y)
    coordinate = stroke5_data[:, :, :2]  # [batch_size, seq_len, 2]

    # 转换pen state为flag_bits
    # stroke-5: [x, y, p1, p2, p3] 其中 p1=1表示pen down, p2=1表示pen up, p3=1表示end of sketch
    pen_states = stroke5_data[:, :, 2:]  # [batch_size, seq_len, 3]

    # 创建flag_bits: 100表示pen down连接, 101表示pen up/lift
    flag_bits = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    # p1=1: pen down (继续画线) -> 100
    # p2=1: pen up (提笔) -> 101
    # p3=1: end of sketch -> 101
    pen_down_mask = pen_states[:, :, 0] == 1  # p1=1
    pen_up_mask = (pen_states[:, :, 1] == 1) | (pen_states[:, :, 2] == 1)  # p2=1 or p3=1

    flag_bits[pen_down_mask] = 100  # f1: stroke continuing or starting
    flag_bits[pen_up_mask] = 101  # f2: stroke ending (pen-up or end-of-sketch)
    # f3: padding points (beyond actual stroke length)
    pad_mask = stroke5_mask == 0
    flag_bits[pad_mask] = 102  # padding flag

    # 计算实际序列长度
    stroke_lengths = stroke5_mask.sum(dim=1).long()  # [batch_size]

    # 生成位置编码 - 限制在MGT字典范围内
    # feat_dict_size = seq_len + 3 (位置编码字典大小)
    max_pos_id = seq_len + 2  # feat_dict_size - 1
    position_encoding = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
    position_encoding = torch.clamp(position_encoding, 0, max_pos_id)

    return coordinate, flag_bits, stroke_lengths, position_encoding


class MGTSketchEncoder(nn.Module):
    """
    基于MGT的草图编码器，适配当前项目的Stroke-5输入格式
    """
    def __init__(self,
                 embed_dim=512,
                 n_layers=4,
                 n_heads=8,
                 feedforward_dim=2048,
                 dropout=0.25,
                 max_seq_len=256,  # 可配置的序列长度
                 pretrained_path=None):
        """
        Args:
            embed_dim: 嵌入维度
            n_layers: Transformer层数
            n_heads: 注意力头数
            feedforward_dim: 前馈网络维度
            dropout: dropout率
            max_seq_len: 最大序列长度
            pretrained_path: 预训练MGT模型路径
        """
        super().__init__()
        # 嵌入维度和最大序列长度配置
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # MGT Graph Transformer Encoder，字典大小 = max_seq_len + 3 (位置编码 + 特殊符号)
        feat_dict_size = self.max_seq_len + 3
        self.mgt_encoder = GraphTransformerEncoder(
            coord_input_dim=2,  # (x, y) 坐标
            feat_input_dim=1,  # flag_bits维度
            feat_dict_size=feat_dict_size,
            n_layers=n_layers,
            n_heads=n_heads,
            embed_dim=embed_dim,
            feedforward_dim=feedforward_dim,
            normalization='batch',
            dropout=dropout
        )

        # 投影到目标嵌入维度
        self.projection = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )

        # 加载预训练权重（如果提供）
        if pretrained_path and os.path.exists(pretrained_path):
            self.load_pretrained_weights(pretrained_path)
            print(f'Loaded pretrained MGT weights from: {pretrained_path}')
        # else:
        #     if not MGT_ENCODER_AVAILABLE:
        #         print('Using simplified MGT encoder for testing (no original MGT available)')

    def load_pretrained_weights(self, pretrained_path):
        """加载预训练的MGT权重"""
        try:
            # if not MGT_ENCODER_AVAILABLE:
            #     print("Using simplified MGT encoder, skipping pretrained weight loading")
            #     return

            checkpoint = torch.load(pretrained_path, map_location='cpu')

            # 提取模型状态字典
            if 'network' in checkpoint:
                state_dict = checkpoint['network']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # 只加载encoder部分的权重
            encoder_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('encoder.'):
                    new_key = key.replace('encoder.', '')
                    encoder_state_dict[new_key] = value
                elif not key.startswith('mlp_classifier'):
                    encoder_state_dict[key] = value

            # 加载权重（允许部分匹配）
            missing_keys, unexpected_keys = self.mgt_encoder.load_state_dict(encoder_state_dict, strict=False)

            if missing_keys:
                print(f'Missing keys when loading MGT weights: {missing_keys[:5]}...' if len(
                    missing_keys) > 5 else missing_keys)
            if unexpected_keys:
                print(f'Unexpected keys when loading MGT weights: {unexpected_keys[:5]}...' if len(
                    unexpected_keys) > 5 else unexpected_keys)

        except Exception as e:
            print(f'Error loading pretrained MGT weights: {e}')

    def forward(self, stroke5_data, stroke5_mask):
        """
        前向传播

        Args:
            stroke5_data: [batch_size, seq_len, 5] - Stroke-5格式输入
            stroke5_mask: [batch_size, seq_len] - 有效点掩码

        Returns:
            sketch_features: [batch_size, embed_dim] - 草图特征向量
        """
        batch_size, seq_len, _ = stroke5_data.shape
        device = stroke5_data.device

        # 调整到MGT的固定长度
        if seq_len != self.max_seq_len:
            # 创建固定长度的tensor
            padded_data = torch.zeros(batch_size, self.max_seq_len, 5, device=device, dtype=stroke5_data.dtype)
            padded_mask = torch.zeros(batch_size, self.max_seq_len, device=device, dtype=stroke5_mask.dtype)

            actual_len = min(seq_len, self.max_seq_len)
            padded_data[:, :actual_len] = stroke5_data[:, :actual_len]
            padded_mask[:, :actual_len] = stroke5_mask[:, :actual_len]

            stroke5_data = padded_data
            stroke5_mask = padded_mask

        # 转换为MGT格式
        coordinate, flag_bits, stroke_lengths, position_encoding = convert_stroke5_to_mgt_format(
            stroke5_data, stroke5_mask
        )

        # 生成邻接矩阵
        attention_mask_2 = produce_adjacent_matrix_2_neighbors(flag_bits, stroke_lengths)
        attention_mask_4 = produce_adjacent_matrix_4_neighbors(flag_bits, stroke_lengths)
        attention_mask_joint = produce_adjacent_matrix_joint_neighbors(flag_bits, stroke_lengths)

        # 生成padding mask
        padding_mask = generate_padding_mask(stroke_lengths, self.max_seq_len).to(device)

        # 调整输入维度以匹配MGT期望的格式
        # flag_bits和position_encoding已经是正确的[batch_size, seq_len]格式

        # MGT Encoder前向传播
        node_features = self.mgt_encoder(
            coord=coordinate,
            flag=flag_bits,
            pos=position_encoding,
            attention_mask1=attention_mask_2,
            attention_mask2=attention_mask_4,
            attention_mask3=attention_mask_joint
        )  # [batch_size, seq_len, embed_dim * 3]

        # 应用padding mask并聚合为图级特征
        masked_features = node_features * padding_mask  # [batch_size, seq_len, embed_dim * 3]

        # 平均池化（只考虑有效点）
        valid_lengths = stroke_lengths.float().unsqueeze(-1)  # [batch_size, 1]
        valid_lengths = torch.clamp(valid_lengths, min=1.0)  # 避免除零

        graph_features = masked_features.sum(dim=1) / valid_lengths  # [batch_size, embed_dim * 3]

        # 投影到目标维度
        sketch_features = self.projection(graph_features)  # [batch_size, embed_dim]

        return sketch_features


def test():
    # 测试MGT草图编码器
    print("Testing MGT Sketch Encoder...")

    # 创建测试数据
    batch_size = 8
    seq_len = 256

    # Stroke-5格式测试数据
    stroke5_data = torch.randn(batch_size, seq_len, 5)
    stroke5_data[:, :, 2:] = torch.randint(0, 2, (batch_size, seq_len, 3)).float()  # pen states

    stroke5_mask = torch.ones(batch_size, seq_len)
    # 模拟不同的序列长度
    for i in range(batch_size):
        actual_len = torch.randint(50, seq_len, (1,)).item()
        stroke5_mask[i, actual_len:] = 0

    # 创建MGT编码器
    mgt_encoder = MGTSketchEncoder(embed_dim=512)

    # 前向传播测试
    with torch.no_grad():
        features = mgt_encoder(stroke5_data, stroke5_mask)
        print(f"Input shape: {stroke5_data.shape}")
        print(f"Output shape: {features.shape}")
        print(f"Feature stats: mean={features.mean().item():.4f}, std={features.std().item():.4f}")

    print("MGT Sketch Encoder test completed successfully!")


if __name__ == '__main__':
    test()

