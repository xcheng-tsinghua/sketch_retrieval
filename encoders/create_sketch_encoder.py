"""
添加 Encoders 时，需要在 supported_encoders 中添加对应的信息
"""
import torch.nn as nn
import timm

from encoders import lstm
from sdgraph import sdgraph_sel, sdgraph_endsnap
from encoders import gru
from encoders import sketch_transformer
import options


class PNGSketchEncoder(nn.Module):
    """
    PNG草图编码器，基于Vision Transformer
    """
    def __init__(self, 
                 model_name='vit_base_patch16_224',
                 pretrained=True,
                 freeze_backbone=False,
                 output_dim=512,
                 dropout_rate=0.1):
        """
        初始化PNG草图编码器
        
        Args:
            model_name: 预训练模型名称
            pretrained: 是否使用预训练权重
            freeze_backbone: 是否冻结主干网络
            output_dim: 输出特征维度
            dropout_rate: Dropout率
        """
        super(PNGSketchEncoder, self).__init__()
        print('create vit_base_patch16_224 encoder')
        
        self.model_name = model_name
        self.output_dim = output_dim
        self.freeze_backbone = freeze_backbone
        
        # 创建ViT模型
        if 'vit' in model_name.lower():
            self.vision_model = timm.create_model(
                model_name, 
                pretrained=pretrained,
                num_classes=0,  # 移除分类头
                global_pool=''  # 移除全局池化
            )
            
            # 获取特征维度
            if hasattr(self.vision_model, 'embed_dim'):
                hidden_dim = self.vision_model.embed_dim
            elif hasattr(self.vision_model, 'num_features'):
                hidden_dim = self.vision_model.num_features
            else:
                hidden_dim = 768  # ViT-Base默认维度
                
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        # 冻结主干网络参数
        if freeze_backbone:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            print(f"已冻结{model_name}的主干网络参数")
        
        # 投影层
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout_rate)
        )
        
        # 初始化投影层权重
        self._init_projection_weights()
        
        # print(f"PNGSketchEncoder initialized:")
        # print(f"  Model: {model_name}")
        # print(f"  Pretrained: {pretrained}")
        # print(f"  Freeze backbone: {freeze_backbone}")
        # print(f"  Hidden dim: {hidden_dim}")
        # print(f"  Output dim: {output_dim}")
    
    def _init_projection_weights(self):
        """初始化投影层权重"""
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, sketch_images):
        """
        前向传播
        
        Args:
            sketch_images: PNG草图张量 [batch_size, 3, 224, 224]
            
        Returns:
            sketch_features: 草图特征 [batch_size, output_dim]
        """
        batch_size = sketch_images.size(0)
        
        # 通过ViT模型获取特征
        if hasattr(self.vision_model, 'forward_features'):
            # timm模型的特征提取方法
            features = self.vision_model.forward_features(sketch_images)
            
            # 获取CLS token特征或全局平均池化
            if features.dim() == 3:  # [batch_size, seq_len, hidden_dim]
                # 使用CLS token（第一个token）
                features = features[:, 0, :]  # [batch_size, hidden_dim]
            elif features.dim() == 2:  # 已经是[batch_size, hidden_dim]
                pass
            else:
                # 如果是其他维度，进行全局平均池化
                features = features.mean(dim=[-2, -1])  # [batch_size, hidden_dim]
        else:
            # 标准的前向传播
            features = self.vision_model(sketch_images)
            if features.dim() > 2:
                features = features.mean(dim=[-2, -1])
        
        # 通过投影层
        sketch_features = self.projection(features)
        
        # L2归一化
        sketch_features = nn.functional.normalize(sketch_features, p=2, dim=1)

        # sketch_features = MLP(sketch_features)
        # cls_logis = F.log_softmax(sketch_features, dim=1)
        # return cls_logis, sketch_features

        return sketch_features
    
    def get_trainable_parameters(self):
        """获取可训练参数"""
        trainable_params = []
        frozen_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append((name, param))
            else:
                frozen_params.append((name, param))
        
        return trainable_params, frozen_params
    
    def get_parameter_count(self):
        """获取参数数量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params
        }


class PNGSketchEncoderWithAttention(PNGSketchEncoder):
    """
    带注意力机制的PNG草图编码器
    """
    def __init__(self, 
                 model_name='vit_base_patch16_224',
                 pretrained=True,
                 freeze_backbone=False,
                 output_dim=512,
                 dropout_rate=0.1,
                 use_cross_attention=False):
        super().__init__(model_name, pretrained, freeze_backbone, output_dim, dropout_rate)
        print('create vit_base_patch16_224 with attention encoder')
        
        self.use_cross_attention = use_cross_attention
        
        if use_cross_attention:
            # 获取隐藏维度
            if hasattr(self.vision_model, 'embed_dim'):
                hidden_dim = self.vision_model.embed_dim
            else:
                hidden_dim = 768
            
            # 交叉注意力层
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True
            )
            
            # 添加层归一化
            self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, sketch_images, context_features=None):
        """
        前向传播（带可选的交叉注意力）
        
        Args:
            sketch_images: PNG草图张量 [batch_size, 3, 224, 224]
            context_features: 上下文特征（用于交叉注意力） [batch_size, seq_len, hidden_dim]
            
        Returns:
            sketch_features: 草图特征 [batch_size, output_dim]
        """
        batch_size = sketch_images.size(0)
        
        # 通过ViT模型获取特征
        if hasattr(self.vision_model, 'forward_features'):
            features = self.vision_model.forward_features(sketch_images)
        else:
            features = self.vision_model(sketch_images)
        
        # 如果启用交叉注意力且提供了上下文特征
        if self.use_cross_attention and context_features is not None:
            if features.dim() == 3:  # [batch_size, seq_len, hidden_dim]
                # 使用交叉注意力
                attended_features, _ = self.cross_attention(
                    query=features,
                    key=context_features,
                    value=context_features
                )
                
                # 残差连接和层归一化
                features = self.layer_norm(features + attended_features)
                
                # 使用CLS token
                features = features[:, 0, :]
            else:
                # 如果features不是3维，直接使用
                if features.dim() > 2:
                    features = features.mean(dim=[-2, -1])
        else:
            # 标准处理
            if features.dim() == 3:
                features = features[:, 0, :]  # 使用CLS token
            elif features.dim() > 2:
                features = features.mean(dim=[-2, -1])
        
        # 通过投影层
        sketch_features = self.projection(features)
        
        # L2归一化
        sketch_features = nn.functional.normalize(sketch_features, p=2, dim=1)
        
        return sketch_features


def create_sketch_encoder(model_name,
                          output_dim=512,
                          pretrained=False,
                          freeze_backbone=False,
                          dropout=0.1,
                          use_attention=False,
                          sketch_format=None,
                          ):
    """
    创建PNG草图编码器
    
    Args:
        model_name: 预训练模型名称
        pretrained: 是否使用预训练权重
        freeze_backbone: 是否冻结主干网络
        output_dim: 输出特征维度
        dropout: Dropout率
        use_attention: 是否使用注意力机制
        sketch_format: 额外参数表
        
    Returns:
        encoder: PNG草图编码器
    """
    sketch_format = options.parse_sketch_format(sketch_format)

    if model_name == 'vit':
        if use_attention:
            encoder = PNGSketchEncoderWithAttention(
                model_name='vit_base_patch16_224',
                pretrained=pretrained,
                freeze_backbone=freeze_backbone,
                output_dim=output_dim,
                dropout_rate=dropout,
                use_cross_attention=True
            )
        else:
            encoder = PNGSketchEncoder(
                model_name='vit_base_patch16_224',
                pretrained=pretrained,
                freeze_backbone=freeze_backbone,
                output_dim=output_dim,
                dropout_rate=dropout
            )

    elif model_name == 'lstm':
        encoder = lstm.BiLSTMEncoder(
            embed_dim=output_dim,
            bidirectional=False,
            dropout=dropout
        )

    elif model_name == 'bidir_lstm':
        encoder = lstm.BiLSTMEncoder(
            embed_dim=output_dim,
            bidirectional=True,
            dropout=dropout
        )

    elif model_name == 'sdgraph':
        encoder = sdgraph_sel.SDGraphEmbedding(
            channel_out=output_dim,
            n_stk=sketch_format['n_stk'],
            n_stk_pnt=sketch_format['n_stk_pnt'],
            dropout=dropout
        )

    elif model_name == 'sketch_transformer':
        encoder = sketch_transformer.SketchTransformer(
            max_length=sketch_format['max_length'],
            embed_dim=output_dim,
            dropout=dropout
        )

    elif model_name == 'gru':
        encoder = gru.GRUEncoder(
            embed_dim=output_dim,
            bidirectional=False,
            dropout=dropout
        )

    elif model_name == 'bidir_gru':
        encoder = gru.GRUEncoder(
            embed_dim=output_dim,
            bidirectional=True,
            dropout=dropout
        )

    else:
        raise TypeError('unsupported encoder name')

    return encoder


if __name__ == '__main__':
    # # 测试PNG草图编码器
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"使用设备: {device}")
    #
    # # 创建编码器
    # encoder = create_sketch_encoder(
    #     model_name='vit_base_patch16_224',
    #     pretrained=True,
    #     freeze_backbone=True,
    #     output_dim=512
    # )
    # encoder.to(device)
    #
    # # 测试前向传播
    # batch_size = 4
    # test_sketches = torch.randn(batch_size, 3, 224, 224).to(device)
    #
    # with torch.no_grad():
    #     features = encoder(test_sketches)
    #
    # print(f"\\n测试结果:")
    # print(f"输入形状: {test_sketches.shape}")
    # print(f"输出形状: {features.shape}")
    # print(f"特征范数: {torch.norm(features, dim=1)}")
    #
    # # 参数统计
    # param_counts = encoder.get_parameter_count()
    # print(f"\\n参数统计:")
    # print(f"总参数: {param_counts['total']:,}")
    # print(f"可训练参数: {param_counts['trainable']:,}")
    # print(f"冻结参数: {param_counts['frozen']:,}")
    pass

