"""
PNG草图-图像统一跨模态对齐模型
专门用于PNG格式草图与图像的对齐训练
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

# 导入编码器
from encoders.png_sketch_encoder import create_png_sketch_encoder
from encoders.optimized_vision_model import create_optimized_vision_model
from encoders.sketchrnn import BiLSTMEncoder


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class PNGSketchImageAlignmentModel(nn.Module):
    """
    PNG草图-图像对齐模型
    使用冻结的图像编码器和可训练的草图编码器
    """
    
    def __init__(self, 
                 embed_dim=512,
                 sketch_model_name='vit_base_patch16_224',
                 image_model_name='vit_base_patch16_224',
                 freeze_image_encoder=True,
                 freeze_sketch_backbone=False,
                 dropout_rate=0.1,
                 temperature=0.07,
                 **kwargs):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.freeze_image_encoder = freeze_image_encoder
        self.freeze_sketch_backbone = freeze_sketch_backbone
        
        # 可学习的温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        
        # 初始化编码器
        self._init_encoders(sketch_model_name, image_model_name, dropout_rate)
        
        # 获取编码器输出维度
        self._get_encoder_dims()
        
        # 初始化投影层
        self._init_projections()
        
        # 初始化参数
        self._initialize_parameters()
        
        print(f"PNGSketchImageAlignmentModel initialized:")
        print(f"  Embed dim: {embed_dim}")
        print(f"  Sketch model: {sketch_model_name}")
        print(f"  Image model: {image_model_name}")
        print(f"  Freeze image encoder: {freeze_image_encoder}")
        print(f"  Freeze sketch backbone: {freeze_sketch_backbone}")
    
    def _init_encoders(self, sketch_model_name, image_model_name, dropout_rate):
        """初始化编码器"""
        
        # PNG草图编码器（可训练）
        # self.sketch_encoder = create_png_sketch_encoder(
        #     model_name=sketch_model_name,
        #     pretrained=True,
        #     freeze_backbone=self.freeze_sketch_backbone,
        #     output_dim=self.embed_dim,
        #     dropout_rate=dropout_rate
        # )

        self.sketch_encoder = BiLSTMEncoder()
        
        # 图像编码器（通常冻结）
        if self.freeze_image_encoder:
            # 使用优化的确定性图像模型（冻结权重）
            self.image_encoder = create_optimized_vision_model(
                model_name=image_model_name
            )
            # 冻结图像编码器参数
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            print("Image encoder weights frozen")
        else:
            # 可训练的图像编码器
            import timm
            self.image_encoder = timm.create_model(
                image_model_name,
                pretrained=True,
                num_classes=0,
                global_pool=''
            )
            print("Image encoder weights trainable")
    
    def _get_encoder_dims(self):
        """获取编码器输出维度"""
        
        # 草图编码器输出维度
        self.sketch_feat_dim = self.embed_dim  # PNG草图编码器直接输出目标维度
        
        # 图像编码器输出维度
        if hasattr(self.image_encoder, 'embed_dim'):
            self.image_feat_dim = self.image_encoder.embed_dim
        elif hasattr(self.image_encoder, 'num_features'):
            self.image_feat_dim = self.image_encoder.num_features
        else:
            self.image_feat_dim = 768  # ViT-Base默认维度
    
    def _init_projections(self):
        """初始化投影层"""
        
        # 草图投影层（如果需要）
        if self.sketch_feat_dim != self.embed_dim:
            self.sketch_projection = nn.Sequential(
                nn.Linear(self.sketch_feat_dim, self.embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                LayerNorm(self.embed_dim)
            )
        else:
            self.sketch_projection = nn.Identity()
        
        # 图像投影层
        self.image_projection = nn.Sequential(
            nn.Linear(self.image_feat_dim, self.embed_dim),
            nn.ReLU(), 
            nn.Dropout(0.1),
            LayerNorm(self.embed_dim)
        )
    
    def _initialize_parameters(self):
        """初始化参数"""
        for module in [self.image_projection]:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0)
    
    def encode_sketch(self, sketch_images):
        """
        编码PNG草图
        
        Args:
            sketch_images: 草图张量 [batch_size, 3, 224, 224]
            
        Returns:
            sketch_features: 归一化后的草图特征 [batch_size, embed_dim]
        """
        # PNG草图编码器已经包含投影和归一化
        sketch_features = self.sketch_encoder(sketch_images)
        
        # 如果需要额外投影
        sketch_features = self.sketch_projection(sketch_features)
        
        # L2归一化
        sketch_features = nn.functional.normalize(sketch_features, p=2, dim=1)
        
        return sketch_features
    
    def encode_image(self, images):
        """
        编码图像
        
        Args:
            images: 图像张量 [batch_size, 3, 224, 224]
            
        Returns:
            image_features: 归一化后的图像特征 [batch_size, embed_dim]
        """
        # 通过图像编码器
        if hasattr(self.image_encoder, 'forward_features'):
            image_features = self.image_encoder.forward_features(images)
            if image_features.dim() == 3:  # [batch_size, seq_len, hidden_dim]
                image_features = image_features[:, 0, :]  # 使用CLS token
        else:
            image_features = self.image_encoder(images)
            if image_features.dim() > 2:
                image_features = image_features.mean(dim=[-2, -1])
        
        # 投影到统一维度
        image_features = self.image_projection(image_features)
        
        # L2归一化
        image_features = nn.functional.normalize(image_features, p=2, dim=1)
        
        return image_features
    
    def forward(self, sketch_masks, images):
        """
        前向传播
        
        Args:
            sketch_images: 草图张量 [batch_size, 3, 224, 224]
            images: 图像张量 [batch_size, 3, 224, 224]
            
        Returns:
            sketch_features: 草图特征 [batch_size, embed_dim]
            image_features: 图像特征 [batch_size, embed_dim]
            logit_scale: 温度参数
        """
        sketch_features = self.encode_sketch(sketch_masks)
        image_features = self.encode_image(images)
        
        return sketch_features, image_features, self.logit_scale.exp()
    
    def compute_similarity(self, sketch_features, image_features, temperature=None):
        """
        计算相似度矩阵
        
        Args:
            sketch_features: 草图特征 [N, embed_dim]
            image_features: 图像特征 [M, embed_dim]
            temperature: 温度参数，如果为None则使用学习的logit_scale
            
        Returns:
            similarity_matrix: 相似度矩阵 [N, M]
        """
        if temperature is None:
            temperature = self.logit_scale.exp()
        
        # 计算余弦相似度
        similarity_matrix = torch.matmul(sketch_features, image_features.t()) * temperature
        
        return similarity_matrix
    
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
        """获取参数数量统计"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params
        }


def create_png_sketch_image_model(embed_dim=512,
                                  sketch_model_name='vit_base_patch16_224',
                                  image_model_name='vit_base_patch16_224',
                                  freeze_image_encoder=True,
                                  freeze_sketch_backbone=False,
                                  dropout_rate=0.1,
                                  temperature=0.07):
    """
    创建PNG草图-图像对齐模型
    
    Args:
        embed_dim: 嵌入维度
        sketch_model_name: 草图编码器模型名称
        image_model_name: 图像编码器模型名称
        freeze_image_encoder: 是否冻结图像编码器
        freeze_sketch_backbone: 是否冻结草图编码器主干网络
        dropout_rate: Dropout率
        temperature: 初始温度参数
        
    Returns:
        model: PNG草图-图像对齐模型
    """
    model = PNGSketchImageAlignmentModel(
        embed_dim=embed_dim,
        sketch_model_name=sketch_model_name,
        image_model_name=image_model_name,
        freeze_image_encoder=freeze_image_encoder,
        freeze_sketch_backbone=freeze_sketch_backbone,
        dropout_rate=dropout_rate,
        temperature=temperature
    )
    
    return model


if __name__ == '__main__':
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    model = create_png_sketch_image_model(
        embed_dim=512,
        freeze_image_encoder=True,
        freeze_sketch_backbone=False
    )
    model.to(device)
    
    # 测试数据
    batch_size = 4
    sketch_images = torch.randn(batch_size, 3, 224, 224).to(device)
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # 前向传播
    with torch.no_grad():
        sketch_features, image_features, logit_scale = model(sketch_images, images)
        
        # 计算相似度
        similarity = model.compute_similarity(sketch_features, image_features)
    
    print(f"Test results:")
    print(f"  Sketch features shape: {sketch_features.shape}")
    print(f"  Image features shape: {image_features.shape}")
    print(f"  Similarity matrix shape: {similarity.shape}")
    print(f"  Temperature parameter: {logit_scale.item():.4f}")
    
    # 参数统计
    param_counts = model.get_parameter_count()
    print(f"Parameter statistics:")
    print(f"  Total parameters: {param_counts['total']:,}")
    print(f"  Trainable parameters: {param_counts['trainable']:,}")
    print(f"  Frozen parameters: {param_counts['frozen']:,}")
    
    # 可训练参数详情
    trainable_params, frozen_params = model.get_trainable_parameters()
    print(f"Trainable parameter modules:")
    for name, param in trainable_params[:10]:  # 只显示前10个
        print(f"  {name}: {param.shape}")
