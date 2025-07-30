"""
基于ULIP_models结构优化的统一跨模态对齐模型
集成草图、图片、文本、点云等多模态编码器
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import timm
import open_clip

# 导入现有编码器
from encoders.sketch_encoder import SketchBERT
from encoders.ImageEncoder_ULIP import ImageEncoder_ULIP
from encoders.optimized_vision_model import create_optimized_vision_model


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class UnifiedCrossModalModel(nn.Module):
    """
    统一的跨模态对齐模型，参照ULIP_models设计
    支持草图、图片、文本、点云等多模态
    """
    
    def __init__(self, 
                 embed_dim=512,
                 sketch_points=256,
                 use_sketch=True,
                 use_image=True,
                 use_text=True,
                 use_pointcloud=True,
                 **kwargs):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.use_sketch = use_sketch
        self.use_image = use_image
        self.use_text = use_text
        self.use_pointcloud = use_pointcloud
        
        # 统一的logit_scale参数（参照ULIP）
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        # 初始化各模态编码器
        self._init_encoders(sketch_points, **kwargs)
        
        # 初始化投影层
        self._init_projections()
        
        # 初始化参数
        self._initialize_parameters()
    
    def _init_encoders(self, sketch_points, **kwargs):
        """初始化各模态编码器"""
        
        # 草图编码器
        if self.use_sketch:
            self.sketch_encoder = SketchBERT(sketch_points)
            self.sketch_feat_dim = 512  # SketchBERT输出维度
        
        # 图片编码器 - 使用优化的确定性模型
        if self.use_image:
            # 使用优化的确定性视觉模型（带缓存，更高效）
            self.vision_model = create_optimized_vision_model(
                model_name='vit_base_patch16_224'
            )
            self.vision_feat_dim = 768  # ViT-Base输出维度
            print("使用优化的确定性预训练ViT模型")
        
        # 文本编码器（暂时禁用，避免依赖问题）
        if self.use_text:
            # self.text_encoder = TextEncoder_ULIP()
            self.text_feat_dim = 512  # TextEncoder输出维度
            print("Warning: Text encoder temporarily disabled")
        
        # 点云编码器（暂时禁用，避免依赖问题）
        if self.use_pointcloud:
            # self.point_encoder = PointBERT_ULIP2()
            self.pc_feat_dim = 768  # PointBERT输出维度
            print("Warning: Point cloud encoder temporarily disabled")
    
    def _init_projections(self):
        """初始化投影层，将各模态特征投影到统一维度"""
        
        if self.use_sketch:
            self.sketch_projection = nn.Parameter(
                torch.empty(self.sketch_feat_dim, self.embed_dim)
            )
        
        if self.use_image:
            self.image_projection = nn.Parameter(
                torch.empty(self.vision_feat_dim, self.embed_dim)
            )
        
        if self.use_text:
            self.text_projection = nn.Parameter(
                torch.empty(self.text_feat_dim, self.embed_dim)
            )
        
        if self.use_pointcloud:
            self.pc_projection = nn.Parameter(
                torch.empty(self.pc_feat_dim, self.embed_dim)
            )
    
    def _initialize_parameters(self):
        """初始化参数（参照ULIP方法）"""
        
        if self.use_sketch:
            nn.init.normal_(self.sketch_projection, std=self.sketch_feat_dim ** -0.5)
        
        if self.use_image:
            nn.init.normal_(self.image_projection, std=self.vision_feat_dim ** -0.5)
        
        if self.use_text:
            nn.init.normal_(self.text_projection, std=self.text_feat_dim ** -0.5)
        
        if self.use_pointcloud:
            nn.init.normal_(self.pc_projection, std=self.pc_feat_dim ** -0.5)
    
    def encode_sketch(self, sketch_data, sketch_mask):
        """编码草图数据"""
        if not self.use_sketch:
            raise ValueError("Sketch encoder not enabled")
        
        sketch_feat = self.sketch_encoder(sketch_data, sketch_mask)
        sketch_embed = sketch_feat @ self.sketch_projection
        return sketch_embed
    
    def encode_image(self, image):
        """编码图片数据"""
        if not self.use_image:
            raise ValueError("Image encoder not enabled")
        
        image_feat = self.vision_model(image)
        image_embed = image_feat @ self.image_projection
        return image_embed
    
    def encode_text(self, text):
        """编码文本数据"""
        if not self.use_text:
            raise ValueError("Text encoder not enabled")
        
        # 暂时禁用，避免依赖问题
        raise NotImplementedError("Text encoder temporarily disabled")
        # text_feat = self.text_encoder(text)
        # text_embed = text_feat @ self.text_projection
        # return text_embed
    
    def encode_pointcloud(self, pc):
        """编码点云数据"""
        if not self.use_pointcloud:
            raise ValueError("Point cloud encoder not enabled")
        
        # 暂时禁用，避免依赖问题
        raise NotImplementedError("Point cloud encoder temporarily disabled")
        # pc_feat = self.point_encoder(pc)
        # pc_embed = pc_feat @ self.pc_projection
        # return pc_embed
    
    def forward(self, **inputs):
        """
        前向传播，支持多模态输入
        
        Args:
            sketch_data: 草图数据 [batch, seq_len, 3]
            sketch_mask: 草图掩码 [batch, seq_len]
            image: 图片数据 [batch, 3, 224, 224]
            text: 文本数据 [batch, seq_len]
            pointcloud: 点云数据 [batch, n_points, 3]
            
        Returns:
            dict: 包含各模态嵌入和logit_scale的字典
        """
        outputs = {'logit_scale': self.logit_scale.exp()}
        
        # 编码各模态
        if 'sketch_data' in inputs and 'sketch_mask' in inputs:
            sketch_embed = self.encode_sketch(inputs['sketch_data'], inputs['sketch_mask'])
            outputs['sketch_embed'] = sketch_embed
        
        if 'image' in inputs:
            image_embed = self.encode_image(inputs['image'])
            outputs['image_embed'] = image_embed
        
        if 'text' in inputs:
            text_embed = self.encode_text(inputs['text'])
            outputs['text_embed'] = text_embed
        
        if 'pointcloud' in inputs:
            pc_embed = self.encode_pointcloud(inputs['pointcloud'])
            outputs['pc_embed'] = pc_embed
        
        return outputs


class SketchImageAlignmentModel(UnifiedCrossModalModel):
    """
    专门用于草图-图片对齐的模型
    继承自UnifiedCrossModalModel，只启用草图和图片模态
    """
    
    def __init__(self, sketch_points=256, embed_dim=512, **kwargs):
        super().__init__(
            embed_dim=embed_dim,
            sketch_points=sketch_points,
            use_sketch=True,
            use_image=True,
            use_text=False,
            use_pointcloud=False,
            **kwargs
        )
    
    def forward(self, sketch_data, sketch_mask, image):
        """
        草图-图片对齐专用前向传播
        
        Args:
            sketch_data: 草图数据 [batch, seq_len, 3]
            sketch_mask: 草图掩码 [batch, seq_len]
            image: 图片数据 [batch, 3, 224, 224]
            
        Returns:
            dict: 包含sketch_embed, image_embed, logit_scale
        """
        # 编码草图
        sketch_embed = self.encode_sketch(sketch_data, sketch_mask)
        
        # 编码图片
        image_embed = self.encode_image(image)
        
        return {
            'sketch_embed': sketch_embed,
            'image_embed': image_embed,
            'logit_scale': self.logit_scale.exp()
        }


def create_sketch_image_model(sketch_points=256, embed_dim=512, pretrained_image_path=None):
    """
    创建草图-图片对齐模型的工厂函数
    
    Args:
        sketch_points: 草图点数
        embed_dim: 嵌入维度
        pretrained_image_path: 预训练图片编码器路径
        
    Returns:
        SketchImageAlignmentModel: 配置好的模型
    """
    model = SketchImageAlignmentModel(
        sketch_points=sketch_points,
        embed_dim=embed_dim
    )
    
    # 加载预训练图片编码器权重
    if pretrained_image_path:
        try:
            pretrained_dict = torch.load(pretrained_image_path, map_location='cpu')
            
            # 提取图片编码器相关权重
            image_dict = {}
            for k, v in pretrained_dict.items():
                if k.startswith('vision_model.') or k.startswith('image_projection'):
                    image_dict[k] = v
            
            # 加载权重
            model.load_state_dict(image_dict, strict=False)
            print(f"Successfully loaded pretrained image encoder from {pretrained_image_path}")
            
        except Exception as e:
            print(f"Warning: Failed to load pretrained image encoder: {e}")
    
    return model


def get_loss():
    """获取损失函数"""
    from encoders.loss_func import SketchImageAlignmentLoss
    return SketchImageAlignmentLoss()


def get_metric_names():
    """获取评估指标名称"""
    return ['loss', 'sketch_image_acc']


# 测试函数
def test_model():
    """测试模型功能"""
    print("Testing SketchImageAlignmentModel...")
    
    # 创建模型
    model = create_sketch_image_model(sketch_points=256, embed_dim=512)
    model.eval()
    
    # 创建测试数据 - 修复：使用正确的stroke-5格式
    batch_size = 4
    sketch_data = torch.randn(batch_size, 256, 5)  # stroke-5格式：[x, y, p1, p2, p3]
    sketch_mask = torch.ones(batch_size, 256, dtype=torch.bool)
    image_data = torch.randn(batch_size, 3, 224, 224)
    
    # 前向传播
    with torch.no_grad():
        outputs = model(sketch_data, sketch_mask, image_data)
    
    print(f"Sketch embed shape: {outputs['sketch_embed'].shape}")
    print(f"Image embed shape: {outputs['image_embed'].shape}")
    print(f"Logit scale: {outputs['logit_scale'].item():.4f}")
    print("Model test passed!")


if __name__ == "__main__":
    test_model()
