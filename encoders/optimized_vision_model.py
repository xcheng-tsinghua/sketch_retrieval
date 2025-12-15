"""
更优雅的确定性视觉模型实现
使用单例模式避免重复加载预训练权重
"""
import torch
import torch.nn as nn
import timm
import os
from functools import lru_cache


# 全局缓存，避免重复加载相同的预训练模型
_MODEL_CACHE = {}


@lru_cache(maxsize=8)
def _load_pretrained_weights(model_name):
    """
    缓存预训练权重，避免重复下载和加载
    """
    print(f"首次加载预训练模型权重：{model_name}")
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    return model.state_dict()


class CachedVisionModel(nn.Module):
    """
    使用缓存权重的确定性视觉模型
    避免重复加载预训练权重
    """
    
    def __init__(self, model_name='vit_base_patch16_224'):
        super().__init__()
        self.model_name = model_name
        
        # 创建模型结构（不加载权重）
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0)
        
        # 加载缓存的预训练权重
        cached_weights = _load_pretrained_weights(model_name)
        self.model.load_state_dict(cached_weights)
        
        # print(f"使用缓存权重创建确定性模型：{model_name}")
    
    def forward(self, x):
        return self.model(x)


def create_optimized_vision_model(model_name='vit_base_patch16_224', checkpoint_path=None):
    """
    创建优化的确定性视觉模型
    
    Args:
        model_name: 模型名称
        checkpoint_path: 检查点路径（用于加载特定权重）
    
    Returns:
        确定性的视觉模型
    """
    real_model_name = 'vit_base_patch16_224' if model_name == 'vit' else None
    
    # 优先从检查点加载
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"从检查点加载视觉模型：{checkpoint_path}")
        model = timm.create_model(real_model_name, pretrained=False, num_classes=0)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 提取视觉模型权重
        vision_weights = {}
        for key, value in checkpoint.items():
            if key.startswith('vision_model.'):
                new_key = key[len('vision_model.'):]
                vision_weights[new_key] = value
        
        if vision_weights:
            model.load_state_dict(vision_weights, strict=False)
            print("成功加载检查点中的视觉模型权重")
            return model
        else:
            print("检查点中未找到视觉模型权重，使用缓存预训练权重")
    
    # 使用缓存的预训练权重
    return CachedVisionModel(real_model_name).model


def test_optimized_model():
    """测试优化模型的确定性和性能"""
    print("=== 测试优化的确定性模型 ===")
    
    import time
    
    # 测试首次加载时间
    start_time = time.time()
    model1 = create_optimized_vision_model()
    first_load_time = time.time() - start_time
    
    # 测试后续加载时间（应该更快，因为使用了缓存）
    start_time = time.time()
    model2 = create_optimized_vision_model()
    second_load_time = time.time() - start_time
    
    print(f"首次加载时间: {first_load_time:.2f}秒")
    print(f"后续加载时间: {second_load_time:.2f}秒")
    print(f"加速比: {first_load_time/second_load_time:.1f}x")
    
    # 验证确定性
    same_weights = True
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if not torch.equal(param1, param2):
            same_weights = False
            print(f"权重不同: {name1}")
            break
    
    print(f"权重一致性: {'✅' if same_weights else '❌'}")
    
    # 测试前向传播
    test_input = torch.randn(2, 3, 224, 224)
    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        output1 = model1(test_input)
        output2 = model2(test_input)
    
    forward_same = torch.allclose(output1, output2, atol=1e-6)
    print(f"前向传播一致性: {'✅' if forward_same else '❌'}")
    
    return same_weights and forward_same


if __name__ == "__main__":
    test_optimized_model()
