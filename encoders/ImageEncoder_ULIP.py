import timm
import torch.nn as nn
import torch
from collections import OrderedDict


class ImageEncoder_ULIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
        self.image_projection = nn.Parameter(torch.empty(768, 512))

    @torch.inference_mode()
    def forward(self, image):
        x = self.vision_model(image)
        x = x @ self.image_projection

        return x


def save_weights_from_all():
    ckpt = torch.load('weights_all.pt', map_location='cpu', weights_only=False)
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    image_encoder = ImageEncoder_ULIP().cuda()

    image_encoder.load_state_dict(state_dict, strict=False)

    torch.save(image_encoder.state_dict(), 'weight_image_encoder.pth')


def create_pretrained_imageencoder(root_ckpt: str = './weights/weight_image_encoder.pth'):
    print('create pretrained image encoder, load weight from ' + root_ckpt)

    image_encoder_pretrained = ImageEncoder_ULIP()

    try:
        image_encoder_pretrained.load_state_dict(torch.load(root_ckpt), strict=True)
    except:
        raise ValueError('can not load pretrained model weight: ', root_ckpt)

    # 设为评估模式
    image_encoder_pretrained = image_encoder_pretrained.eval()

    # 禁用梯度计算，提升速度
    image_encoder_pretrained.requires_grad_(False)

    return image_encoder_pretrained


def test():
    amodel = ImageEncoder_ULIP()
    emb = torch.rand(9, 3, 224, 224)

    out = amodel(emb)
    print(out.size())


if __name__ == '__main__':
    # test()

    # save_weights_from_all()
    test()


    pass


