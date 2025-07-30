import torch.nn as nn
import torch
from collections import OrderedDict


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class TextEncoder_ULIP(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(49408, 512)
        self.positional_embedding = nn.Parameter(torch.empty(77, 512))

        self.transformer = Transformer(
            width=512,
            layers=12,
            heads=8,
            attn_mask=self.build_attention_mask(),
        )

        self.ln_final = LayerNorm(512)
        self.text_projection = nn.Parameter(torch.empty(512, 512))

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(77, 77)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @torch.inference_mode()
    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


def create_pretrained_textencoder(root_ckpt: str = './weights/weight_text_encoder.pth'):
    print('create pretrained text encoder, load weight from ' + root_ckpt)

    text_encoder_pretrained = TextEncoder_ULIP()

    try:
        text_encoder_pretrained.load_state_dict(torch.load(root_ckpt), strict=True)
    except:
        raise ValueError('can not load pretrained model weight: ', root_ckpt)

    # 设为评估模式
    text_encoder_pretrained = text_encoder_pretrained.eval()

    # 禁用梯度计算，提升速度
    text_encoder_pretrained.requires_grad_(False)

    return text_encoder_pretrained


def save_weights_from_all():
    ckpt = torch.load('weights_all.pt', map_location='cpu', weights_only=False)
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    text_encoder = TextEncoder_ULIP().cuda()

    text_encoder.load_state_dict(state_dict, strict=False)

    torch.save(text_encoder.state_dict(), 'weight_text_encoder.pth')


def test_input():
    # ckpt = torch.load('weights_all.pt', map_location='cpu', weights_only=False)
    # state_dict = OrderedDict()
    # for k, v in ckpt['state_dict'].items():
    #     state_dict[k.replace('module.', '')] = v

    # text_encoder = TextEncoder_ULIP().cuda()
    #
    # text_encoder.load_state_dict(state_dict, strict=False)

    text_encoder = create_pretrained_textencoder().cuda()

    test_emb = torch.randint(1, 200, [5, 77]).cuda()

    res = text_encoder(test_emb)

    print(res.size())


if __name__ == '__main__':

    # ckpt = torch.load('weights_all.pt', map_location='cpu', weights_only=False)
    # state_dict = OrderedDict()
    # for k, v in ckpt['state_dict'].items():
    #     state_dict[k.replace('module.', '')] = v
    #
    # text_encoder = TextEncoder_ULIP().cuda()
    #
    # text_encoder.load_state_dict(state_dict, strict=False)
    #
    # torch.save(text_encoder.state_dict(), 'weight_text_encoder.pth')
    #
    #
    # test_emb = torch.randint(1, 200, [5, 77]).cuda()
    #
    # res = text_encoder(test_emb)
    test_input()
    pass

