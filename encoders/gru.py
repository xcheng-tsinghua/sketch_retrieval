import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    def __init__(self, input_dim=5, embed_dim=512, num_layers=2, bidirectional=True, dropout=0.4):
        super().__init__()

        if bidirectional:
            hidden_dim = embed_dim // 2
        else:
            hidden_dim = embed_dim

        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        )

        # 如果是双向GRU，输出维度翻倍
        self.out_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x):
        # x: [bs, n_pnts, 3]
        _, h_n = self.gru(x)  # h_n: [num_layers * num_directions, bs, hidden_dim]

        if self.bidirectional:
            # 拼接最后一层正向和反向的 hidden state
            h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)  # [bs, hidden_dim * 2]
        else:
            h_last = h_n[-1]  # [bs, hidden_dim]

        return h_last  # [bs, out_dim]


if __name__ == '__main__':
    pass








