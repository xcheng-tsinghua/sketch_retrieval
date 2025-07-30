import torch
import torch.nn as nn


class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=256, num_layers=2, bidirectional=True, dropout=0.4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        )

        self.out_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x):
        # x: [bs, n_pnts, 3]
        _, (h_n, _) = self.lstm(x)  # h_n: [num_layers * num_directions, bs, hidden_dim]

        if self.bidirectional:
            # 取最后一层的正向和反向 hidden state
            h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)  # [bs, hidden_dim * 2]
        else:
            h_last = h_n[-1]  # [bs, hidden_dim]

        return h_last  # [bs, out_dim]


if __name__ == "__main__":


    pass









