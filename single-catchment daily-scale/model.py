import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, hidden_size: int, dropout_rate: float = 0.0):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        self.lstm = nn.LSTM(input_size=5,
                            hidden_size=self.hidden_size,
                            num_layers=2,
                            bias=True,
                            batch_first=True)

        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=1)

    def forward(self, x: torch.Tensor, basin_ids: torch.Tensor = None) -> torch.Tensor:
        # LSTM 前向传播
        output, (h_n, c_n) = self.lstm(x)
        # 修改2: 两层LSTM时，h_n的形状为[2, batch_size, hidden_size]
        # 取最后一层（第二层）的最后一个隐藏状态
        pred = self.fc(self.dropout(h_n[-1, :, :]))
        return pred