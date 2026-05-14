import torch
import torch.nn as nn
import config

class Model(nn.Module):
    def __init__(self, hidden_size: int, dropout_rate: float = 0.0):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        self.lstm = nn.LSTM(input_size=config.site,
                            hidden_size=self.hidden_size,
                            num_layers=2,
                            bias=True,
                            batch_first=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fa = nn.Linear(self.hidden_size, 1)

    def forward(self, input, flowbase, area):
        _, (h_n, _) = self.lstm(input)  # 忽略输出序列，仅保留最后一个隐藏状态
        last_hidden = h_n[-1]
        last_hidden = self.dropout(last_hidden)
        output = self.fa(last_hidden).squeeze(1) * area + flowbase  # 去除隐藏状态中的第一个维度和输出的第二个维度
        return output

if __name__ == "__main__":
    net = Model()
    print(net)
    param = net.named_parameters()
    for i in param:
        print(i)