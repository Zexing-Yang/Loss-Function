'''
2021/12/14
增加gru方法
2021/12/16
将序列-残差-注意力整合到一起
'''


import torch.nn as nn
import torch
import torch.nn.functional as F
import config
from  attention import Attention






class Flowconf(nn.Module):
    """
    此为单站点的径流聚合函数
    """
    def __init__(self, method = "selfattention"):
        super(Flowconf, self).__init__()
        self.method = method
        self.relu = F.relu
        self.softmax = F.softmax
        self.Wa = nn.Parameter(torch.Tensor(1, config.timesteps))

        if self.method == "gru1":
            self.gru = nn.GRU(input_size=config.conv1d_out_channel,
                              hidden_size=config.hidden_size,
                              num_layers=config.num_layer,
                              batch_first=True)
            self.dropout = nn.Dropout(p=0)
            self.fa = nn.Linear(config.hidden_size, 1)
        if self.method == "seqresatten":
            self.gru = nn.GRU(input_size=config.conv1d_out_channel,
                              hidden_size=config.hidden_size,
                              num_layers=config.num_layer,
                              batch_first=True)
            self.attn = Attention(method=config.attn_method)
            self.fa = nn.Linear(config.hidden_size * 2, 1)




    def forward(self, inputline):


        if self.method == "selfattention":
            return self.selfattention_score(inputline)
        elif self.method == "unitattention":
            return self.unitattention_score(inputline)
        elif self.method == "gru1":
            return self.gru1_score(inputline)
        elif self.method == "seqresatten":
            return self.seqresatten_score(inputline)


    def selfattention_score(self, inputline):


        attn= self.softmax(self.relu(self.Wa*inputline),dim= -1)
        output = inputline*attn


        return output.sum(dim=-1)

    def unitattention_score(self, inputline):
        '''

        :param inputline:[batch-size,out-channel, timesteps]
        :return:
        '''

        onetensor = torch.ones(config.timesteps)
        attn = self.softmax(self.relu(self.Wa * onetensor.to(config.device)), dim=-1)
        output = inputline * attn

        return output.sum(dim=-1)




    def gru1_score(self, inputline):
        '''

        :param inputline: [batch-size,out-channel, timesteps]
        :return:
        '''
        _, input1 = self.gru(inputline.transpose(1,2))
        input1 = self.dropout(input1)
        output = self.fa(input1.squeeze(0))
        return output.squeeze(1)

    def seqresatten_score(self, inputline):
        '''

        :param input: [batch-size, timesteps, out-channel]
        :return:
        '''
        input = inputline.transpose(1, 2)
        outputs, hidden = self.gru(input)

        attention_weight = self.attn(hidden, outputs).unsqueeze(1)
        # attn_weights [batch_size,1,timesteps] * [batch_size,timesteps,hidden_size]
        context = attention_weight.bmm(outputs) #[batch_size, 1, hidden_size]
        concat_input = torch.cat((hidden.transpose(0, 1), context), -1).squeeze(1)  # [batch_size, 2*hidden_size]
        output = self.fa(concat_input)  ##[batch_size, hidden_size]
        return output.squeeze(1)








