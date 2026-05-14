'''
准备数据集, 导入数据, 并且只对输入数据进行归一化
'''
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import config
import os
import datadeal
import pandas as pd

class NumDataset(Dataset):
    def __init__(self, dataset1 = 'train'):
        # 修正文件路径拼接错误
        raindistrain = pd.read_csv(
            f"./data_divided/{config.zhanming}/raindistrain{config.timesteps}_{config.leadtimes}.csv",
            index_col=0)
        raindistest = pd.read_csv(
            f"./data_divided/{config.zhanming}/raindistest{config.timesteps}_{config.leadtimes}.csv",
            index_col=0)
        raindisval = pd.read_csv(
            f"./data_divided/{config.zhanming}/raindisval{config.timesteps}_{config.leadtimes}.csv",
            index_col=0)
        if dataset1 == 'train':
            rd = raindistrain.values
        if dataset1 == 'test':
            rd = raindistest.values
        if dataset1 == 'val':
            rd = raindisval.values
        self.data = rd.reshape((rd.shape[0] // config.timesteps, config.timesteps, rd.shape[1]))

    def __getitem__(self, item):
        #不同模型的lable不完全一样,需要根据模型做具体的修改
        #time只取了timesteps最后一个维度的
        time = self.data[item,-1,:2 ]
        input = self.data[item,:,2:3 ]
        flowbase = self.data[item,-1, 2+config.site ]
        area = self.data[item,-1, 3+config.site ]
        lable = self.data[item,-1, 5+config.site]
        return time, input, flowbase, area,lable,

    def __len__(self):
        return self.data.shape[0]
def collate_fn(batch):
    time, input, flowbase, area, lable = zip(*batch)
    input = np.array(input).astype(float)
    input = torch.FloatTensor(input)
    flowbase = torch.FloatTensor(flowbase)
    area = torch.FloatTensor(area)
    lable = torch.FloatTensor(lable)
    return time, input, flowbase,area,  lable

def get_dataloader(dataset1 = 'train'):
    batch_size = config.batch_size
    if dataset1 == 'train':
        shuffle_value = True
    else:
        shuffle_value = False
    return DataLoader(NumDataset(dataset1),batch_size=batch_size, shuffle=shuffle_value,collate_fn=collate_fn)
'''#在深度学习中，dataloader( )中的shute参数用于打乱数据顺序，以避免模型对数据顺序的依赖性，从而提高横型的泛化能力，对于训练集，
 通常会将shuffle设置为True，以便在每个epoch中对数据进行随机排序，可以避免模型过度依赖数据集中某些特定的数据顺序，从而防止横型过拟合。
对于验证集和测试集，通常会将shuffle设置为False，以确保验证集和测试集的数据顺序不变，从而能够稳定地评估模型的性能。
如果验证集和测试集的数据顺序随机变化，模型的性能评估结果可能会出现波动，从而影响评估结果。因此，合理的设置是在训练集中将shuffle设置为True，
在验证集和测试集中将shuffle设置为False，以便在训练模型和评估模型时都能达到最佳性能。参数drop_last=True的含义是当数据集的大小无法被batch_size整除时，
丢弃最后一个未满的batch。如果数据集中的样本数无法被batch_size整除，那么最后一个batch的样本数将少于 batch_size。若直接对其进行丢弃，会导致数据集的样本数量减少，
这可能使训练数据的多样性减少，影响模型的泛化能力（特别是对于小数据集）。同时，如果训练数据在分布上有任何排序（例如，按标签或特征排列），丢弃末尾的数据可能使模型学习到的分布偏离完整数据分布。
如果丢弃的数据占训练集比例较大，剩余数据被反复训练，特别是当数据集容量有限时，可能导致模型更倾向于记住训练集中已用样本（过拟合）。
'''
if __name__ == '__main__':
    for j, (time, input, flowbase,area, target) in enumerate(get_dataloader(dataset1='val')):
        # input: [batch_size, timesteps, feature]
        # target: [batch_size, timesteps, leadtimes]
        if j == 0:
            print(j)
            # print(input)
            print(input)
            print(flowbase)
            # print(target)
            print(target)
            break
