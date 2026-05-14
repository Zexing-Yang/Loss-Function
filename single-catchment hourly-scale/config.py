'''
参数文件
'''
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import copy
# 所有需要运行的站点列表
ZHANMING_LIST = ['test']  # 自行修改站点列表'chenda','yutan','longmen','yangjiafang','maiyuan','yongding','guanyinqiao'

# 所有需要运行的time参数列表
TIME_VALUES = range(5)  # 自行修改time参数列表
# 动态参数（运行时会被覆盖）
zhanming = 'test'
time = 1


#通用的一些参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
timesteps = 30
leadtimes = 1
site = 1
target_dim = 1
hidden_size = 8
num_layer = 1
conv1d_out_channel = 1
attn_method ='general'
MODEL_CONFIG = {
    'hidden_size': 20,
    'dropout_rate': 0.1,
    'learning_rate': 1e-3,
    'sequence_length': 365,
    'n_epochs': 20,  # 增加训练轮数以获得更好收敛
    'train_batch_size': 512,
    'eval_batch_size': 1024,
}

patience = 50

# dataset
scaler = MinMaxScaler(feature_range=(0, 1))

#结果处理相关参数---test.py

plot_leadtime = 1

def get_model_path():
    return f"./model/{zhanming}/seq2seq_{time + 1}_attention.model"

def get_optimizer_path():
    return f"./model/{zhanming}/optimizer{time + 1}_attention_model"

def get_save_paths():
    datestr = datetime.strftime(datetime.now(), "%Y%m%d")
    return {
        "save_nse": f"./result/{zhanming}/nse_rmse_{time + 1}.xlsx",
        "save_result": f"./result/{zhanming}/result_{time + 1}.xlsx",
        "save_plotloss": f'./picture/{zhanming}/loss_plot_{time + 1}.png',
        "save_resultpicture": f"./hydrograph/{zhanming}/predictions_run_{time + 1}.png",
    }