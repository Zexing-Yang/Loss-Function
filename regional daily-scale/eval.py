import torch
from config import DEVICE
from utils import calculate_metrics  # 导入新的多指标计算函数
import numpy as np

def eval_model(model, basin_loaders, period='test', device='cpu'):
    """

    参数:
        model: 要评估的模型
        basin_loaders: 流域数据加载器字典
        period: 评估的数据集类型 ('val' 或 'test')
        device: 计算设备
        basin_to_idx: 流域ID到索引的映射

    返回:
        dict: 每个流域指标的结果
    """
    model.eval()
    basin_results = {}

    # 遍历每个流域
    for basin, data in basin_loaders.items():
        loader = data[period][1]  # 获取数据加载器
        # 获取对应数据集对象
        ds = data[period][0]
        obs_list, preds_list = [], []
        with torch.no_grad():
            # 处理当前流域的所有批次
            for xs, ys in loader:
                # 准备输入数据和流域ID
                xs = xs.to(device)
                # 模型预测
                y_hat = model(xs)
                # 收集结果
                obs_list.append(ys.cpu().numpy())
                preds_list.append(y_hat.cpu().numpy())

                # 合并所有批次的结果
            obs = np.concatenate(obs_list)
            preds = np.concatenate(preds_list)
            # 反标准化预测结果
            preds = ds.local_rescale(preds, variable='output')
            # 计算所有指标
            metrics = calculate_metrics(obs, preds)

            basin_results[basin] = {
                'NSE': metrics['NSE'],
                'KGE': metrics['KGE'],
                'TPE': metrics['TPE'],
                'obs': obs,
                'preds': preds
            }

    return basin_results