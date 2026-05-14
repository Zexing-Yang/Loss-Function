import pandas as pd
from torch.utils.data import DataLoader
from dataloader import CamelsTXT
import config
from typing import Dict, Tuple, Any
from tqdm import tqdm  # 正确导入 tqdm 类

def create_basin_datasets(basin: str):
    """
    为单个流域创建训练集、验证集和测试集数据集

    参数:
        basin (str): 流域标识符

    返回:
        tuple: 包含:
            ds_train (CamelsTXT): 训练数据集
            ds_val (CamelsTXT): 验证数据集
            ds_test (CamelsTXT): 测试数据集
            means: 训练数据特征均值（用于标准化）
            stds: 训练数据特征标准差（用于标准化）
    """
    # 创建训练数据集
    ds_train = CamelsTXT(
        basin,  # 流域ID
        seq_length=config.MODEL_CONFIG['sequence_length'],  # 序列长度（时间步长）
        period="train",  # 数据周期标识
        # 将配置中的日期字符串转换为datetime对象
        dates=[pd.to_datetime(d, format="%Y-%m-%d") for d in config.TRAIN_PERIOD]
    )

    # 从训练数据集中获取特征均值和标准差（用于后续数据标准化）
    means = ds_train.get_means()
    stds = ds_train.get_stds()

    # 创建验证数据集，使用训练集的统计量保持标准化一致性
    ds_val = CamelsTXT(
        basin,
        seq_length=config.MODEL_CONFIG['sequence_length'],
        period="eval",  # 评估模式（通常不会应用数据增强）
        dates=[pd.to_datetime(d, format="%Y-%m-%d") for d in config.VAL_PERIOD],
        means=means,  # 传递训练集的均值
        stds=stds  # 传递训练集的标准差
    )

    # 创建测试数据集（使用与验证集相同的标准化参数）
    ds_test = CamelsTXT(
        basin,
        seq_length=config.MODEL_CONFIG['sequence_length'],
        period="eval",
        dates=[pd.to_datetime(d, format="%Y-%m-%d") for d in config.TEST_PERIOD],
        means=means,
        stds=stds
    )

    return ds_train, ds_val, ds_test, means, stds


def get_data_loaders():
    basin_datasets = {}
    print(f"Loading data for {len(config.BASIN_LIST)} basins...")

    for basin in config.BASIN_LIST:
        try:
            # 为当前流域创建数据集
            ds_train, ds_val, ds_test, means, stds = create_basin_datasets(basin)

            # 创建数据加载器
            train_loader = DataLoader(
                ds_train,
                batch_size=config.MODEL_CONFIG['train_batch_size'],
                shuffle=True
            )
            val_loader = DataLoader(
                ds_val,
                batch_size=config.MODEL_CONFIG['eval_batch_size'],
                shuffle=False
            )
            test_loader = DataLoader(
                ds_test,
                batch_size=config.MODEL_CONFIG['eval_batch_size'],
                shuffle=False
            )
            print("-"*80)
            # 添加到结果字典
            basin_datasets[basin] = {
                'train': (ds_train, train_loader),
                'val': (ds_val, val_loader),
                'test': (ds_test, test_loader),
                'stats': (means, stds)
            }
        except Exception as e:
            # 处理异常，继续处理其他流域
            tqdm.write(f"\nError loading basin {basin}: {str(e)}")
            continue
    # 统计成功加载的流域数量
    loaded_count = len(basin_datasets)
    print(f"Successfully loaded {loaded_count}/{len(config.BASIN_LIST)} basins")

    # 如果没有加载到任何流域，抛出异常
    if loaded_count == 0:
        raise RuntimeError("No basins loaded successfully")

    return basin_datasets