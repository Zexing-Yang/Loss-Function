import glob
import numpy as np
import pandas as pd
import torch
import os
from typing import Tuple, List
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
import config

def find_folder(basin: str, base_path: str) -> str:
    """
    根据流域ID查找对应的文件夹
    '01','02','07','11','17'
    优化：缓存文件夹查找结果避免重复扫描
    """
    target_basins = config.REGION_LIST
    # 检查流域ID是否符合格式（应为8位数字）
    if not basin.isdigit() or len(basin) != 8:
        raise ValueError(f"Invalid basin ID format: {basin}")

    for folder_id in target_basins:
        folder_path = os.path.join(base_path, folder_id)

        # 构建匹配模式
        pattern = os.path.join(folder_path, f"{basin}_*.txt")
        matches = glob.glob(pattern)

        if matches:
            return folder_path

    raise FileNotFoundError(f"No data folder found for basin {basin}")


def load_forcing(basin: str) -> Tuple[pd.DataFrame, int]:
    """
    加载指定流域的气象驱动数据

    修改：自动查找流域数据所在的文件夹
    """
    base_path = './basin_mean_forcing/daymet'
    folder_path = find_folder(basin, base_path)

    # 构建气象数据文件路径
    files = list(glob.glob(f"{folder_path}/{basin}_*.txt"))

    # 错误处理：未找到文件
    if len(files) == 0:
        raise RuntimeError(f'No forcing file found for Basin {basin} in {folder_path}')

    file_path = files[0]  # 获取第一个匹配的文件

    # 读取CSV数据（空格分隔，跳过前3行表头）
    with open(file_path) as fp:
        df = pd.read_csv(fp, sep=r'\s+', header=3)

    # 构建日期时间索引
    dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str))
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")

    # 从文件头读取流域面积
    with open(file_path) as fp:
        content = fp.readlines()
        area = int(content[2])  # 第三行包含流域面积

    return df, area

def load_discharge(basin: str, area: int) -> pd.Series:
    """
    加载指定流域的径流数据并进行单位转换

    修改：自动查找流域数据所在的文件夹
    """
    base_path = './usgs_streamflow'
    folder_path = find_folder(basin, base_path)

    # 使用glob匹配文件路径
    files = list(glob.glob(f"{folder_path}/{basin}_*.txt"))

    # 错误处理：未找到文件
    if len(files) == 0:
        raise RuntimeError(f'No discharge file found for Basin {basin} in {folder_path}')

    file_path = files[0]  # 获取第一个匹配的文件

    # 读取数据并设置列名（原始文件没有列名）
    col_names = ['basin', 'Year', 'Mnth', 'Day', 'QObs', 'flag']
    with open(file_path) as fp:
        df = pd.read_csv(fp, sep=r'\s+', header=None, names=col_names)

    # 构建日期时间索引
    dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str))
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")

    """
    单位转换公式详解：
      将径流量从立方英尺每秒(cfs)转换为毫米每天(mm/day)
      转换因子：28316846.592 = 
        (1立方英尺 = 0.028316846592立方米) * 10^9 (单位统一因子)
      再乘以一天的秒数(86400秒)
      最后除以流域面积(平方米)并调整单位(10^6转换为平方公里)

      最终公式：
        mm/day = (28316846.592 * Q_cfs * 86400) / (area * 10^6)
    """
    df.QObs = 28316846.592 * df.QObs * 86400 / (area * 10 ** 6)
    # 检查径流量是否有负值
    if (df.QObs < 0).any():
        print(f"Warning: Basin {basin} has negative discharge values")
    return df.QObs  # 返回径流时间序列


def reshape_data(x: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape matrix data into sample shape for LSTM training.

    :param x: Matrix containing input features column wise and time steps row wise
    :param y: Matrix containing the output feature.
    :param seq_length: Length of look back days for one day of prediction

    :return: Two np.ndarrays, the first of shape (samples, length of sequence,
        number of features), containing the input data for the LSTM. The second
        of shape (samples, 1) containing the expected output for each input
        sample.
    """
    num_samples, num_features = x.shape

    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    y_new = np.zeros((num_samples - seq_length + 1, 1))

    for i in range(0, x_new.shape[0]):
        x_new[i, :, :num_features] = x[i:i + seq_length, :]
        y_new[i, :] = y[i + seq_length - 1, 0]

    return x_new, y_new

class CamelsTXT(Dataset):
    """
    PyTorch Dataset 类，用于加载和处理特定流域的水文气象数据

    该类实现以下核心方法：
    - __init__(): 初始化数据集对象
    - __len__(): 返回数据集中的样本数量
    - __getitem__(): 获取索引对应的样本（输入序列+目标值）
    - _load_data(): 内部数据加载和预处理方法

    继承自 torch.utils.data.Dataset，可与 PyTorch DataLoader 配合使用
    """

    def __init__(self, basin: str, seq_length: int = 365, period: str = None,
                 dates: List = None, means: pd.Series = None, stds: pd.Series = None):
        """
        初始化流域数据集

        Args:
            basin (str): 8位流域编码
            seq_length (int, optional): 时间窗口大小（历史天数）。默认为365天
            period (str, optional): 数据子集标识（'train'训练集/'eval'验证集）。None表示全数据集
            dates (List, optional): 日期范围 [开始日期, 结束日期]。None表示所有可用日期
            means (pd.Series, optional): 特征标准化所需的均值（训练集统计值）
            stds (pd.Series, optional): 特征标准化所需的标准差（训练集统计值）
        """
        self.basin = basin
        self.seq_length = seq_length
        self.period = period
        self.dates = dates
        self.means = means  # 各特征的均值
        self.stds = stds  # 各特征的标准差

        # 加载数据到内存
        self.x, self.y = self._load_data()

        # 存储样本数量（作为类属性）
        self.num_samples = self.x.shape[0]

    def __len__(self):
        """返回数据集中的样本总数"""
        return self.num_samples

    def __getitem__(self, idx: int):
        """获取指定索引的样本（输入序列和对应目标值）"""
        return self.x[idx], self.y[idx]

    def _load_data(self):
        """
        从文本文件加载输入输出数据并进行预处理

        处理步骤:
        1. 加载气象数据
        2. 加载径流数据并进行单位转换
        3. 按指定日期范围截取数据
        4. 训练集统计特征均值和标准差
        5. 特征提取与标准化
        6. 重塑为LSTM所需的序列格式
        7. 清洗无效样本
        8. 转换为PyTorch张量

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 输入序列和对应目标值
        """
        # 1. 加载气象数据和流域面积
        df, area = load_forcing(self.basin)

        # 2. 加载径流数据并添加为DataFrame列
        df['QObs(mm/d)'] = load_discharge(self.basin, area)

        # 3. 按日期范围截取数据
        if self.dates is not None:
            # If meteorological observations exist before start date
            # use these as well. Similiar to hydrological warmup period.
            if self.dates[0] - pd.DateOffset(days=self.seq_length) > df.index[0]:
                start_date = self.dates[0] - pd.DateOffset(days=self.seq_length)
            else:
                start_date = self.dates[0]
            df = df[start_date:self.dates[1]]

        # 4. 训练集：计算并存储特征的均值和标准差
        if self.period == 'train':
            self.means = df.mean()  # 各列均值
            self.stds = df.std()  # 各列标准差

        # 5. 从DataFrame中提取特征矩阵（使用指定的气象变量）
        x = np.array([
            df['prcp(mm/day)'].values,  # 降水量
            df['srad(W/m2)'].values,  # 下行短波辐射
            df['tmax(C)'].values,  # 最高温度
            df['tmin(C)'].values,  # 最低温度
            df['vp(Pa)'].values  # 水汽压
        ]).T  # 转置为 [时间步, 特征数] 形状

        y = np.array([df['QObs(mm/d)'].values]).T  # 径流目标值

        # 5.1 输入特征标准化 (Z-score归一化)
        x = self._local_normalization(x, variable='inputs')
        tqdm.write(f"Basin {self.basin} data head:")
        tqdm.write(str(df.head()))
        # 6. 重塑为LSTM序列格式
        x, y = reshape_data(x, y, self.seq_length)

        # 7. 训练集专用数据清洗
        if self.period == "train":
            # Delete all samples, where discharge is NaN
            if np.sum(np.isnan(y)) > 0:
                print(f"Deleted some records because of NaNs {self.basin}")
                x = np.delete(x, np.argwhere(np.isnan(y)), axis=0)
                y = np.delete(y, np.argwhere(np.isnan(y)), axis=0)

            # Deletes all records, where no discharge was measured (-999)
            x = np.delete(x, np.argwhere(y < 0)[:, 0], axis=0)
            y = np.delete(y, np.argwhere(y < 0)[:, 0], axis=0)

            # 7.3 径流目标值标准化
            y = self._local_normalization(y, variable='output')

        # 8. 转换为PyTorch张量（单精度浮点）
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))

        return x, y

        # 辅助方法：本地归一化函数
    def _local_normalization(self, feature: np.ndarray, variable: str) -> np.ndarray:
        """
        使用存储的统计量对输入特征或输出目标进行Z-score标准化

        Args:
            feature: 待标准化的特征值数组
            variable: 特征类型标识符 ('inputs' 或 'output')

        Returns:
            标准化后的特征数组

        Raises:
            RuntimeError: 如果传入未知的变量类型

        标准化公式:
            标准化值 = (原始值 - 均值) / 标准差
        """
        if variable == 'inputs':
            means = np.array([
                self.means['prcp(mm/day)'],
                self.means['srad(W/m2)'],
                self.means['tmax(C)'],
                self.means['tmin(C)'],
                self.means['vp(Pa)']
            ])

            stds = np.array([
                self.stds['prcp(mm/day)'],
                self.stds['srad(W/m2)'],
                self.stds['tmax(C)'],
                self.stds['tmin(C)'],
                self.stds['vp(Pa)']
            ])

            # 对整个特征数组执行向量化标准化
            # 利用numpy广播机制，对每列应用对应的统计量
            feature = (feature - means) / stds

        elif variable == 'output':
            # 对输出目标(径流量)进行标准化
            # 注意：径流量是单列数据
            feature = ((feature - self.means["QObs(mm/d)"]) /
                        self.stds["QObs(mm/d)"])

        else:
            # 避免未知变量类型的错误处理
            raise RuntimeError(f"Unknown variable type {variable}")

        return feature

    def local_rescale(self, feature: np.ndarray, variable: str) -> np.ndarray:
        """
        将标准化后的特征/目标值逆变换回原始尺度（Z-score标准化逆操作）

        Args:
            feature: 标准化后的特征值数组
            variable: 特征类型标识符 ('inputs' 或 'output')

        Returns:
            恢复原始尺度的特征数组

        Raises:
            RuntimeError: 如果传入未知的变量类型

        逆变换公式（标准化公式的逆操作）：
            原始值 = 标准化值 × 标准差 + 均值
        """
        if variable == 'inputs':
            # 重建气象特征的均值数组（按固定顺序）
            means = np.array([
                self.means['prcp(mm/day)'],  # 降水量原始均值
                self.means['srad(W/m2)'],  # 太阳辐射原始均值
                self.means['tmax(C)'],  # 最高温度原始均值
                self.means['tmin(C)'],  # 最低温度原始均值
                self.means['vp(Pa)']  # 水汽压原始均值
            ])

            # 重建气象特征的标准差数组
            stds = np.array([
                self.stds['prcp(mm/day)'],  # 降水量原始标准差
                self.stds['srad(W/m2)'],  # 太阳辐射原始标准差
                self.stds['tmax(C)'],  # 最高温度原始标准差
                self.stds['tmin(C)'],  # 最低温度原始标准差
                self.stds['vp(Pa)']  # 水汽压原始标准差
            ])

            # 对整个特征数组执行逆变换
            # 利用numpy广播机制，对每列应用对应的统计量
            feature = feature * stds + means

        elif variable == 'output':
            feature = (feature * self.stds["QObs(mm/d)"] +
                       self.means["QObs(mm/d)"])

        else:
            # 避免未知变量类型的错误处理
            raise RuntimeError(f"Unknown variable type {variable}")

        return feature

    def get_means(self):
        return self.means

    def get_stds(self):
        return self.stds