import torch.nn as nn
from dataset import get_dataloader
import torch
import config
import numpy as np



class MAE(nn.Module):
    def __init__(self):
        super(MAE, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs(y_true - y_pred))


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.mean((y_true - y_pred) ** 2)



class RAE(nn.Module):
    def __init__(self):
        super(RAE, self).__init__()

    def forward(self, y_pred, y_true):
        abs_error = torch.sum(torch.abs(y_true - y_pred))
        mean_true = torch.mean(y_true)
        abs_deviation = torch.sum(torch.abs(y_true - mean_true))
        return abs_error / abs_deviation


class RSE(nn.Module):
    def __init__(self):
        super(RSE, self).__init__()

    def forward(self, y_pred, y_true):
        squared_error = torch.sum((y_true - y_pred) ** 2)
        mean_true = torch.mean(y_true)
        variance = torch.sum((y_true - mean_true) ** 2)
        return squared_error / variance


class MAPE(nn.Module):
    def __init__(self):
        super(MAPE, self).__init__()

    def forward(self, y_pred, y_true):
        epsilon = 1e-8  # 避免除零
        return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100


class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, y_pred, y_true):
        mse = torch.mean((y_true - y_pred) ** 2)
        return torch.sqrt(mse)


class MSLE(nn.Module):
    def __init__(self):
        super(MSLE, self).__init__()

    def forward(self, y_pred, y_true):

        return torch.mean((torch.log1p(y_true) - torch.log1p(y_pred)) ** 2)


class RMSLE(nn.Module):
    def __init__(self):
        super(RMSLE, self).__init__()

    def forward(self, y_pred, y_true):

        msle = torch.mean((torch.log1p(y_true) - torch.log1p(y_pred)) ** 2)
        return torch.sqrt(msle)


class NRMSE(nn.Module):
    def __init__(self, normalization='mean'):
        super(NRMSE, self).__init__()
        self.normalization = 'mean'  # 'mean' 或 'range'

    def forward(self, y_pred, y_true):
        rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2))
        if self.normalization == 'mean':
            norm_factor = torch.mean(y_true)
        elif self.normalization == 'range':
            norm_factor = torch.max(y_true) - torch.min(y_true)
        else:
            raise ValueError("Normalization must be 'mean' or 'range'")
        return rmse / norm_factor


class RRMSE(nn.Module):
    def __init__(self):
        super(RRMSE, self).__init__()

    def forward(self, y_pred, y_true):
        rmse = torch.mean((y_true - y_pred) ** 2)
        mean_pred = torch.sum((y_pred) ** 2)
        return torch.sqrt(rmse / mean_pred)


class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
        error = y_true - y_pred
        abs_error = torch.abs(error)
        condition = abs_error < self.delta
        squared_loss = 0.5 * error ** 2
        linear_loss = self.delta * abs_error - 0.5 * self.delta ** 2
        return torch.mean(torch.where(condition, squared_loss, linear_loss))



class QuantileLoss(nn.Module):
    def __init__(self, quantile=0.8):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile

    def forward(self, y_pred, y_true):
        error = y_true - y_pred
        loss = torch.max(self.quantile * error, (self.quantile - 1) * error)
        return torch.mean(loss)


class NSELoss(nn.Module):
    """纳什-萨特克利夫效率系数损失函数"""

    def __init__(self):
        super(NSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: 预测值
            y_true: 真实观测值
        Returns:
            loss: NSE (越小越好)
        """
        # 计算观测值的均值
        y_true_mean = torch.mean(y_true)

        # 分子：预测值与观测值的平方差之和
        numerator = torch.sum((y_true - y_pred) ** 2)

        # 分母：观测值与均值的平方差之和
        denominator = torch.sum((y_true - y_true_mean) ** 2)

        # NSE =  (分子/分母)
        loss = (numerator / denominator)

        # 返回损失：NSE (因为NSE越接近1越好，所以损失越小越好)
        return loss


class KGELoss(nn.Module):
    """科尔莫戈罗夫-斯米尔诺夫效率系数损失函数（PyTorch版）"""

    def __init__(self):
        super(KGELoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: 预测值 (位于GPU的Tensor)
            y_true: 真实观测值 (位于GPU的Tensor)
        Returns:
            loss: 1 - KGE (越小越好), 保持自动求导能力
        """
        # 使用PyTorch函数计算均值、方差和协方差
        mean_obs = torch.mean(y_true)
        mean_pred = torch.mean(y_pred)

        # 避免除零错误，增加一个极小值
        # 计算相关系数r
        covariance = torch.sum((y_true - mean_obs) * (y_pred - mean_pred))
        std_obs = torch.sqrt(torch.sum((y_true - mean_obs) ** 2))
        std_pred = torch.sqrt(torch.sum((y_pred - mean_pred) ** 2))

        # 计算三个分量
        w = mean_pred / mean_obs # 偏置项
        v = std_pred / std_obs  # 变异性项
        r = covariance / (std_obs * std_pred) # 相关系数

        loss = torch.sqrt((w - 1) ** 2 + (v - 1) ** 2 + (r - 1) ** 2)

        return loss
