import tqdm
import torch
import numpy as np
import random
import config


def train_epoch(model, optimizer, basin_loaders, loss_func, epoch, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for basin, data in basin_loaders.items():
        loader = data['train'][1]
        for xs, ys in loader:
            xs, ys = xs.to(device), ys.to(device)

            optimizer.zero_grad()
            y_hat = model(xs)
            loss = loss_func(y_hat, ys)
            loss.backward()
            optimizer.step()

            batch_size = xs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples

def eval_loss(model, basin_loaders, loss_func,  device=config.DEVICE):
    """计算验证集上的平均损失"""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        # 遍历每个流域的验证数据
        for data in basin_loaders.values():
            loader = data['val'][1]
            for xs, ys in loader:
                xs = xs.to(device)
                ys = ys.to(device)

                # 前向传播
                y_hat = model(xs)
                loss = loss_func(y_hat, ys)
                batch_size = xs.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

    avg_loss = total_loss / total_samples
    return avg_loss