import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
import config
from model import Model
from datadeal import get_data_loaders
from train import train_epoch, eval_loss
from eval import eval_model
from utils import calculate_metrics
import time
import Loss

def train_basin(region_id, basin_id):
    """为单个流域训练模型（需要传入正确的区域ID）"""
    region_id_padded = region_id.zfill(2)  # 确保两位格式

    print(f"\n{'=' * 60}")
    print(f"START TRAINING BASIN {basin_id} (Region {region_id})")
    print(f"{'=' * 60}")

    # 设置流域专属的结果目录：使用正确的区域ID
    basin_results_dir = os.path.join(config.RESULTS_DIR, region_id_padded, basin_id)
    os.makedirs(basin_results_dir, exist_ok=True)
    print(f"Results will be saved to: {basin_results_dir}")

    # ===================== 1. 数据准备 =====================
    print(f"\nLoading data for basin {basin_id}...")

    # 动态设置当前要训练的流域列表（仅包含当前流域）
    original_basin_list = getattr(config, 'BASIN_LIST', [])
    config.BASIN_LIST = [basin_id]

    try:
        # 加载单个流域的数据
        basin_datasets = get_data_loaders()
    except Exception as e:
        print(f"Error loading data for basin {basin_id}: {str(e)}")
        config.BASIN_LIST = original_basin_list  # 恢复原始列表
        return None

    # 检查是否成功加载
    if basin_id not in basin_datasets:
        print(f"Basin {basin_id} not successfully loaded.")
        config.BASIN_LIST = original_basin_list
        return None

    # ===================== 2. 模型初始化 =====================
    model = Model(
        hidden_size=config.MODEL_CONFIG['hidden_size'],
        dropout_rate=config.MODEL_CONFIG['dropout_rate'],
    ).to(config.DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.MODEL_CONFIG['learning_rate']
    )
    loss_func = Loss.QuantileLoss()
    print(f'Loss function: {loss_func}')

    # ===================== 3. 训练循环 =====================
    n_epochs = config.MODEL_CONFIG['n_epochs']
    best_val_loss = float('inf')
    best_epoch = 0

    train_history = {
        'epochs': [], 'train_loss': [], 'val_loss': [],
        'val_nse': [], 'val_kge': [], 'val_tpe': [], 'duration': []
    }

    basin_start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()

        # 训练和验证（数据集仅包含当前流域）
        train_loss = train_epoch(model, optimizer, basin_datasets, loss_func, epoch, config.DEVICE)
        val_loss = eval_loss(model, basin_datasets, loss_func, config.DEVICE)

        # 在验证集上评估性能指标
        val_results = eval_model(model, basin_datasets, period='val', device=config.DEVICE)
        # 由于是单流域，val_results 字典只有一个键值对
        basin_val_result = val_results.get(basin_id, {})
        val_nse = basin_val_result.get('NSE', np.nan)
        val_kge = basin_val_result.get('KGE', np.nan)
        val_tpe = basin_val_result.get('TPE', np.nan)

        epoch_duration = time.time() - epoch_start

        # 记录历史
        train_history['epochs'].append(epoch)
        train_history['train_loss'].append(train_loss)
        train_history['val_loss'].append(val_loss)
        train_history['val_nse'].append(val_nse)
        train_history['val_kge'].append(val_kge)
        train_history['val_tpe'].append(val_tpe)
        train_history['duration'].append(epoch_duration)

        # 打印进度
        tqdm.write(f"Basin {basin_id} | Epoch {epoch}/{n_epochs} | "
                   f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                   f"Val NSE: {val_nse:.4f} | Time: {epoch_duration:.1f}s")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_nse': val_nse,
                'val_kge': val_kge,
                'val_tpe': val_tpe,
                'basin_id': basin_id,
                'region_id': region_id,
                'model_config': config.MODEL_CONFIG
            }, os.path.join(basin_results_dir, 'best_model.pth'))

    # ===================== 4. 测试评估 =====================
    print(f"\nEvaluating on test set for basin {basin_id}...")
    # 加载最佳模型进行评估
    checkpoint_path = os.path.join(basin_results_dir, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Warning: Best model checkpoint not found for {basin_id}, using final model for test.")

    test_results = eval_model(model, basin_datasets, period='test', device=config.DEVICE)
    basin_test_result = test_results.get(basin_id, {})

    # ===================== 5. 保存结果 =====================
    print(f"\nSaving results for basin {basin_id}...")
    # 保存测试结果CSV（单行数据，格式与之前一致）
    results_data = [{
        'Region': region_id,
        'Basin': basin_id,
        'NSE': basin_test_result.get('NSE', np.nan),
        'KGE': basin_test_result.get('KGE', np.nan),
        'TPE': basin_test_result.get('TPE', np.nan)
    }]
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(os.path.join(basin_results_dir, 'test_results.csv'), index=False)

    # 保存训练历史
    history_df = pd.DataFrame(train_history)
    history_df.to_csv(os.path.join(basin_results_dir, 'training_history.csv'), index=False)

    # 保存流域总结报告
    with open(os.path.join(basin_results_dir, 'summary.txt'), 'w') as f:
        f.write(f"Basin {basin_id} Training Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Region ID: {region_id}\n")
        f.write(f"Basin ID: {basin_id}\n")
        f.write(f"Best epoch: {best_epoch}/{n_epochs}\n")
        f.write(f"Best validation loss: {best_val_loss:.4f}\n")
        f.write(f"Test NSE: {basin_test_result.get('NSE', np.nan):.4f}\n")
        f.write(f"Test KGE: {basin_test_result.get('KGE', np.nan):.4f}\n")
        f.write(f"Test TPE: {basin_test_result.get('TPE', np.nan):.4f}\n")
        f.write(f"Total training time: {time.time() - basin_start_time:.1f} seconds\n")

    # ===================== 6. 绘制图表（单个流域）=====================
    print(f"Generating plot for basin {basin_id}...")
    plot_basin_results(basin_id, region_id, basin_test_result, basin_results_dir)

    # 恢复原始流域列表
    config.BASIN_LIST = original_basin_list

    print(f"Basin {basin_id} processing completed!")
    return {
        'region_id': region_id,
        'basin_id': basin_id,
        'test_nse': basin_test_result.get('NSE', np.nan),
        'test_kge': basin_test_result.get('KGE', np.nan),
        'test_tpe': basin_test_result.get('TPE', np.nan)
    }


def plot_basin_results(basin_id, region_id, test_result, save_dir):
    """为单个流域绘制结果图表"""
    if not test_result or 'obs' not in test_result or 'preds' not in test_result:
        print(f"No valid test results to plot for basin {basin_id}")
        return

    fig, ax = plt.subplots(figsize=(12, 4))

    # 获取时间范围
    test_start, test_end = config.TEST_PERIOD
    start_date = pd.to_datetime(test_start, format="%Y-%m-%d")
    end_date = pd.to_datetime(test_end, format="%Y-%m-%d")
    date_range = pd.date_range(start_date, end_date)

    # 确保长度匹配
    obs = test_result['obs'].flatten()
    preds = test_result['preds'].flatten()
    n = min(len(date_range), len(obs))
    dates = date_range[:n]
    obs_plot = obs[:n]
    preds_plot = preds[:n]

    # 绘制观测和预测值
    ax.plot(dates, obs_plot, 'b-', label='Observation', linewidth=1.2)
    ax.plot(dates, preds_plot, 'r-', label='Prediction', linewidth=1.2, alpha=0.8)
    ax.set_title(f"Region {region_id} - Basin {basin_id}\n"
                 f"NSE: {test_result.get('NSE', np.nan):.3f}, "
                 f"KGE: {test_result.get('KGE', np.nan):.3f}, "
                 f"TPE: {test_result.get('TPE', np.nan):.3f}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Discharge (mm/d)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"basin_{basin_id}_results.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Plot saved to: {plot_path}")


def save_region_summary(region_id, basin_results_list, results_dir):
    """将一个区域的所有流域结果保存为一份CSV文件（格式与MSE.csv一致）"""
    if not basin_results_list:
        return None

    # 构建区域文件夹路径
    region_id_padded = region_id.zfill(2)
    region_dir = os.path.join(results_dir, region_id_padded)
    os.makedirs(region_dir, exist_ok=True)

    # 准备数据，格式与您的MSE.csv示例完全一致
    summary_data = []
    for result in basin_results_list:
        summary_data.append({
            'Region': region_id,
            'Basin': result['basin_id'],
            'NSE': result['test_nse'],
            'KGE': result['test_kge'],
            'TPE': result['test_tpe']
        })

    summary_df = pd.DataFrame(summary_data)
    # 按流域ID排序，使文件更规整
    summary_df = summary_df.sort_values(by='Basin').reset_index(drop=True)

    # 保存文件，例如：region_01_results.csv
    region_summary_path = os.path.join(region_dir, f'region_{region_id_padded}_results.csv')
    summary_df.to_csv(region_summary_path, index=False)
    print(f"区域 {region_id} 的汇总结果已保存至：{region_summary_path}")
    return region_summary_path


if __name__ == "__main__":
    """主函数 - 按流域循环训练，并按区域汇总结果"""
    start_time = time.time()
    config.RESULTS_DIR = "training_results"
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # 获取按区域分组的流域字典
    basin_dict = config.discover_basins()

    print("发现的流域分组：")
    for region_id, basin_list in basin_dict.items():
        print(f"  区域 {region_id}: {len(basin_list)} 个流域")
        if basin_list:
            print(f"    示例流域: {basin_list[:3]}...")  # 显示前3个流域作为示例

    all_processed_results = []  # 保存所有流域结果用于最终总结
    regional_summaries = {}  # 记录每个区域的汇总文件路径

    # 按区域循环
    for region_id, basin_list in basin_dict.items():
        if not basin_list:
            print(f"\n跳过区域 {region_id} - 未找到流域")
            continue

        print(f"\n开始处理区域 {region_id}，共 {len(basin_list)} 个流域...")
        region_basin_results = []  # 临时保存当前区域所有流域的结果

        # 遍历该区域的每个流域进行训练
        for basin_id in basin_list:
            # 传入正确的区域ID和流域ID
            basin_result = train_basin(region_id, basin_id)
            if basin_result:
                region_basin_results.append(basin_result)
                all_processed_results.append(basin_result)

        # 当前区域所有流域训练完成后，生成区域汇总文件
        if region_basin_results:
            summary_path = save_region_summary(region_id, region_basin_results, config.RESULTS_DIR)
            if summary_path:
                regional_summaries[region_id] = summary_path

    # ===================== 生成最终的总结合并报告 =====================
    print(f"\n{'=' * 60}")
    print("所有区域训练完成，生成最终报告")
    print(f"{'=' * 60}")

    # 1. 保存一份包含所有流域的全局汇总文件（便于整体分析）
    if all_processed_results:
        global_summary_data = []
        for result in all_processed_results:
            global_summary_data.append({
                'Region': result['region_id'],
                'Basin': result['basin_id'],
                'NSE': result['test_nse'],
                'KGE': result['test_kge'],
                'TPE': result['test_tpe']
            })
        global_df = pd.DataFrame(global_summary_data).sort_values(by=['Region', 'Basin'])
        global_summary_path = os.path.join(config.RESULTS_DIR, 'all_basins_global_summary.csv')
        global_df.to_csv(global_summary_path, index=False)
        print(f"\n全局流域汇总文件已保存至：{global_summary_path}")

    # 2. 打印区域汇总文件清单
    print("\n已生成的按区域汇总的结果文件：")
    for region_id, filepath in regional_summaries.items():
        print(f"  区域 {region_id}: {filepath}")

    print(f"\n总处理时间：{time.time() - start_time:.1f} 秒")
    print("训练完成！您现在可以直接使用各区域文件夹下的 'region_XX_results.csv' 文件进行画图。")