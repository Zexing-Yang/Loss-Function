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
from datadeal import get_data_loaders


def train_region(region_id, basin_list):
    """为单个区域训练模型"""
    print(f"\n{'=' * 60}")
    print(f"START TRAINING REGION {region_id}")
    print(f"{'=' * 60}")

    # 设置区域专属的结果目录 - 修改为两位数字格式
    region_id_padded = region_id.zfill(2)  # 确保两位格式
    region_results_dir = os.path.join(config.RESULTS_DIR, region_id_padded)
    os.makedirs(region_results_dir, exist_ok=True)
    print(f"Results will be saved to: {region_results_dir}")

    # ===================== 1. 数据准备 =====================
    print(f"\nLoading data for region {region_id} - {len(basin_list)} basins...")

    # 临时设置当前区域的流域列表
    original_basin_list = getattr(config, 'BASIN_LIST', [])
    config.BASIN_LIST = basin_list  # 动态设置当前区域的流域

    try:
        basin_datasets = get_data_loaders()
    except Exception as e:
        print(f"Error loading data for region {region_id}: {str(e)}")
        config.BASIN_LIST = original_basin_list  # 恢复原始列表
        return None

    # 获取实际加载的流域数量
    loaded_basins = list(basin_datasets.keys())
    num_basins = len(loaded_basins)

    if num_basins == 0:
        print(f"No basins successfully loaded for region {region_id}")
        config.BASIN_LIST = original_basin_list
        return None

    print(f"Successfully loaded {num_basins} basins for region {region_id}")

    # ===================== 2. 模型初始化 =====================
    model = Model(
        hidden_size=config.MODEL_CONFIG['hidden_size'],
        dropout_rate=config.MODEL_CONFIG['dropout_rate'],
    ).to(config.DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.MODEL_CONFIG['learning_rate']
    )

    loss_func = Loss.MAE()
    print(f'Loss function: {loss_func}')

    # ===================== 3. 训练循环 =====================
    n_epochs = config.MODEL_CONFIG['n_epochs']
    best_val_loss = float('inf')
    best_epoch = 0

    train_history = {
        'epochs': [], 'train_loss': [], 'val_loss': [],
        'val_nse': [], 'val_kge': [], 'val_tpe': [], 'duration': []
    }

    region_start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()

        # 训练和验证
        train_loss = train_epoch(model, optimizer, basin_datasets, loss_func, epoch, config.DEVICE)
        val_loss = eval_loss(model, basin_datasets, loss_func, config.DEVICE)

        # 性能评估
        val_results = eval_model(model, basin_datasets, period='val', device=config.DEVICE)
        val_nses = [result['NSE'] for result in val_results.values()]
        avg_nse = np.nanmean(val_nses)
        val_kges = [result['KGE'] for result in val_results.values()]
        avg_kge = np.nanmean(val_kges)
        val_tpes = [result['TPE'] for result in val_results.values()]
        avg_tpe = np.nanmean(val_tpes)

        epoch_duration = time.time() - epoch_start

        # 记录历史
        train_history['epochs'].append(epoch)
        train_history['train_loss'].append(train_loss)
        train_history['val_loss'].append(val_loss)
        train_history['val_nse'].append(avg_nse)
        train_history['val_kge'].append(avg_kge)
        train_history['val_tpe'].append(avg_tpe)
        train_history['duration'].append(epoch_duration)

        # 打印进度
        tqdm.write(f"Region {region_id} | Epoch {epoch}/{n_epochs} | "
                   f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                   f"Val NSE: {avg_nse:.4f} | Time: {epoch_duration:.1f}s")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_nse': avg_nse,
                'val_kge': avg_kge,
                'val_tpe': avg_tpe,
                'basin_list': loaded_basins,
                'region_id': region_id,
                'model_config': config.MODEL_CONFIG
            }, os.path.join(region_results_dir, 'best_model.pth'))

    # ===================== 4. 测试评估 =====================
    print(f"\nEvaluating on test set for region {region_id}...")

    # 加载最佳模型
    checkpoint = torch.load(os.path.join(region_results_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_results = eval_model(
        model, basin_datasets, period='test', device=config.DEVICE
    )

    # ===================== 5. 保存结果 =====================
    print(f"\nSaving results for region {region_id}...")

    # 保存测试结果CSV
    results_data = []
    for basin, result in test_results.items():
        results_data.append({
            'Region': region_id,
            'Basin': basin,
            'NSE': result['NSE'],
            'KGE': result['KGE'],
            'TPE': result['TPE']
        })

    results_df = pd.DataFrame(results_data)
    results_df.to_csv(os.path.join(region_results_dir, 'test_results.csv'), index=False)

    # 保存训练历史
    history_df = pd.DataFrame(train_history)
    history_df.to_csv(os.path.join(region_results_dir, 'training_history.csv'), index=False)

    # 保存区域总结报告
    avg_nse = results_df['NSE'].mean()
    avg_kge = results_df['KGE'].mean()
    avg_tpe = results_df['TPE'].mean()

    with open(os.path.join(region_results_dir, 'summary.txt'), 'w') as f:
        f.write(f"Region {region_id} Training Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Region ID: {region_id}\n")
        f.write(f"Number of basins: {num_basins}\n")
        f.write(f"Best epoch: {best_epoch}/{n_epochs}\n")
        f.write(f"Best validation loss: {best_val_loss:.4f}\n")
        f.write(f"Average NSE: {avg_nse:.4f}\n")
        f.write(f"Average KGE: {avg_kge:.4f}\n")
        f.write(f"Average TPE: {avg_tpe:.4f}\n")
        f.write(f"Total training time: {time.time() - region_start_time:.1f} seconds\n")
        f.write(f"Basins: {', '.join(loaded_basins)}\n")

    # ===================== 6. 绘制图表 =====================
    print(f"Generating plots for region {region_id}...")
    plot_region_results(region_id, test_results, region_results_dir)

    # 恢复原始流域列表
    config.BASIN_LIST = original_basin_list

    print(f"Region {region_id} processing completed!")
    return {
        'region_id': region_id,
        'num_basins': num_basins,
        'avg_nse': avg_nse,
        'avg_kge': avg_kge,
        'avg_tpe': avg_tpe,
        'test_results': test_results
    }


def plot_region_results(region_id, test_results, save_dir):
    """为单个区域绘制结果图表"""
    num_basins = len(test_results)
    num_cols = 3
    num_rows = (num_basins + num_cols - 1) // num_cols

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))
    if num_basins > 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    # 按NSE排序
    sorted_basins = sorted(test_results.items(), key=lambda x: x[1]['NSE'], reverse=True)

    for idx, (basin, result) in enumerate(sorted_basins):
        ax = axs[idx]

        # 获取时间范围
        test_start, test_end = config.TEST_PERIOD
        start_date = pd.to_datetime(test_start, format="%Y-%m-%d")
        end_date = pd.to_datetime(test_end, format="%Y-%m-%d")
        date_range = pd.date_range(start_date, end_date)

        # 确保长度匹配
        n = min(len(date_range), len(result['obs']))
        dates = date_range[:n]
        obs = result['obs'][:n]
        preds = result['preds'][:n]

        # 绘制观测和预测值
        ax.plot(dates, obs, 'b-', label='Observation', alpha=0.7, linewidth=1)
        ax.plot(dates, preds, 'r-', label='Prediction', alpha=0.8, linewidth=1)

        # 设置子图标题和标签
        ax.set_title(f"Basin {basin}\nNSE: {result['NSE']:.3f}, KGE: {result['KGE']:.3f}", fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # 隐藏多余子图
    for idx in range(len(sorted_basins), len(axs)):
        axs[idx].axis('off')

    plt.suptitle(f"Region {region_id} - Model Performance ({num_basins} Basins)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 修改图片文件名，去掉region_前缀
    plt.savefig(os.path.join(save_dir, f'{region_id}_results.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 为每个流域保存详细图表
    for basin, result in test_results.items():
        fig, ax = plt.subplots(figsize=(12, 4))

        test_start, test_end = config.TEST_PERIOD
        start_date = pd.to_datetime(test_start, format="%Y-%m-%d")
        end_date = pd.to_datetime(test_end, format="%Y-%m-%d")
        date_range = pd.date_range(start_date, end_date)

        n = min(len(date_range), len(result['obs']))
        dates = date_range[:n]
        obs = result['obs'][:n]
        preds = result['preds'][:n]

        ax.plot(dates, obs, 'b-', label='Observation', linewidth=1.2)
        ax.plot(dates, preds, 'r-', label='Prediction', linewidth=1.2, alpha=0.8)
        ax.set_title(
            f"Region {region_id} - Basin {basin}\nNSE: {result['NSE']:.3f}, KGE: {result['KGE']:.3f}, TPE: {result['TPE']:.3f}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Discharge (mm/d)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"basin_{basin}.png"), dpi=200)
        plt.close()


if __name__ == "__main__":
    """主函数 - 按区域循环训练"""
    start_time = time.time()

    # 创建总结果目录
    config.RESULTS_DIR = "training_results"
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    print(f"All regional results will be saved in: {config.RESULTS_DIR}")

    # 获取按区域分组的流域字典
    basin_dict = config.discover_basins()

    # 存储所有区域的结果
    all_region_results = {}

    # 按区域循环训练
    for region_id, basin_list in basin_dict.items():
        if not basin_list:
            print(f"Skipping region {region_id} - no basins found")
            continue

        region_result = train_region(region_id, basin_list)
        if region_result:
            all_region_results[region_id] = region_result

    # ===================== 生成总结合并报告 =====================
    print(f"\n{'=' * 60}")
    print("GENERATING SUMMARY REPORT FOR ALL REGIONS")
    print(f"{'=' * 60}")

    # 保存跨区域比较结果
    summary_data = []
    for region_id, result in all_region_results.items():
        summary_data.append({
            'Region': region_id,
            'Number_of_Basins': result['num_basins'],
            'Average_NSE': result['avg_nse'],
            'Average_KGE': result['avg_kge'],
            'Average_TPE': result['avg_tpe']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(config.RESULTS_DIR, 'regional_summary.csv'), index=False)

    # 打印总结报告
    print("\nREGIONAL TRAINING SUMMARY:")
    print("-" * 60)
    for region_id, result in all_region_results.items():
        print(f"Region {region_id}: {result['num_basins']} basins | "
              f"NSE: {result['avg_nse']:.4f} | "
              f"KGE: {result['avg_kge']:.4f} | "
              f"TPE: {result['avg_tpe']:.4f}")

    print(f"\nTotal processing time: {time.time() - start_time:.1f} seconds")
    print("All regional trainings completed successfully!")