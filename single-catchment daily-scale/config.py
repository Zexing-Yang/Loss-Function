import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

# 配置Matplotlib绘图参数
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 14
plt.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['font.weight'] = 'normal'
mpl.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 设置根目录和计算设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 在config.py中添加区域配置
REGION_LIST = ['01']  # 目标区域文件夹
RESULTS_DIR = "training_results"

BASIN_LIST = []
# 修改discover_basins函数，按区域返回流域字典
def discover_basins():
    """发现并返回按区域分组的流域字典"""
    basin_dict = {}

    for region_id in REGION_LIST:
        folder_id = region_id.zfill(2)  # 确保两位格式
        forcing_path = f'./basin_mean_forcing/daymet/{folder_id}'
        basin_set = set()

        # 检查文件夹是否存在
        if not os.path.exists(forcing_path):
            print(f"Warning: Region folder {forcing_path} not found")
            basin_dict[region_id] = []
            continue

        # +++ 修改点：优先使用自定义流域列表 +++
        # 如果用户指定了BASIN_LIST，则只使用这些流域
        if BASIN_LIST:
            # 验证自定义流域是否在指定区域中存在
            available_basins = []
            for basin_id in BASIN_LIST:
                # 构建流域文件路径模式进行检查
                pattern = os.path.join(forcing_path, f"{basin_id}_*.txt")
                matches = glob.glob(pattern)
                if matches:
                    available_basins.append(basin_id)
                else:
                    print(f"Warning: Basin {basin_id} not found in region {region_id}")
            basin_dict[region_id] = sorted(available_basins)
            print(f"Region {region_id}: using {len(available_basins)} user-specified basins")

        # 如果未指定BASIN_LIST，则自动发现该区域所有流域
        else:
            # 查找该区域下的所有流域文件 (您原有的自动发现逻辑)
            files = glob.glob(f"{forcing_path}/*.txt")
            for file in files:
                basin_id = os.path.basename(file).split('_')[0]
                basin_set.add(basin_id)

            basin_dict[region_id] = sorted(basin_set)
            print(f"Region {region_id}: auto-discovered {len(basin_set)} basins")

    return basin_dict

# 打印发现的流域数量
TRAIN_PERIOD = ["1980-10-01", "1995-09-30"]
VAL_PERIOD = ["1995-10-01", "2000-09-30"]
TEST_PERIOD = ["2000-10-01", "2010-09-30"]

# 在MODEL_CONFIG中添加贝叶斯优化相关配置
MODEL_CONFIG = {
    'hidden_size': 20,
    'dropout_rate': 0.1,
    'learning_rate': 1e-3,
    'sequence_length': 365,
    'n_epochs': 50,  # 增加训练轮数以获得更好收敛
    'train_batch_size': 512,
    'eval_batch_size': 1024,
}
