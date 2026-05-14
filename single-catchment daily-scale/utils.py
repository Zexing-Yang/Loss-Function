import numpy as np


def calculate_metrics(obs: np.array, sim: np.array) -> dict:
    # 创建有效数据掩码
    valid_mask = (obs >= 0) & (~np.isnan(obs))

    # 同时过滤obs和sim
    obs_valid = obs[valid_mask]
    sim_valid = sim[valid_mask]

    # 检查是否有有效数据
    if len(obs_valid) == 0:
        return {'NSE': np.nan, 'KGE': np.nan, 'TPE': np.nan}

    # ================= NSE计算 =================
    denominator = np.power((obs_valid.mean() - obs_valid), 2).sum()


    nse_val = 1 - np.power((obs_valid - sim_valid), 2).sum() / denominator

    # 检查NSE是否异常
    if nse_val > 1.5:  # 如果NSE异常大，可能是数据问题
        print(f"警告: NSE值异常: {nse_val}")
        # 可以返回nan或重新检查数据
    # ================= KGE计算 =================
    mean_sim = sim_valid.mean()
    mean_obs = obs_valid.mean()
    w = mean_sim / mean_obs
    var_real = np.power(obs - obs.mean(), 2).sum()
    var_pred = np.power(sim - sim.mean(), 2).sum()
    v = np.power(var_pred / var_real, 0.5)
    var1 = ((obs - obs.mean()) * (sim - sim.mean())).sum()
    r = var1 / (np.power(var_real, 0.5) * np.power(var_pred, 0.5))
    kge_val = 1 - np.power(np.power(w - 1, 2) + np.power(v - 1, 2) + np.power(r - 1, 2), 0.5)

    # ================= TPE计算 =================
    obs = obs_valid.flatten()
    sim = sim_valid.flatten()
    obs_sort = np.sort(obs)[::-1]
    obs_argsort = np.argsort(-obs)
    sim_sort = np.array(sim)[obs_argsort]
    tpe_val = abs(obs_sort[0:len(obs) // 50] - sim_sort[0:len(obs) // 50]).sum() / obs_sort[0:len(obs) // 50].sum()
    # h = max(1, int(len(obs) * 0.02))
    #
    # # 获取峰值索引 (降序排列)
    # top_idx = np.argsort(obs)[::-1][:h]
    # top_obs = obs[top_idx]
    # top_sim = sim[top_idx]
    #
    # # 计算TPE
    # a = np.sum(np.abs(top_obs - top_sim))
    # b = np.sum(top_obs)
    #
    # tpe_val = a / b
    return {'NSE': nse_val, 'KGE': kge_val, 'TPE': tpe_val}