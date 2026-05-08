import pickle
import numpy as np
import torch
from pathlib import Path
from config import path
import os

def load_emissivity_data(filename):
    """加载发射率数据文件"""
    file_path = Path(path.INPUT_DIR) / filename
    return np.load(file_path, allow_pickle=True)

def process_emissivity_data(emissivity_dict, T_ref):
    """
    处理发射率数据并装配出mole_fraction_ratios、temperatures、path_lengths
    和三维epsilon数组
    
    参数:
    emissivity_dict (dict): 加载的发射率数据字典
    T_ref (float): 参考温度(K)
    
    返回:
    Tuple: (mole_fracs, temps, paths, epsilon_3d)
    """
    # 提取唯一值
    mole_fracs = np.array(list(emissivity_dict.keys()))
    temps = np.array(list(emissivity_dict[mole_fracs[0]].keys()))
    paths = sorted(set([item[0] for item in next(iter(emissivity_dict[mole_fracs[0]].values()))]))
    
    # 初始化三维数组
    epsilon_3d = np.zeros((len(mole_fracs), len(temps), len(paths)))
    
    # 填充epsilon_3d数组
    for mfr_idx, mfr in enumerate(mole_fracs):
        for T_idx, T in enumerate(temps):
            for pl_idx, pl in enumerate(paths):
                # 查找对应epsilon值
                epsilon_values = emissivity_dict[mfr][T]
                for pl_value, epsilon in epsilon_values:
                    if pl_value == pl:
                        epsilon_3d[mfr_idx, T_idx, pl_idx] = epsilon
                        break
    
    # 温度归一化
    temps_normalized = temps / T_ref
    
    return mole_fracs, temps_normalized, np.array(paths), epsilon_3d

def prepare_tensors(train_data, val_data, T_ref, device):
    """
    准备训练和验证数据集
    
    参数:
    train_data, val_data: 训练和验证数据字典
    T_ref (float): 参考温度(K)
    device: 计算设备
    
    返回:
    Tuple: (X_train, y_train, X_val, y_val)
    """
    # 处理数据
    train_mole, train_temp, train_path, train_epsilon = process_emissivity_data(train_data, T_ref)
    val_mole, val_temp, val_path, val_epsilon = process_emissivity_data(val_data, T_ref)
    
    # 转换为张量
    def to_tensor(data):
        return torch.tensor(data, dtype=torch.float32).to(device)
    
    mole_train = to_tensor(train_mole)
    temp_train = to_tensor(train_temp)
    path_train = to_tensor(train_path)
    epsilon_3d_train = to_tensor(train_epsilon)
    
    mole_val = to_tensor(val_mole)
    temp_val = to_tensor(val_temp)
    path_val = to_tensor(val_path)
    epsilon_3d_val = to_tensor(val_epsilon)
    
    # 生成三维网格并展平
    Mr_grid_train, Tr_grid_train, L_grid_train = torch.meshgrid(mole_train, temp_train, path_train, indexing='ij')
    epsilon_flat_train = epsilon_3d_train.reshape(-1)
    
    Mr_grid_val, Tr_grid_val, L_grid_val = torch.meshgrid(mole_val, temp_val, path_val, indexing='ij')
    epsilon_flat_val = epsilon_3d_val.reshape(-1)

    # 训练集数据
    X_train = (Mr_grid_train.reshape(-1), Tr_grid_train.reshape(-1), L_grid_train.reshape(-1))
    y_train = epsilon_flat_train

    # 验证集数据
    X_val = (Mr_grid_val.reshape(-1), Tr_grid_val.reshape(-1), L_grid_val.reshape(-1))
    y_val = epsilon_flat_val
    
    return X_train, y_train, X_val, y_val