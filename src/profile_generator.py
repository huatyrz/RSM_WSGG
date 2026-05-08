#!/usr/bin/env python3
"""
Radiative Profile Generator for Ammonia-Hydrogen Combustion
Author: OpenAI Assistant
Usage: python profile_generator.py
"""

import numpy as np
import os
from typing import Callable, Dict, List, Optional

# 在这里修改要生成的案例ID列表
# SELECTED_CASES = ["1.1", "2.1"]   # 只生成1.1和2.1
SELECTED_CASES = []  # 空列表表示生成所有案例

# ------------------------------
# 可配置参数 (按需修改)
# ------------------------------
OUTPUT_DIR = "profiles"        # 输出目录
L_VALUES = [1.0, 5.0]    # 空间尺度 (m)
N_POINTS = 20                # 空间离散点数


# ------------------------------
# 物理场函数定义
# ------------------------------

def gradient_temp_v1(x: np.ndarray, L: float) -> np.ndarray:
    """Case2.1/3.1/3.2温度分布 (Eq.25)"""
    return 1000 - 400 * np.cos(2* np.pi * x /L )

def gradient_temp_v2(x: np.ndarray, L: float) -> np.ndarray:
    """Case2.2温度分布 (Eq.26)"""
    return 1900 - 400 * np.cos(2* np.pi * x /L )

# Case1的温度函数组
def case11_temp(x: np.ndarray, _: float) -> np.ndarray:
    """Case1.1 700K恒定温度"""
    return np.full_like(x, 600.0)

def case12_temp(x: np.ndarray, _: float) -> np.ndarray:
    """Case1.2 1100K恒定温度"""
    return np.full_like(x, 1300.0)

def case13_temp(x: np.ndarray, _: float) -> np.ndarray:
    """Case1.3 1500K恒定温度"""
    return np.full_like(x, 1900.0)

# def case14_temp(x: np.ndarray, _: float) -> np.ndarray:
#     """Case1.3 1500K恒定温度"""
#     return np.full_like(x, 550.0)

# def case15_temp(x: np.ndarray, _: float) -> np.ndarray:
#     """Case1.3 1500K恒定温度"""
#     return np.full_like(x, 600.0)

# def case16_temp(x: np.ndarray, _: float) -> np.ndarray:
#     """Case1.3 1500K恒定温度"""
#     return np.full_like(x, 650.0)

# Case3的浓度分布函数组
def case31_h2o(x: np.ndarray, L: float) -> np.ndarray:
    """Case3.1 H2O分布 (Eq.27)"""
    return 0.06 - 0.02 * np.cos(2 * np.pi * x / L)

def case31_nh3(x: np.ndarray, L: float) -> np.ndarray:
    """Case3.1 NH3分布 (Eq.28)"""
    return 0.14 + 0.02 * np.cos(2 * np.pi * x / L)

def case32_h2o(x: np.ndarray, L: float) -> np.ndarray:
    """Case3.2 H2O分布 (Eq.29)"""
    return 0.14 - 0.02 * np.cos(2 * np.pi * x / L)

def case32_h2o(x: np.ndarray, L: float) -> np.ndarray:
    """Case3.2 NH3分布 (Eq.30)"""
    return 0.06 + 0.02 * np.cos(2 * np.pi * x / L)

# ------------------------------
# 工况配置
# ------------------------------
CASE_CONFIGS = [
    # Case1 子工况组 (不同固定温度)
    {"case_id": "1.1", "temp_func": case11_temp, "y_h2o": 0.04, "y_nh3": 0.16, "mix_ratio": 0.25},
    {"case_id": "1.2", "temp_func": case12_temp, "y_h2o": 0.04, "y_nh3": 0.16, "mix_ratio": 0.25},
    {"case_id": "1.3", "temp_func": case13_temp, "y_h2o": 0.04, "y_nh3": 0.16, "mix_ratio": 0.25},
    # {"case_id": "1.4", "temp_func": case14_temp, "y_h2o": 0.04, "y_nh3": 0.16, "mix_ratio": 0.25},
    # {"case_id": "1.5", "temp_func": case15_temp, "y_h2o": 0.04, "y_nh3": 0.16, "mix_ratio": 0.25},
    # {"case_id": "1.6", "temp_func": case16_temp, "y_h2o": 0.04, "y_nh3": 0.16, "mix_ratio": 0.25},
    
    # Case2 子工况组 (不同温度梯度)
    {"case_id": "2.1", "temp_func": gradient_temp_v1, "y_h2o": 0.1, "y_nh3": 0.1, "mix_ratio": 1.0},
    {"case_id": "2.2", "temp_func": gradient_temp_v2, "y_h2o": 0.1, "y_nh3": 0.1, "mix_ratio": 1.0},
    
    # Case3 子工况组 (不同H2O、NH3分布)
    {"case_id": "3.1", "temp_func": gradient_temp_v1, "y_h2o": case31_h2o, "y_nh3": case31_nh3, "mix_ratio": None},
    {"case_id": "3.2", "temp_func": gradient_temp_v1, "y_h2o": case32_h2o, "y_nh3": case31_nh3, "mix_ratio": None}
]


# ------------------------------
# 核心生成函数
# ------------------------------
def generate_profiles(
    case_configs: List[Dict],
    l_values: List[float],
    n_points: int,
    output_dir: str,
    selected_cases: List[str]
) -> None:
    """生成所有工况的辐射参数文件
    
    Args:
        case_configs: 工况配置列表
        l_values: 空间尺度列表 (m)
        n_points: 空间离散点数
        output_dir: 输出目录路径
        selected_cases: 要生成的案例ID列表（空列表表示全部）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 过滤选择的案例
    configs_to_generate = [
        cfg for cfg in case_configs 
        if not selected_cases or cfg["case_id"] in selected_cases
    ]
    
    print(f"将生成 {len(configs_to_generate)} 个案例: " 
          f"{[c['case_id'] for c in configs_to_generate]}")

    for config in configs_to_generate:
        for L in l_values:
            # 生成空间坐标
            x = np.linspace(0, L, n_points)
            
            # 计算温度场
            T = config["temp_func"](x, L)
            
            # 计算H2O浓度
            y_h = (
                config["y_h2o"](x, L) 
                if callable(config["y_h2o"]) 
                else np.full_like(x, config["y_h2o"])
            )
            
            # 计算NH3浓度
            if config["y_nh3"] is not None:
                # 处理函数或固定值
                y_n = (
                    config["y_nh3"](x, L)
                    if callable(config["y_nh3"])
                    else np.full_like(x, config["y_nh3"])
                )
            else:
                # 当y_nh3未提供时，根据mix_ratio或假设计算
                if config["mix_ratio"] is not None:
                    # 已知混合比，计算Y_NH3 = Y_H2O / mix_ratio
                    with np.errstate(divide='ignore', invalid='ignore'):
                        y_n = np.divide(y_h, config["mix_ratio"])
                        y_n = np.nan_to_num(y_n, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    # 假设总浓度为1，Y_NH3 = 1 - Y_H2O
                    y_n = 1.0 - y_h
            
            # 计算混合比
            mr = (
                np.full_like(x, config["mix_ratio"])
                if config["mix_ratio"] is not None
                else np.divide(y_h, y_n, out=np.zeros_like(y_h), where=y_n != 0)
            )
            
            # 保存文件
            save_profile(
                x=x, y_h=y_h, y_n=y_n, mr=mr, T=T,
                case_id=config["case_id"], L=L, output_dir=output_dir
            )


def save_profile(
    x: np.ndarray,
    y_h: np.ndarray,
    y_n: np.ndarray,
    mr: np.ndarray,
    T: np.ndarray,
    case_id: str,
    L: float,
    output_dir: str
) -> None:
    """保存工况数据为.npy二进制文件
    
    Args:
        x: 空间坐标 [m], shape=(N,)
        y_h: H2O摩尔分数, shape=(N,)
        y_n: NH3摩尔分数, shape=(N,)
        mr: 混合比 (Y_H/Y_N), shape=(N,)
        T: 温度 [K], shape=(N,)
        case_id: 工况编号 (如 "1.1")
        L: 空间长度 [m]
        output_dir: 输出目录
    """
    # 合并为结构化数组 (便于后续按字段访问)
    data = np.rec.fromarrays(
        [x, y_h, y_n, mr, T],
        names="x, Y_H2O, Y_NH3, M_ratio, T",
        formats="float64, float64, float64, float64, float64"
    )
    
    # 保存为.npy文件
    filename = f"Case{case_id}_L{L:.1f}m.npy"
    np.save(os.path.join(output_dir, filename), data)


# ------------------------------
# 主程序入口
# ------------------------------
if __name__ == "__main__":
    generate_profiles(
        case_configs=CASE_CONFIGS,
        l_values=L_VALUES,
        n_points=N_POINTS,
        output_dir=OUTPUT_DIR,
        selected_cases=SELECTED_CASES  # 传入选择的案例ID列表
    )
    print(f"生成完成! 文件保存在: {os.path.abspath(OUTPUT_DIR)}")