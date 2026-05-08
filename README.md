# RSM-WSGG 模型与辐射传热求解项目

## 项目概述
本项目基于响应面方法（RSM）与加权和灰气体（WSGG）模型，开发了一套用于气体辐射特性建模与辐射传热方程（RTE）求解的计算工具。项目结构清晰，涵盖了从环境配置、模型训练、核心求解到结果可视化的完整流程。

---

## 项目结构

- `setup_env.ps1`          # Windows 下环境配置脚本
- `notebooks/`             # Jupyter Notebook 文件
  - `model_training.ipynb` # 模型训练与 WSGG 系数生成
  - `validation.ipynb`     # 模型验证与误差分析
- `src/`                   # 核心源代码
  - `models/`              # RSM-WSGG 数学模型定义
  - `gas_rte_solver.py`    # RTE 求解器实现
  - `utils.py`             # 辅助工具函数
- `results/`               # 输出结果目录
  - `coefficients.xlsx`    # 生成的 WSGG 系数表
  - `img/`                 # 计算结果可视化图表
---

## 目录与文件说明

### 1. 环境配置 (`setup_env.ps1`)
- **作用**：Windows 系统下的一键环境配置脚本（PowerShell）。
- **功能**：自动创建 Python 虚拟环境，安装项目依赖的第三方库（如 `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `torch` 等）。

### 2. 交互式工作区 (`notebooks/`)
此目录用于模型开发、调试和结果演示。
- **`model_training.ipynb`**：
    - 加载训练数据，执行响应面模型（RSM）的训练。
    - 基于训练好的 RSM 模型，生成适用于不同工况的 WSGG 模型系数。
- **`validation.ipynb`**：
    - 使用测试数据验证生成的 WSGG 系数的准确性。
    - 进行误差分析（如计算与基准解或实验数据的偏差），并生成误差分析图表。

### 3. 核心源代码 (`src/`)
项目的主要计算逻辑和模型定义。
- **`models/`**：
    - 包含 RSM 和 WSGG 数学模型的类定义。
    - 实现了气体辐射特性与温度、组分浓度等参数的关系。
- **`gas_rte_solver.py`**：
    - 实现辐射传热方程（RTE）的数值求解器。
    - 利用生成的 WSGG 系数，计算气体介质中的辐射热流、热源项等。
- **`utils.py`**：
    - 存放通用工具函数，如数据读取、预处理、结果保存和绘图函数。

### 4. 输出结果 (`results/`)
存放程序运行后生成的所有文件。
- **`coefficients.xlsx`**：
    - 由 `model_training.ipynb` 生成，以表格形式存储 WSGG 模型的拟合系数（如吸收系数、权重因子）。
- **`img/`**：
    - 存储由验证脚本或求解器生成的所有可视化图片，例如：
        - 温度场、热流密度分布云图。
        - 模型预测值与参考值的对比曲线。
        - 误差分布直方图等。

---

## 使用流程
1.  **环境准备**：首先运行 `.\setup_env.ps1` 配置 Python 环境并安装依赖。
2.  **模型开发**：进入 `notebooks/` 目录，按顺序运行：
    - `model_training.ipynb`：训练模型并生成 `coefficients.xlsx`。
    - `validation.ipynb`：验证系数精度，初步分析结果。
3.  **集成应用**：在您自己的主程序中，引用 `src/` 下的模块（特别是 `gas_rte_solver.py`），并加载 `results/coefficients.xlsx` 中的系数进行辐射传热计算。
4.  **结果分析**：最终的计算结果图表将保存于 `results/img/` 目录中。

---

## 总结
该项目结构体现了从**数据驱动建模**到**物理方程求解**的完整闭环：
- **`notebooks/`** 负责前端的模型训练与验证分析。
- **`src/`** 提供了可集成、可调用的核心计算模块。
- **`results/`** 集中管理所有输出数据与图表，便于结果复现与报告撰写。
