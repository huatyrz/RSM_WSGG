# RSM-WSGG 模型与辐射传热求解项目

> English version: [`README_en.md`](./README_en.md)

## 项目概述

本项目基于响应面方法（RSM）与加权和灰气体（WSGG）模型，开发了一套用于 H₂O/NH₃ 混合气体
辐射特性建模与辐射传热方程（RTE）求解的计算工具，并包含论文中全部图表与算例的复现数据与脚本。

本仓库公开提供：

1. **逐线法（LBL）参考计算** —— 用于训练与验证的发射率数据库，以及全部算例的一维 RTE
   LBL 参考解；
2. **Case 1.1–3.2 的验证数据** —— 给定的温度与组分剖面及其 LBL 参考解，另含两个真实火焰
   算例（Case 4.1、4.2）；
3. **图表复现脚本** —— 可重新生成论文中每一幅图与每一张表；
4. **英文说明文档** [`README_en.md`](./README_en.md) 与本文（中文）。

---

## 项目结构

```
RSM_WSGG/
├── README.md                        # 中文说明（本文件）
├── README_en.md                     # 英文说明
│
├── config/
│   ├── path.py                      # 路径配置（首次使用请按本地环境修改）
│   └── coefficients.json            # 默认系数集选择
│
├── notebooks/                       # Jupyter Notebook
│   ├── gasmix_emiss_calc.ipynb      # LBL 发射率数据库生成（需 AbscDB）
│   ├── model_training.ipynb         # RSM-WSGG 训练与 WSGG 系数生成
│   ├── validation.ipynb             # 发射率与一维 RTE 验证
│   ├── validation_publication_plots.ipynb  # 论文风格图表复现
│   ├── ng_benchmark.ipynb           # N_g = 4–7 精度与 RTE 耗时基准
│   ├── spectral_convergence.ipynb   # 光谱分辨率收敛性研究
│   └── separator.ipynb              # 辅助工具
│
├── src/                             # 核心源代码
│   ├── gas_rte_solver.py            # RTE 求解器（solve_rte、solve_wsgg_dom、
│   │                                #             getLBLsolution、get_kappas）
│   ├── get_emis_abso.py             # LBL 吸收系数 / 发射率接口
│   ├── data_preparer.py             # 训练与验证数据集准备
│   ├── profile_generator.py         # Case 1.1–3.2 的合成 T / Y_H2O / Y_NH3 剖面
│   ├── generate_validation_plots.py # 论文全部图表生成
│   ├── ExtractFortranDB.py          # 光谱数据库提取
│   ├── utils.py                     # I/O 辅助函数
│   └── models/
│       ├── wsgg_model.py            # WSGGModel（beta、gamma）容器与前向计算
│       ├── rsm_wsgg.py              # RSM-WSGG 拟合
│       ├── wang_wsgg.py             # Wang 等 WSGG 模型（对比模型）
│       └── wsgg.py                  # 通用 WSGG 辅助函数
│
├── input/
│   ├── MixEmiss_*.npy               # LBL 发射率数据库（主数据集与独立验证集）
│   ├── coefficients_from_wang.xlsx  # Wang 等模型系数
│   └── profiles/                    # 各算例的给定剖面
│       ├── Case1.1 … Case3.2_L1.0m.npy
│       ├── Case4.1_L1.0m.npy        # 真实火焰，L = 1.0 m
│       └── Case4.2_L5.0m.npy        # 同一火焰，L = 5.0 m
│
└── results/
    ├── coefficients_<时间戳>_Ng<N>_...xlsx   # 拟合得到的 RSM-WSGG 系数
    ├── coefficients_<…>.log                  # 训练日志
    ├── lbl/                         # 全部 9 个算例的 LBL 参考解
    ├── publication_notebook/        # 论文图表（PNG / PDF / TeX）
    ├── ng_benchmark/                # N_g 基准测试输出
    ├── spectral_convergence/        # 收敛性研究输出
    └── img/                         # 探索性图片
```

---

## 环境配置

仓库中**未提供一键配置脚本**，请直接安装依赖：

```bash
pip install numpy scipy pandas matplotlib openpyxl numba tqdm
# 仅训练模型时需要：torch
```

首次运行前请修改 `config/path.py`，将其中的绝对路径指向本地的 `AbscDB` 光谱数据库与工作目录。
`input/` 中的 LBL **发射率数据库**是自包含的，可直接用于训练与验证；但从零重新生成该数据库
（`gasmix_emiss_calc.ipynb`）或重算 LBL 参考解，则需要 `AbscDB`。

---

## 使用流程

1. **环境准备**：安装上述依赖，并按本地环境修改 `config/path.py`。
2. **模型训练**：运行 `notebooks/model_training.ipynb`，生成
   `results/coefficients_<时间戳>_Ng<N>_Exp4_Tref1000_bs<B>.xlsx`。
3. **图表复现**：运行 `src/generate_validation_plots.py` 或
   `notebooks/validation_publication_plots.ipynb`，重新生成论文全部图表。
4. **结果分析**：输出保存在 `results/publication_notebook/`。

```bash
cd src
python generate_validation_plots.py --groups all \
       --coefficients ../results/coefficients_20260326_122927_Ng6_Exp4_Tref1000_bs16384.xlsx
```

`--groups` 可选 `parity`、`emissivity`、`cases` 或 `all`。

---

## 算例数据说明

`input/profiles/CaseX.Y_LZ.Zm.npy` 为结构化数组，字段为
`x`（m）、`Y_H2O`（–）、`Y_NH3`（–）、`M_ratio`（–）、`T`（K）；
对应的 LBL 参考解为 `results/lbl/CaseX.Y_LZ.Zm.npy`，形状 `(N, 3)`，列为
`[x, q, -dq]`，其中 `q` 为辐射热流，`-dq` 为辐射源项。

| 算例 | 域长 $L$ | 网格点数 | 说明 |
|---|---|---|---|
| 1.1–1.3 | 1.0 m | 100 | 等温均匀介质 |
| 2.1–2.2 | 1.0 m | 100 | 非等温均匀介质 |
| 3.1–3.2 | 1.0 m | 100 | 非等温非均匀介质 |
| 4.1 | 1.0 m | 81 | 真实火焰剖面 |
| 4.2 | 5.0 m | 81 | 同一火焰，较长光程 |

Case 4.1 / 4.2 的温度与组分场直接取自已发表的**层流预混 NH₃/H₂/N₂–空气对冲火焰**
一维解（Richter, Barlow, Dawson, Dreizler, Geyer, *Proc. Combust. Inst.* **42** (2026)，
TU-Datalib，DOI [10.48328/tudatalib-2271](https://doi.org/10.48328/tudatalib-2271)，
CC BY 4.0），不求解化学反应。所取区间为 $M_r$ 保持在训练包络 $0.1 \le M_r \le 4.0$ 内的
连续火焰前锋区域（$T$ 从 423 到 1471 K）；已燃尽区因 NH₃ 被消耗、$M_r \to \infty$ 而剔除。

---

## 论文图表与输出文件对照

| 论文条目 | 输出文件 |
|---|---|
| 图 2（发射率 parity 图） | `parity_models_vs_lbl.png` |
| 图 3（发射率–温度） | `emissivity_vs_temperature_mr_0p500.png`、`…_mr_3p500.png` |
| 图 4（发射率–光程，变 $M_r$） | `emissivity_vs_path_length_mr_0p500.png`、`…_mr_3p500.png` |
| 图 5（发射率–光程，变 $T$） | `emissivity_vs_path_length_t_1100K.png`、`…_t_1700K.png` |
| 图 6（等温均匀算例） | `case_validation_Case1.1/1.2/1.3_L1.0m.png` |
| 图 7（非等温均匀算例） | `case_validation_Case2.1/2.2_L1.0m.png` |
| 图 8（非等温非均匀算例） | `case_validation_Case3.1/3.2_L1.0m.png` |
| **图 9（真实火焰算例，新增）** | `case_validation_Case4.1_L1.0m.png`、`case_validation_Case4.2_L5.0m.png` |
| **表 3（误差统计）** | `case_error_summary.tex` / `.csv` / `.png` |

> **关于图表编号的说明**：在论文前一版本中，误差统计汇总被编为**图 9**，现改为**表 3**；
> 新增的真实火焰图则编号为**图 9**。因此回复信中早期出现的"Figs. 2–9"实际指"全部插图
> 及该汇总表"，现应理解为**"图 2–9 与表 3"**。最终编号请以论文排版稿为准。

---

## 关于对比模型（Wang 等）的说明

Wang 等模型声明的适用区间为 300–2300 K、压力光程 0.1–20 m。图中对超出该区间的部分已明确标注：

* parity 图中，区间外的 Wang 数据点以灰色空心方块绘制，并在独立的
  `Wang (extrapolated)` 统计框中给出指标；
* 发射率–温度图中，$P_tL > 20$ bar·m 的曲线以虚线绘制（标注 "outside Wang range"）；
* 发射率–光程图中，$P_tL > 20$ bar·m 的区域以灰色底纹标出。

严格的定量比较应仅在上述区间内进行。此外，两个模型基于不同的光谱数据库
（本文为 HITEMP2010/HITRAN2020，Wang 等为 HITEMP2016/HITRAN2010），这也进一步限制了
二者的直接可比性。

---

## 模型常数

论文所用系数集为
`results/coefficients_20260326_122927_Ng6_Exp4_Tref1000_bs16384.xlsx`：

| 工作表 | 形状 | 含义 |
|---|---|---|
| `beta` | `(N_g+1, 15)` | 灰气体权重 logits（softmax），首行为透明气体 |
| `gamma` | `(N_g+1, 5)` | 吸收系数多项式（softplus），首行 $\kappa = 0$ |

模型常数：$N_g = 6$ 个吸收灰气体 + 1 个透明气体，多项式阶数 `Exp = 4`，
$T_\text{ref} = 1000$ K，$X = X_{H_2O} + X_{NH_3} = 0.2$（固定），$P_t = 1$ bar。

---

## 引用

若使用本项目的代码或数据，请引用原论文及本仓库：

```
https://github.com/huatyrz/RSM_WSGG
```

Case 4.1 / 4.2 的真实火焰剖面源自 Richter 等的数据集
（TU-Datalib，DOI 10.48328/tudatalib-2271，CC BY 4.0），使用这些算例时请一并引用该数据集。
