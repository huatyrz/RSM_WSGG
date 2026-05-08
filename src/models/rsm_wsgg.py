# 此程序用于 WSGG 模型系数拟合，包含一种透明气体和四种灰气体
# From paper by M.H. Bordbar, G. Wecel, T. Hyppanen, Combustion and Flame 161 (2014) 2435-2445.
import numpy as np
import statsmodels.api as sm
import pickle
from scipy.optimize import minimize
from joblib import Parallel, delayed
from scipy.interpolate import griddata
from pyDOE3 import lhs
import os
import pandas as pd

# ------------------------------------------------------------------------函数分割线----------------------------------------------------------------------
# 此函数使用非线性最小二乘法拟合并返回 WSGG 模型系数 c和 d
def fit_wsgg_model_parallel(mole_fraction_ratios, temperatures, path_lengths, emissivity_data, Ng, Pt, X, T_ref, Exp, initial_guess=None):
    # 计算归一化温度 Tr
    Tr = temperatures / T_ref

    # 定义目标函数（残差的平方和）
    def objective(params, L, epsilon_data):
        a = params[:Ng]  # 灰气体权重因子
        K = params[Ng:]  # 吸收系数
        a0 = 1 - np.sum(a)  # 透明气体权重因子
        epsilon = np.zeros_like(L)
        for i in range(Ng):
            epsilon += a[i] * (1 - np.exp(-np.exp(K[i]) * Pt * X * L))
        # epsilon += a0 * (1 - np.exp(0 * Pt * L))  # 透明气体的贡献，K0=0
        return np.sum((epsilon - epsilon_data)**2)
        # relative_error = (epsilon - epsilon_data) / (epsilon_data + 1e-8)   # 防止除以零
        # return np.sum(relative_error**2)
    
    # 初始猜测值
    # 如果 initial_guess_K 为 None，则使用函数内部定义的默认值
    if initial_guess is None:
        # 这里定义默认的初始猜测值
        initial_guess_K = np.array([-1.0, 2.0, -3.0, 4.0])
        initial_guess_a = np.array([0.2, 0.2, 0.2, 0.2])
        # 将它们合并为一个初始猜测值数组
        initial_guess = np.concatenate((initial_guess_a, initial_guess_K))


    # 定义约束函数
    # def constraint(params, Ng):
    #     K = params[Ng:]
    #     return np.array([K[1] - K[0], K[2] - K[1], K[3] - K[2]])

    # 定义约束字典列表
    cons = [
        {'type': 'ineq', 'fun': lambda params: params[Ng+1] - params[Ng]},
        {'type': 'ineq', 'fun': lambda params: params[Ng+2] - params[Ng+1]},
        {'type': 'ineq', 'fun': lambda params: params[Ng+3] - params[Ng+2]},
        # {'type': 'ineq', 'fun': lambda params: params[Ng+4] - params[Ng+3]},
        # {'type': 'ineq', 'fun': lambda params: params[Ng+5] - params[Ng+4]},
        {'type': 'ineq', 'fun': lambda params: 1 - np.sum(params[:Ng])}
    ]

    # 参数边界
    # 单独指定每个 a 参数的边界
    bounds_a = [(0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0)]
    # 单独指定每个 K 参数的边界
    # bounds_K = [(0, 0.1), (0, 1.0), (0, 2.0), (0, 50.0)]
    bounds_K = [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)]

    # 将两个边界列表合并，形成完整的边界列表
    bounds = bounds_a + bounds_K

    # 拟合a, K
    def fit_params(params, L, epsilon_data):
        result = minimize(objective, params, args=(L, epsilon_data), method='SLSQP', bounds=bounds, constraints=cons)
        return result.x
    
    # 使用Parallel和delayed并行化拟合所有摩尔分数和温度下的参数
    results = Parallel(n_jobs=-1)(
        delayed(fit_params)(initial_guess, path_lengths, emissivity_data[m_index, t_index, :]) 
        for m_index, m in enumerate(mole_fraction_ratios) for t_index, t in enumerate(temperatures))
    
    # 将结果填充到三维数组中
    a = np.empty((Ng, len(temperatures), len(mole_fraction_ratios)))
    K = np.empty((Ng, len(temperatures), len(mole_fraction_ratios)))
    for result_index, result in enumerate(results):
        m_index = result_index // len(temperatures)
        t_index = result_index % len(temperatures)
        a[:, t_index, m_index] = result[:Ng]
        K[:, t_index, m_index] = result[Ng:]
    
    # 拟合beta_jk
    # 初始化系数矩阵beta
    beta = np.zeros((Ng, 15))  # 包括截距项，所以是15项
    # beta = np.zeros((Ng, 10))  # 包括截距项，所以是10项
 
    # 对每个灰气体进行回归分析
    for i in range(Ng):
        # 重塑数据为二维形式
        Tr_reshaped = Tr.repeat(len(mole_fraction_ratios))
        Mr_reshaped = np.tile(mole_fraction_ratios, len(Tr))
        a_reshaped = a[i].flatten()
        
        # 创建设计矩阵，包括一次项、二次项、三次项和四次项以及所有交叉项
        X = np.column_stack((
            Tr_reshaped, Mr_reshaped,
            Tr_reshaped**2, Mr_reshaped**2, Tr_reshaped * Mr_reshaped,
            Tr_reshaped**3, Mr_reshaped**3, Tr_reshaped**2 * Mr_reshaped, Tr_reshaped * Mr_reshaped**2,
            Tr_reshaped**4, Mr_reshaped**4, Tr_reshaped**3 * Mr_reshaped, Tr_reshaped * Mr_reshaped**3, Tr_reshaped**2 * Mr_reshaped**2
        ))
        
        # 添加截距项
        X = sm.add_constant(X)
        
        # 创建模型
        model = sm.OLS(a_reshaped, X)
        
        # 拟合模型
        results = model.fit()
        # result = model.fit_regularized(alpha=0.1, L1_wt=0.3)
        
        # 存储系数到beta矩阵
        beta[i, 0] = results.params[0]  # 截距项
        beta[i, 1] = results.params[1]  # Tr_reshaped (一次项)
        beta[i, 2] = results.params[2]  # Mr_reshaped (一次项)
        beta[i, 3] = results.params[3]  # Tr_reshaped**2 (二次项)
        beta[i, 4] = results.params[4]  # Mr_reshaped**2 (二次项)
        beta[i, 5] = results.params[5]  # Tr_reshaped * Mr_reshaped (交叉项)
        beta[i, 6] = results.params[6]  # Tr_reshaped**3 (三次项)
        beta[i, 7] = results.params[7]  # Mr_reshaped**3 (三次项)
        beta[i, 8] = results.params[8]  # Tr_reshaped**2 * Mr_reshaped (三次交叉项)
        beta[i, 9] = results.params[9]  # Tr_reshaped * Mr_reshaped**2 (三次交叉项)
        beta[i, 10] = results.params[10]  # Tr_reshaped**4 (四次项)
        beta[i, 11] = results.params[11]  # Mr_reshaped**4 (四次项)
        beta[i, 12] = results.params[12]  # Tr_reshaped**3 * Mr_reshaped (四次交叉项)
        beta[i, 13] = results.params[13]  # Tr_reshaped * Mr_reshaped**3 (四次交叉项)
        beta[i, 14] = results.params[14]  # Tr_reshaped**2 * Mr_reshaped**2 (四次交叉项)

    # 拟合d{i,k}
    # 特定温度
    specific_temperature = T_ref  # 我们想要查找的特定温度
    
    # 找到特定温度对应的索引 j_value
    matching_indices = np.where(temperatures == specific_temperature)[0]
    
    if len(matching_indices) == 0:
        # 如果没有找到匹配的温度，选择最接近的温度
        closest_index = np.argmin(np.abs(temperatures - specific_temperature))
        j_value = closest_index
    #   print(f"特定温度 {specific_temperature} 未在温度列表中找到，使用最接近的温度 {temperatures[j_value]}。")
    else:
        j_value = matching_indices[0]  # 假设只有一个匹配的温度
    
    # 继续后续计算
    gamma = np.empty((Ng, Exp + 1))
    
    for i in range(Ng):
        K_values = K[i, j_value, :]  # 只取特定温度下的 K 值
        gamma_coeffs = np.polyfit(mole_fraction_ratios, K_values, Exp)
        gamma[i, :] = np.flip(gamma_coeffs)  # 将拟合系数赋值给 gamma 的第 i 行    
    
    return beta, gamma


# ------------------------------------------------------------------------函数分割线----------------------------------------------------------------------
def fit_wsgg_model_parallel_optimized(mole_fraction_ratios, temperatures, path_lengths, emissivity_data, Ng, Pt, X, T_ref, Exp, initial_guess=None, n_jobs=-1):
    """
    优化后的WSGG模型参数拟合函数，支持多层次并行计算
    
    参数：
        mole_fraction_ratios : 摩尔分数比数组 (M,)
        temperatures : 温度数组 (T,)
        path_lengths : 路径长度 (L,)
        emissivity_data : 发射率数据立方体 (M, T, L)
        Ng : 灰气体数量
        Pt : 总压强
        X : 参与性气体浓度和
        T_ref : 参考温度
        Exp : 多项式阶数
        initial_guess : 初始猜测值 (可选)
        n_jobs : 并行作业数
    返回：
        beta : 灰气体权重系数矩阵 (Ng, 15)
        gamma : 吸收系数多项式系数矩阵 (Ng, Exp+1)
    """
    # ==================== 初始化部分 ====================
    Tr = temperatures / T_ref  # 归一化温度
    M = len(mole_fraction_ratios)
    T = len(temperatures)
    L = len(path_lengths)
    
    # 参数边界设置
    bounds_a = [(0, 1)] * Ng
    bounds_K = [(-10, 10)] * Ng
    bounds = bounds_a + bounds_K
    
    # 约束条件：K_i < K_{i+1} 和 sum(a) <= 1
    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[Ng+i+1] - x[Ng+i]} for i in range(Ng-1)
    ] + [
        {'type': 'ineq', 'fun': lambda x: 1 - np.sum(x[:Ng])}
    ]
    

    # 初始猜测值
    # 如果 initial_guess_K 为 None，则使用函数内部定义的默认值
    if initial_guess is None:
        # 这里定义默认的初始猜测值
        initial_guess_K = np.array([-1.0, 2.0, -3.0, 4.0])
        initial_guess_a = np.array([0.2, 0.2, 0.2, 0.2])
        # 将它们合并为一个初始猜测值数组
        initial_guess = np.concatenate((initial_guess_a, initial_guess_K))

    # ==================== 核心优化部分 ====================
    def vectorized_objective(params, L_vec, epsilon_obs):
        """向量化目标函数"""
        a = params[:Ng]
        K = np.exp(params[Ng:])  # 使用指数保证正性
        a0 = 1 - a.sum()
        
        # 向量化计算发射率
        optical_thickness = K[:, None] * Pt * X * L_vec
        epsilon_calc = a @ (1 - np.exp(-optical_thickness)) + a0 * 0  # a0项
        relative_error = (epsilon_calc - epsilon_obs) / (epsilon_obs + 1e-8)   # 防止除以零
        return np.sum(relative_error**2)

    # 并行优化每个温度/摩尔组合
    def fit_single_case(m_idx, t_idx):
        """并行优化单个工况"""
        res = minimize(
            vectorized_objective,
            initial_guess,
            args=(path_lengths, emissivity_data[m_idx, t_idx, :]),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            # options={'maxiter': 200, 'ftol': 1e-6}
        )
        return res.x

    # 第一层并行：摩尔分数维度
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(lambda m: [fit_single_case(m, t) for t in range(T)])(m)
        for m in range(M)
    )

    # ==================== 结果后处理 ====================
    # 重构结果数组
    a_matrix = np.zeros((Ng, T, M))
    K_matrix = np.zeros((Ng, T, M))
    
    for m_idx in range(M):
        for t_idx in range(T):
            params = results[m_idx][t_idx]
            a_matrix[:, t_idx, m_idx] = params[:Ng]
            K_matrix[:, t_idx, m_idx] = params[Ng:]

    # ==================== Beta系数拟合 ====================
    def fit_beta(i):
        """并行拟合单个灰气体的beta系数"""
        # 准备数据
        Tr_flat = np.repeat(Tr, M)
        Mr_flat = np.tile(mole_fraction_ratios, T)
        a_flat = a_matrix[i].T.flatten()
        
        # 构建多项式特征矩阵
        design_matrix = np.column_stack([
            Tr_flat**p * Mr_flat**q 
            for p in range(5) 
            for q in range(5) 
            if 0 <= p + q <= 4
        ])
        # design_matrix = sm.add_constant(design_matrix)
        
        # 带正则化的回归
        model = sm.OLS(a_flat, design_matrix)
        result = model.fit_regularized(alpha=0.1, L1_wt=0.3)
        return result.params

    beta = np.array(Parallel(n_jobs=n_jobs)(delayed(fit_beta)(i) for i in range(Ng)))

    # ==================== Gamma系数拟合 ====================
    def fit_gamma(i):
        """并行拟合单个灰气体的gamma系数"""
        # 找到最接近参考温度的索引
        t_idx = np.argmin(np.abs(temperatures - T_ref))
        return np.polyfit(mole_fraction_ratios, K_matrix[i, t_idx], Exp)

    gamma = np.array(Parallel(n_jobs=n_jobs)(delayed(fit_gamma)(i) for i in range(Ng)))

    return beta, gamma

# ------------------------------------------------------------------------函数分割线----------------------------------------------------------------------
# ------------------------------------------------------------------------
# 修改后的 fit_wsgg_model_parallel 函数（GPU版本）
# ------------------------------------------------------------------------
def fit_wsgg_model_parallel_gpu(mole_fraction_ratios, temperatures, path_lengths, emissivity_data, Ng, Pt, X, T_ref, Exp, initial_guess=None):
    # 将输入数据转换为PyTorch张量并移至GPU
    mole_fraction_ratios = torch.tensor(mole_fraction_ratios, dtype=torch.float32, device=device)
    temperatures = torch.tensor(temperatures, dtype=torch.float32, device=device)
    path_lengths = torch.tensor(path_lengths, dtype=torch.float32, device=device)
    emissivity_data = torch.tensor(emissivity_data, dtype=torch.float32, device=device)
    
    Tr = temperatures / T_ref  # 计算归一化温度

    # 定义GPU优化的目标函数
    def objective_gpu(params, L, epsilon_data):
        a = params[:Ng]  # 灰气体权重因子
        K = params[Ng:]  # 吸收系数
        a0 = 1 - torch.sum(a)  # 透明气体权重因子
        
        # 向量化计算发射率
        exponent = -torch.exp(K.view(-1,1)) * Pt * X * L
        epsilon = torch.sum(a.view(-1,1) * (1 - torch.exp(exponent)), dim=0)
        return torch.sum((epsilon - epsilon_data)**2)

    # 初始猜测值（GPU张量）
    if initial_guess is None:
        initial_guess = torch.cat((
            torch.tensor([0.2]*4, dtype=torch.float32, device=device),
            torch.tensor([-1.0, 2.0, -3.0, 4.0], dtype=torch.float32, device=device)
        ))

    # 约束条件转换为GPU兼容格式
    cons = [
        {'type': 'ineq', 'fun': lambda params: params[Ng+1] - params[Ng]},
        {'type': 'ineq', 'fun': lambda params: params[Ng+2] - params[Ng+1]},
        {'type': 'ineq', 'fun': lambda params: params[Ng+3] - params[Ng+2]},
        {'type': 'ineq', 'fun': lambda params: 1 - torch.sum(params[:Ng])}
    ]

    # 参数边界（GPU兼容）
    bounds = [(0,1)]*4 + [(-10,10)]*4

    # 批量优化函数（使用PyTorch优化器）
    def fit_params_batch(initial_guess, L, epsilon_data):
        params = nn.Parameter(initial_guess.clone().detach().requires_grad_(True))
        optimizer = optim.LBFGS([params], lr=0.01, max_iter=100)
        
        def closure():
            optimizer.zero_grad()
            loss = objective_gpu(params, L, epsilon_data)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        return params.detach()

    # 批量处理数据
    L_grid, epsilon_grid = torch.meshgrid(path_lengths, emissivity_data.view(-1))
    
    # 使用DataLoader加速数据加载
    dataset = torch.utils.data.TensorDataset(L_grid, epsilon_grid)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False)

    # 并行优化（GPU加速）
    results = []
    with torch.no_grad():
        for L_batch, epsilon_batch in dataloader:
            batch_results = [fit_params_batch(initial_guess, L, eps) 
                            for L, eps in zip(L_batch, epsilon_batch)]
            results.extend(batch_results)

    # 重组结果到三维张量
    a = torch.zeros((Ng, len(temperatures), len(mole_fraction_ratios)), device=device)
    K = torch.zeros_like(a)
    
    for idx, res in enumerate(results):
        m_idx = idx // len(temperatures)
        t_idx = idx % len(temperatures)
        a[:, t_idx, m_idx] = res[:Ng]
        K[:, t_idx, m_idx] = res[Ng:]

    # 多项式拟合（GPU加速）
    def poly_fit_gpu(x, y, degree):
        X = torch.vander(x, degree+1)
        coeffs = torch.linalg.lstsq(X, y).solution
        return coeffs.flip(0)  # 调整系数顺序

    # 拟合beta矩阵（向量化计算）
    Tr_ext = Tr.repeat(len(mole_fraction_ratios))
    Mr_ext = mole_fraction_ratios.repeat(len(temperatures))
    
    beta = torch.zeros((Ng, 15), device=device)
    for i in range(Ng):
        X = torch.stack([
            Tr_ext, Mr_ext,
            Tr_ext**2, Mr_ext**2, Tr_ext*Mr_ext,
            Tr_ext**3, Mr_ext**3, Tr_ext**2*Mr_ext, Tr_ext*Mr_ext**2,
            Tr_ext**4, Mr_ext**4, Tr_ext**3*Mr_ext, Tr_ext*Mr_ext**3, Tr_ext**2*Mr_ext**2
        ], dim=1)
        
        X = torch.cat([torch.ones(X.shape[0],1,device=device), X], dim=1)
        beta[i] = torch.linalg.lstsq(X, a[i].flatten()).solution

    # 拟合gamma矩阵
    j_value = torch.argmin(torch.abs(temperatures - T_ref))
    gamma = torch.stack([poly_fit_gpu(mole_fraction_ratios, K[i,j_value], Exp) 
                        for i in range(Ng)])

    return beta.cpu().numpy(), gamma.cpu().numpy()


# ------------------------------------------------------------------------函数分割线----------------------------------------------------------------------
def load_and_process_emissivity_data(filename):
    """
    从文件中读取发射率数据库，并装配出mole_fraction_ratios、temperatures、path_lengths三个一维数组
    和一个三维的epsilon_3d数组。

    参数:
    filename (str): 发射率数据库文件的名称。

    返回:
    mole_fraction_ratios (np.array): 摩尔分数比数组。
    temperatures (np.array): 温度数组。
    path_lengths (np.array): 路径长度数组。
    epsilon_3d (np.array): 三维的发射率epsilon数组。
    """
    with open(filename, 'rb') as f:
        all_data = pickle.load(f)

    # 提取唯一的 mole_fraction_ratio、Tgas 和 path_length_value
    unique_mole_fraction_ratios = list(all_data.keys())
    unique_temperatures = list(next(iter(all_data.values())).keys())
    unique_path_lengths = sorted(set([item[0] for item in next(iter(next(iter(all_data.values())).values()))]))
    
    # 保存一维数组
    mole_fraction_ratios = np.array(unique_mole_fraction_ratios)
    temperatures = np.array(unique_temperatures)
    path_lengths = np.array(unique_path_lengths)

    # 初始化 epsilon 的三维数组
    epsilon_3d = np.zeros((len(unique_mole_fraction_ratios), len(unique_temperatures), len(unique_path_lengths)))

    # 填充 epsilon_3d 数组
    for idx_mfr, mole_fraction_ratio in enumerate(unique_mole_fraction_ratios):
        for idx_T, Tgas in enumerate(unique_temperatures):
            for idx_pl, path_length_value in enumerate(unique_path_lengths):
                # 查找对应的 epsilon 值
                epsilon_values = all_data[mole_fraction_ratio][Tgas]
                for pl_value, epsilon in epsilon_values:
                    if pl_value == path_length_value:
                        epsilon_3d[idx_mfr, idx_T, idx_pl] = epsilon
                        break
    # 打印结果以验证
    # print("摩尔分数比数组:", mole_fraction_ratios_array)
    # print("温度数组:", temperatures_array)
    # print("路径长度数组:", path_lengths_array)
    # print("发射率数组:", epsilon_3d)
    return mole_fraction_ratios, temperatures, path_lengths, epsilon_3d


# ------------------------------------------------------------------------函数分割线----------------------------------------------------------------------
# 此函数用以按照格式打印出系数矩阵beta和gamma
# 定义格式化打印函数
def print_formatted_arrays(beta, gamma):
    # 打印Beta表格
    print("\nBeta Coefficients:")
    print(f"{'Coef':<8} | {'i=1':<8} | {'i=2':<8} | {'i=3':<8} | {'i=4':<8}")
    print("-" * 50)
    for i in range(15):
        name = f"β_i{i}"  # 自动生成系数名称
        print(f"{name:<8} | {beta[0, i]:.7f} | {beta[1, i]:.7f} | {beta[2, i]:.7f} | {beta[3, i]:.7f}")
    
    # 打印Gamma表格
    print("\nGamma Coefficients:")
    print(f"{'Coef':<8} | {'i=1':<8} | {'i=2':<8} | {'i=3':<8} | {'i=4':<8}")
    print("-" * 50)
    for i in range(5):
        name = f"γ_i{i}"  # 自动生成系数名称
        print(f"{name:<8} | {gamma[0, i]:.7f} | {gamma[1, i]:.7f} | {gamma[2, i]:.7f} | {gamma[3, i]:.7f}")


# ------------------------------------------------------------------------函数分割线----------------------------------------------------------------------
def save_to_excel(beta, gamma, Ng, Exp, filename):
    """使用pandas库导出到Excel，支持可变灰气体数量Ng"""
    
    # 生成动态列名，如i=1到i=Ng
    columns = [f'i={i+1}' for i in range(Ng)]
    
    # 转换Beta数组为DataFrame（假设beta形状为(15, Ng)）
    beta_df = pd.DataFrame(
        data=beta.T,   # 转置保证维度对应
        index=[f'β_i{i}' for i in range(15)],
        # index=[f'β_i{i}' for i in range(10)],
        columns=columns
    ).apply(lambda x: x.map(lambda y: f"{y:.7f}" if abs(y) > 1e-7 else "0.0000000"))
    
    # 转换Gamma数组为DataFrame（假设gamma形状为(5, Ng)）
    gamma_df = pd.DataFrame(
        data=gamma.T,
        index=[f'γ_i{i}' for i in range(Exp+1)],
        # index=[f'γ_i{i}' for i in range(4)],
        columns=columns
    ).apply(lambda x: x.map(lambda y: f"{y:.7f}" if abs(y) > 1e-7 else "0.0000000"))
    
    # 创建Excel写入对象
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        beta_df.to_excel(writer, sheet_name='Beta')
        gamma_df.to_excel(writer, sheet_name='Gamma')
        
        # 获取工作簿对象并设置格式
        workbook = writer.book
        _format_sheet(workbook, 'Beta', Ng)
        _format_sheet(workbook, 'Gamma', Ng)
    
    print(f"文件已保存至：{filename}")

def _format_sheet(workbook, sheet_name, Ng):
    """设置工作表格式（列宽+居中），支持动态列数"""
    worksheet = workbook.get_worksheet_by_name(sheet_name)
    
    # 设置列宽：A列固定12，数值列B开始，宽度15
    worksheet.set_column('A:A', 15)  # 系数名称列
    if Ng >= 1:
        # 数值列范围：B列到B+Ng-1列（列索引1到1+Ng-1）
        worksheet.set_column(1, 1 + Ng - 1, 20)
    
    # 设置所有列居中：从A列到数值列最后一列
    cell_format = workbook.add_format({'align': 'center'})
    worksheet.set_column(0, 1 + Ng - 1, None, cell_format)


# ------------------------------------------------------------------------函数分割线----------------------------------------------------------------------
def read_from_excel(filename):
    """从Excel文件读取Beta和Gamma系数"""
    # 读取Beta系数表
    beta_df = pd.read_excel(filename, sheet_name='Beta', index_col=0)
    beta = beta_df.astype(float).values.T  # 转置恢复原始维度
    
    # 读取Gamma系数表
    gamma_df = pd.read_excel(filename, sheet_name='Gamma', index_col=0)
    gamma = gamma_df.astype(float).values.T  # 转置恢复原始维度
    
    return beta, gamma


# ------------------------------------------------------------------------函数分割线----------------------------------------------------------------------
# rsm_wsgg根据拟合的beta和gamma计算权重系数 a和吸收系数 k
def get_k_a(Tr, Mr, Ng, Exp, beta, gamma):
    a = np.zeros(Ng)
    K = np.zeros(Ng)
    for i in range(Ng):
        a[i] = (
                beta[i, 0] +  # 截距项
                beta[i, 1] * Tr +  # Tr (一次项)
                beta[i, 2] * Mr +  # Mr (一次项)
                beta[i, 3] * Tr**2 +  # Tr^2 (二次项)
                beta[i, 4] * Mr**2 +  # Mr^2 (二次项)
                beta[i, 5] * Tr * Mr +  # Tr * Mr (交叉项)
                beta[i, 6] * Tr**3 +  # Tr^3 (三次项)
                beta[i, 7] * Mr**3 +  # Mr^3 (三次项)
                beta[i, 8] * Tr**2 * Mr +  # Tr^2 * Mr (三次交叉项)
                beta[i, 9] * Tr * Mr**2 + # Tr * Mr^2 (三次交叉项)
                beta[i, 10] * Tr**4 +  # Tr^4 (四次项)
                beta[i, 11] * Mr**4 +  # Mr^4 (四次项)
                beta[i, 12] * Tr**3 * Mr +  # Tr^3 * Mr (四次交叉项)
                beta[i, 13] * Tr * Mr**3 +  # Tr * Mr^3 (四次交叉项)
                beta[i, 14] * Tr**2 * Mr**2  # Tr^2 * Mr^2 (四次交叉项)
        )
        K[i] = np.exp(np.sum(gamma[i,:]*Mr**np.arange(Exp+1)))
    
    ans_a = np.zeros(Ng+1)
    ans_a[0] = 1.0 - np.sum(a)
    ans_a[1:] = a

    # Calculation of "kk" coefficients
    ans_k = np.zeros(Ng+1)
    ans_k[1:] = K

    return ans_a, ans_k


# ------------------------------------------------------------------------函数分割线----------------------------------------------------------------------
# rsm_wsgg根据拟合的beta和gamma计算权重系数 a和吸收系数 k
def get_k_a_ML(Tr, Mr, Ng, Exp, beta, gamma):
    # a = np.zeros(Ng)
    # K = np.zeros(Ng)

    # features = []
    # for p in range(5):
    #     for q in range(5):
    #         if p + q <= 4:
    #             features.append((Tr ** p) * (Mr ** q))
    # features = np.stack(features, axis=-1)  # (..., 15)
    # # 计算权重因子 a: 爱因斯坦求和 (Ng, ...)
    # a = np.einsum('nf,...f->n...', beta, features)  # (Ng, ...)
    # # 使用sigmoid约束每个灰气体的权重在(0,1)之间
    # clamped_a = 1 / (1 + np.exp(-a))  # Sigmoid激活
    # # 添加透明气体权重约束
    # sum_a = np.sum(clamped_a, axis=0, keepdims=True)
    # denominator = np.maximum(sum_a, 1.0)
    # clamped_a = clamped_a / denominator  # 归一化总权重不超过1

    # # 计算吸收系数k
    # powers = np.arange(Exp, -1, -1, dtype=np.float32)  # 创建幂次项
    # mr_powers = Mr[..., np.newaxis] ** powers          # (..., Exp+1)
    # K = np.einsum('nf,...f->n...', gamma, mr_powers)   # (Ng, ...)
    # K_clamped = np.maximum(K, 0)
    # K_sum = np.cumsum(K_clamped, axis=0)               # 累积求和

    # # 组装最终结果
    # ans_a = np.zeros(Ng + 1)
    # ans_a[0] = 1.0 - np.sum(clamped_a)
    # ans_a[1:] = clamped_a  # 修正权重赋值

    # ans_k = np.zeros(Ng + 1)
    # ans_k[1:] = K_sum      # 吸收系数累积和

    # return ans_a, ans_k
    #############################################################################################################
    #############################################################################################################

    # 计算特征矩阵 (..., 15)
    features = np.stack([
        (Tr ** p) * (Mr ** q)
        for p in range(5)
        for q in range(5) 
        if p + q <= 4
    ], axis=-1)
    
    # 计算权重系数a [Ng, ...] 
    logits = np.einsum('nf,...f->n...', beta, features)
    a = np.exp(logits) / np.sum(np.exp(logits), axis=0)  # softmax归一化
    
    # 计算吸收系数K [Ng, ...]
    powers = np.arange(Exp, -1, -1, dtype=np.float32)  # 降序幂次[Exp,..,0]
    mr_powers = Mr[..., np.newaxis] ** powers         # (..., Exp+1)
    K = np.einsum('nf,...f->n...', gamma, mr_powers)  # 原始吸收系数
    
    # 应用softplus确保非负 (与PyTorch一致)
    K = np.log(1 + np.exp(K))  # softplus(K)
    
    # 强制第一个气体(透明气体)吸收系数=0
    K[0] = 0.0  # 透明气体无吸收

    return a, K  # 形状均为(Ng, ...)

# ------------------------------------------------------------------------函数分割线----------------------------------------------------------------------

if __name__=="__main__":
    
    # 示例数据，用于本模块测试
    # mole_fraction_ratios = np.array([0.01, 0.125])  # 摩尔分数
    # temperatures = np.array([300, 400, 500])  # 温度
    # path_lengths = np.array([0.01, 0.05, 0.1, 0.2])  # 路径长度
    # emissivity_data = np.random.rand(2, 3, 4)  # 随机生成的发射率数据
    
    # 存储摩尔分数、温度、路径长度及对应发射率的数据库文件
    filename = 'absorption_mix_p1.013_mf0.01_T3.npy'
    mole_fraction_ratios, temperatures, path_lengths, epsilon_3d = load_and_process_emissivity_data(filename)
    # 调用 fit_WSGG_model 拟合 WSGG 模型系数
    c, d = fit_wsgg_model(mole_fraction_ratios, temperatures, path_lengths, epsilon_3d)
    # for k in range(Ng+1):
    #     print(c[:, :, k])