# 此程序用于 WSGG 模型系数拟合，包含一种透明气体和四种灰气体
# From paper by M.H. Bordbar, G. Wecel, T. Hyppanen, Combustion and Flame 161 (2014) 2435-2445.
import numpy as np
import pickle
from scipy.optimize import minimize
from joblib import Parallel, delayed

Ng = 4  # WSGG模型中使用一种透明气体和四种灰气体
Pt = 1.0  # 总压为 1.0 bar

# ------------------------------------------------------------------------函数分割线----------------------------------------------------------------------
# 此函数使用非线性最小二乘法拟合并返回 WSGG 模型系数 c和 d
# 你可以使用以下方式调用这个函数：
# c, d = fit_wsgg_model_parallel(mole_fraction_ratios, temperatures, path_lengths, emissivity_data)
def fit_wsgg_model_parallel(mole_fraction_ratios, temperatures, path_lengths, emissivity_data, T_ref, Exp, initial_guess=None):
    
    # 计算归一化温度 Tr
    Tr = temperatures / T_ref

    # 定义目标函数（残差的平方和）
    def objective(params, L, epsilon_data):
        a = params[:Ng]  # 灰气体权重因子
        K = params[Ng:]  # 吸收系数
        a0 = 1 - np.sum(a)  # 透明气体权重因子
        epsilon = np.zeros_like(L)
        for i in range(Ng):
            epsilon += a[i] * (1 - np.exp(-np.exp(K[i]) * Pt * L))
        # epsilon += a0 * (1 - np.exp(0 * Pt * L))  # 透明气体的贡献，K0=0
        return np.sum((epsilon - epsilon_data)**2)
    
    # 初始猜测值
    # 如果 initial_guess_K 为 None，则使用函数内部定义的默认值
    if initial_guess is None:
        # 这里定义默认的初始猜测值
        initial_guess_K = np.array([-1.0, 2.0, -3.0, 4.0])
        initial_guess_a = np.array([0.2, 0.2, 0.2, 0.2])
        # 将它们合并为一个初始猜测值数组
        initial_guess = np.concatenate((initial_guess_a, initial_guess_K))

    # 定义约束字典列表
    cons = [
        {'type': 'ineq', 'fun': lambda params: params[Ng+1] - params[Ng]},
        {'type': 'ineq', 'fun': lambda params: params[Ng+2] - params[Ng+1]},
        {'type': 'ineq', 'fun': lambda params: params[Ng+3] - params[Ng+2]},
        {'type': 'ineq', 'fun': lambda params: 1 - np.sum(params[:Ng])}
    ]

    # 参数边界
    # 单独指定每个 a 参数的边界
    bounds_a = [(0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0)]
    # 单独指定每个 K 参数的边界
    # bounds_K = [(-10.0, 2.5), (-10.0, 2.5), (-10.0, 2.5), (2.5, 5.0)]
    bounds_K = [(-10.0, 5.0), (-10.0, 5.0), (-10.0, 5.0), (-10.0, 5.0)]

    # 将两个边界列表合并，形成完整的边界列表
    bounds = bounds_a + bounds_K
    
    # # 拟合a, K
    # def fit_params(initial_guess, L, epsilon_data_for_mole_fraction):
    #     results = []
    #     for t_index in range(len(temperatures)):
    #         epsilon_data = epsilon_data_for_mole_fraction[t_index, :]
    #         result = minimize(objective, initial_guess, args=(L, epsilon_data), method='SLSQP', bounds=bounds, constraints=cons)
    #         results.append(result.x)
    #     return results

    # # 使用Parallel和delayed并行化拟合每个摩尔分数下的所有温度的参数
    # results = Parallel(n_jobs=-1)(
    #     delayed(fit_params)(initial_guess, path_lengths, emissivity_data[m_index, :, :]) 
    #     for m_index in range(len(mole_fraction_ratios))
    # )
    
    # # 将结果填充到三维数组中
    # a = np.empty((Ng, len(temperatures), len(mole_fraction_ratios)))
    # K = np.empty((Ng, len(temperatures), len(mole_fraction_ratios)))
    # for result_index, result in enumerate(results):
    #     m_index = result_index
    #     for t_index, params in enumerate(result):
    #         a[:, t_index, m_index] = params[:Ng]
    #         K[:, t_index, m_index] = params[Ng:]

    # 拟合a, K
    def fit_params(params, L, epsilon_data):
        result = minimize(objective, initial_guess, args=(L, epsilon_data), method='SLSQP', bounds=bounds, constraints=cons)
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
    
    # 拟合b{i,j}
    b = np.empty((Ng, Exp+1, len(mole_fraction_ratios)))
    for i in range(Ng):
        for j in range(len(mole_fraction_ratios)):
            a_values = a[i, :, j]
            b_coeffs = np.polyfit(Tr, a_values, Exp)
            b[i, :, j] = np.flip(b_coeffs)

    # 拟合 c{i,j,k}
    c = np.empty((Ng, Exp+1, Exp+1))
    for i in range(Ng):
        for j in range(Exp+1):
            b_values = b[i, j, :]
            c_coeffs = np.polyfit(mole_fraction_ratios, b_values, Exp)
            c[i, j, :] = np.flip(c_coeffs)

    # 拟合d{i,k}
    # 特定温度
    specific_temperature = T_ref  # 我们想要查找的特定温度
    # 找到特定温度对应的j值
    j_value = np.where(temperatures == specific_temperature)[0][0]  # 假设只有一个匹配的温度
    d = np.empty((Ng, Exp+1))

    for i in range(Ng):
        K_values = K[i, j_value, :]  # 只取特定温度下的K值
        d_coeffs = np.polyfit(mole_fraction_ratios, K_values, Exp)
        d_coeffs_reversed = np.flip(d_coeffs)
        d[i, :] = d_coeffs_reversed  # 将拟合系数赋值给d的第i行
    
    return c, d

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
# ----------------------------------------------------------

d = np.array([
[-3.1033125, 1.4714387, -2.6395497, 2.2706910, -0.7651195],
[-0.4935761, -0.0976231, 0.9289737, -1.5629763, 0.7507532],
[1.7816689, -0.1538956, 0.7230548, -1.0127203, 0.4489266],
[3.9578413, 0.2881624, -0.9723673, 1.1433244, -0.4514768]
])

c = np.empty((4,5,5))

k = 0
c[:,:,k] = [
[-0.0591312, 0.3482228, 1.1940291, -1.5271528, 0.4536846],
[0.6540789, -3.0437067, 6.9054142, -6.1603529, 1.8212429],
[-0.0196246, 3.1115484, -7.5208097, 6.0887802, -1.6379344],
[0.4684779, -0.6372538, -0.6222242, 1.2377738, -0.4544188]
]

k = 1
c[:,:,k] = [
[-0.2004003, 0.8893549, -1.6479178, 2.9397371, -1.0896192],
[-0.2157803, -0.7875167, 4.4086164, -3.3083838, 0.6776469],
[0.5914104, -3.5949712, 9.5197093, -9.0218685, 2.7452612],
[-0.1256695, 1.8353946, -4.8264901, 4.3648643, -1.2868319]
]

k = 2
c[:,:,k] = [
[-0.3138009, 4.4046502, -11.7529478, 4.7936546, -0.3264374],
[0.5257927, 2.6433532, -14.0801290, 12.5885772, -3.3259342],
[-0.7371477, 4.4600319, -14.0309814, 14.8147649, -4.8135602],
[0.0913505, -4.4848078, 13.4852405, -12.8732426, 3.9058930]
]

k = 3
c[:,:,k] = [
[0.8284367, -8.8100387, 22.5670249, -13.0465287, 2.4498642],
[-0.7353246, -1.4058828, 11.5791086, -10.5502175, 2.7442666],
[0.5657939, -3.4633159, 12.4692028, -13.9834045, 4.6763074],
[0.0124240, 4.2808716, -13.3896791, 12.8817117, -3.9089138]
]

k = 4
c[:,:,k] = [
[-0.4415235, 4.4499267, -11.3619730, 7.2419782, -1.5657980],
[0.3387871, 0.0719901, -3.1316653, 2.7629948, -0.6444527],
[-0.1783211, 1.0537690, -4.2673028, 4.9943824, -1.6980663],
[-0.0282658, -1.4934377, 4.7323391, -4.5326028, 1.3648784]
]

#----------------------------------------------------------

def get_k_a(Tr, Mr, Exp):
    
    b = np.zeros((Ng, Exp+1))
    a = np.zeros(Ng)
    K = np.zeros(Ng)
    
    for i in range(Ng):
        for j in range(Exp+1):
            b[i,j] = np.sum(c[i,j,:]*Mr**np.arange(Exp+1))
   
    for i in range(Ng):
        a[i] = np.sum(b[i,:]*Tr**np.arange(Exp+1))

    for i in range(Ng):
        K[i] = np.exp(np.sum(d[i,:]*Mr**np.arange(Exp+1)))

    ans_a = np.zeros(5)
    ans_a[0] = 1.0 - np.sum(a)
    ans_a[1:] = a

    # Calculation of "kk" coefficients
    ans_k = np.zeros(5)
    ans_k[1:] = K

    return ans_a, ans_k



# ------------------------------------------------------------------------函数分割线----------------------------------------------------------------------
# 此函数用以按照格式打印出系数矩阵c和d
# 定义格式化打印函数
def print_formatted_arrays(c, d):
    def print_slice(slice_array, label):
        formatted_output = '[' + ',\n'.join([
            '[' + ', '.join([f'{x:.7f}' for x in row]) + ']'
            for row in slice_array
        ]) + ']'
        print(f"{label}:")
        print(formatted_output)
        print()  # 在每个切片后面打印一个空行

    # 打印c的每个第三维的切片
    for k in range(Ng+1):
        print_slice(c[:, :, k], f"Array c, k = {k}")

    # 打印d
    print_slice(d, "Array d")



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