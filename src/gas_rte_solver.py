"""
文件名: gas_radiation_transport_solver.py

该程序用于求解一维气体混合物的辐射传输方程，计算辐射通量(G)、热流(q)及其梯度(dq)。
主要应用于燃烧模拟、大气辐射传输等场景，通过考虑H2O和NH3的吸收系数，计算给定温度和浓度剖面下的辐射特性。

程序流程：
1. 加载温度、浓度剖面数据
2. 计算各气体组分的吸收系数（考虑温度和浓度插值）
3. 计算黑体辐射强度
4. 并行求解辐射传输方程（分波数段计算后积分）
5. 保存计算结果

输入：包含温度、H2O/NH3浓度剖面的.npy文件
输出：包含G、q、dq的.npy结果文件
"""

import numpy as np
from numba import njit
from scipy import constants
from tqdm import trange
from concurrent.futures import ProcessPoolExecutor
from scipy.integrate import solve_ivp
from scipy.constants import sigma
import glob
import os
import math

# 定义黑体辐射计算相关常数
C1 = 2*constants.pi*constants.h*constants.c**2 * 1e8  # 第一辐射常数（转换单位后）
C2 = constants.h*constants.c/constants.k*1e2       # 第二辐射常数（转换单位后）

# 吸收系数数据库路径和参数网格定义
ABSC_DB = "../AbscDB"
X_GRID = np.linspace(0, 1, 5)          # 浓度插值网格 (H2O, NH3的摩尔分数)
T_GRID = np.arange(300.0, 3001.0, 100.0)  # 温度插值网格 (K)

def i_b_eta(temperature:float, eta:float, n:float = 1): # pylint: disable=invalid-name
    """
    return the black body radiation intensity

    Parameters
    ----------
    temperature : float
        In Kelvin
    eta : float
        In cm-1
    n : float, optional

    Returns
    -------
    black body intensity : float
        in terms of wavenumber with dimension cm-1
    """
    e_b_lambda = C1*eta**3/(n**2*(np.exp(C2*eta/(n*temperature))-1))
    return e_b_lambda/np.pi

def to_same_grid(kappas):
    max_n = max([len(kappa) for kappa in kappas])
    for i, kappa in enumerate(kappas):
        n = len(kappa)
        if n < max_n:
            kappas[i] = np.interp(np.linspace(0, 1, max_n), np.linspace(0, 1, n), kappas[i])
    return kappas


def get_specie_kappas(specie: str, x: float, T: float):
    id_x = np.searchsorted(X_GRID, x, side='right') - 1
    id_T = np.searchsorted(T_GRID, T, side='right') - 1
    if x == X_GRID[id_x] and T == T_GRID[id_T]:
        return np.load(f"{ABSC_DB}/01.0/{specie}/{x:.2f}_{int(T):04d}.npy")
    elif x == X_GRID[id_x]:
        Tl = T_GRID[id_T]
        Tr = T_GRID[id_T + 1]
        kl = np.load(f"{ABSC_DB}/01.0/{specie}/{x:.2f}_{int(Tl):04d}.npy")
        kr = np.load(f"{ABSC_DB}/01.0/{specie}/{x:.2f}_{int(Tr):04d}.npy")
        kl, kr = to_same_grid([kl, kr])
        return kl + (kr - kl) * (T - Tl) / (Tr - Tl)
    elif T == T_GRID[id_T]:
        xl = X_GRID[id_x]
        xr = X_GRID[id_x + 1]
        kl = np.load(f"{ABSC_DB}/01.0/{specie}/{xl:.2f}_{int(T):04d}.npy")
        kr = np.load(f"{ABSC_DB}/01.0/{specie}/{xr:.2f}_{int(T):04d}.npy")
        kl, kr = to_same_grid([kl, kr])
        return kl + (kr - kl) * (x - xl) / (xr - xl)
    else:
        xl = X_GRID[id_x]
        xr = X_GRID[id_x + 1]
        Tl = T_GRID[id_T]
        Tr = T_GRID[id_T + 1]
        kll = np.load(f"{ABSC_DB}/01.0/{specie}/{xl:.2f}_{int(Tl):04d}.npy")
        klr = np.load(f"{ABSC_DB}/01.0/{specie}/{xl:.2f}_{int(Tr):04d}.npy")
        krl = np.load(f"{ABSC_DB}/01.0/{specie}/{xr:.2f}_{int(Tl):04d}.npy")
        krr = np.load(f"{ABSC_DB}/01.0/{specie}/{xr:.2f}_{int(Tr):04d}.npy")
        kll, klr, krl, krr = to_same_grid([kll, klr, krl, krr])
        kl = kll + (klr - kll) * (T - Tl) / (Tr - Tl)
        kr = krl + (krr - krl) * (T - Tl) / (Tr - Tl)
        return kl + (kr - kl) * (x - xl) / (xr - xl)



def get_mixture_kappas(xs: np.ndarray, T: float):
    species = ["H2O", "NH3"]
    specie_kappas = [get_specie_kappas(specie, xs[i], T) for i, specie in enumerate(species)]
    max_n = max(len(kappa) for kappa in specie_kappas)
    mixture_kappas = np.zeros(max_n)
    for i, specie_kappa in enumerate(specie_kappas):
        if len(specie_kappa) != max_n:
            specie_kappa = np.interp(np.linspace(0, 1, max_n), np.linspace(0, 1, len(specie_kappa)), specie_kappa)
        mixture_kappas += specie_kappa * xs[i]
    return mixture_kappas


def get_kappas(xss, Ts):
    kappas = [get_mixture_kappas(xs, T) for xs, T in zip(xss, Ts)]
    return np.array(to_same_grid(kappas))


def get_i_b_etas(Ts, etas):
    return np.array([i_b_eta(T, etas) for T in Ts])

@njit
def e1(x):
    if x <= 1.0:
        a0 = -0.57721566
        a1 = 0.99999193
        a2 = -0.24991055
        a3 = 0.05519968
        a4 = -0.00976004
        a5 = 0.00107857
        e1_value = (
            a0
            + x * (a1 + x * (a2 + x * (a3 + x * (a4 + x * a5))))
            - np.log(x + 1e-8)
        )
    else:
        a1 = 8.5733287401
        a2 = 18.0590169730
        a3 = 8.6347608925
        a4 = 0.2677737343
        b1 = 9.5733223454
        b2 = 25.6329561486
        b3 = 21.0996530827
        b4 = 3.9584969228
        e1_value = (
            (x * (a3 + x * (a2 + x * (a1 + x))) + a4)
            / (x * (b3 + x * (b2 + x * (b1 + x))) + b4)
            * np.exp(-x)
            / x
        )
    return e1_value

@njit
def expn_float(n, x):
    """
    Exponential integral of order n, based on the method in the Modest book.
    """
    if n < 1:
        raise ValueError("Order N must be at least 1.")
    
    en_value = e1(x)
    if n > 1:
        ex = np.exp(-x)
        for i in range(2, n + 1):
            en_value = (ex - x * en_value) / (i - 1)
    return en_value

@njit
def expn(n, xs):
    return np.array([expn_float(n, x) for x in xs])


def solve_rte(xs, kappas, I_bs, I_b1, I_b2):
    """求解辐射传输方程
    返回：
    G : 辐射通量 (W/m²)
    q : 热流 (W/m²)
    dq : 热流梯度 (W/m³)
    """
    EPSILON = 0.0
    n = len(xs)
    d_xs = np.diff(np.ascontiguousarray(xs))
    
    # 计算光学厚度tau
    d_taus = (kappas[1:] + kappas[:-1])/2 * d_xs  # 中点积分
    taus = np.zeros(n)
    taus[1:] = np.cumsum(d_taus)
    taus[0] = EPSILON
    tau_L = taus[-1]  # 总光学厚度
    
    # 计算辐射通量G
    G = np.zeros(n)
    term1 = I_b1 * expn(2, taus)        # 左边界贡献
    term2 = I_b2 * expn(2, tau_L - taus) # 右边界贡献

    for i, tau in enumerate(taus):
        # 积分项计算：分为tau左右两部分
        idx1 = taus <= tau
        idx2 = taus >= tau

        integral_term1 = np.trapz(I_bs[idx1], expn(2, tau - taus[idx1]))
        integral_term2 = np.trapz(np.ascontiguousarray(I_bs[idx2][::-1]), np.ascontiguousarray(expn(2, taus[idx2] - tau)[::-1]))
        
        G[i] = 2 * np.pi * (term1[i] + term2[i] + integral_term1 + integral_term2)
    
    # 计算热流q及其梯度dq
    q = np.zeros(n)
    term1 = I_b1 * expn(3, taus)
    term2 = I_b2 * expn(3, tau_L - taus)

    for i, tau in enumerate(taus):
        idx1 = taus <= tau
        idx2 = taus >= tau

        integral_term1 = np.trapz(I_bs[idx1], expn(3, tau - taus[idx1]))
        integral_term2 = np.trapz(np.ascontiguousarray(I_bs[idx2][::-1]), np.ascontiguousarray(expn(3, taus[idx2] - tau)[::-1]))
        
        q[i] = 2 * np.pi * (term1[i] - term2[i] + integral_term1 - integral_term2)

    dq = (4 * np.pi * I_bs - G) * kappas  # 热流梯度
    return G, q, dq


def compute_chunk(kappas_chunk, i_b_etas_chunk, xs, I_b1_chunk, I_b2_chunk, d_wn):
    """
    为分块数据计算 G 和 q。
    """
    G_chunk = np.zeros_like(xs)
    q_chunk = np.zeros_like(xs)
    dq_chunk = np.zeros_like(xs)
    for wn_idx in range(kappas_chunk.shape[1]):
        wn_kappas = kappas_chunk[:, wn_idx]
        wn_i_b_etas = i_b_etas_chunk[:, wn_idx]
        G, q, dq = solve_rte(xs, wn_kappas, wn_i_b_etas, I_b1_chunk[wn_idx], I_b2_chunk[wn_idx])
        G_chunk += G * d_wn
        q_chunk += q * d_wn
        dq_chunk += dq * d_wn
    return G_chunk, q_chunk, dq_chunk

def getLBLsolution(T_l, T_r, P_tot):

    # 数据存储目录
    DATA_DIR = "../input/profiles"
    
    # 获取所有.npy文件路径
    case_files = glob.glob(os.path.join(DATA_DIR, "Case*.npy"))

    # 遍历加载每个文件
    for file_path in case_files:
        # 从文件名解析工况ID和长度参数
        filename = os.path.basename(file_path).replace(".npy", "")
        case_id, L_tag = filename.split("_")  # 示例: "Case1_L0.1m" → case_id="Case1", L_tag="L0.1m"
        L = float(L_tag[1:-1])  # 提取长度数值 (0.1, 1.0, 5.0)
        
        # 加载当前工况数据
        data = np.load(file_path, allow_pickle=False)
        
        # 结构化数组通过字段名访问
        xs = data["x"]                # 空间坐标 [m]
        x_h2o = data["Y_H2O"]         # H2O摩尔分数
        x_nh3 = data["Y_NH3"]         # NH3摩尔分数
        m_ratio = data["M_ratio"]     # 混合比
        temperature_profiles = data["T"]       # 温度 [K]

        p_h2o = x_h2o         # H2O的分压
        p_nh3 = x_nh3         # NH3的分压
        
        print(f"Processing {case_id} (L={L} m)")
 
        xss = np.vstack([p_h2o, p_nh3]).T
        
        # 计算吸收系数和黑体辐射
        kappas = get_kappas(xss, temperature_profiles) * 100.0  # 单位转换
        etas = np.linspace(0.1, 15000.0, len(kappas[0]))  # 波数网格
        i_b_etas = get_i_b_etas(temperature_profiles, etas)
        
        # 初始化结果数组
        final_G = np.zeros_like(xs)
        final_q = np.zeros_like(xs)
        final_dq = np.zeros_like(xs)
        
        # 边界辐射强度
        I_b1 = i_b_eta(T_l, etas)  # 左边界
        I_b2 = i_b_eta(T_r, etas)  # 右边界
        emis_left = 1.0
        emis_right = 1.0
        d_wn = etas[1] - etas[0]   # 波数间隔
        
        # 并行计算设置
        num_cores = os.cpu_count()
        chunk_size = (kappas.shape[1] + num_cores - 1) // num_cores
        kappas_chunks = [kappas[:, i * chunk_size:(i + 1) * chunk_size] for i in range(num_cores)]
        i_b_etas_chunks = [i_b_etas[:, i * chunk_size:(i + 1) * chunk_size] for i in range(num_cores)]
        I_b1_chunks = [I_b1[i * chunk_size:(i + 1) * chunk_size] for i in range(num_cores)]
        I_b2_chunks = [I_b2[i * chunk_size:(i + 1) * chunk_size] for i in range(num_cores)]
        
        # 并行计算
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = []
            for kappas_chunk, i_b_etas_chunk, I_b1_chunk, I_b2_chunk in zip(kappas_chunks, i_b_etas_chunks, I_b1_chunks, I_b2_chunks):
                futures.append(executor.submit(compute_chunk, kappas_chunk, i_b_etas_chunk, xs, I_b1_chunk, I_b2_chunk, d_wn))
            
            # 汇总结果
            for future in futures:
                G_part, q_part, dq_part = future.result()
                final_G += G_part
                final_q += q_part
                final_dq += dq_part
        
        # 保存结果
        ans = np.vstack([final_G, final_q, final_dq]).T
        # np.save(f"./results/h2o_nh3_{i:02d}.npy", ans)
        # 保存文件时添加区间标识（例如：h2o_nh3_00_0.0_0.1.npy）
        np.save(f"../results/{case_id}_L{L}m.npy", ans)  # 关键修改点


def solve_wsgg_dom(temperature_profiles, rsm_a, rsm_k, xs, T_l, T_r, epsilon=1.0, n_angles=8):
    """
    使用离散坐标法(DOM)求解WSGG辐射传递方程
    严格遵循Modest书中的公式(20.89)-(20.92)
    
    参数:
        temperature_profiles : 温度分布 [K], 形状为 (n,)
        rsm_a : 灰气体权重系数, 形状为 (n, Ng)
        rsm_k : 吸收系数 [m⁻¹], 形状为 (n, Ng)
        xs : 位置坐标 [m], 形状为 (n,)
        T_l, T_r : 左右边界温度 [K]
        epsilon : 壁面发射率 (默认为1.0)
        n_angles : 角度离散数 (必须为偶数, 默认为8)
    
    返回:
        q : 辐射热流分布 [W/m²], 形状为 (n,)
    """
    # 获取离散角度和权重（高斯-勒让德积分）
    def get_gauss_quadrature(n):
        """生成高斯-勒让德积分点和权重（对称分布在[-1,1]）"""
        mu, w = np.polynomial.legendre.leggauss(n)
        return mu, w
    
    n = len(xs)
    Ng = rsm_a.shape[1]
    mu, weights = get_gauss_quadrature(n_angles)
    
    # 计算位置相关的dx (用于射线追踪)
    dx = np.zeros(n)
    if n > 1:
        dx[0] = xs[1] - xs[0]
        for i in range(1, n-1):
            dx[i] = 0.5 * (xs[i+1] - xs[i-1])
        dx[n-1] = xs[n-1] - xs[n-2]
    
    # 计算黑体辐射强度 (公式20.89中的I_b)
    I_b = sigma * np.power(temperature_profiles, 4) / np.pi  # [W/m²·sr]
    
    # 初始化强度场 [灰气体, 方向, 位置]
    I_plus = np.zeros((Ng, n_angles//2, n))   # μ > 0 的辐射强度
    I_minus = np.zeros((Ng, n_angles//2, n))   # μ < 0 的辐射强度
    
    # 分离正负角度
    pos_mu = mu[mu > 0]
    pos_weights = weights[mu > 0]
    neg_mu = mu[mu < 0]
    neg_weights = weights[mu < 0]
    n_pos = len(pos_mu)
    
    # 边界黑体辐射强度
    I_b_l = epsilon * sigma * T_l**4 / np.pi
    I_b_r = epsilon * sigma * T_r**4 / np.pi
    
    # 主求解循环（每个灰气体独立求解）
    for k in range(Ng):
        k_gas = rsm_k[:, k]
        a_gas = rsm_a[:, k]
        
        # 处理正方向角度 (μ > 0)
        for m_idx, (mu_val, w) in enumerate(zip(pos_mu, pos_weights)):
            # 正向扫描：从左到右 (公式20.91的离散形式)
            # 左边界条件 (公式20.92)
            I_plus[k, m_idx, 0] = epsilon * a_gas[0] * I_b_l + (1 - epsilon) * I_minus[k, m_idx, 0]
            
            for i in range(1, n):
                # 吸收系数和光学厚度
                kappa_i = k_gas[i]
                delta_tau = kappa_i * dx[i] / mu_val
                
                # 指数格式求解 (公式20.91)
                if delta_tau > 1e-6:
                    alpha = np.exp(-delta_tau)
                else:
                    # 小光学厚度的线性近似
                    alpha = 1 - delta_tau
                
                # 源项计算 (公式20.91右侧)
                source = a_gas[i] * I_b[i] * (1 - alpha)
                
                # 更新辐射强度
                I_plus[k, m_idx, i] = I_plus[k, m_idx, i-1] * alpha + source
        
        # 处理负方向角度 (μ < 0)
        for m_idx, (mu_val, w) in enumerate(zip(neg_mu, neg_weights)):
            # 负向扫描：从右到左
            mu_abs = abs(mu_val)
            # 右边界条件 (公式20.92)
            I_minus[k, m_idx, n-1] = epsilon * a_gas[n-1] * I_b_r + (1 - epsilon) * I_plus[k, m_idx, n-1]
            
            for i in range(n-2, -1, -1):
                # 吸收系数和光学厚度
                kappa_i = k_gas[i]
                delta_tau = kappa_i * dx[i] / mu_abs
                
                # 指数格式求解
                if delta_tau > 1e-6:
                    alpha = np.exp(-delta_tau)
                else:
                    alpha = 1 - delta_tau
                
                # 源项计算
                source = a_gas[i] * I_b[i] * (1 - alpha)
                
                # 更新辐射强度
                I_minus[k, m_idx, i] = I_minus[k, m_idx, i+1] * alpha + source
    
    # 计算辐射热流 (公式20.89积分形式)
    q = np.zeros(n)
    for i in range(n):
        q_total = 0
        # 正方向贡献
        for m_idx, (mu_val, w) in enumerate(zip(pos_mu, pos_weights)):
            for k in range(Ng):
                q_total += w * mu_val * I_plus[k, m_idx, i]
        # 负方向贡献
        for m_idx, (mu_val, w) in enumerate(zip(neg_mu, neg_weights)):
            for k in range(Ng):
                q_total += w * mu_val * I_minus[k, m_idx, i]
        # 立体角积分因子
        q[i] = q_total * 2 * np.pi  # 2π来自方位角积分
    
    return q


def solve_rte_DOM(xs, kappas, I_bs, I_b1, I_b2, eps1=1.0, eps2=1.0, N_quad=8):
    """
    使用离散坐标法(DOM)求解辐射传输方程
    
    参数:
    xs : 空间坐标数组 [m]
    kappas : 吸收系数数组 [m⁻¹] (len=n)
    I_bs : 黑体辐射强度 [W/m³·sr] (len=n)
    I_b1, I_b2 : 边界辐射强度 [W/m³·sr]
    eps1, eps2 : 边界发射率 (默认黑体边界)
    N_quad : 离散方向数 (支持2,4,8阶)
    
    返回:
    G : 辐射通量 [W/m²]
    q : 辐射热流 [W/m²]
    dq : 热流梯度 [W/m³]
    """
    # 1. 方向离散配置
    if N_quad == 2:
        # S2离散方向 (正负各1个方向)
        mus = np.array([0.57735, -0.57735])  # 方向余弦
        weights = np.array([0.5, 0.5])       # 积分权重
    elif N_quad == 4:
        # S4离散方向 (正负各2个方向)
        mus = np.array([0.295876, 0.908248, -0.295876, -0.908248])
        weights = np.array([0.523598, 0.476402, 0.523598, 0.476402])
    elif N_quad == 8:
        # S8离散方向
        mus = np.array([0.174564, 0.525758, 0.796254, 0.960312,
                       -0.174564, -0.525758, -0.796254, -0.960312])
        weights = np.array([0.134911, 0.300704, 0.240348, 0.324037,
                            0.134911, 0.300704, 0.240348, 0.324037])
    else:
        raise ValueError("仅支持2/4/8阶离散方向")

    n = len(xs)                  # 空间点数
    n_dir = len(mus)             # 方向数
    dx = np.diff(xs)             # 空间步长
    
    # 2. 初始化辐射强度场 (各方向初始化为黑体辐射)
    I = np.outer(I_bs, np.ones(n_dir)).T  # 形状: (n_dir, n)

    # 3. 迭代求解
    max_iter = 1000
    tol = 1e-8
    converged = False
    
    for iter in range(max_iter):
        I_prev = I.copy()
        
        # 4. 边界条件更新
        # 左边界 (x=0)
        for i in range(n_dir//2):  # 仅正方向
            # 收集负方向强度 (从介质射出)
            I_out = 0.0
            for j in range(n_dir//2, n_dir):  # 负方向索引
                I_out += weights[j] * abs(mus[j]) * I[j, 0]
            # 更新正方向强度 (边界发射 + 反射)
            I[i, 0] = eps1 * I_b1 + (1 - eps1) * 2.0 * I_out
        
        # 右边界 (x=L)
        for i in range(n_dir//2, n_dir):  # 仅负方向
            # 收集正方向强度
            I_in = 0.0
            for j in range(n_dir//2):  # 正方向索引
                I_in += weights[j] * mus[j] * I[j, -1]
            # 更新负方向强度
            I[i, -1] = eps2 * I_b2 + (1 - eps2) * 2.0 * I_in
        
        # 5. 空间扫描 (每个方向独立求解)
        # 正方向扫描 (从左到右)
        for i in range(n_dir//2):  # 正方向
            for j in range(1, n):  # 从第二个点到最后一个点
                # 隐式差分格式
                a = abs(mus[i]) / dx[j-1] if j > 0 else abs(mus[i]) / dx[0]
                denom = a + kappas[j]
                num = a * I[i, j-1] + kappas[j] * I_bs[j]
                I[i, j] = num / denom
                
        # 负方向扫描 (从右到左)
        for i in range(n_dir//2, n_dir):  # 负方向
            for j in range(n-2, -1, -1):  # 从倒数第二个点反向扫描
                # 隐式差分格式
                a = abs(mus[i]) / dx[j] if j < n-1 else abs(mus[i]) / dx[-1]
                denom = a + kappas[j]
                num = a * I[i, j+1] + kappas[j] * I_bs[j]
                I[i, j] = num / denom
        
        # 6. 收敛检查
        max_error = np.max(np.abs(I - I_prev) / (np.abs(I_prev) + 1e-16))
        if max_error < tol:
            converged = True
            break
    
    if not converged:
        print(f"DOM求解在{max_iter}次迭代后未收敛")
    
    # 7. 计算辐射量
    G = np.zeros(n)  # 入射辐射 [W/m²]
    q = np.zeros(n)  # 辐射热流 [W/m²]
    
    # 积分求总辐射量
    for j in range(n):
        for i in range(n_dir):
            G[j] += 2 * np.pi * weights[i] * I[i, j]
            q[j] += 2 * np.pi * weights[i] * mus[i] * I[i, j]
    
    # 计算辐射源项 (能量守恒)
    dq = kappas * (4 * np.pi * I_bs - G)  # [W/m³]
    
    return G, q, dq

# 示例调用 -----------------------------------------------------
if __name__ == "__main__":
    Nx = 50
    xs = np.linspace(0, 1, Nx)
    T_profile = 800 - 500*xs  # 线性温度分布
    
    # 随机生成权重和吸收系数（示例用）
    ng = 5  
    np.random.seed(42)
    a_matrix = np.random.rand(Nx, ng)
    a_matrix /= a_matrix.sum(axis=1, keepdims=True)  # 归一化
    k_matrix = 0.1 + 0.05*np.random.rand(Nx, ng)
    
    # 调用求解器
    q, I_total, S_total = solve_rte_DOM(
        xs=xs,
        temperature_profile=T_profile,
        a_matrix=a_matrix,
        k_matrix=k_matrix,
        T1=800.0,
        T2=300.0,
        eps1=0.8,
        eps2=0.6,
        S_N=4
    )
