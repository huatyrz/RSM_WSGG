import pandas as pd
import numpy as np
# ------------------------------------------------------------------------函数分割线----------------------------------------------------------------------
# ----------------------
# 1. 读取带摩尔分数范围的Excel
# ----------------------
def read_coefficients_with_ranges(file_path, sheet_name):
    """读取包含摩尔分数范围的多区间系数表"""
    # 读取双行表头并统一区间符号
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=[1,2], skiprows=0)
    # 处理特殊符号（如将"1~5"转换为"1-5"）
    columns = df.columns.to_frame()
    columns.iloc[0] = columns.iloc[0].str.replace('~', '-')  # 统一区间符号
    df.columns = pd.MultiIndex.from_frame(columns)

    # 获取唯一摩尔分数区间及对应列范围
    m_ranges = []
    current_range = None
    start_col = 1  # 从第1列开始（第0列为索引列）
    
    # 遍历第一层表头建立区间映射
    for idx, col in enumerate(df.columns.get_level_values(0)[1:], 1):  # 从第1列开始
        if col != current_range:
            if current_range is not None:
                m_ranges.append({
                    "range": current_range,
                    "start": start_col,
                    "end": idx-1,
                    "columns_count": idx - start_col
                })
            current_range = col
            start_col = idx
    
    # 添加最后一个区间
    m_ranges.append({
        "range": current_range,
        "start": start_col,
        "end": len(df.columns)-1,
        "columns_count": len(df.columns) - start_col
    })

    # 构建区间映射字典 (示例数据中的区间顺序已知)
    valid_ranges = ['0.05-0.2', '0.2-1', '1-5', '5-99']
    range_dict = {}
    for idx, r in enumerate(valid_ranges):
        lower, upper = map(float, r.split('-'))
        range_dict[(lower, upper)] = idx  # 列索引为0,1,2,3
    
    # 1. 原始数据读取（跳过空行并处理合并单元格）
    raw_df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, skiprows=3)  # 根据图片数据跳过多余表头

    # 2. 索引解析（匹配图片中的"1, 0"格式）
    def parse_index(val):
        try:
            parts = str(val).replace(" ", "").split(",")
            return (int(parts[0]), int(parts[1]))
        except:
            return (np.nan, np.nan)
    
    # 提取索引列（假设索引在A列）
    raw_df[['i', 'j']] = raw_df.iloc[:, 0].apply(
        lambda x: pd.Series(parse_index(x)) if pd.notnull(x) else (np.nan, np.nan)
    )
    
    c_df = raw_df.dropna(subset=['i','j']).astype({'i':int, 'j':int})

    # ======================
    # 第二部分：单独读取K系数（图片中的35-39行）
    # ======================
    # 直接定位K系数区域（图片中i=1-4对应的K1-K3）
    k_df = pd.read_excel(file_path, sheet_name=sheet_name, 
                        header=None,
                        skiprows=36,  # 从第36行开始
                        nrows=4) 

    # 3. 建立三维数据结构（适配图片中的列分布）
    coefficients = {}
    valid_ranges = ['0.05-0.2', '0.2-1', '1-5', '5-99']
    param_types = ['C1', 'C2', 'C3', 'K1', 'K2', 'K3']
    
    # 存储C系数（三维字典：i -> j -> 参数）
    for (i, j), row in c_df.groupby(['i','j']):
        # 获取当前行的所有列（排除i,j后的参数列）
        data = row.iloc[0, 1:].values  # 假设i,j是前两列，参数从第三列开始
        
        # 计算每个参数的索引：每区间3列，四个区间
        c1_values = [data[3*k] for k in range(4)]      # 每个区间的第一个参数是C1
        c2_values = [data[3*k + 1] for k in range(4)]  # 第二个是C2
        c3_values = [data[3*k + 2] for k in range(4)]  # 第三个是C3

        coefficients.setdefault(i, {})[j] = {
            'C1': c1_values,
            'C2': c2_values,
            'C3': c3_values
        }

    # 结构化存储K系数（每个i对应四个区间的参数组）
    for i in [1,2,3,4]:
        # 提取当前i的K系数行（应为12个数值）
        k_values = k_df.iloc[i-1, 1:].tolist()  # ← 关键修改：切片排除第一列

        # 拆分为四组，每组三个参数（对应四个摩尔分数区间）
        k_grouped = [ 
            k_values[idx*3 : (idx+1)*3]  # 每区间3个参数
            for idx in range(4) 
        ]
        
        # 存储到字典（与C系数共享同一数据结构）
        coefficients[i]['K'] = k_grouped

    return coefficients, range_dict

def select_range(M, range_dict):
    """根据M值选择对应的区间索引"""
    for (lower, upper), idx in range_dict.items():
        if lower <= M < upper:
            return idx
    if M >= max(u for (_, u) in range_dict.keys()):
        return max(range_dict.values())
    raise ValueError(f"M={M}超出定义范围")
    
# ------------------------------------------------------------------------函数分割线----------------------------------------------------------------------
def calculate_a_i(T, M, range_dict, coefficients, i, T_ref=1400):
    col_idx = select_range(M, range_dict)
    a_i = 0
    for j in range(1,9):  # j=1-8
        coeff = coefficients[i][j]
        term = (coeff['C1'][col_idx] * M**2 +
                coeff['C2'][col_idx] * M +
                coeff['C3'][col_idx])
        a_i += term * (T/T_ref)**(8-j)
    return a_i

# ------------------------------------------------------------------------函数分割线----------------------------------------------------------------------
def calculate_k_i(M, range_dict, coefficients, i):
    col_idx = select_range(M, range_dict)
    k_coeffs = coefficients[i]['K'][col_idx]  # 取出对应区间的[K1,K2,K3]
    return k_coeffs[0]*M**2 + k_coeffs[1]*M + k_coeffs[2]


# ------------------------------------------------------------------------函数分割线----------------------------------------------------------------------
# wang_wsgg根据拟合的系数C和K计算权重系数 a和吸收系数 k
def get_k_a(T, Mr, Ng, coefficients, range_dict):
    a = np.zeros(Ng)  # 对应gas_i 1-4
    K = np.zeros(Ng)
    for gas_i in [1,2,3,4]:
        a[gas_i-1] = calculate_a_i(T, Mr, range_dict, coefficients, gas_i)
        K[gas_i-1] = calculate_k_i(Mr, range_dict, coefficients, gas_i)
    
    ans_a = np.zeros(Ng+1)
    ans_a[0] = 1.0 - np.sum(a)
    ans_a[1:] = a

    # Calculation of "kk" coefficients
    ans_k = np.zeros(Ng+1)
    ans_k[1:] = K

    return ans_a, ans_k

# ------------------------------------------------------------------------函数分割线----------------------------------------------------------------------

if __name__ == "__main__":
    # 读取数据
    coefficients, range_dict = read_coefficients_with_ranges("wang_data.xlsx", sheet_name="Px=1bar")
    
    # 示例计算
    M = 1.3
    T = 1500
    results = []
    for gas_i in [1,2,3,4]:
        a_i = calculate_a_i(T, M, range_dict, coefficients, gas_i)
        k_i = calculate_k_i(M, range_dict, coefficients, gas_i)
        results.append((gas_i, a_i, k_i))
    
    # 打印结果
    print(f"计算结果（M={M}, T={T}K）:")
    print("| 灰气体 | 权重系数 (a_i) | 吸收系数 (k_i) |")
    print("|--------|----------------|----------------|")
    for gas_i, a, k in results:
        print(f"| {gas_i}     | {a:.4e}      | {k:.4e}      |")