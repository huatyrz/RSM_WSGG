import numpy as np
from scipy.io import FortranFile
import matplotlib.pyplot as plt

# 本程序用于读取Fortran生成的二进制吸收系数文件
# 生成该二进制吸收系数文件的Fortran程序为SPECTRAL RADIATION CALCULATION SOFTWARE ~ SRCS ~ by MICHAEL F. MODES and Tao Ren

def Read_Fortran_AbscData(file_path, temperature_index):
    # 初始化存储温度的列表和存储数据的字典
    temperatures = []
    temperature_data = {}
    
    # 打开文件
    with FortranFile(file_path, 'r') as file:
        # 读取文件头信息
        # 根据 Fortran 代码，头信息包括：
        # - P（压力）
        # - x（摩尔分数）
        # - wvnm_b（波长起始）
        # - wvnm_e（波长结束）
        # - nT（温度数量）
        P = file.read_reals(dtype=np.float32)
        x = file.read_reals(dtype=np.float32)
        wvnm_b = file.read_reals(dtype=np.float32)
        wvnm_e = file.read_reals(dtype=np.float32)
        nT = file.read_ints(dtype=np.int32)
        
        # 打印头信息
        print(f"Pressure (P): {P[0]}")
        print(f"Mole Fraction (x): {x[0]}")
        print(f"Wavelength Start (wvnm_b): {wvnm_b[0]}")
        print(f"Wavelength End (wvnm_e): {wvnm_e[0]}")
        print(f"Number of Temperatures (nT): {nT[0]}\n")
    
         # 初始化存储数据的数组
        temperatures = []
        wvnmst = []
        absc_r4 = []
        
        # 读取温度和波长步长
        for it in range(nT[0]):
            T = file.read_reals(dtype=np.float32)  # 温度
            temperatures.append(T[0])  # 将温度值添加到列表中
            wvnmst = file.read_reals(dtype=np.float32)
            number = file.read_ints(dtype=np.int32)
            absc_r4 = file.read_reals(dtype=np.float64)
            
            #打印温度和波长步长
            print(f"Temperature (T): {T[0]}")
            print(f"Wavelength Step (wvnmst): {wvnmst[0]}")
            print(f"Number of Wavelengths (number): {number[0]}")
            print(f"Absorption Coefficients: {absc_r4}")
            
            # 计算波数
            wvnm = np.arange(wvnm_b[0], wvnm_e[0] + wvnmst[0], wvnmst[0])
            
             # 将数据存储在字典中
            temperature_data[T[0]] = {'wvnm': wvnm, 'absc_r4': absc_r4}
        
    # 将温度列表转换为 NumPy 数组
    temperatures_array = np.array(temperatures)
    
    # 选择要绘制的温度下标
    # temperature_index = 2  # 例如，绘制第三个温度下的数据
    
    # 检查选择的温度下标是否在范围内
    if 0 <= temperature_index < len(temperatures_array):
        target_temperature = temperatures_array[temperature_index]
        wvnm = temperature_data[target_temperature]['wvnm']
        absc_r4 = temperature_data[target_temperature]['absc_r4']
        # plt.plot(wvnm, absc_r4, label=f'Temperature {target_temperature} K')
    
        # plt.xlabel('Wavenumber (cm⁻¹)')
        # plt.ylabel('Absorption Coefficient')
        # plt.title(f'Absorption Coefficient vs Wavenumber at Temperature {target_temperature} K')
        # plt.legend()
        # plt.show()
        return wvnm, absc_r4  # 返回波数和吸收系数
    else:
        print(f"Temperature index {temperature_index} is out of range.")
        return None  # 如果温度索引超出范围，返回 None

# 以下代码用于测试模块功能
# if __name__ == "__main__":
#     file_path = 'abscH2O.28.P.01.0m.0.01.sp.vt.dat'
#     temperature_index = 2
#     read_and_plot_absorption_coefficient(file_path, temperature_index)