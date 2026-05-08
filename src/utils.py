import pandas as pd
import numpy as np
import torch

def print_formatted_arrays(beta, gamma):
    """格式化打印系数矩阵"""
    # 打印Beta表格
    print("\nBeta Coefficients:")
    print(f"{'Coef':<8} | {'i=1':<8} | {'i=2':<8} | {'i=3':<8} | {'i=4':<8}")
    print("-" * 50)
    for i in range(15):
        name = f"β_i{i}"
        print(f"{name:<8} | {beta[0, i]:.7f} | {beta[1, i]:.7f} | {beta[2, i]:.7f} | {beta[3, i]:.7f}")
    
    # 打印Gamma表格
    print("\nGamma Coefficients:")
    print(f"{'Coef':<8} | {'i=1':<8} | {'i=2':<8} | {'i=3':<8} | {'i=4':<8}")
    print("-" * 50)
    for i in range(5):
        name = f"γ_i{i}"
        print(f"{name:<8} | {gamma[0, i]:.7f} | {gamma[1, i]:.7f} | {gamma[2, i]:.7f} | {gamma[3, i]:.7f}")

def save_to_excel(beta, gamma, Ng, Exp, filename):
    """使用pandas库导出到Excel，支持可变灰气体数量Ng"""
    
    # 生成动态列名，如i=0到i=Ng
    columns = [f'i={i}' for i in range(Ng+1)]
    
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
        worksheet.set_column(1, 1 + Ng, 20)
    
    # 设置所有列居中：从A列到数值列最后一列
    cell_format = workbook.add_format({'align': 'center'})
    worksheet.set_column(0, 1 + Ng, None, cell_format)

def read_from_excel(filename):
    """从Excel读取参数"""
    beta_df = pd.read_excel(filename, sheet_name='Beta', index_col=0)
    gamma_df = pd.read_excel(filename, sheet_name='Gamma', index_col=0)
    return beta_df.values.T, gamma_df.values.T

def setup_device():
    """设置计算设备并返回"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

def get_timestamp():
    """获取当前时间戳"""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")