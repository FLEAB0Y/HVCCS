import pandas as pd
import numpy as np
import os

def clean_excel_data(file_path, output_path=None):
    """
    清洗Excel数据，修正异常值。
    - 从第8列开始处理数据（228*4列）
    - 第2-21行是20个受试者数据
    - 第22行是均值行
    - 方差将由脚本计算
    """
    # 设置输出路径
    if output_path is None:
        file_name, file_ext = os.path.splitext(file_path)
        output_path = f"{file_name}_cleaned{file_ext}"
    
    # 读取Excel文件
    print(f"正在读取文件: {file_path}")
    df = pd.read_excel(file_path)
    
    # 打印基本信息
    print(f"文件包含 {df.shape[0]} 行和 {df.shape[1]} 列")
    
    # 定义数据范围
    data_start_col = 7  # 第8列（索引从0开始）
    subjects_start_row = 1  # 第2行（索引从0开始）
    subjects_end_row = 20  # 第21行
    mean_row = 21  # 第22行（索引从0开始）
    
    # 检查实际行数是否足够
    if df.shape[0] <= mean_row:
        print(f"警告：文件只有 {df.shape[0]} 行，少于预期的 {mean_row+1} 行")
        return
    
    # 统计修改数量
    modified_cols = 0
    modified_values = 0
    
    # 遍历每一列数据
    for col_idx in range(data_start_col, df.shape[1]):
        col_name = df.columns[col_idx]
        
        # 获取该列受试者的数据
        subjects_data = df.iloc[subjects_start_row:subjects_end_row+1, col_idx].copy()
        
        # 自己计算方差
        valid_data = subjects_data[pd.notna(subjects_data)]
        if len(valid_data) < 2:  # 需要至少两个值才能计算方差
            continue
            
        variance = valid_data.var()
        
        # 检查方差是否大于1
        if np.isnan(variance) or variance <= 1:
            continue
        
        modified_cols += 1
        print(f"处理列 '{col_name}' (计算方差 = {variance:.2f})")
        
        # 计算有效的替代值（1-5之间的整数）的中位数
        valid_values = []
        for v in subjects_data:
            if pd.notna(v) and 1 <= v <= 5 and float(v).is_integer():
                valid_values.append(v)
        
        # 确定替代值
        if valid_values:
            replacement_value = int(round(np.median(valid_values)))
        else:
            replacement_value = 3  # 默认值
        
        # 迭代减小方差直到小于1
        iteration = 0
        max_iterations = 20  # 防止无限循环
        current_data = subjects_data.copy()
        current_var = variance
        
        # 开始从最极端的值进行替换
        modified_rows = []
        
        while current_var > 1 and iteration < max_iterations:
            iteration += 1
            
            # 计算数据的均值和偏差
            valid_data = current_data[pd.notna(current_data)]
            if len(valid_data) < 2:
                break
                
            mean_val = valid_data.mean()
            
            # 计算每个值与均值的偏差
            deviations = {}
            for row_offset, value in enumerate(current_data):
                if pd.isna(value):
                    continue
                dev = abs(value - mean_val)
                deviations[subjects_start_row + row_offset] = dev
            
            # 找出偏差最大的行
            if not deviations:
                break
                
            max_dev_row = max(deviations, key=deviations.get)
            
            # 替换该行的值
            old_val = df.iloc[max_dev_row, col_idx]
            df.iloc[max_dev_row, col_idx] = replacement_value
            current_data = df.iloc[subjects_start_row:subjects_end_row+1, col_idx].copy()
            
            # 重新计算方差
            valid_new_data = current_data[pd.notna(current_data)]
            current_var = valid_new_data.var()
            
            modified_rows.append((max_dev_row, old_val, replacement_value))
            print(f"  迭代 {iteration}: 行 {max_dev_row+1} 的值从 {old_val} 修改为 {replacement_value}，当前方差 = {current_var:.2f}")
            
            if current_var <= 1:
                print(f"  成功: 方差降至 {current_var:.2f}")
        
        # 记录修改的值数量
        modified_values += len(modified_rows)
        
        # 更新均值
        new_values = df.iloc[subjects_start_row:subjects_end_row+1, col_idx]
        df.iloc[mean_row, col_idx] = new_values.mean()
    
    print(f"总共修改了 {modified_cols} 列中的 {modified_values} 个值")
    
    # 保存修改后的数据
    print(f"正在保存修改后的数据到: {output_path}")
    df.to_excel(output_path, index=False)
    
    return output_path

# 主程序
if __name__ == "__main__":
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 设置文件路径
    wjx_file_path = os.path.join(script_dir, "..", "wjx", "wjx.xlsx")
    
    # 检查文件是否存在
    if not os.path.exists(wjx_file_path):
        print(f"错误：文件不存在 - {wjx_file_path}")
        print(f"当前工作目录: {os.getcwd()}")
        exit(1)
    
    # 设置输出路径
    wjx_output_path = os.path.join(script_dir, "..", "wjx", "wjx_cleaned.xlsx")
    
    # 执行数据清洗
    print(f"使用自动检测的文件路径: {wjx_file_path}")
    clean_excel_data(wjx_file_path, wjx_output_path)
    print(f"处理完成，清洗后的数据已保存到: {wjx_output_path}")