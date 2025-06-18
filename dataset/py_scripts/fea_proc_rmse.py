import numpy as np
import os
import re
import pandas as pd
from pathlib import Path

def get_txt_files(folder_path):
    """获取指定文件夹中的所有txt文件"""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"文件夹 {folder_path} 不存在")
    return [f for f in os.listdir(folder_path) if f.endswith('.txt')]

def parse_filename(filename):
    """解析文件名，提取序号、标签、码率控制等级和丢包率"""
    pattern = r"(\d+)_id(\d+)_q(\d+)_l(\d+)\.txt"
    match = re.match(pattern, filename)
    if match:
        sequence_num = int(match.group(1))
        label = int(match.group(2))
        bit_rate_level = int(match.group(3))
        packet_loss_rate = int(match.group(4)) / 100  # 将丢包率除以100
        return sequence_num, label, bit_rate_level, packet_loss_rate
    return None

def calculate_rmse(original_file_path, processed_file_path):
    """计算两个文件之间的RMSE，跳过每行前52个数字，只处理后99个数字"""
    try:
        # 读取原始文件
        original_points = []
        with open(original_file_path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('//'):  # 跳过空行和注释行
                    values = [float(val) for val in line.strip().split(',') if val]
                    # 跳过前52个数字
                    values = values[52:]
                    # 每三个数字形成一个3D点
                    for i in range(0, len(values), 3):
                        if i+2 < len(values):  # 确保有完整的三个值
                            original_points.append(values[i:i+3])
        
        # 读取处理后的文件
        processed_points = []
        with open(processed_file_path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('//'):  # 跳过空行和注释行
                    values = [float(val) for val in line.strip().split(',') if val]
                    # 跳过前52个数字
                    values = values[52:]
                    # 每三个数字形成一个3D点
                    for i in range(0, len(values), 3):
                        if i+2 < len(values):  # 确保有完整的三个值
                            processed_points.append(values[i:i+3])
        
        # 确保两个数组长度相同
        min_len = min(len(original_points), len(processed_points))
        original_points = np.array(original_points[:min_len])
        processed_points = np.array(processed_points[:min_len])
        
        # 计算RMSE
        squared_diff = np.sum((original_points - processed_points) ** 2, axis=1)  # 逐点计算平方差
        rmse = np.sqrt(np.mean(squared_diff))  # 计算均方根误差
        
        return rmse
    
    except Exception as e:
        print(f"计算RMSE时出错: {e}")
        return None

def create_excel(data, output_file):
    """创建并保存Excel表格"""
    # 更新表头名称，添加RMSE列
    df = pd.DataFrame(data, columns=["视频编号", "人物标签", "CRF", "drop", "RMSE"])
    df = df.sort_values(by="视频编号")
    df.to_excel(output_file, index=False)
    return output_file

def process_files(orig_folder_path, proc_folder_path, output_file):
    """处理文件夹中的所有文件并生成Excel表格"""
    try:
        proc_files = get_txt_files(proc_folder_path)
        
        # 存储解析结果
        data = []
        
        # 解析每个文件名并计算RMSE
        for file in proc_files:
            result = parse_filename(file)
            if result:
                sequence_num, label, bit_rate_level, packet_loss_rate = result
                
                # 构建对应的原始文件名（不带q和l参数）
                orig_file = f"{sequence_num:04d}_id{label:02d}.txt"
                orig_file_path = os.path.join(orig_folder_path, orig_file)
                proc_file_path = os.path.join(proc_folder_path, file)
                
                # 如果原始文件存在，计算RMSE
                if os.path.exists(orig_file_path):
                    rmse = calculate_rmse(orig_file_path, proc_file_path)
                    data.append((sequence_num, label, bit_rate_level, packet_loss_rate, rmse))
                else:
                    print(f"原始文件 {orig_file} 不存在")
        
        if not data:
            print("没有找到符合格式的文件")
            return None
        
        # 创建并保存Excel表格
        output_path = create_excel(data, output_file)
        print(f"数据已成功保存到 {output_path}")
        return output_path
    
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        return None

if __name__ == "__main__":
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建相对路径
    contra_features_dir = os.path.join(script_dir, "..", "contra_features")
    proc_features_dir = os.path.join(script_dir, "..", "proc_features")
    output_file = os.path.join(script_dir, "..", "wjx", "RMSE.xlsx")
    
    print(f"原始文件夹: {contra_features_dir}")
    print(f"处理文件夹: {proc_features_dir}")
    print(f"输出文件路径: {output_file}")
    
    process_files(contra_features_dir, proc_features_dir, output_file)