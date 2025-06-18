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

def create_excel(data, output_file):
    """创建并保存Excel表格"""
    # 更新表头名称
    df = pd.DataFrame(data, columns=["视频编号", "人物标签", "CRF", "drop"])
    df = df.sort_values(by="视频编号")
    df.to_excel(output_file, index=False)
    return output_file

def process_files(folder_path, output_file):
    """处理文件夹中的所有文件并生成Excel表格"""
    try:
        files = get_txt_files(folder_path)
        
        # 存储解析结果
        data = []
        
        # 解析每个文件名
        for file in files:
            result = parse_filename(file)
            if result:
                data.append(result)
        
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
    features_dir = os.path.join(script_dir, "..", "proc_features")
    output_file = os.path.join(script_dir, "..", "wjx", "file_parameters.xlsx")
    
    print(f"读取文件夹: {features_dir}")
    print(f"输出文件路径: {output_file}")
    
    process_files(features_dir, output_file)