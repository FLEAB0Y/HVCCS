import os
import numpy as np
import random

def quantize_data(data, bit_depth):
    """将数据量化到指定的位深度
    
    Args:
        data: 原始数据数组
        bit_depth: 量化等级，可以是32, 16, 8, 4
        
    Returns:
        量化后的数据
    """
    # 对于32位，不做任何量化处理
    if bit_depth == 32:
        return data.copy()
    
    # 计算量化级别
    if bit_depth == 16:
        levels = 2**16 - 1  # 65535级
    elif bit_depth == 8:
        levels = 2**8 - 1   # 255级
    elif bit_depth == 4:
        levels = 2**4 - 1   # 15级
    else:
        raise ValueError(f"不支持的位深度: {bit_depth}")
    
    # 创建结果数组
    result = np.zeros_like(data)
    
    # 对每列分别进行量化
    for col in range(data.shape[1]):
        col_data = data[:, col]
        
        # 跳过全零列
        if np.all(col_data == 0):
            continue
        
        # 找出该列的最大和最小值
        min_val = np.min(col_data)
        max_val = np.max(col_data)
        
        if min_val == max_val:
            # 如果最大值等于最小值，不需要量化
            result[:, col] = col_data
            continue
        
        # 归一化到[0, 1]范围
        normalized = (col_data - min_val) / (max_val - min_val)
        
        # 量化到指定级别
        quantized_normalized = np.round(normalized * levels) / levels
        
        # 反归一化回原始范围
        result[:, col] = quantized_normalized * (max_val - min_val) + min_val
    
    return result

def apply_packet_loss(data, loss_rate):
    """应用丢包率，随机将部分数据点置为0
    
    Args:
        data: 原始数据，形状为[帧数, 特征数]
        loss_rate: 丢包率，0到1之间
        
    Returns:
        应用丢包后的数据
    """
    # 创建数据的副本
    result = data.copy()
    
    # 对于0丢包率，直接返回原始数据
    if loss_rate == 0:
        return result
    
    # 生成随机矩阵，确定哪些点要丢弃
    random_matrix = np.random.random(data.shape)
    # 创建掩码，True表示保留，False表示丢弃
    mask = random_matrix >= loss_rate
    
    # 应用掩码，将被丢弃的点置为0
    result = result * mask
    
    return result

def process_feature_file(input_path, output_dir, bit_depths, loss_rates):
    """处理单个特征文件，应用不同的量化等级和丢包率
    
    Args:
        input_path: 输入文件路径
        output_dir: 输出目录
        bit_depths: 量化等级列表
        loss_rates: 丢包率列表
    """
    # 从文件名中提取基本名称
    base_name = os.path.basename(input_path)
    name_without_ext = os.path.splitext(base_name)[0]
    
    print(f"正在处理: {name_without_ext}")
    
    # 读取原始特征数据
    try:
        with open(input_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"读取文件 {input_path} 失败: {e}")
        return
    
    # 解析数据
    data = []
    for line in lines:
        if line.strip():  # 忽略空行
            try:
                # 移除行末的逗号和换行符，然后分割数据
                values = line.rstrip(',\n').split(',')
                # 过滤掉空字符串并转换为浮点数
                values = [float(v) for v in values if v]
                if values:  # 确保有数据
                    data.append(values)
            except Exception as e:
                print(f"解析行数据失败: {e}")
                continue
    
    if not data:
        print(f"警告: 文件 {input_path} 中没有有效数据")
        return
    
    # 检查所有行是否有相同的特征数量
    feature_lengths = [len(row) for row in data]
    if len(set(feature_lengths)) > 1:
        print(f"警告: 文件 {input_path} 中不同行的特征数量不一致")
        # 找出最常见的特征数量
        from collections import Counter
        most_common_length = Counter(feature_lengths).most_common(1)[0][0]
        # 过滤掉不符合最常见长度的行
        data = [row for row in data if len(row) == most_common_length]
        print(f"已过滤保留具有 {most_common_length} 个特征的行")
    
    # 转换为NumPy数组
    data = np.array(data)
    
    # 对每种量化等级和丢包率组合进行处理
    for bit_depth in bit_depths:
        for loss_rate in loss_rates:
            # 创建输出文件名
            output_filename = f"{name_without_ext}_q{bit_depth}_l{int(loss_rate*100)}.txt"
            output_path = os.path.join(output_dir, output_filename)
            
            # 量化数据
            quantized_data = quantize_data(data, bit_depth)
            
            # 应用丢包
            processed_data = apply_packet_loss(quantized_data, loss_rate)
            
            # 保存到文件
            try:
                with open(output_path, 'w') as f:
                    for row in processed_data:
                        # 将数据转换回字符串并添加逗号
                        line = ','.join([f"{value:.6f}" for value in row]) + ',\n'
                        f.write(line)
                print(f"  已保存: {output_filename}")
            except Exception as e:
                print(f"保存文件 {output_path} 失败: {e}")
    
    print(f"完成: {name_without_ext}")

def main():
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 设置输入和输出目录
    input_dir = os.path.join(script_dir, "..", "ori_features")
    output_dir = os.path.join(script_dir, "..", "proc_features")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义量化等级和丢包率
    bit_depths = [32, 16, 8, 4]
    loss_rates = [0, 0.1, 0.2, 0.4, 0.7]
    
    # 获取所有.txt文件
    try:
        txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    except Exception as e:
        print(f"读取目录 {input_dir} 失败: {e}")
        return
    
    if not txt_files:
        print(f"警告: 在 {input_dir} 中没有找到.txt文件")
        return
    
    print(f"找到 {len(txt_files)} 个.txt文件")
    
    # 处理每个文件
    for txt_file in txt_files:
        input_path = os.path.join(input_dir, txt_file)
        process_feature_file(input_path, output_dir, bit_depths, loss_rates)
    
    print("所有文件处理完成！")

if __name__ == "__main__":
    # 设置随机种子以便结果可重现
    np.random.seed(42)
    main()