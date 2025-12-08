import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import PchipInterpolator, CubicHermiteSpline

# 获取当前脚本所在的绝对路径 (.../Codes/visualization_tools)
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (.../Codes)
project_root = os.path.dirname(script_dir)

def process_distance_single_feature(file_path, segment_length):
    """
    处理单个特征文件的距离可视化，进行频域分析，使用傅立叶变换绘制幅频响应图。
    
    参数:
    - file_path: str, 特征文件路径 (.txt)
    - segment_length: int, 每段的帧数长度（保留参数，但不用于拟合）
    
    返回:
    - dict: 包含保存图片路径和参数文件路径的dict
    """
    # 定义保存目录
    save_dir = os.path.join(project_root, 'res', 'feature_visualization_distance_frequency')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    filename = os.path.basename(file_path)
    
    # 读取所有行
    with open(file_path, 'r') as f:
        lines = [ln.rstrip('\n') for ln in f if ln.strip()]
    selected_lines = lines  # 读取全部行

    if not selected_lines:
        print(f"{filename} 未找到数据，跳过。")
        return None

    # 解析所有行的数据，读取所有33个landmarks的xyz列（52为blendshapes结束，共33*3=99列）
    colume_start = 52
    num_landmarks = 33
    data = []
    for line in selected_lines:
        vals = [float(x) for x in line.split(',')[colume_start:colume_start + num_landmarks * 3] if x.strip()]
        data.append(vals)
    data = np.array(data)

    if data.size == 0:
        print(f"文件 {filename} 解析后无数据，跳过。")
        return None

    num_frames, num_features = data.shape  # 行数为帧数，列数为特征数（99）

    # 创建图形
    plt.figure(figsize=(20, 8))
    
    # 获取颜色循环
    colors = plt.cm.tab20(np.linspace(0, 1, num_landmarks))
    
    # 保存所有landmarks的FFT参数
    all_params = {}
    
    for landmarks_index in range(num_landmarks):
        # 每个landmarks的xyz列索引（0-based: landmarks_index*3, landmarks_index*3+1, landmarks_index*3+2）
        x_col = landmarks_index * 3
        y_col = landmarks_index * 3 + 1
        z_col = landmarks_index * 3 + 2
        # 计算距离：sqrt(x^2 + y^2 + z^2)
        distances = np.sqrt(data[:, x_col]**2 + data[:, y_col]**2 + data[:, z_col]**2)
        
        # 进行傅立叶变换
        fft_result = np.fft.fft(distances)
        freqs = np.fft.fftfreq(num_frames)  # 频率，假设采样频率为1 Hz
        
        # 计算幅值（取绝对值）
        amplitudes = np.abs(fft_result)
        
        # 只绘制正频率部分
        positive_freqs = freqs[:num_frames//2]
        positive_amplitudes = amplitudes[:num_frames//2]
        
        # 绘制曲线
        plt.plot(positive_freqs, positive_amplitudes, color=colors[landmarks_index], linewidth=2, label=f'Landmark {landmarks_index}')
        
        # 保存参数
        all_params[f'landmark_{landmarks_index}'] = {
            'frequency': list(positive_freqs),
            'amplitude': list(positive_amplitudes)
        }
    
    # 设置图表属性
    plt.title(f'{filename} landmark distance frequency analysis (all 33 landmarks)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend(loc='upper right', fontsize='small')
    
    # 保存图片到指定文件夹
    save_path = os.path.join(save_dir, f'{filename}_all_landmarks.png')
    plt.savefig(save_path)
    plt.close()  # 关闭图形以释放内存
    
    # 保存FFT参数到txt文件（保存所有landmarks的频率和幅值）
    txt_path = os.path.join(save_dir, f'{filename}_all_landmarks_fft_params.txt')
    with open(txt_path, 'w') as f:
        for key, value in all_params.items():
            f.write(f"{key}:\n")
            f.write("Frequency (Hz): " + str(value['frequency']) + "\n")
            f.write("Amplitude: " + str(value['amplitude']) + "\n\n")
    
    print(f"已保存 {filename} 的可视化结果到 {save_path}")
    print(f"已保存 {filename} 的FFT参数到 {txt_path}")
    
    return {
        'image_path': save_path,
        'params_path': txt_path
    }

# 示例调用（可选，用于测试）
if __name__ == "__main__":
    # 定义文件夹路径
    contra_features_dir = os.path.join(project_root, 'features')
    
    # 遍历 contra_features 文件夹中的所有 .txt 文件
    for filename in os.listdir(contra_features_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(contra_features_dir, filename)
            segment_length = 8  # 保留参数，但不使用
            result = process_distance_single_feature(file_path, segment_length)
            if result:
                print(f"处理完成: {result}")
    
    print("所有文件处理完成。")
