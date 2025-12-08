import os
import numpy as np
import matplotlib.pyplot as plt

# 获取当前脚本所在的绝对路径 (.../Codes/visualization_tools)
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (.../Codes)
project_root = os.path.dirname(script_dir)

# 定义文件夹路径
contra_features_dir = os.path.join(project_root, 'features')
save_dir = os.path.join(project_root, 'res', 'feature_visualization_distance')

# 确保输出文件夹存在
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 遍历 contra_features 文件夹中的所有 .txt 文件
for filename in os.listdir(contra_features_dir):
    if filename.endswith('.txt'):
        filepath = os.path.join(contra_features_dir, filename)
        
        # 读取所有行
        with open(filepath, 'r') as f:
            lines = [ln.rstrip('\n') for ln in f if ln.strip()]
        selected_lines = lines  # 读取全部行

        if not selected_lines:
            print(f"{filename} 未找到数据，跳过。")
            continue

        # 解析所有行的数据，只读取第53-151列（0-based索引52:151）
        data = []
        for line in selected_lines:
            vals = [float(x) for x in line.split(',')[52:151] if x.strip()]
            data.append(vals)
        data = np.array(data)

        if data.size == 0:
            print(f"文件 {filename} 解析后无数据，跳过。")
            continue

        num_frames, num_features = data.shape  # 行数为帧数，列数为特征数

        # 创建图形并保存，不在线显示
        plt.figure(figsize=(20, 8))  # 加长图片尺寸，增加宽度
        x = np.arange(num_frames)
        
        # 计算33个特征的距离（每个特征有3列xyz）
        num_landmarks = 33
        for i in range(num_landmarks):
            # 每个特征的xyz列索引（0-based: i*3, i*3+1, i*3+2）
            x_col = i * 3
            y_col = i * 3 + 1
            z_col = i * 3 + 2
            # 计算距离：sqrt(x^2 + y^2 + z^2)
            distances = np.sqrt(data[:, x_col]**2 + data[:, y_col]**2 + data[:, z_col]**2)
            plt.plot(x, distances, label=f'landmark {i+1}')  # 标签显示特征号（1-based）
        
        plt.title(f'{filename} landmark distance visualization (33 landmarks)')
        plt.xlabel('frames')
        plt.ylabel('distance from origin')
        # 设置major ticks每5帧显示标签，minor ticks每帧用于网格
        ticks = np.arange(0, num_frames, 5)
        labels = np.arange(1, num_frames + 1, 5)
        plt.xticks(ticks, labels)
        plt.gca().set_xticks(np.arange(num_frames), minor=True)
        plt.grid(True, which='both', axis='x')  # 在major和minor ticks上都绘制网格，确保每帧有网格
        # 特征较多时不显示全部图例以免遮挡
        if num_landmarks <= 20:
            plt.legend(loc='upper right', fontsize='small')
        # 保存图片到指定文件夹
        save_path = os.path.join(save_dir, f'{filename}.png')
        plt.savefig(save_path)
        plt.close()  # 关闭图形以释放内存
        
        print(f"已保存 {filename} 的可视化结果到 {save_path}")
        
print("所有文件处理完成。")