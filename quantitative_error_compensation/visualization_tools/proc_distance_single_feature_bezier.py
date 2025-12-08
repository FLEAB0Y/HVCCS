import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import PchipInterpolator, CubicHermiteSpline

# 获取当前脚本所在的绝对路径 (.../Codes/visualization_tools)
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (.../Codes)
project_root = os.path.dirname(script_dir)

def process_distance_single_feature(file_path, landmarks_index, segment_length):
    """
    处理单个特征文件的距离可视化，使用Hermite插值分段拟合贝塞尔曲线，确保C1连续性。
    
    参数:
    - file_path: str, 特征文件路径 (.txt)
    - landmarks_index: int, 姿势关键点索引 (0-32)
    - segment_length: int, 每段的帧数长度
    
    返回:
    - dict: 包含保存的图片路径和参数文件路径
    """
    # 定义保存目录
    save_dir = os.path.join(project_root, 'res', 'single_feature_visualization_distance_Bezier')
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

    # 解析所有行的数据，只读取指定关键点的xyz列
    colume_index = 52 + landmarks_index * 3  # 52为blendshapes结束，landmarks_index从0开始
    data = []
    for line in selected_lines:
        vals = [float(x) for x in line.split(',')[colume_index:colume_index+3] if x.strip()]
        data.append(vals)
    data = np.array(data)

    if data.size == 0:
        print(f"文件 {filename} 解析后无数据，跳过。")
        return None

    num_frames, num_features = data.shape  # 行数为帧数，列数为特征数

    # 创建图形并保存，不在线显示
    plt.figure(figsize=(20, 8))  # 加长图片尺寸，增加宽度
    x = np.arange(num_frames)
    
    # 计算1个特征的距离（每个特征有3列xyz）
    num_landmarks = 1
    distances = None
    for i in range(num_landmarks):
        # 每个特征的xyz列索引（0-based: i*3, i*3+1, i*3+2）
        x_col = i * 3
        y_col = i * 3 + 1
        z_col = i * 3 + 2
        # 计算距离：sqrt(x^2 + y^2 + z^2)
        distances = np.sqrt(data[:, x_col]**2 + data[:, y_col]**2 + data[:, z_col]**2)
        # 移除原始折线绘制
    
    # 分段拟合贝塞尔曲线（使用Hermite插值，每segment_length帧一段，重叠1帧，确保C1连续性）
    overlap = 1
    step = segment_length - overlap
    num_segments = max(1, (num_frames - overlap) // step)
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
    segments = []
    prev_deriv = None
    for i in range(num_segments):
        start = i * step
        end = min(start + segment_length, num_frames)
        x_seg = x[start:end]
        y_seg = distances[start:end]
        if len(x_seg) > 1:
            # 使用PCHIP估算导数
            pchip = PchipInterpolator(x_seg, y_seg)
            dydx = pchip.derivative()(x_seg)
            if i > 0 and prev_deriv is not None:
                # 后续段的第一个点使用上一段最后一个点的导数
                dydx[0] = prev_deriv
            # 使用CubicHermiteSpline拟合
            interp = CubicHermiteSpline(x_seg, y_seg, dydx)
            segments.append((x_seg, y_seg, dydx, interp))
            x_fine = np.linspace(x_seg[0], x_seg[-1], 100)
            y_fine = interp(x_fine)
            plt.plot(x_fine, y_fine, color=colors[i % len(colors)], linewidth=2, label=f'Fitted segment {i+1}')
            # 保存上一段最后一个点的导数
            prev_deriv = dydx[-1]
    
    plt.title(f'{filename} landmark distance visualization (landmark {landmarks_index})')
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
    
    # 保存Hermite插值参数到txt文件（保存每段的x、y和导数数据，用于重建插值器）
    txt_path = os.path.join(save_dir, f'{filename}_hermite_params.txt')
    with open(txt_path, 'w') as f:
        for i, (x_seg, y_seg, dydx, interp) in enumerate(segments):
            f.write(f"Segment {i+1}:\n")
            f.write(f"x: {list(x_seg)}\n")
            f.write(f"y: {list(y_seg)}\n")
            f.write(f"dydx: {list(dydx)}\n")
            f.write("\n")
    
    print(f"已保存 {filename} 的可视化结果到 {save_path}")
    print(f"已保存 {filename} 的Hermite参数到 {txt_path}")
    
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
            landmarks_index = 13  # 左肘
            segment_length = 8
            result = process_distance_single_feature(file_path, landmarks_index, segment_length)
            if result:
                print(f"处理完成: {result}")
    
    print("所有文件处理完成。")
