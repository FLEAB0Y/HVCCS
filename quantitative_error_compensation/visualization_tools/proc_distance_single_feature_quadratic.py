import os
import numpy as np
import matplotlib.pyplot as plt

# 获取当前脚本所在的绝对路径 (.../quantitative_error_compensation/visualization_tools)
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (.../quantitative_error_compensation)
project_root = os.path.dirname(script_dir)


def process_distance_single_feature(file_path, landmarks_index, segment_length):
    """
    处理单个特征文件中【一个指定关键点】的距离可视化，
    使用分段二次函数拟合，相邻段重叠 1 帧，并保证相邻段函数值和一阶导数连续。
    仅用于绘图与保存该特征的二次拟合参数。
    """
    # 定义保存目录（改名为 Quadratic）
    save_dir = os.path.join(project_root, 'res', 'single_feature_visualization_distance_Quadratic')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = os.path.basename(file_path)

    # 读取所有行
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [ln.rstrip('\n') for ln in f if ln.strip()]
    selected_lines = lines  # 读取全部行

    if not selected_lines:
        print(f"{filename} 未找到数据，跳过。")
        return None

    # 解析所有行的数据，只读取指定关键点的 xyz 列
    colume_index = 52 + landmarks_index * 3  # 52 为 blendshapes 结束，landmarks_index 从 0 开始
    data = []
    for line in selected_lines:
        parts = [x for x in line.split(',') if x.strip() != '']
        # 防止越界
        if len(parts) < colume_index + 3:
            continue
        vals = [float(x) for x in parts[colume_index:colume_index + 3]]
        data.append(vals)
    data = np.array(data, dtype=np.float64)

    if data.size == 0:
        print(f"文件 {filename} 解析后无数据，跳过。")
        return None

    num_frames, num_features = data.shape  # 行数为帧数，列数为特征数(这里应为3: xyz)

    # 计算一个特征点的距离（每个特征有3列 xyz）
    x_idx, y_idx, z_idx = 0, 1, 2
    distances = np.sqrt(
        data[:, x_idx] ** 2 +
        data[:, y_idx] ** 2 +
        data[:, z_idx] ** 2
    )

    x_all = np.arange(num_frames)

    # ==== 1. 分段：相邻段重叠 1 帧 ====
    step = segment_length - 1
    if step <= 0:
        print("segment_length 必须 >= 2")
        return None

    segments_x = []
    segments_y = []

    start = 0
    while True:
        end = start + segment_length
        if end > num_frames:
            break  # 不补最后不足一整段的尾巴
        x_seg = x_all[start:end]
        y_seg = distances[start:end]
        segments_x.append(x_seg)
        segments_y.append(y_seg)
        start += step  # 重叠 1 帧（下一段从 start+step 开始）

    num_segments = len(segments_x)
    if num_segments < 1:
        print(f"{filename} 帧数不足一段（segment_length={segment_length}），跳过。")
        return None

    # ==== 2. 构造全局线性方程 A * theta = d ====
    # 对每段有 3 个未知数：a_i, b_i, c_i
    num_params = num_segments * 3

    eq_rows = []
    rhs = []

    # 2.1 每段的“拟合点”约束
    for i in range(num_segments):
        x_seg = segments_x[i]
        y_seg = segments_y[i]
        for x_val, y_val in zip(x_seg, y_seg):
            row = np.zeros(num_params, dtype=np.float64)
            base = i * 3  # a_i 在 base, b_i 在 base+1, c_i 在 base+2
            row[base + 0] = x_val ** 2
            row[base + 1] = x_val
            row[base + 2] = 1.0
            eq_rows.append(row)
            rhs.append(y_val)

    # 2.2 相邻段的函数值、导数连续约束
    for i in range(num_segments - 1):
        x_end = segments_x[i][-1]        # 第 i 段的最后一帧
        x_next = segments_x[i + 1][0]    # 第 i+1 段的第一帧（通常和 x_end 相同）

        base_i = i * 3
        base_j = (i + 1) * 3

        # 函数值连续： y_i(x_end) - y_{i+1}(x_next) = 0
        row_val = np.zeros(num_params, dtype=np.float64)
        row_val[base_i + 0] = x_end ** 2
        row_val[base_i + 1] = x_end
        row_val[base_i + 2] = 1.0

        row_val[base_j + 0] = -x_next ** 2
        row_val[base_j + 1] = -x_next
        row_val[base_j + 2] = -1.0

        eq_rows.append(row_val)
        rhs.append(0.0)

        # 导数连续： y_i'(x_end) - y_{i+1}'(x_next) = 0
        row_deriv = np.zeros(num_params, dtype=np.float64)
        # y_i'(x_end) = 2*a_i*x_end + b_i
        row_deriv[base_i + 0] = 2.0 * x_end
        row_deriv[base_i + 1] = 1.0
        # -y_{i+1}'(x_next) = -(2*a_{i+1}*x_next + b_{i+1})
        row_deriv[base_j + 0] = -2.0 * x_next
        row_deriv[base_j + 1] = -1.0

        eq_rows.append(row_deriv)
        rhs.append(0.0)

    A = np.vstack(eq_rows)
    d_vec = np.array(rhs, dtype=np.float64)

    # 2.3 求解全局最小二乘
    coeffs_all, *_ = np.linalg.lstsq(A, d_vec, rcond=None)

    # 解析每一段的 (a, b, c)
    segments_params = []
    for i in range(num_segments):
        base = i * 3
        a = float(coeffs_all[base + 0])
        b = float(coeffs_all[base + 1])
        c = float(coeffs_all[base + 2])
        x_seg = segments_x[i]
        segments_params.append({
            'segment_index': i + 1,
            'start_frame': int(x_seg[0]),
            'end_frame': int(x_seg[-1]),
            'a': a,
            'b': b,
            'c': c
        })

    # ==== 3. 绘图（只绘制这个特征点的距离曲线及其分段二次拟合） ====
    plt.figure(figsize=(20, 8))
    colors = ['red', 'green', 'orange', 'purple', 'brown',
              'pink', 'gray', 'olive', 'cyan', 'magenta']

    # 原始距离曲线
    plt.plot(x_all, distances, color='lightgray', linewidth=1, label='Original distance')

    for i, seg in enumerate(segments_params):
        a, b, c = seg['a'], seg['b'], seg['c']
        x_seg = segments_x[i]
        # 为了平滑显示，在段内插值
        x_fine = np.linspace(x_seg[0], x_seg[-1], 100)
        y_fine = a * x_fine ** 2 + b * x_fine + c
        plt.plot(
            x_fine,
            y_fine,
            color=colors[i % len(colors)],
            linewidth=2,
            label=f'Segment {i + 1}'
        )

    plt.title(f'{filename} landmark distance visualization (landmark {landmarks_index})')
    plt.xlabel('Frames')
    plt.ylabel('Distance from origin')

    ticks = np.arange(0, num_frames, 5)
    labels = np.arange(1, num_frames + 1, 5)
    plt.xticks(ticks, labels)
    plt.gca().set_xticks(np.arange(num_frames), minor=True)
    plt.grid(True, which='both', axis='x')

    if num_segments <= 20:
        plt.legend(loc='upper right', fontsize='small')

    save_dir_img = save_dir
    save_path = os.path.join(save_dir_img, f'{filename}_landmark{landmarks_index}.png')
    plt.savefig(save_path, dpi=200)
    plt.close()

    # 保存该特征点的二次拟合参数
    txt_path = os.path.join(save_dir, f'{filename}_landmark{landmarks_index}_quadratic_params.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        for seg in segments_params:
            f.write(f"Segment {seg['segment_index']} (frames {seg['start_frame']} - {seg['end_frame']}):\n")
            f.write(f"a: {seg['a']}\n")
            f.write(f"b: {seg['b']}\n")
            f.write(f"c: {seg['c']}\n")
            f.write("\n")

    print(f"已保存 {filename} 的特征点 {landmarks_index} 可视化结果到 {save_path}")
    print(f"已保存 {filename} 的特征点 {landmarks_index} 二次拟合参数到 {txt_path}")

    return {
        'image_path': save_path,
        'params_path': txt_path
    }


# ====== 新增：对 151 维全部特征做分段二次拟合 + 上采样 ======

def _fit_quadratic_segments_1d(y, segment_length):
    """
    对单通道序列 y (长度 N) 做分段二次拟合，段长为 segment_length，段间重叠1帧，
    保证相邻段在重叠点的函数值与一阶导数连续。
    返回：
      segments_params: list[{start_frame, end_frame, a,b,c}]
    """
    y = np.asarray(y, dtype=np.float64)
    num_frames = len(y)
    x_all = np.arange(num_frames, dtype=np.float64)

    step = segment_length - 1
    if step <= 0:
        raise ValueError("segment_length 必须 >= 2")

    segments_x, segments_y = [], []
    start = 0
    while True:
        end = start + segment_length
        if end > num_frames:
            break
        x_seg = x_all[start:end]
        y_seg = y[start:end]
        segments_x.append(x_seg)
        segments_y.append(y_seg)
        start += step

    num_segments = len(segments_x)
    if num_segments < 1:
        raise ValueError(f"帧数不足一段 (segment_length={segment_length})")

    num_params = num_segments * 3
    eq_rows = []
    rhs = []

    # 拟合点约束
    for i in range(num_segments):
        x_seg = segments_x[i]
        y_seg = segments_y[i]
        for x_val, y_val in zip(x_seg, y_seg):
            row = np.zeros(num_params, dtype=np.float64)
            base = i * 3
            row[base + 0] = x_val ** 2
            row[base + 1] = x_val
            row[base + 2] = 1.0
            eq_rows.append(row)
            rhs.append(y_val)

    # 相邻段的函数值 & 导数连续
    for i in range(num_segments - 1):
        x_end = segments_x[i][-1]
        x_next = segments_x[i + 1][0]
        base_i = i * 3
        base_j = (i + 1) * 3

        # 函数值连续
        row_val = np.zeros(num_params, dtype=np.float64)
        row_val[base_i + 0] = x_end ** 2
        row_val[base_i + 1] = x_end
        row_val[base_i + 2] = 1.0
        row_val[base_j + 0] = -x_next ** 2
        row_val[base_j + 1] = -x_next
        row_val[base_j + 2] = -1.0
        eq_rows.append(row_val)
        rhs.append(0.0)

        # 导数连续
        row_deriv = np.zeros(num_params, dtype=np.float64)
        row_deriv[base_i + 0] = 2.0 * x_end
        row_deriv[base_i + 1] = 1.0
        row_deriv[base_j + 0] = -2.0 * x_next
        row_deriv[base_j + 1] = -1.0
        eq_rows.append(row_deriv)
        rhs.append(0.0)

    A = np.vstack(eq_rows)
    d_vec = np.array(rhs, dtype=np.float64)
    coeffs_all, *_ = np.linalg.lstsq(A, d_vec, rcond=None)

    segments_params = []
    for i in range(num_segments):
        base = i * 3
        a = float(coeffs_all[base + 0])
        b = float(coeffs_all[base + 1])
        c = float(coeffs_all[base + 2])
        x_seg = segments_x[i]
        segments_params.append({
            "start_frame": int(x_seg[0]),
            "end_frame": int(x_seg[-1]),
            "a": a,
            "b": b,
            "c": c,
        })
    return segments_params


def _eval_quadratic_segments_at_times(segments_params, t_tgt, fps):
    """
    已知分段二次参数 segments_params（按帧索引定义），在目标时间轴 t_tgt 上评估。
    fps: 源帧率，用于 t -> x(连续帧索引) 的映射: x = t * fps
    返回：y_tgt (len = len(t_tgt))
    """
    y_tgt = np.zeros_like(t_tgt, dtype=np.float64)
    for idx, t in enumerate(t_tgt):
        x = t * fps  # 连续帧坐标
        chosen = None
        for seg in segments_params:
            if seg["start_frame"] <= x <= seg["end_frame"]:
                chosen = seg
                break
        if chosen is None:
            if x < segments_params[0]["start_frame"]:
                chosen = segments_params[0]
            else:
                chosen = segments_params[-1]
        a, b, c = chosen["a"], chosen["b"], chosen["c"]
        y_tgt[idx] = a * x ** 2 + b * x + c
    return y_tgt


def upsample_all_151_features_quadratic(file_path, segment_length, src_fps, target_fps):
    """
    对【整个文件中的 151 维特征】做：
      1) 分段二次拟合（每一维单独拟合）
      2) 按 target_fps 在时间轴上重新采样
    结果保存到 quantitative_error_compensation\\upsample_features 目录。
    不参与绘图，只做数值上采样。
    """
    upsample_dir = os.path.join(project_root, 'upsample_features')
    os.makedirs(upsample_dir, exist_ok=True)

    filename = os.path.basename(file_path)

    # 读取所有行
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    if not lines:
        print(f"{filename} 未找到数据，跳过上采样(151维)。")
        return None

    # 现在明确：一行总共 151 维 = 52 blendshape + 33*3 landmarks
    FEATURE_START_COL = 0      # 从第 0 列开始读
    NUM_FEATURES_TOTAL = 151   # 整行 151 维

    data = []
    for line in lines:
        parts = [x for x in line.split(",") if x.strip() != ""]
        if len(parts) < FEATURE_START_COL + NUM_FEATURES_TOTAL:
            # 这一帧列数不够 151，跳过
            continue
        vals = [float(x) for x in parts[FEATURE_START_COL:FEATURE_START_COL + NUM_FEATURES_TOTAL]]
        data.append(vals)
    data = np.array(data, dtype=np.float64)

    if data.size == 0:
        print(f"文件 {filename} 解析后无 151 维特征数据，跳过上采样。")
        return None

    num_frames, num_feats = data.shape
    if num_feats != NUM_FEATURES_TOTAL:
        print(f"{filename} 中实际特征数 {num_feats} != 预期 151，跳过上采样。")
        return None

    # 时间轴（秒）
    t_src = np.arange(num_frames, dtype=np.float64) / float(src_fps)

    # 目标时间轴
    t_end = t_src[-1]
    dt = 1.0 / float(target_fps)
    num_target = int(np.round(t_end / dt)) + 1
    t_tgt = np.linspace(0.0, t_end, num_target, dtype=np.float64)

    data_tgt = np.zeros((len(t_tgt), num_feats), dtype=np.float64)

    for feat_idx in range(num_feats):
        y = data[:, feat_idx]
        try:
            seg_params = _fit_quadratic_segments_1d(y, segment_length)
            y_tgt = _eval_quadratic_segments_at_times(seg_params, t_tgt, fps=src_fps)
            data_tgt[:, feat_idx] = y_tgt
        except ValueError as e:
            print(f"{filename} 特征 {feat_idx} 拟合失败: {e}，采用线性插值替代。")
            data_tgt[:, feat_idx] = np.interp(t_tgt, t_src, y)

    # 保存：每行 151 维
    save_name = os.path.splitext(filename)[0] + f"_151features_quadratic_upsampled_{target_fps}fps.txt"
    save_path = os.path.join(upsample_dir, save_name)
    with open(save_path, "w", encoding="utf-8") as f:
        for row in data_tgt:
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")

    print(f"{filename} 的 151 维特征二次拟合上采样结果已保存到 {save_path}")
    return {
        "upsampled_path": save_path,
        "num_original_frames": num_frames,
        "num_upsampled_frames": len(t_tgt),
        "num_features": num_feats,
    }


# 示例调用
if __name__ == "__main__":
    # 定义特征文件夹路径
    contra_features_dir = os.path.join(project_root, 'features')

    # 配置参数
    landmarks_index = 13      # 只指定一个特征点用于绘图
    segment_length = 5        # 分段长度（绘图和拟合都用这个）
    src_fps = 30              # 原始帧率
    target_fps = 120          # 目标帧率

    for filename in os.listdir(contra_features_dir):
        if not filename.endswith('.txt'):
            continue

        file_path = os.path.join(contra_features_dir, filename)

        # 1) 保留：指定一个特征点的距离分段二次拟合 + 绘图
        result_vis = process_distance_single_feature(file_path, landmarks_index, segment_length)
        if result_vis:
            print(f"可视化处理完成: {result_vis}")

        # 2) 新增：对该文件中全部 151 维特征做二次拟合 + 上采样
        result_upsample_all = upsample_all_151_features_quadratic(
            file_path,
            segment_length=segment_length,
            src_fps=src_fps,
            target_fps=target_fps
        )
        if result_upsample_all:
            print(f"151 维特征上采样完成: {result_upsample_all}")

    print("所有文件的单特征绘图 + 151 维二次拟合上采样完成。")
