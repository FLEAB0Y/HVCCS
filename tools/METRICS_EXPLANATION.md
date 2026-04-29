# 姿态样条评估与可视化指标（与当前代码同步）

本文档对应当前实现：
- `tools/splines_metrics_batch.py`
- `tools/plot_splines_metrics_batch.py`

记号：
- 文件（样本）索引：$i=1,\dots,N$
- 帧索引：$t=1,\dots,T_i$
- 关节数：$K=17$
- GT 关节坐标：$\mathbf{g}_{i,t,j}\in\mathbb{R}^3$，预测坐标：$\hat{\mathbf{p}}_{i,t,j}\in\mathbb{R}^3$

---

## 1. 评估对象与时间对齐

对每个文件 $i$：
1. 用预测样条在统一高帧率（`upsample_fps`，默认 120Hz）上重采样，得到 $\hat{\mathbf{p}}_{i,t,j}$。
2. 将 GT pose 插值到同一时间戳，得到 $\mathbf{g}_{i,t,j}$。
3. 仅在有效重叠时间戳上计算误差。

线性基线（`linear_*`）的当前实现为：
- 先将 **预测样条** 以低帧率（`linear_downsample_fps`，默认 30Hz）采样为控制点；
- 再做分段线性插值回评估时间戳。

---

## 2. 文件内指标（single-file）

### 2.1 MPJPE（mm）

逐帧关节平均误差：
$$
e_{i,t}=\frac{1}{K}\sum_{j=1}^{K}\left\|\hat{\mathbf{p}}_{i,t,j}-\mathbf{g}_{i,t,j}\right\|_2
$$

文件内统计：
$$
\text{V-MPJPE}_i=\frac{1}{T_i}\sum_t e_{i,t},\quad
\text{p95-MPJPE}_i=\operatorname{Perc}_{95}(\{e_{i,t}\}),\quad
\text{Max-MPJPE}_i=\max_t e_{i,t}
$$

### 2.2 MPJVE（mm/s）

速度误差按相邻帧差分：
$$
\mathbf{v}^{\text{gt}}_{i,t,j}=\frac{\mathbf{g}_{i,t+1,j}-\mathbf{g}_{i,t,j}}{\Delta t_t},\quad
\mathbf{v}^{\text{pr}}_{i,t,j}=\frac{\hat{\mathbf{p}}_{i,t+1,j}-\hat{\mathbf{p}}_{i,t,j}}{\Delta t_t}
$$

$$
e^v_{i,t}=\frac{1}{K}\sum_{j=1}^{K}\left\|\mathbf{v}^{\text{pr}}_{i,t,j}-\mathbf{v}^{\text{gt}}_{i,t,j}\right\|_2
$$

再取 $\text{V/p95/Max}$（与 MPJPE 同型）。

### 2.3 MPJPE\_BL（%）

对非根关节 $j\neq 0$，以 GT 骨长均值归一化：
$$
\bar L_{i,j}=\frac{1}{T_i}\sum_t\left\|\mathbf{g}_{i,t,j}-\mathbf{g}_{i,t,\pi(j)}\right\|_2
$$
$$
E^{\text{BL}}_{i,t,j}=\frac{\|\hat{\mathbf{p}}_{i,t,j}-\mathbf{g}_{i,t,j}\|_2}{\bar L_{i,j}}
$$

全局（文件内）对有效 $(t,j)$ 聚合，并乘 $100\%$，得到
$\text{V/p95/Max-MPJPE\_BL}_i$。

### 2.4 RTE（%）

先对根轨迹做刚体对齐（Kabsch）：
$$
\hat{\mathbf{r}}^*_{i,t}=\mathbf{R}_i\hat{\mathbf{r}}_{i,t}+\mathbf{t}_i
$$

GT 根轨迹路径长度：
$$
D_i=\sum_{t=1}^{T_i-1}\|\mathbf{r}^{\text{gt}}_{i,t+1}-\mathbf{r}^{\text{gt}}_{i,t}\|_2
$$

$$
\text{RTE}_i=\frac{1}{T_i}\sum_t\frac{\|\hat{\mathbf{r}}^*_{i,t}-\mathbf{r}^{\text{gt}}_{i,t}\|_2}{D_i}\times100\%
$$

### 2.5 Jitter（10 m/s³）

三阶差分 jerk：
$$
\mathbf{j}_{i,t,j}=\left(\hat{\mathbf{p}}_{i,t+3,j}-3\hat{\mathbf{p}}_{i,t+2,j}+3\hat{\mathbf{p}}_{i,t+1,j}-\hat{\mathbf{p}}_{i,t,j}\right)f^3
$$

先做关节均值，再做时间均值，并做单位换算：
$$
\text{Jitter}_i=\frac{1}{10}\cdot\frac{1}{1000}\cdot
\frac{1}{T_i-3}\sum_t\frac{1}{K}\sum_j\|\mathbf{j}_{i,t,j}\|_2
$$

对应键名：`jitter_10mps3_pose_upsampled`。

---

## 3. “文件内” 与 “文件间”统计口径

这是当前最重要的两层统计。

### 3.1 文件内（每个 sample_id）

在 `metrics_per_file.csv` 中，每个文件输出两行：
- `method = pred`
- `method = linear`

每行里的 `V/p95/Max`（如 `V-MPJPE(mm)`）是**该文件内部**按时间（或时间×关节）统计得到的值。

### 3.2 文件间（跨 sample_id）

在 `metrics_summary.json` 的 `per_file_stats` 中，对“文件内标量”再做一次跨文件统计：

设某指标的文件内标量为 $m_i$（例如 $m_i=\text{V-MPJPE}_i$），则：
$$
\text{mean}=\frac{1}{N}\sum_{i=1}^{N}m_i,\quad
\text{median}=\operatorname{Perc}_{50}(\{m_i\}),
$$
$$
\text{p95}=\operatorname{Perc}_{95}(\{m_i\}),\quad
\text{max}=\max_i m_i
$$

因此：
- `per_file_stats.xxx.mean/max/p95` 是**文件间统计**；
- `metrics_per_file.csv` 的 `V/p95/Max` 是**文件内统计**。

---

## 4. 可视化数学口径（plot_splines_metrics_batch.py）

每张图固定一个指标 `metric_name` 和一个跨文件统计 `stat_name ∈ {mean, median, p95, max}`。

### 4.1 纵轴

纵轴取 `metrics_summary.json -> per_file_stats[metric_name][stat_name]`。

### 4.2 横轴（Bitrate per Keypoint）

从 `res/pose_metrics_batch_test/baseline_q*/codec_metrics_summary.json` 读取：
$$
B_q^{(s)} = \text{bitrate\_kbps at q-level }q\text{ and stat }s
$$
再换算：
$$
X_q^{(s)} = \frac{B_q^{(s)}}{17}
$$
即 `Bitrate per Keypoint (kbps)`。

### 4.3 q16→q64 断轴缩略（仅显示变换）

令 $x_{16}=X_{q16}^{(s)}$, $x_{64}=X_{q64}^{(s)}$，压缩系数 $\alpha=0.08$。
对显示坐标做分段变换：
$$
\tilde x =
\begin{cases}
x, & x\le x_{16}\\
x_{16}+\alpha(x-x_{16}), & x>x_{16}
\end{cases}
$$

拟合与绘制在 $\tilde x$ 上进行；刻度文本仍显示原始 $x$（未压缩值）。

---

## 5. 当前用于汇总/绘图的核心键

`pred`：
- `mpjpe_pose_upsampled_mm`, `mpjpe_pose_upsampled_p95_mm`, `mpjpe_pose_upsampled_max_mm`
- `mpjve_pose_upsampled_mmps`, `mpjve_pose_upsampled_p95_mmps`, `mpjve_pose_upsampled_max_mmps`
- `bl_mpjpe_percent_pose_upsampled`, `bl_mpjpe_p95_percent_pose_upsampled`, `bl_mpjpe_max_percent_pose_upsampled`
- `rte_percent_pose_upsampled`, `jitter_10mps3_pose_upsampled`

`linear`：
- 对应同名 `linear_*` 键。

以上即为“最新计算 + 最新可视化”与代码一致的数学定义。
