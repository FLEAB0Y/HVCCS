# 姿态样条曲线质量指标说明

本文档描述 `splines_metrics_batch.py`（批量评估）与 `splines_metrics.py`（单文件可视化）中所有评估指标的数学定义、物理含义、好坏方向及典型取值范围。

---

## 评估框架

评估在以下两个场景下同时进行：

- **pred（预测）**：将预测样条曲线以高帧率（默认 120 Hz）重采样，与对应时间戳的 GT pose 比较。
- **linear（线性基准）**：将 GT pose 以低帧率（默认 30 Hz）降采样后做分段线性插值，再以同样的高帧率重采样，与 GT pose 比较。该基准反映了不使用任何曲线拟合时的朴素性能下界。

所有指标均在两种方法的**相同时间窗口**内计算，保证可比性。

---

## 一、MPJPE — 平均关节位置误差

### 定义

设序列共 $T$ 帧，$K = 17$ 个关节，各帧 GT 位置为 $\mathbf{g}_{t,j} \in \mathbb{R}^3$，预测位置为 $\hat{\mathbf{p}}_{t,j} \in \mathbb{R}^3$（单位：mm）。

每帧的平均关节误差为：

$$
e_t = \frac{1}{K} \sum_{j=1}^{K} \left\| \hat{\mathbf{p}}_{t,j} - \mathbf{g}_{t,j} \right\|_2
$$

三个统计量：

$$
\text{V-MPJPE} = \frac{1}{T}\sum_{t=1}^{T} e_t, \quad
\text{p95-MPJPE} = \text{Percentile}_{95}(\{e_t\}), \quad
\text{Max-MPJPE} = \max_t e_t
$$

### 含义

衡量预测姿态与 GT 之间的**绝对空间误差**（无任何归一化）。V-MPJPE 反映平均精度，p95 和 Max 反映误差的长尾分布，适合发现偶发性大误差帧。

### 方向与取值

- **越低越好**
- 优秀：< 10 mm；可接受：10–30 mm；较差：> 50 mm
- 该指标对坐标系的全局偏移敏感；若预测坐标系与 GT 存在系统性偏差，数值会虚高

---

## 二、MPJVE — 平均关节速度误差

### 定义

每帧的速度用相邻帧差分除以时间间隔 $\Delta t = 1/\text{fps}$ 估算：

$$
\mathbf{v}_{t,j} = \frac{\mathbf{p}_{t+1,j} - \mathbf{p}_{t,j}}{\Delta t}, \quad t = 1, \ldots, T-1
$$

每帧平均速度误差：

$$
e^v_t = \frac{1}{K}\sum_{j=1}^{K} \left\| \hat{\mathbf{v}}_{t,j} - \mathbf{v}_{t,j} \right\|_2
$$

三个统计量定义同 MPJPE：V（均值）、p95（95 百分位）、Max（最大值）。

### 含义

衡量预测姿态的**运动速度精度**。MPJPE 低但 MPJVE 高，说明预测在空间位置上接近 GT，但运动节奏（快慢变化）存在明显抖动或滞后。

### 方向与取值

- **越低越好**
- 优秀：< 50 mm/s；可接受：50–200 mm/s；较差：> 500 mm/s
- 线性插值的 MPJVE 通常高于样条，因为分段线性在控制点处存在速度不连续跳变

---

## 三、MPJPE_BL — 骨长归一化 MPJPE

### 定义

对每个关节 $j \neq 0$（根关节 hip 除外），以其父骨的平均长度作为归一化参考：

$$
\bar{L}_j = \frac{1}{T}\sum_t \left\| \mathbf{g}_{t,j} - \mathbf{g}_{t,\text{parent}(j)} \right\|_2
$$

归一化误差（无量纲）：

$$
E^{\text{BL}}_{t,j} = \frac{\left\| \hat{\mathbf{p}}_{t,j} - \mathbf{g}_{t,j} \right\|_2}{\bar{L}_j}
$$

全局 BL-MPJPE（以百分比表示）：

$$
\text{V-MPJPE\_BL} = \frac{1}{T(K-1)} \sum_{t,j \neq 0} E^{\text{BL}}_{t,j} \times 100\%
$$

p95 和 Max 统计量对所有有效的 $(t,j)$ 样本同样计算。

### 含义

将误差用**人体骨段长度**归一化，消除了不同数据集或不同身高人体之间的量纲差异。1% 表示误差约为该关节所在骨段长度的 1%，具有直观的相对意义。根关节（hip）因无父骨而排除在外。

### 方向与取值

- **越低越好**
- 优秀：< 5%；可接受：5–20%；较差：> 50%
- 上肢关节（肘、腕）因骨段短，绝对误差相同时百分比更高

---

## 四、RTE — 根节点平移误差

### 定义

取所有帧的根关节（hip，关节 0）3D 轨迹 $\{\mathbf{r}^{\text{GT}}_t\}$ 与 $\{\hat{\mathbf{r}}_t\}$。

**第一步：刚性对齐**（Kabsch 算法，旋转 + 平移，无缩放）

$$
\hat{\mathbf{r}}^*_t = \mathbf{R}\,\hat{\mathbf{r}}_t + \mathbf{t},
\quad \text{其中 } (\mathbf{R}, \mathbf{t}) = \arg\min \sum_t \left\| \mathbf{R}\,\hat{\mathbf{r}}_t + \mathbf{t} - \mathbf{r}^{\text{GT}}_t \right\|_2^2
$$

消除全局坐标系偏移和初始朝向差异，使比较聚焦于**轨迹形状误差**。

**第二步：计算 GT 根节点总位移（路径长度）**

$$
D = \sum_{t=1}^{T-1} \left\| \mathbf{r}^{\text{GT}}_{t+1} - \mathbf{r}^{\text{GT}}_t \right\|_2
$$

**第三步：归一化均值误差**

$$
\text{RTE} = \frac{1}{D} \cdot \frac{1}{T}\sum_{t=1}^{T} \left\| \hat{\mathbf{r}}^*_t - \mathbf{r}^{\text{GT}}_t \right\|_2 \times 100\%
$$

若 GT 几乎静止（$D \approx 0$），则 RTE 置为 NaN。

### 含义

RTE 衡量**根节点（全局位移）的轨迹跟踪精度**，归一化后与运动幅度无关。1% 表示平均误差为总行走/运动距离的 1%。刚性对齐确保测量的是相对轨迹形状而非坐标系对齐误差。

### 方向与取值

- **越低越好**
- 优秀：< 2%；可接受：2–10%；较差：> 20%
- 慢速动作（$D$ 小）时数值不稳定，参考意义降低
- 线性插值的 RTE 通常较低（无预测延迟），样条的优势体现在 MPJVE 和 Jitter

---

## 五、Jitter — 运动抖动

### 定义

抖动通过计算预测姿态的三阶差分（加速度的变化率，即 **jerk**）来量化。

对全身 $K$ 个关节的三维坐标，以帧率 $f$（Hz）计算离散 jerk：

$$
\mathbf{j}_{t,k} = \left(\hat{\mathbf{p}}_{t+3,k} - 3\hat{\mathbf{p}}_{t+2,k} + 3\hat{\mathbf{p}}_{t+1,k} - \hat{\mathbf{p}}_{t,k}\right) \cdot f^3, \quad t = 1,\ldots,T-3
$$

每帧全关节 jerk 范数均值：

$$
\bar{j}_t = \frac{1}{K}\sum_{k=1}^{K} \left\| \mathbf{j}_{t,k} \right\|_2
$$

最终指标（单位换算到 $10\,\text{m/s}^3$）：

$$
\text{Jitter} = \frac{1}{T-3}\sum_t \bar{j}_t \cdot \frac{1}{1000} \cdot \frac{1}{10} \quad [\text{单位: } 10\,\text{m/s}^3]
$$

其中除以 1000 将 mm/s³ 转换为 m/s³，再除以 10 得到以 $10\,\text{m/s}^3$ 为单位的数值（避免数值过大）。

### 含义

Jitter 衡量预测序列的**时间平滑性**，即运动是否自然流畅。较高的 Jitter 意味着存在高频抖动或帧间突变，在渲染虚拟人时会造成明显的视觉瑕疵。注意：Jitter 仅取决于预测序列本身，与 GT 无关。

### 方向与取值

- **越低越好**
- 优秀：< 0.5（单位 $10\,\text{m/s}^3$）；可接受：0.5–2；较差：> 5
- 线性插值在控制点处存在速度突变，jerk 理论上趋于无穷大（Dirac δ），因此实际数值通常远高于样条方法
- GT 的 Jitter 若通过同样方法计算，一般低于 0.3

---

## 六、统计聚合方式

对每个指标，同时报告三种统计量：

| 统计量 | 符号 | 含义 |
|--------|------|------|
| 均值 | V- | 所有时间帧/样本的平均，代表整体水平 |
| 95 百分位 | p95- | 排除最差 5% 后的上界，抗异常值干扰，反映绝大多数帧的表现 |
| 最大值 | Max- | 所有样本中的最差情况，用于发现极端误差帧 |

RTE 和 Jitter 仅报告一个标量（本身已是全局统计量）。

---

## 七、指标汇总

| 指标 | 单位 | 好坏方向 | 优秀阈值 | 可接受范围 |
|------|------|----------|----------|------------|
| MPJPE | mm | ↓ 越低越好 | < 10 mm | 10–30 mm |
| MPJVE | mm/s | ↓ 越低越好 | < 50 mm/s | 50–200 mm/s |
| MPJPE\_BL | % | ↓ 越低越好 | < 5% | 5–20% |
| RTE | % | ↓ 越低越好 | < 2% | 2–10% |
| Jitter | 10 m/s³ | ↓ 越低越好 | < 0.5 | 0.5–2 |

---

## 八、关节结构（H36M-17 骨架）

本项目使用 Human3.6M 数据集的 17 关节骨架。父子关系如下：

```
hip (0)
├── rhip (1) → rknee (2) → rfoot (3)
├── lhip (4) → lknee (5) → lfoot (6)
└── spine (7) → thorax (8)
                ├── neck (9) → head (10)
                ├── lshoulder (11) → lelbow (12) → lwrist (13)
                └── rshoulder (14) → relbow (15) → rwrist (16)
```

- 根关节 hip（0）无父节点，不参与 MPJPE\_BL 的归一化计算，但参与 RTE 和 Jitter 计算。
- MPJPE\_BL 的归一化骨长取关节到其父关节距离的时间均值。

---

## 九、典型场景解读

**样条 pred 优于 linear 基准的预期表现：**
- MPJPE：pred 应低于 linear（样条能拟合关节间运动规律）
- MPJVE：pred 应显著低于 linear（线性插值在控制点处有速度跳变）
- Jitter：pred 应远低于 linear（三次样条天然具有连续二阶导数）
- RTE：两者相近，取决于根节点的跟踪策略；若样条无法捕捉大幅移动，可能不如 linear

**若 pred 的 p95 或 Max 远大于 V（均值）：**
说明误差分布有较长尾部，存在少数极差帧，建议进一步分析这些帧的发生时刻（通常对应动作突变、遮挡等困难场景）。
