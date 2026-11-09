# Copilot 活动进度记录

## 近期提交概览

| 提交 SHA | 日期 | 提交信息 |
|---------|------|--------|
| `2126f80` | 2026-04-30 20:40 | add residual codec to grpc2socket and avatar_fea_sender |
| `19d99c0` | 2026-04-30 15:20 | add splines fit function and upsamnple ufnction to grpc2socket |

---

## 提交 1：`19d99c0` — 项目初始化 + 样条拟合与上采样

**日期**：2026-04-30 15:20  
**作者**：weixin_46069645

### 主要内容

#### 项目整体初始化
首次提交包含了 HVCCS 项目的完整基础框架，涵盖以下模块：

- **`fea_extr_py_scripts/`**：核心特征提取与传输脚本
  - `grpc2socket.py`：gRPC 转 Socket 的中继服务，接收 gRPC Avatar 特征流后通过 TCP 转发到 Unity
  - `grpc_avatar_fea_sender.py`：整合姿势（Pose）与面部表情（Blendshapes）识别的发送器
  - `grpc_online_splines_sender.py` / `grpc_online_splines_receiver.py`：在线样条编解码发送接收端
  - `grpc_offline_splines_sender.py` / `grpc_offline_splines_receiver.py`：离线样条编解码发送接收端
  - `realtime_offline_splines_fit.py`：实时/离线样条拟合核心算法（Cubic Hermite）
  - `client.py` / `server.py`：gRPC 客户端与服务端
  - `THStreamData.py`：数据包结构定义（`THStreamDataPayload`、`THDataWarehouse`）
  - `splines_entropy_codec_train.py`：样条熵编码训练脚本
  - `splines_fit_train.py`：样条拟合模型训练

- **`checkpoints/`**：配置与模型权重
  - `grpc_online_splines_codec_config.json`：在线样条编解码配置
  - `grpc_offline_splines_codec_config.json`：离线样条编解码配置
  - `grpc_online_splines_entropy_codebook_q{4,6,8,10,12,14,16}.json`：各量化精度（4~16 bit）的 Huffman 熵编码码本
  - `pose_codec_config.json` / `pose_codec_config_v3.json`：姿势编解码配置
  - `splines_mamba_runs/train_gpu0_e1000/ckpt/best.pt`：Mamba 样条拟合最优模型权重

- **`unity_cs_scripts/`**：Unity C# 脚本
  - `BDCtrl_girl1.cs` / `BDCtrl_nezha.cs`：骨骼驱动控制器
  - `BSCtrl.cs`：BlendShape 控制器
  - `FaceDataReceiver.cs`：面部数据接收器
  - `PointCloud.cs` / `PointCloudDataReceiver.cs`：点云渲染与接收
  - `PointCloudCSHelper.cs`：点云辅助工具
  - `dracoreceiver.cs`：Draco 压缩点云接收

- **`Metrics/`**：质量评估指标
  - `avatar_metrics.py`：Avatar 渲染质量评估
  - `FDD_no_GT.py`：无 GT 的帧级失真度量
  - `pose_quality_report_30fps.csv` / `pose_quality_report_120fps.csv`：30fps / 120fps 姿势质量报告

- **`dataset/`**：数据集与处理脚本
  - `avatar_QoE/`：主观 QoE 评估数据集（包含 Avatar Net 和 PointCloud Net 权重）
  - `py_scripts/`：特征提取、预处理脚本
  - `cs_scripts/`：girl / nezha 数据集对应 C# 脚本

- **`tools/`**：工具脚本
  - `splines_fit.py` / `splines_metrics.py` / `splines_metrics_batch.py`：样条拟合精度评估
  - `plot_pose_metrics_batch.py` / `plot_splines_metrics_batch.py`：批量指标可视化
  - `MPJPE.py`：MPJPE（Mean Per Joint Position Error）计算
  - `ffmpeg_client.py` / `ffmpeg_server.py` / `ffmpeg_ui.py`：基于 FFmpeg 的视频流工具
  - `time_diff_cal_sender.py` / `time_diff_cal_receiver.py`：时延测量工具

- **`quantitative_error_compensation/`**：误差补偿分析
  - 多项可视化工具：距离/速度分析、单特征 Bézier/二次曲线分析、空间误差可视化

- **`Setup_GUI.py`**：系统配置 GUI

#### `grpc2socket.py` 新增功能：样条拟合上采样（StreamingSplinesUpsampler）

新增 `StreamingSplinesUpsampler` 类，实现实时帧间样条插值：

- **算法**：基于 Cubic Hermite 样条，利用历史加速度外推切线斜率
  - 首帧仅缓存，无输出
  - 第二帧起用前向差分估计切线
  - 第三帧起用二阶差分加速度外推：`v_k = (3*x_k - 4*x_{k-1} + x_{k-2}) / (2*Δt)`
- **接口**：`generate_upsampled_frames(curr_ts_ms, limb_data_str) → [(ts_ms, limb_csv), ...]`
- **上采样倍率**：可配置 `upsample_factor`（默认 1，即不上采样）
- **输出策略**：严格一帧延迟，只发送 `[k-1, k)` 区间内的插值点，避免右端点重复

---

## 提交 2：`2126f80` — 残差编解码器接入 grpc2socket 与 avatar_fea_sender

**日期**：2026-04-30 20:40  
**作者**：weixin_46069645

### 变更文件

| 文件 | 状态 | 变更规模 |
|------|------|--------|
| `checkpoints/grpc_online_avatar_fea_codec_config.json` | 新增 | 37 行 |
| `fea_extr_py_scripts/grpc2socket.py` | 修改 | +344 行 |
| `fea_extr_py_scripts/grpc_avatar_fea_sender.py` | 修改 | +920 行（重写） |

### 主要内容

#### 1. 新增编解码配置文件 `grpc_online_avatar_fea_codec_config.json`

针对 Avatar 特征（姿势 + 面部）定义残差编解码参数：

```json
{
  "common": {
    "packet_tag": "AVATAR_FEA_RES_V1",
    "num_keypoints": 33,
    "coord_dims": 3,
    "face_dims": 52,
    "i_frame_interval": 30,
    "quantize_p_frame": true,
    "quant_bits": 12,
    "clip_abs": 256.0,
    "entropy_enabled": true,
    "entropy_codec": "huffman",
    "entropy_codebook_path": "checkpoints/grpc_online_splines_entropy_codebook_q12.json"
  }
}
```

- 支持 I 帧（关键帧）/ P 帧（残差帧）分离量化
- 姿势和面部 BlendShape 分别配置量化精度与剪裁范围
- 基于 Huffman 熵编码

#### 2. `grpc2socket.py` — 新增残差解码器基础设施

新增以下函数，负责在接收端对 Avatar 特征包进行解码：

| 函数 | 功能 |
|------|------|
| `load_runtime_config(config_path)` | 加载并校验编解码配置 JSON |
| `_is_within_root(path, root)` | 路径安全校验（防目录穿越） |
| `resolve_runtime_path(path, root)` | 将相对路径解析为安全的绝对路径 |
| `resolve_codebook_path_by_bits(path, bits)` | 按量化位数自动定位对应码本文件 |
| `load_huffman_codebook(path)` | 加载并校验 Huffman 码本 |
| `build_huffman_decode_tree(encode_map)` | 从编码映射构建 Huffman 解码树 |
| `dequantize_uniform(q_values, bits, clip)` | 均匀反量化 |
| `decode_payload(...)` | 核心解码函数，支持 `qint16` 和 `qidx_huff` 两种格式 |
| `parse_meta(ext_desc)` | 解析数据包元信息（帧类型、帧号等） |
| `build_decoder_context(config_path)` | 构建完整解码上下文（含 Huffman 树） |
| `decode_avatar_feature_payload(...)` | 端到端解码 Avatar 特征包，还原姿势和面部数据 |

**解码流程**：
1. 解析 `ext_desc` 元信息（帧类型：I/P 帧，帧编号）
2. I 帧：直接反序列化为浮点坐标
3. P 帧：Huffman 解码 → 反量化 → 加上前一帧（残差还原）
4. 输出归一化姿势关键点（33×3）和面部 BlendShape（52 维）

#### 3. `grpc_avatar_fea_sender.py` — 完整重写为带残差编码的发送器

原版本为简单姿势平滑发送器，新版本全面升级：

**新增编码基础设施**：

| 函数/类 | 功能 |
|---------|------|
| `load_runtime_config` / `resolve_runtime_path` | 配置加载与路径安全 |
| `load_huffman_codebook` | 加载 Huffman 码本 |
| `quantize_uniform(values, bits, clip)` | 均匀量化 |
| `huffman_encode_symbols(symbols, encode_map)` | Huffman 编码，返回字节流和有效 bit 长度 |
| `build_huffman_decode_tree` | 构建 Huffman 解码树 |
| `encode_payload(...)` | 核心编码：支持 I 帧直接量化和 P 帧残差+Huffman |
| `decode_payload_local(...)` | 本地解码（用于编码验证） |
| `build_pose_codec_context(config_path)` | 构建完整编解码上下文 |

**`FrameDataManager` 类重写**：

- 新增 `_encode_pose_with_codec(pose_bytes, timestamp_ms)`：
  - 按 `i_frame_interval` 决定是否发送 I 帧或 P 帧
  - P 帧计算残差（当前帧 - 前一帧）后量化 + Huffman 编码
  - 将编码结果打包进 `ext_desc` 元信息（包含帧类型、帧号、量化参数、bit 长度）
- 新增 `_encode_face_with_codec(face_bytes, timestamp_ms)`：
  - 同样支持 I/P 帧编码
  - 面部 BlendShape 52 维独立编码
- `_try_send_complete_frame`：收集完整的姿势+面部编码包后发送

**编码包格式（ext_desc 元信息）**：
```
{frame_type}|{frame_no}|{pose_dtype}|{pose_quant_bits}|{pose_clip_abs}|{pose_bit_len}|{face_dtype}|{face_quant_bits}|{face_clip_abs}|{face_bit_len}
```

---

## 总结

| 阶段 | 完成内容 |
|------|--------|
| **基础框架搭建** | 项目整体代码库初始化，覆盖 gRPC 传输、样条编解码、Unity 渲染、QoE 评估 |
| **样条上采样** | `grpc2socket.py` 实现 Cubic Hermite 实时插值上采样，支持任意倍率 |
| **残差编解码（发送端）** | `grpc_avatar_fea_sender.py` 完整重写，支持 I/P 帧残差编码 + Huffman 熵编码 |
| **残差编解码（接收端）** | `grpc2socket.py` 新增完整解码管线，支持 Huffman 解码 + 反量化 + 残差还原 |
| **配置体系** | 新增 `grpc_online_avatar_fea_codec_config.json`，统一管理编解码参数 |
