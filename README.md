# HVCCS
Hologram Virtual Conference Communication System
## 配置虚拟环境
- 从官网安装anaconda3
- 运行以下命令创建虚拟环境
```
conda env create -n face_detec -f HVCCS.yaml
conda activate face_detec
```

`HVCCS.yaml` 已包含实时链路的 Python 依赖；仓库当前没有单独的
`requirements.txt`，不需要再执行额外的 `pip install`。如果 YAML 中的
`prefix` 指向了另一台机器上的路径，请删除该行后再创建环境。

## Unity & Quest 3 Setup

- 安装 Unity Hub，并通过 Unity Hub 安装 Unity `2022.3.55f1`。
- 在浏览器输入`unityhub://2022.3.55f1/9f374180d209`自动跳转到Unity Hub安装unity 2022.3.55f1，安装时选择`Android Build Support`和`Android SDK & NDK Tools`
- 新建 HDRP 项目并进入；如果要编译 Android 应用部署到 Quest 3，使用 URP 项目。

### 1 Setup Meta Quest 3
#### 1.1 Install Link & MQDH
- 安装Meta Quest Link，登陆账号，连接Quest 3
- 安装Meta Quets Developer Hub，登录账号。在'setting'中选择ADB和NDK，选择安装Android SDK和NDK，找到unity安装目录下的`Android SDK`和`Android NDK`，选择对应的文件夹。

#### 1.2 Unity Quest 3 Setup

- 详情参考[Meta官方文档](https://developers.meta.com/horizon/documentation/unity/unity-tutorial-hello-vr)
- 点击`Edit`下拉中的`Project Setting`选项，找到左侧下拉列表最下方`XR Plugin Mnagement`，选择`install`
- Asset Store无法下载时，通过`Window/Package Manager`，点击`+`按钮，选择`Add package from name`，在[Meta官方文档](https://npm.developer.oculus.com/)中查看`com.meta.xr.sdk.core`和`com.meta.xr.sdk.interaction`，点击`Add`，等待安装完成。
- 在Meta XR Tools中，选择`Project Setup Tool`，点击`Fix All`和`Apply All`，等待完成。

### 2 Avatar Setup
#### 2.1 Avatar connects to c# scripts

- 将`unity_cs_scripts/BDCtrl_girl1.cs`，`/unity_cs_scripts/BSCtrl.cs`和`unity_cs_scripts/FaceDataReceiver.cs`的3个csharp脚本复制到`Assets/Scripts/`文件夹下。
- 建立多个人物模型`girl1`和`girl2`，注意其中用于绑定BSCtrl.cs脚本的骨骼`Face6`需要重命名为不同名字。点击`Add Component`找到`Scripts`文件夹，将三个脚本添加到模型。选中模型，在inspector面板中设置各个参数。
- 模型不能直接调整位置，需要通过inspector面板中的位置偏移调整位置。
- 为每个模型设置socket端口号，具体配置方式参见`fea_extr_py_scripts/grpc2socket.py`。

#### 2.2 Avatar armatures relationships
- 用于修改bug的骨骼参考，如无必要不用阅读。
- 世界坐标系中，x是左右，y是上下，z是前后。
- `nezha`的骨骼结构是
```
Hips # x是-左+右，y是-前+后，z是+上-下
    Spine # 其他骨骼，x是+左-右，y是+上-下，z是+前-后
        Spine1
            Spine2
                LeftShoulder
                    LeftArm
                        LeftForeArm
                            LeftHand
                                LeftHandThumb1
                                LeftHandIndex1
                                LeftHandMiddle1
                                LeftHandRing1
                                LeftHandPinky1
                                    LeftHandThumb2
                                    LeftHandIndex2
                                    LeftHandMiddle2
                                    LeftHandRing2
                                    LeftHandPinky2
                                        LeftHandThumb3
                                        LeftHandIndex3
                                        LeftHandMiddle3
                                        LeftHandRing3
                                        LeftHandPinky3                               
                Neck
                    Head
                RightShoulder
                    RightArm
                        RightForeArm
                            RightHand
                                RightHandThumb1
                                RightHandIndex1
                                RightHandMiddle1
                                RightHandRing1
                                RightHandPinky1
                                    RightHandThumb2
                                    RightHandIndex2
                                    RightHandMiddle2
                                    RightHandRing2
                                    RightHandPinky2
                                        RightHandThumb3
                                        RightHandIndex3
                                        RightHandMiddle3
                                        RightHandRing3
                                        RightHandPinky3
    LeftUpLeg
        LeftLeg
            LeftFoot
                LeftToeBase
    RightUpLeg
        RightLeg
            RightFoot
                RightToeBase
```
- `t-pose`的骨骼结构是
```
Bip001 Pelvis # x是-前+后，y是+左-右，z是+上-下
    Bip001 L Thigh # 其他骨骼x是-上+下，y是+前-后，z是+左-右
        Bip001 L Calf
            Bip001 L Foot
                Bip001 L Toe0
    Bip001 R Thigh
        Bip001 R Calf
            Bip001 R Foot
                Bip001 R Toe0
    Bip001 Spine
        Bip001 Spine1
            Bip001 Spine2
                Bip001 Spine3
                    Bip001 L Clavicle
                        Bip001 L UpperArm
                            Bip001 L Forearm
                                Bip001 L Hand
                                    Bip001 L Finger0
                                    Bip001 L Finger1
                                    Bip001 L Finger2
                                    Bip001 L Finger3
                                    Bip001 L Finger4
                    Bip001 Neck
                        Bip001 Head
                    Bip001 R Clavicle
                        Bip001 R UpperArm
                            Bip001 R Forearm
                                Bip001 R Hand
                                    Bip001 R Finger0
                                    Bip001 R Finger1
                                    Bip001 R Finger2
                                    Bip001 R Finger3
                                    Bip001 R Finger4
                    
```

### 3 Unity Render Streaming & WebRTC
#### 3.1 prepare
- Unity Render Streaming 是浏览器查看 Unity 画面的链路，与 gRPC 特征传输和
  Unity 端口（8890、9890 等）相互独立。先在 Unity 2022.3.55f1 项目中打开
  `Window/Package Manager`。
- 选择 `Add package by name...`，添加 WebRTC 包：
```
Name: com.unity.webrtc
Version: 3.0.0-pre.5
```
- 再次选择 `Add package by name...`，添加 Unity Render Streaming：
```
Name: com.unity.renderstreaming
Version: 3.1.0-exp.6
```
- 安装后打开 `Render Streaming Wizard`，执行 `Fix all`。
- 在`Render Streaming Wizard`窗口点击`Download latest version web app`，下载最新的web app。如果无法下载请访问[github](https://github.com/Unity-Technologies/UnityRenderStreaming)下载`WebApp`文件夹。
- 在 `Window/Package Manager` 中打开 `Unity Render Streaming` 的 `Samples`，导入示例场景。
- 在[nodejs](https://nodejs.org/en/)下载对应版本node.js并安装。
- 进入下载的 `WebApp` 目录安装依赖：
```bash
npm install
```
- Windows 运行 `run.bat`；macOS/Linux 运行 `./run.sh`。如果脚本没有执行权限，先运行 `chmod +x run.sh`。
- 在 Unity 场景的 `Signaling Manager` 中打开项目设置并配置 signaling URL；WebApp 的
  signaling 端口和浏览器访问地址以 `run.bat`/`run.sh` 的输出为准，不要与 8890/9890
  的数字人数据和反馈端口混用。

#### 3.2 setup unity scene
- 在unity中打开`Assets/Samples/Unity Render Streaming/3.1.0-exp.6/Example/`，将里面子文件夹中的场景拖入`Hierarchy`中。
- 在`Assets`中新建`models`文件夹，将需要渲染的模型，如`nezha.fbx`放入该文件夹中。然后将模型拖入`Hierarchy/WebBrowserInput/`中。
- 当前 LHZ 实时链路推荐只挂载 `real_time_lhz_cs_scripts/RealTimeLhzAvatarEndpoint.cs`。
  将脚本复制到 `Assets/Scripts/`，把 `Avatar Root`、`Face Renderer`、数据端口、反馈端口、
  `Clock Offset Ms` 和 Grounding 选项配置好；同一个输入流只连接一个数字人端点。
- 旧版 `unity_cs_scripts/FaceDataReceiver.cs + BSCtrl.cs` 仍可用于旧场景，但不要与
  `RealTimeLhzAvatarEndpoint` 同时挂载到同一个数字人，否则会重复读取和驱动骨骼。
- 进入`WebAPP`文件夹，运行`WebApp/run.bat`，启动web服务，可以看到ip地址。
- 在unity中点击`play`按钮，运行unity场景。
- 在本机浏览器中输入`127.0.0.1`，或其他电脑浏览器中输入`run.bat`运行后显示的sigaling server的ip地址，可以看到unity渲染的画面。


## Python Scripts Setup

- `HVCCS/fea_extr_py_scripts/`中存放了实时链路所需的 Python 脚本。先在项目根目录
  激活 `face_detec` 环境，并确保 `data/pose_landmarker_full.task` 和
  `data/face_landmarker_v2_with_blendshapes.task` 存在。

### 1 Server

- **step1：启动 Unity。** 按上面的 Unity 场景说明运行 WebRTC 场景和
  `RealTimeLhzAvatarEndpoint`。端点监听数据端口 8890，向反馈端口 9890 发送时延反馈。
- **step2：校准跨机器时钟。** 如果 Sender 和 Server 不在同一台电脑，运行
  `tools/time_diff_cal_receiver.py`（Server）和 `tools/time_diff_cal_sender.py`（Sender），
  计算“Server 时间 − Sender 时间”的毫秒值。将该值填入 Unity 端点的 `Clock Offset Ms`。
  同机运行时填 `0`。旧版三字段 Unity 反馈才使用 Server 的 `--clock_offset_ms` 参数。
- **step3：启动 gRPC 转发。** 默认命令会监听 `50051..50055`，转发到 `8890..8894`，并保留
  `50055/8894` 作为点云端口：
  ```bash
  python fea_extr_py_scripts/grpc2socket.py --spline_upsample 4
  ```
  如果需要自定义端口，`--grpc_ports` 和 `--socket_ports` 必须同时提供且数量相同；默认映射
  的最后一项会作为点云端口。`--hide_point_cloud_ui` 只隐藏监控面板，不会改变协议端口。
- `grpc2socket.py` 通过持久 TCP 连接向 Unity 逐帧发送 UTF-8 文本：时间戳、52 个面部值和
  99 个姿势值；`--spline_upsample 4` 将 30 FPS 输入上采样到约 120 FPS。MTP 只统计原始输入帧。

### 2 Sender
- Sender 是客户端，必须在 Server 的 gRPC 端口启动后运行。默认读取
  `checkpoints/grpc_online_avatar_fea_codec_config.json` 中的 `sender.server_addr`、
  `sender.port_num` 和 `sender.fps`（当前默认 `127.0.0.1:50051`、30 FPS）：
  ```bash
  python fea_extr_py_scripts/grpc_avatar_fea_sender.py
  ```
- 远程 Server 或不同端口使用命令行覆盖：
  ```bash
  python fea_extr_py_scripts/grpc_avatar_fea_sender.py \
    --server_addr 192.168.1.11 --port_num 50051 --fps 30
  ```
- Sender 自动读取系统摄像头列表中的第一个摄像头；如模型文件不在 `data/`，通过
  `--pose_model_path` 和 `--face_model_path` 指定。`--codec_config_path` 可替换编解码配置。

实时链路的输入帧率由 `checkpoints/grpc_online_avatar_fea_codec_config.json` 中的
`sender.fps` 控制，默认是 `30.0` FPS；也可以用 `--fps` 临时覆盖。若使用
`grpc2socket.py --spline_upsample 4`，应保持输入为 30 FPS，使样条输出稳定为
约 120 FPS。`spline_upsample` 是相对输入帧率的倍数，并不是固定的目标帧率。
实时样条对 52 个面部通道和 99 个姿势通道组成的 151 维特征一起拟合；MTP
反馈缓存只登记原始输入帧时间戳，插值帧不会单独计入总时延。

端口可以通过命令行覆盖：`grpc2socket.py` 使用
`--grpc_ports 50051 --socket_ports 8890`，sender 使用 `--port_num 50051`
和 `--server_addr 127.0.0.1`。省略这些参数时，Server 使用代码中的默认映射
`50051→8890`、`50052→8891`、`50053→8892`、`50054→8893`、`50055→8894`；
Sender 使用配置文件中的 `sender.port_num`，配置缺失时回退到代码默认端口
`50051`。因此无参数运行仍可使用默认端口。

#### Expected Effect
- 运行后，请确保身体距离Sender三米左右以确保整个身体进入画面。当看到`fea_extr_py_scripts/grpc_avatar_fea_sender.py`不断打印发送信息说明运行正常
- 在Server中弹出用户网络参数监控画面，Sender对应用户的统计表中不断更新折线图。
- unity中对应数字人做出相应动作。如果画面中未找到数字人，可以切换Scend窗口，双击左侧Hierarchy窗口的Avatar对象，视角会自动追踪到该数字人，通过修改偏移量可以调整位置（而不是数字人本身的位置和旋转）。

### 3 splines codec（仿真实验）

本节是离线特征回放、残差编解码和样条拟合的仿真实验流程，**与前文的实时数字人链路无关**。实时数字人仍然使用 `grpc_avatar_fea_sender.py`、`grpc2socket.py` 和 Unity；不要用本节脚本替换实时链路。

当前仓库实际使用的实验入口是：

- `fea_extr_py_scripts/grpc_offline_splines_sender.py`：从 `.npy` 或文本特征文件读取姿势帧，按指定 `fps` 发送 gRPC 数据。
- `fea_extr_py_scripts/grpc_offline_splines_receiver.py`：接收数据、解码、统计，并对每个通道执行样条拟合后保存结果。
- `fea_extr_py_scripts/splines_entropy_codec_train.py`：根据实验数据训练 Huffman 熵编码码本，可选执行。
- `fea_extr_py_scripts/realtime_offline_splines_fit.py`：提供样条 predictor 和拟合函数，被 receiver 调用，也可用于后续离线拟合和指标计算。
- `checkpoints/grpc_offline_splines_codec_config.json`：sender、receiver 和码本训练的统一配置。

README 旧版本中提到的 `grpc_online_splines_sender.py` 和 `grpc_online_splines_receiver.py` 已从仓库删除，不要再按旧命令运行。

#### 3.1 仿真实验数据流

```text
特征文件（.npy / 文本）
    ↓
grpc_offline_splines_sender.py
    ├── I/P 帧残差编码
    ├── 可选量化
    └── Huffman 熵编码
    ↓ gRPC
grpc_offline_splines_receiver.py
    ├── Huffman 解码、反量化
    ├── P 帧残差重建
    ├── 按 channel 分组
    └── predictor 样条拟合
    ↓
res/ 下的 .npz 样条结果
```

sender 和 receiver 必须使用一致的 `common` 编解码参数、同一份码本和相同的 gRPC 端口。sender 会在输入残差编码前按 `sender.fps` 节拍发送；receiver 在收到数据后按 `receiver.fps` 执行拟合。receiver 默认在发送结束后空闲 `idle_timeout_sec` 秒自动停止并保存结果。

#### 3.2 配置文件

配置文件：`checkpoints/grpc_offline_splines_codec_config.json`

`common` 控制数据形状和编解码：

- `packet_tag`：数据包标签，通常为 `POSE_RES_V1`。
- `num_keypoints`、`coord_dims`：输入姿势形状；当前默认是 `17 x 3`。
- `i_frame_interval`：I 帧间隔，其他帧使用相对上一重建帧的 P 帧残差。
- `quantize`：量化总开关；未单独设置 I/P 帧开关时，作为它们的默认值。
- `quantize_i_frame`、`quantize_p_frame`：分别控制 I/P 帧是否量化。
- `quant_scale`：非熵编码路径的缩放系数。
- `quant_bits`、`clip_abs`：均匀量化的位数和裁剪范围。
- `entropy_enabled`、`entropy_codec`：当前实现支持 Huffman；启用时应设为 `true` 和 `huffman`。
- `entropy_codebook_path`：Huffman 码本路径，例如 `checkpoints/grpc_online_splines_entropy_codebook_q8.json`。

当前 receiver/sender 不支持 README 旧版本描述的 zlib 路径；代码对非 Huffman 熵编码会报错。

`sender` 控制输入和发送：

- `feature_file`：输入文件，支持 `.npy` 和文本；`.npy` 默认按 `(T, 17, 3)` 或可推断的扁平布局读取。
- `server_addr`、`port_num`：receiver 地址和 gRPC 端口。
- `fps`：发送节拍，例如 `30.0`。
- `start_col`：文本输入的起始列。
- `max_frames`：最大帧数，`0` 表示发送全部帧。
- `buffer_limit`：发送缓存上限。
- `channel`：通道号，范围为 `0-50`；多通道实验应为每个输入流设置不同的值。
- `debug`：是否打印逐包调试信息。

`receiver` 控制接收、拟合和保存：

- `grpc_port`：监听端口，应与 `sender.port_num` 一致。
- `report_interval`、`poll_interval`：统计和缓冲区轮询周期，单位为秒。
- `idle_timeout_sec`：发送结束后的自动退出等待时间；设为 `0` 或负数可关闭自动退出。
- `debug`：是否打印逐包解码和接收调试信息。
- `save_max_frames`：最多接收的帧数，`0` 表示不限制。
- `save_dir`：结果目录，相对路径以项目根目录为基准。
- `spline_fit_enabled`：是否执行样条拟合。
- `spline_save_file`：输出 `.npz` 文件名；多通道时会自动追加 `_ch<channel>`。
- `predictor_type`：样条 predictor，可选 `baseline`、`kalman`、`abg` 或 `mamba`。
- `fps`：样条拟合帧率；通常与 sender 的 `fps` 一致。
- `process_acc_var`、`measurement_var`、`init_pos_var`、`init_vel_var`：Kalman/Mamba 等 predictor 的噪声和初始方差参数。
- `alpha`、`beta`、`gamma`：`abg` predictor 参数。
- `mamba_checkpoint_path`、`mamba_history_len`、`mamba_cuda_device`：Mamba predictor 参数。

`train` 控制码本训练：

- `input_path`：用于统计符号分布的 `.npy` 文件或目录。若填写单个 `.npy` 文件，训练代码会改为扫描该文件所在目录中的 `.npy` 文件。
- `output_json`：码本输出路径。
- `quant_bits_list`：要训练的量化位数列表，例如 `[4, 6, 8, 10, 12, 14, 16]`。
- `clip_percentile`：计算 `clip_abs` 时使用的百分位数。
- `include_i_frames`：是否将 I 帧纳入码本训练。

修改 `num_keypoints`、`coord_dims`、量化位数、码本路径、端口或 predictor 参数时，应同步检查 sender 和 receiver 的配置是否仍然匹配。

#### 3.3 可选：训练熵编码码本

如果已有匹配的码本，可以直接运行 sender/receiver；只有在更换训练数据、量化位数或需要重新估计裁剪范围时，才运行码本训练：

```bash
conda activate face_detec
cd /Users/twz/demo_sys_user/HVCCS
python fea_extr_py_scripts/splines_entropy_codec_train.py \
  --config-path checkpoints/grpc_offline_splines_codec_config.json
```

训练结果由 `train.output_json` 和 `train.quant_bits_list` 决定。若要让训练器按 `train.clip_percentile` 重新估计 `clip_abs`，应先将 `common.clip_abs` 设为 `0`（或删除该字段）；当前配置中的正值会被直接复用。运行实验前，还应确认 `common.entropy_codebook_path` 能解析到与 `common.quant_bits` 一致的码本文件。

#### 3.4 手动运行仿真实验

sender 和 receiver 不依赖命令行参数，运行参数全部从 JSON 配置读取。必须先启动 receiver，再启动 sender，建议使用两个终端：

终端一：

```bash
conda activate face_detec
cd /Users/twz/demo_sys_user/HVCCS
python fea_extr_py_scripts/grpc_offline_splines_receiver.py
```

终端二：

```bash
conda activate face_detec
cd /Users/twz/demo_sys_user/HVCCS
python fea_extr_py_scripts/grpc_offline_splines_sender.py
```

receiver 会在 sender 结束并达到 `idle_timeout_sec` 后停止，然后将每个通道的样条结果保存到 `receiver.save_dir`。单通道输出使用 `receiver.spline_save_file`；多通道输出会按通道追加文件名。

#### 3.5 一键实验脚本

`test.sh` 封装了完整的仿真实验流程：

1. 训练熵编码码本。
2. 启动 offline receiver。
3. 延迟启动 offline sender。
4. 运行 `realtime_offline_splines_fit.py`。
5. 运行 `tools/splines_metrics.py`。
6. 运行 `tools/splines_metrics_batch.py`。

```bash
conda activate face_detec
cd /Users/twz/demo_sys_user/HVCCS
bash test.sh
```

运行前请检查 `grpc_offline_splines_codec_config.json` 中的 `sender.feature_file`、`train.input_path`、码本路径和输出目录；这些路径决定实验输入、码本和结果保存位置。
