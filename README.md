# HVCCS
Hologram Virtual Conference Communication System
## 配置虚拟环境
- 从官网安装anaconda3
- 运行以下命令创建虚拟环境
```
conda env create -f HVCCS.yaml
pip install -r requirements.txt
pip install mediapipe==0.10.20 --no-deps
```

## Unity & Quest 3 Setup

- Unity Hub安装unity h3.3.3-c2
- 在浏览器输入`unityhub://2022.3.55f1/9f374180d209`自动跳转到Unity Hub安装unity 2022.3.55f1，安装时选择`Android Build Support`和`Android SDK & NDK Tools`
- 新建HDRP项目（如果需要编译为app部署到Quest上，应该新建URP项目，因为Quest部署Anroid应用目前不支持HDRP），并进入。

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
- 在unity中打开`Window/Package Manager`，点击`+`按钮，选择`Add package from git URL...`，输入以下地址：
```
com.unity.webrtc@3.0.0-pre.5
```
- 点击`Add`，等待安装完成
- 在unity中打开`Window/Package Manager`，点击`+`按钮，选择`Add package by name...`，输入以下文字：
```
com.unity.renderstreaming
```
- 下方可选安装版本，输入`3.1.0-exp.6`点击`Add`，等待安装完成。安装包后会自动打开`Render Streaming Wizard`窗口。选择`Fix all`。
- 在`Render Streaming Wizard`窗口点击`Download latest version web app`，下载最新的web app。如果无法下载请访问[github](https://github.com/Unity-Technologies/UnityRenderStreaming)下载`WebApp`文件夹。
- 在unity中打开`Window/Package Manager`，找到`Unity Render Streaming`点击进入详情，找到`samples`，点击`Import`，等待完成。
- 在[nodejs](https://nodejs.org/en/)下载对应版本node.js并安装。
- 可以通过修改源码来修改传输端口和模式，源码在`/WebApp/src/index.ts`中。选中场景`HDRP/RenderStreaming`中`signaling manager`选择`open project settings`，在面板选择`create new setting assets`，在一个地方保存即可。然后再次选择`open project settings`，即可修改URL。
- node.js完成安装后进入`WebApp`文件夹，打开命令行窗口，输入以下命令安装依赖：
```bash
npm install
```
- Windows直接双击运行`run.bat`，linux在命令行运行`./run.sh`

#### 3.2 setup unity scene
- 在unity中打开`Assets/Samples/Unity Render Streaming/3.1.0-exp.6/Example/`，将里面子文件夹中的场景拖入`Hierarchy`中。
- 在`Assets`中新建`models`文件夹，将需要渲染的模型，如`nezha.fbx`放入该文件夹中。然后将模型拖入`Hierarchy/WebBrowserInput/`中。
- 在`Assets`中新建`scripts`文件夹，将`unity_cs_scripts`文件夹中的`BSCtrl.cs`和`FaceDataReceiver.cs`拖入该文件夹中。
- 选中`nezha`点击`Add Component`，搜索`FaceDataReceiver.cs`，添加该组件。
- 选中`nezha`点击`Add Component`，搜索`BSCtrl`，添加该组件。添加引用`head_m1`。
- 在`inspector`中，为`FaceDataReceiver.cs`添加引用，选中`nezha`。
- 进入`WebAPP`文件夹，运行`WebApp/run.bat`，启动web服务，可以看到ip地址。
- 在unity中点击`play`按钮，运行unity场景。
- 在本机浏览器中输入`127.0.0.1`，或其他电脑浏览器中输入`run.bat`运行后显示的sigaling server的ip地址，可以看到unity渲染的画面。


## Python Scripts Setup

- `HVCCS/fea_extr_py_scripts/`中存放了所有实时系统所需的python脚本。

### 1 Server

- **stpe1**: `tools/time_diff_cal_receiver.py`用于计算Sender和Server的本地时间差。将`RECEIVER_IP`改为Server本机IP地址，运行本程序。

- **step3**: 配置好unity项目，点击运行。

- **step4**： `fea_extr_py_scripts/grpc2socket.py`用于接收Sender的gprc协议推送的数据，通过socket协议转发给unity软件c#脚本`unity_cs_scripts/FaceDataReceiver.cs`。统计发送数据大小，通过Sender数据包中的时间戳，和unity返回的时间戳计算MTP时延。使用时首先为每个端口号对应的Sender设置时延校正，校正值通过`tools/time_diff_cal_receiver.py`获取。

### 2 Sender
- **step2**: `tools/time_diff_cal_sender.py`用于计算Sender和Server的本地时间差。修改`SENDER_IP`为本机IP地址，`RECEIVER_IP`为Server的IP地址。Sender是客户端，应该等Server启动后，再运行Sender。

- **step5**: `fea_extr_py_scripts/grpc_avatar_fea_sender.py`自动获取系统摄像头列表中的第一个摄像头，用于获取直播视频流。需要确保Sender和Server处于同一网络下，将Sender目标IP地址设置为Server的IP地址。为每个客户端分别设置端口号。Sender是客户端，应该等Server启动后，再运行Sender。

### Expected Effect
- 运行后，请确保身体距离Sender三米左右以确保整个身体进入画面。当看到`fea_extr_py_scripts/grpc_avatar_fea_sender.py`不断打印发送信息说明运行正常
- 在Server中弹出用户网络参数监控画面，Sender对应用户的统计表中不断更新折线图。
- unity中对应数字人做出相应动作。如果画面中未找到数字人，可以切换Scend窗口，双击左侧Hierarchy窗口的Avatar对象，视角会自动追踪到该数字人，通过修改偏移量可以调整位置（而不是数字人本身的位置和旋转）。

### 3 splines codec

本节对应脚本：
- `fea_extr_py_scripts/grpc_online_splines_sender.py`
- `fea_extr_py_scripts/grpc_online_splines_receiver.py`
- `checkpoints/grpc_online_splines_codec_config.json`

#### 3.1 发送端模块组成与工作顺序

`grpc_online_splines_sender.py`内部模块及执行顺序如下：

1. 配置加载模块
- 读取 `checkpoints/grpc_online_splines_codec_config.json`。
- 校验 `common/sender/receiver` 三个 section 是否存在。

2. 数据读取模块
- `iter_h36m_frames()` 支持 `.npy` 与文本输入。
- 输出统一为每帧 `num_keypoints x coord_dims`，默认 `17 x 3`。
- 在“数据输入残差编码模块之前”按 `fps` 做节拍控制（不是在编码结束后再 sleep），用于稳定统计编码/量化时延。

3. 输入打点模块（t0）
- 每帧进入残差编码前记录时间戳 `t0_ms`。
- `t0_ms` 会写入数据包元数据 `ext_desc`。

4. 残差编码模块
- 第1帧或满足 `i_frame_interval` 的帧作为 I 帧，直接发送当前姿态。
- 其余为 P 帧，发送 `当前帧 - 上一重建帧` 的残差。

5. 量化与熵编码模块
- 支持 I/P 帧独立量化开关：`quantize_i_frame`、`quantize_p_frame`。
- 量化：`float32 -> int16`（对应帧类型量化开关为 `true` 时）。
- 熵编码：可选 `zlib` 压缩（`entropy_enabled=true` 时）。
- 发送端会对“已编码载荷”做一次本地解码，再更新历史重建帧，保证与接收端重建基准一致。

6. 打包发送模块
- 将编码后的字节放入 gRPC 的 `limb_data`。
- 将 `kind/channel/timestamp/payload_dtype/entropy_codec/quant_scale/t0_ms` 等元数据写入 `ext_desc`（JSON）。

7. 发送缓存模块
- 发送缓存达到 `buffer_limit` 时等待，防止缓存爆满。

#### 3.2 接收端工作顺序

`grpc_online_splines_receiver.py`执行顺序：

1. 读取同一份 JSON 配置并启动 gRPC 服务。
2. 从接收缓冲区取包，统计时延与带宽。
3. 从 `ext_desc` 解析编码元数据。
4. 对 `limb_data` 进行反熵编码（若有）与反量化。
5. 若为 P 帧则执行残差解码：`重建帧 = 上一重建帧 + 当前残差`。
6. 残差解码完成后立即记录 `t1_ms`，并统计时延 `t1_ms - t0_ms`（包含编码、量化、解码路径）。
7. 将重建结果 reshape 为 `(num_keypoints, coord_dims)`。
8. 不区分通道，所有可解码帧统一追加保存，最终落盘为 `.npy`，形状为 `(T, num_keypoints, coord_dims)`。

#### 3.3 JSON 参数说明（含单位与取值范围）

配置文件：`checkpoints/grpc_online_splines_codec_config.json`

`common`：
- `packet_tag`：数据包标签；类型 `string`；建议固定为 `POSE_RES_V1`。
- `num_keypoints`：关键点数；类型 `int`；单位 `个`；范围 `>0`，当前任务为 `17`。
- `coord_dims`：每个关键点坐标维度；类型 `int`；单位 `维`；范围 `>0`，当前任务为 `3`。
- `i_frame_interval`：I 帧间隔；类型 `int`；单位 `帧`；范围 `>=1`（建议 10-120）。
- `quantize`：是否量化；类型 `bool`；取值 `true/false`。
- `quantize_i_frame`：I 帧是否量化；类型 `bool`；取值 `true/false`。
- `quantize_p_frame`：P 帧是否量化；类型 `bool`；取值 `true/false`。
- `quant_scale`：量化缩放系数；类型 `float`；单位 `无`；范围 `>0`（建议 100-5000）。
- `entropy_enabled`：是否启用 zlib 压缩；类型 `bool`；取值 `true/false`。
- `entropy_level`：zlib 压缩等级；类型 `int`；范围 `0-9`，值越大压缩率通常越高、CPU开销越大。

量化与包大小说明：
- 当 `entropy_enabled=false` 且数据类型固定为 `int16` 时，单帧字节数主要由“维度数量”决定，与 `quant_scale` 取值基本无关。
- `quant_scale` 主要影响量化误差，不直接改变定长 `int16` 载荷字节数。
- 若希望码率随数据分布变化，通常需要启用熵编码（如 `zlib`）或可变长编码策略。

`sender`：
- `feature_file`：输入特征文件路径；类型 `string`；支持 `.npy` 或文本。
- `server_addr`：接收端地址；类型 `string`；如 `127.0.0.1`。
- `port_num`：发送目标 gRPC 端口；类型 `int`；范围 `1-65535`。
- `fps`：发送帧率；类型 `float`；单位 `fps`；范围 `>0`（建议 1-240）。
- `start_col`：文本特征起始列；类型 `int`；单位 `列索引`；范围 `>=0`。
- `max_frames`：最大发送帧数；类型 `int`；单位 `帧`；`0` 表示不限制。
- `buffer_limit`：发送缓冲上限；类型 `int`；单位 `包`；范围 `>=1`。
- `channel`：发送通道号；类型 `int`；范围 `0-50`。
- `debug`：是否打印调试日志；类型 `bool`。

`receiver`：
- `grpc_port`：本地监听端口；类型 `int`；范围 `1-65535`。
- `report_interval`：统计打印周期；类型 `float`；单位 `秒`；范围 `>0`。
- `poll_interval`：缓冲轮询周期；类型 `float`；单位 `秒`；范围 `>0`。
- `debug`：是否打印逐包日志；类型 `bool`。
- `save_enabled`：是否保存解码结果；类型 `bool`。
- `save_max_frames`：最多保存帧数；类型 `int`；单位 `帧`；`0` 表示不限制。
- `save_dir`：解码结果目录；类型 `string`；相对路径默认相对于项目根。
- `save_file`：解码结果文件名；类型 `string`；建议后缀 `.npy`。

#### 3.4 运行方式

两个脚本都不再依赖命令行参数，直接运行即可：

```bash
cd HVCCS/fea_extr_py_scripts
python grpc_online_splines_receiver.py
python grpc_online_splines_sender.py
```

如果需要改发送速率、端口、是否压缩、保存位置等，只修改 `checkpoints/grpc_online_splines_codec_config.json` 即可。