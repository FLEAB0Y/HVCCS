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

#### 1.3 Avatar Setup
- 世界坐标系中，x是左右，y是上下，z是前后。
- `nezha`的骨骼结构是
```
Hips # x是-左+右，y是-前+后，z是+上-下
    other # x是+左-右，y是+上-下，z是+前-后
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
                    Bip001 Neck
                        Bip001 Head
                    Bip001 R Clavicle
                        Bip001 R UpperArm
                            Bip001 R Forearm
                                Bip001 R Hand
                    
```

### 2 Unity Render Streaming & WebRTC
#### 2.1 prepare
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

#### 2.2 setup unity scene
- 在unity中打开`Assets/Samples/Unity Render Streaming/3.1.0-exp.6/Example/`，将里面子文件夹中的场景拖入`Hierarchy`中。
- 在`Assets`中新建`models`文件夹，将需要渲染的模型，如`nezha.fbx`放入该文件夹中。然后将模型拖入`Hierarchy/WebBrowserInput/`中。
- 在`Assets`中新建`scripts`文件夹，将`unity_cs_scripts`文件夹中的`BSCtrl.cs`和`FaceDataReceiver.cs`拖入该文件夹中。
- 选中`nezha`点击`Add Component`，搜索`FaceDataReceiver.cs`，添加该组件。
- 选中`nezha`点击`Add Component`，搜索`BSCtrl`，添加该组件。添加引用`head_m1`。
- 在`inspector`中，为`FaceDataReceiver.cs`添加引用，选中`nezha`。
- 进入`WebAPP`文件夹，运行`WebApp/run.bat`，启动web服务，可以看到ip地址。
- 在unity中点击`play`按钮，运行unity场景。
- 在本机浏览器中输入`127.0.0.1`，或其他电脑浏览器中输入`run.bat`运行后显示的sigaling server的ip地址，可以看到unity渲染的画面。