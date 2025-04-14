# HVCCS
Hologram Virtual Conference Communication System
## 配置虚拟环境
- 从官网安装anaconda3
- 运行`conda env create -f face_detec.yaml`来创建虚拟环境

## 配置unity渲染和推流
### 1 Setup Unity
- 安装unity hub 3.3.3-c2
- 安装unity 2022.3.55f1
- 新建项目，并进入

### 2 安装依赖
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
- node.js完成安装后进入`WebApp`文件夹，打开命令行窗口，输入以下命令安装依赖：
```bash
npm install
```
- 可以通过修改源码来修改传输端口和模式，源码在`/WebApp/src/index.ts`中

### 3 配置unity场景
- 在unity中打开`Assets/Samples/Unity Render Streaming/3.1.0-exp.6/Example/`，将里面子文件夹中的场景拖入`Hierarchy`中。
- 在`Assets`中新建`models`文件夹，将需要渲染的模型，如`nezha.fbx`放入该文件夹中。然后将模型拖入`Hierarchy/WebBrowserInput/`中。
- 在`Assets`中新建`scripts`文件夹，将`unity_cs_scripts`文件夹中的`BSCtrl.cs`和`FaceDataReceiver.cs`拖入该文件夹中。
- 选中`nezha`点击`Add Component`，搜索`FaceDataReceiver.cs`，添加该组件。
- 选中`nezha`点击`Add Component`，搜索`BSCtrl`，添加该组件。添加引用`head_m1`。
- 在`inspector`中，为`FaceDataReceiver.cs`添加引用，选中`nezha`。
- 进入`WebAPP`文件夹，运行`WebApp/run.bat`，启动web服务，可以看到ip地址。
- 在unity中点击`play`按钮，运行unity场景。
- 在本机浏览器中输入`127.0.0.1`，或其他电脑浏览器中输入`run.bat`运行后显示的sigaling server的ip地址，可以看到unity渲染的画面。