# 数据集收集
- `HVCCS/dataset/`有所有用于数据集收集的工具。
- 目录结构如下:
```
dataset
    py_scripts //处理特征的python脚本
        fea_extra.py // 1. 从../vidoe/中提取特征，保存到`../ori_features`
        fea_proc.py // 2. 对`../ori_features`中原始特征量化编码和网络损伤，保存到`../proc_features`。
        delete_useless_files.py // 3. 删除12个XXXX_idxx_32b_l0.txt文件
        rename.py //4. 对`../proc_features`重命名，添加4位随机视频序号用于控制播放顺序
        contra_rename.py // 5. 按照`../proc_features`找到对应原始特征，重命名存到`contra_features`
    contra_features //与proc_features中一一对应的未经处理的features
        xxxx_idxx.txt
        ...
    cs_scripts //unity使用的csharp脚本
        girls_scripts //不同人物模型控制脚本不同，这是女孩模型的脚本
            BDCtrl_dataset.cs //控制人物肢体动作，需要手动绑定人物角色`girl1`
            BSCtrl_dataset.cs //控制人物面部表情，需要手动绑定人物头部模型e.g. `face6`
            Dataloader_dataset.cs //从项目本文件夹中的`Assets/proc_dataset`文件夹读取.txt文件
        nezha_scripts //哪吒的脚本文件夹
            ...
    ori_features //从视频提取的原始特征
    proc_features // 对原始特征量化编码和网络损伤
    videos //原始视频
```
- `girl_scripts`中的个csharp脚本复制到`Assets/Scripts/`文件夹下。
- 建立两个人物模型`girl1`和`girl2`，点击`Add Component`找到`Scripts`文件夹，将三个脚本添加到模型。选中模型，在inspector面板中设置各个参数。
- 按顺序运行四个python脚本，然后将`proc_features`文件夹和`contra_features`文件夹复制到unity项目`Assets/`文件夹中。