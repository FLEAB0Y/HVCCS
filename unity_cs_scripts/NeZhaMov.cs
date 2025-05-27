using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using UnityEngine;

//人体各个关节的列表
public enum PositionIndex:int
{
    Hips=0,
    Spine,
    Chest,
    UpperChest,
    LeftShoulder,
    LeftUpperArm,
    LeftLowerArm,
    LeftHand,
    LeftThumbIntermediate,
    LeftIndexIntermediate,
    LeftLittleIntermediate,
    RightShoulder,
    RightUpperArm,
    RightLowerArm,
    RightHand,
    RightThumbIntermediate,
    RightIndexIntermediate,
    RightLittleIntermediate,
    LeftUpperLeg,
    LeftLowerLeg,
    LeftFoot,
    RightUpperLeg,
    RightLowerLeg,
    RightFoot,
    Neck,
    LeftToes,
    RightToes,
    //……
}

public class NeZhaMov : MonoBehaviour
{
    public GameObject NeZha;//哪吒人物物体
    private Animator ani;//挂在哪吒上的动画组件
    
    [SerializeField] private FaceDataReceiver dataReceiver; // 引用FaceDataReceiver组件
    [SerializeField] private bool useRealTimeData = true; // 是否使用实时数据
    
    float[,] raw_position;//读取利用medipipe存储的关键点坐标txt文本文件
    int count;//总共采样的帧数
    List<string> lines;//读取txt的原始list
    Transform[] BodyPart;//avatar人体模型的各个身体部件
    Vector3 raw1;
    Vector3 raw2;
    Vector3 raw;
    Vector3 vec;
    Vector3 forward;//Hips的方向
    Vector3 LHN;//Left Hand Normal，左手的法向(LookRotation中为y方向)
    Vector3 RHN;//Right Hand Normal，右手的法向(LookRotation中为y方向)
    Vector3 gaze;//哪吒头部的朝向

    //一些部位的三维坐标
    Vector3 PosLeftUpperLeg;
    Vector3 PosRightUpperLeg;
    Vector3 PosLeftThumb;
    Vector3 PosLeftIndex;
    Vector3 PosRightThumb;
    Vector3 PosRightIndex;
    Vector3 PosLeftLittle;
    Vector3 PosRightLittle;
    Vector3 PosHead;

    //以下为人物的身体部位到其子部位的方向
    Vector3 LUL_vec;//LeftUpperLeg的z方向
    Vector3 RUL_vec;//RightUpperLeg
    Vector3 LLL_vec;//LeftLowerLeg
    Vector3 RLL_vec;//RightLowerLeg
    Vector3 LF_vec;//LeftFoot
    Vector3 RF_vec;//RightFoot
    Vector3 Sp_vec;//Spine
    Vector3 Ch_vec;//Chest
    Vector3 UC_vec;//UpperChest
    Vector3 LS_vec;//LeftShoulder
    Vector3 LUA_vec;//LeftUpperArm
    Vector3 LLA_vec;//LeftLowerArm
    Vector3 RS_vec;//RightShoulder
    Vector3 RUA_vec;//RightUpperArm
    Vector3 RLA_vec;//RightLowerArm
    Vector3 LT_vec;//LeftThumb
    Vector3 LI_vec;//LeftIndex
    Vector3 LL_vec;//LeftLittle
    Vector3 RT_vec;//RightThumb
    Vector3 RI_vec;//RightIndex
    Vector3 RL_vec;//RightLittle
    Vector3 N_vec;//Neck

    //以下为中间矩阵
    Quaternion MidHips;
    Quaternion MidLeftUpperLeg;
    Quaternion MidRightUpperLeg;
    Quaternion MidLeftLowerLeg;
    Quaternion MidRightLowerLeg;
    Quaternion MidLeftFoot;
    Quaternion MidRightFoot;
    Quaternion MidSpine;
    Quaternion MidChest;
    Quaternion MidUpperChest;
    Quaternion MidLeftShoulder;
    Quaternion MidRightShoulder;
    Quaternion MidLeftUpperArm;
    Quaternion MidRightUpperArm;
    Quaternion MidLeftLowerArm;
    Quaternion MidRightLowerArm;
    Quaternion MidLeftHand;
    Quaternion MidRightHand;
    Quaternion MidLeftThumb;
    Quaternion MidLeftIndex;
    Quaternion MidLeftLittle;
    Quaternion MidRightThumb;
    Quaternion MidRightIndex;
    Quaternion MidRightLittle;
    Quaternion MidNeck;
    float mag;
    int i;
    
    // 实时数据的当前帧
    private float[] currentFrameData;
    private bool hasNewData = false;
    private object dataLock = new object();
    
    // GUI相关变量
    [SerializeField] private bool showDebugGUI = true; // 是否显示调试GUI
    [SerializeField] private int guiMaxPoints = 5; // 显示的最大关键点数量
    private Rect guiWindowRect = new Rect(10, 10, 300, 400); // GUI窗口的位置和大小
    private Vector2 scrollPosition; // 滚动视图的位置
    private int selectedPoint = 0; // 选择显示的特定关键点
    private float dataRate = 0; // 数据接收频率
    private long lastTimestamp = 0; // 上次数据时间戳
    
    // Start is called before the first frame update
    void Start()
    {
        Debug.Log("【初始化】NeZhaMov开始初始化");
        
        // 如果没有指定FaceDataReceiver，尝试查找
        if (dataReceiver == null)
        {
            Debug.Log("【查找组件】未指定FaceDataReceiver，尝试查找");
            dataReceiver = FindObjectOfType<FaceDataReceiver>();
            if (dataReceiver == null)
            {
                Debug.LogWarning("【组件缺失】未找到FaceDataReceiver组件，将使用文件数据");
                useRealTimeData = false;
            }
            else
            {
                Debug.Log($"【组件找到】已找到FaceDataReceiver: {dataReceiver.gameObject.name}");
            }
        }
        else
        {
            Debug.Log($"【组件就绪】已通过Inspector指定FaceDataReceiver: {dataReceiver.gameObject.name}");
        }
        
        // 订阅肢体数据事件
        if (useRealTimeData && dataReceiver != null)
        {
            dataReceiver.OnLimbDataReceived += OnLimbDataReceived;
            Debug.Log("【事件注册】已订阅OnLimbDataReceived事件");
            
            // 初始化一个空的当前帧数据
            currentFrameData = new float[99]; // 33点 * 3坐标 = 99
            Debug.Log("【数据准备】已初始化currentFrameData数组，大小: 99");
        }
        
        if (!useRealTimeData)
        {
            // 从文件读取数据（原来的代码）
            lines = System.IO.File.ReadLines("Assets/Scripts/MotionFile.txt").ToList();
            count = lines.Count;
            i = 0;
            raw_position = new float[count, 99];
            
            for (int i = 0; i <= (count-1); i++)
            {
                string[] points = lines[i].Split(',');
                for (int j = 0; j <= 32; j++)
                {
                    raw_position[i, 0 + j * 3] = float.Parse(points[0 + (j * 3)]) / 100;
                    raw_position[i, 1 + j * 3] = float.Parse(points[1 + (j * 3)]) / 100;
                    raw_position[i, 2 + j * 3] = float.Parse(points[2 + (j * 3)]) / 300;
                }
            }
        }
        else
        {
            // 为实时数据准备一个单帧结构
            count = 1;
            i = 0;
            raw_position = new float[1, 99]; // 只需要一帧
        }
        
        BodyPart = new Transform[38]; // Unity中的身体关节
        ani = NeZha.GetComponent<Animator>();
        InitializeBodyParts();
        
        // 先求初始化的中间/对齐矩阵
        forward = TriangleNormal(BodyPart[0].position, BodyPart[18].position, BodyPart[21].position);
        InitializeAlignmentMatrices();
        
        Debug.Log("【初始化完成】NeZhaMov初始化完成");
    }
    
    private void InitializeBodyParts()
    {
        BodyPart[0] = ani.GetBoneTransform(HumanBodyBones.Hips);
        BodyPart[1] = ani.GetBoneTransform(HumanBodyBones.Spine);
        BodyPart[2] = ani.GetBoneTransform(HumanBodyBones.Chest);
        BodyPart[3] = ani.GetBoneTransform(HumanBodyBones.UpperChest);
        BodyPart[4] = ani.GetBoneTransform(HumanBodyBones.LeftShoulder);
        BodyPart[5] = ani.GetBoneTransform(HumanBodyBones.LeftUpperArm);
        BodyPart[6] = ani.GetBoneTransform(HumanBodyBones.LeftLowerArm);
        BodyPart[7] = ani.GetBoneTransform(HumanBodyBones.LeftHand);
        BodyPart[8] = ani.GetBoneTransform(HumanBodyBones.LeftThumbIntermediate);
        BodyPart[9] = ani.GetBoneTransform(HumanBodyBones.LeftIndexIntermediate);
        BodyPart[10] = ani.GetBoneTransform(HumanBodyBones.LeftLittleIntermediate);
        BodyPart[11] = ani.GetBoneTransform(HumanBodyBones.RightShoulder);
        BodyPart[12] = ani.GetBoneTransform(HumanBodyBones.RightUpperArm);
        BodyPart[13] = ani.GetBoneTransform(HumanBodyBones.RightLowerArm);
        BodyPart[14] = ani.GetBoneTransform(HumanBodyBones.RightHand);
        BodyPart[15] = ani.GetBoneTransform(HumanBodyBones.RightThumbIntermediate);
        BodyPart[16] = ani.GetBoneTransform(HumanBodyBones.RightIndexIntermediate);
        BodyPart[17] = ani.GetBoneTransform(HumanBodyBones.RightLittleIntermediate);
        BodyPart[18] = ani.GetBoneTransform(HumanBodyBones.LeftUpperLeg);
        BodyPart[19] = ani.GetBoneTransform(HumanBodyBones.LeftLowerLeg);
        BodyPart[20] = ani.GetBoneTransform(HumanBodyBones.LeftFoot);
        BodyPart[21] = ani.GetBoneTransform(HumanBodyBones.RightUpperLeg);
        BodyPart[22] = ani.GetBoneTransform(HumanBodyBones.RightLowerLeg);
        BodyPart[23] = ani.GetBoneTransform(HumanBodyBones.RightFoot);
        BodyPart[24] = ani.GetBoneTransform(HumanBodyBones.Neck);
        BodyPart[25] = ani.GetBoneTransform(HumanBodyBones.LeftToes);
        BodyPart[26] = ani.GetBoneTransform(HumanBodyBones.RightToes);
        BodyPart[27] = ani.GetBoneTransform(HumanBodyBones.LeftThumbDistal);
        BodyPart[28] = ani.GetBoneTransform(HumanBodyBones.RightThumbDistal);
        BodyPart[29] = ani.GetBoneTransform(HumanBodyBones.LeftIndexDistal);
        BodyPart[30] = ani.GetBoneTransform(HumanBodyBones.RightIndexDistal);
        BodyPart[31] = ani.GetBoneTransform(HumanBodyBones.LeftLittleDistal);
        BodyPart[32] = ani.GetBoneTransform(HumanBodyBones.RightLittleDistal);
        BodyPart[33] = ani.GetBoneTransform(HumanBodyBones.LeftMiddleIntermediate);
        BodyPart[34] = ani.GetBoneTransform(HumanBodyBones.RightMiddleIntermediate);
        BodyPart[35] = ani.GetBoneTransform(HumanBodyBones.LeftRingIntermediate);
        BodyPart[36] = ani.GetBoneTransform(HumanBodyBones.RightRingIntermediate);
        BodyPart[37] = ani.GetBoneTransform(HumanBodyBones.Head);
    }
    
    private void InitializeAlignmentMatrices()
    {
        // LowerBody
        MidHips = Quaternion.Inverse(BodyPart[0].rotation) * Quaternion.LookRotation(forward);
        MidLeftUpperLeg = Quaternion.Inverse(BodyPart[18].rotation) * Quaternion.LookRotation((BodyPart[18].position - BodyPart[19].position), forward);
        MidRightUpperLeg = Quaternion.Inverse(BodyPart[21].rotation) * Quaternion.LookRotation((BodyPart[21].position - BodyPart[22].position), forward);
        MidLeftLowerLeg = Quaternion.Inverse(BodyPart[19].rotation) * Quaternion.LookRotation((BodyPart[19].position - BodyPart[20].position), forward);
        MidRightLowerLeg = Quaternion.Inverse(BodyPart[22].rotation) * Quaternion.LookRotation((BodyPart[22].position - BodyPart[23].position), forward);
        MidLeftFoot = Quaternion.Inverse(BodyPart[20].rotation) * Quaternion.LookRotation((BodyPart[20].position - BodyPart[25].position), forward);
        MidRightFoot = Quaternion.Inverse(BodyPart[23].rotation) * Quaternion.LookRotation((BodyPart[23].position - BodyPart[26].position), forward);

        // UpperBody
        MidSpine = Quaternion.Inverse(BodyPart[1].rotation) * Quaternion.LookRotation((BodyPart[1].position - BodyPart[2].position), forward);
        MidChest = Quaternion.Inverse(BodyPart[2].rotation) * Quaternion.LookRotation((BodyPart[2].position - BodyPart[3].position), forward);
        MidUpperChest = Quaternion.Inverse(BodyPart[3].rotation) * Quaternion.LookRotation((BodyPart[3].position - BodyPart[24].position), forward);
        MidLeftShoulder = Quaternion.Inverse(BodyPart[4].rotation) * Quaternion.LookRotation((BodyPart[4].position - BodyPart[5].position), forward);
        MidLeftUpperArm = Quaternion.Inverse(BodyPart[5].rotation) * Quaternion.LookRotation((BodyPart[5].position - BodyPart[6].position), forward);
        MidLeftLowerArm = Quaternion.Inverse(BodyPart[6].rotation) * Quaternion.LookRotation((BodyPart[6].position - BodyPart[7].position), forward);
        
        // 左手
        LHN = TriangleNormal(BodyPart[7].position, BodyPart[9].position, BodyPart[8].position);
        MidLeftHand = Quaternion.Inverse(BodyPart[7].rotation) * Quaternion.LookRotation((BodyPart[8].position - BodyPart[9].position), LHN);
        
        // 右身体部分
        MidRightShoulder = Quaternion.Inverse(BodyPart[11].rotation) * Quaternion.LookRotation((BodyPart[11].position - BodyPart[12].position), forward);
        MidRightUpperArm = Quaternion.Inverse(BodyPart[12].rotation) * Quaternion.LookRotation((BodyPart[12].position - BodyPart[13].position), forward);
        MidRightLowerArm = Quaternion.Inverse(BodyPart[13].rotation) * Quaternion.LookRotation((BodyPart[13].position - BodyPart[14].position), forward);
        
        // 右手
        RHN = TriangleNormal(BodyPart[14].position, BodyPart[15].position, BodyPart[16].position);
        MidRightHand = Quaternion.Inverse(BodyPart[14].rotation) * Quaternion.LookRotation((BodyPart[15].position - BodyPart[16].position), RHN);
        
        // 手指
        MidLeftThumb = Quaternion.Inverse(BodyPart[8].rotation) * Quaternion.LookRotation((BodyPart[8].position - BodyPart[27].position), LHN);
        MidLeftIndex = Quaternion.Inverse(BodyPart[9].rotation) * Quaternion.LookRotation((BodyPart[9].position - BodyPart[29].position), LHN);
        MidLeftLittle = Quaternion.Inverse(BodyPart[10].rotation) * Quaternion.LookRotation((BodyPart[10].position - BodyPart[31].position), LHN);
        MidRightThumb = Quaternion.Inverse(BodyPart[15].rotation) * Quaternion.LookRotation((BodyPart[15].position - BodyPart[28].position), RHN);
        MidRightIndex = Quaternion.Inverse(BodyPart[16].rotation) * Quaternion.LookRotation((BodyPart[16].position - BodyPart[30].position), RHN);
        MidRightLittle = Quaternion.Inverse(BodyPart[17].rotation) * Quaternion.LookRotation((BodyPart[17].position - BodyPart[32].position), RHN);

        // 颈部
        gaze = forward;
        MidNeck = Quaternion.Inverse(BodyPart[24].rotation) * Quaternion.LookRotation(gaze);
    }
    
    // 处理接收到的肢体数据
    private void OnLimbDataReceived(float[] limbData, long timestamp)
    {
        // 添加详细调试信息
        Debug.Log($"【接收数据】收到肢体数据，长度: {limbData?.Length}, 时间戳: {timestamp}");
        
        // 输出前3个数据项样本(如果存在)
        if (limbData != null && limbData.Length > 0)
        {
            string sampleData = "数据样本: ";
            for (int i = 0; i < Math.Min(3, limbData.Length); i++)
            {
                sampleData += $"[{i}]={limbData[i]} ";
            }
            Debug.Log(sampleData);
        }

        // 每个关节点数据实际上是一个字符串形式的数组"[x,y,z,visibility]"
        if (limbData == null || limbData.Length < 33) // 应有33个关节点
        {
            Debug.LogWarning($"【数据不完整】接收到的肢体数据不完整: {limbData?.Length} 个关节点 (应为 33)");
            return;
        }
        
        // 计算数据频率
        if (lastTimestamp != 0)
        {
            long timeDiff = timestamp - lastTimestamp;
            if (timeDiff > 0)
            {
                dataRate = 1000.0f / timeDiff; // 转换为Hz
            }
        }
        lastTimestamp = timestamp;

        lock (dataLock)
        {
            try
            {
                int successfullyParsedPoints = 0;
                
                // 解析每个关节点的数据
                for (int j = 0; j < 33 && j < limbData.Length; j++)
                {
                    // 获取原始数据字符串
                    string arrayStr = limbData[j].ToString();
                    
                    // 检查是否是数组字符串格式"[x,y,z,visibility]"
                    if (arrayStr.StartsWith("[") && arrayStr.EndsWith("]"))
                    {
                        // 移除方括号并分割字符串
                        string content = arrayStr.Substring(1, arrayStr.Length - 2);
                        string[] components = content.Split(',');
                        
                        if (components.Length >= 4)
                        {
                            int destIdx = j * 3; // 目标索引
                            bool parseSuccess = true;
                            
                            // 解析x,y,z值并应用缩放
                            if (float.TryParse(components[0], out float x))
                                currentFrameData[destIdx] = x / 100.0f;
                            else
                                parseSuccess = false;
                                
                            if (float.TryParse(components[1], out float y))
                                currentFrameData[destIdx + 1] = y / 100.0f;
                            else
                                parseSuccess = false;
                                
                            if (float.TryParse(components[2], out float z))
                                currentFrameData[destIdx + 2] = z / 300.0f;
                            else
                                parseSuccess = false;
                            
                            if (parseSuccess)
                                successfullyParsedPoints++;
                        }
                        else
                        {
                            Debug.LogWarning($"【格式错误】关节点 {j} 数据格式错误: {arrayStr}，组件数: {components.Length}");
                        }
                    }
                    else
                    {
                        Debug.LogWarning($"【格式错误】关节点 {j} 不是有效的数组格式: {arrayStr}");
                    }
                }
                
                // 更新当前帧数据
                for (int j = 0; j < 99; j++)
                {
                    raw_position[0, j] = currentFrameData[j];
                }
                
                hasNewData = true;
                Debug.Log($"【数据解析】成功解析 {successfullyParsedPoints}/33 个关节点");
                
                // 检查一些关键点的数据是否合理
                if (successfullyParsedPoints > 0)
                {
                    Debug.Log($"【数据检查】Hips位置: ({raw_position[0, 69]}, {raw_position[0, 70]}, {raw_position[0, 71]})");
                    Debug.Log($"【数据检查】左手位置: ({raw_position[0, 45]}, {raw_position[0, 46]}, {raw_position[0, 47]})");
                    Debug.Log($"【数据检查】右手位置: ({raw_position[0, 48]}, {raw_position[0, 49]}, {raw_position[0, 50]})");
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"【解析错误】解析肢体数据时出错: {e.Message}\n{e.StackTrace}");
            }
        }
    }

    // OnGUI方法，用于绘制GUI
    private void OnGUI()
    {
        if (!showDebugGUI) return;
        
        guiWindowRect = GUI.Window(0, guiWindowRect, DrawDebugWindow, "肢体数据监视器");
    }

    // 绘制调试窗口的内容
    private void DrawDebugWindow(int windowID)
    {
        // 开始滚动视图
        scrollPosition = GUILayout.BeginScrollView(scrollPosition);
        
        // 显示基本信息
        GUILayout.Label($"数据源: {(useRealTimeData ? "实时数据" : "文件数据")}");
        GUILayout.Label($"新数据状态: {(hasNewData ? "有新数据" : "无新数据")}");
        
        if (useRealTimeData)
        {
            GUILayout.Label($"数据接收频率: {dataRate:F1} Hz");
        }
        else
        {
            GUILayout.Label($"当前帧索引: {i}/{count-1}");
        }
        
        GUILayout.Space(10);
        
        // 显示关键点数据
        if (raw_position != null && i < raw_position.GetLength(0))
        {
            // 关键点名称(简化版)
            string[] pointNames = {"鼻子", "左眼", "右眼", "左耳", "右耳", "左肩", "右肩", "左肘", "右肘", "左手腕", "右手腕"};
            
            // 选择器
            GUILayout.BeginHorizontal();
            GUILayout.Label("选择关键点:");
            selectedPoint = Mathf.Clamp(GUILayout.SelectionGrid(selectedPoint, pointNames, 3), 0, 10);
            GUILayout.EndHorizontal();
            
            // 显示选定关键点详细数据
            int baseIdx = selectedPoint * 3;
            if (baseIdx + 2 < raw_position.GetLength(1))
            {
                GUILayout.Label($"点 {selectedPoint} ({(selectedPoint < pointNames.Length ? pointNames[selectedPoint] : "未命名")}):");
                GUILayout.Label($"X: {raw_position[i, baseIdx]:F3}");
                GUILayout.Label($"Y: {raw_position[i, baseIdx+1]:F3}");
                GUILayout.Label($"Z: {raw_position[i, baseIdx+2]:F3}");
            }
            
            GUILayout.Space(10);
            
            // 显示所有关键点列表(部分)
            GUILayout.Label("所有关键点数据 (部分):");
            
            // 计算实际显示的点数量
            int pointsToShow = Mathf.Min(guiMaxPoints, raw_position.GetLength(1) / 3);
            
            for (int j = 0; j < pointsToShow; j++)
            {
                baseIdx = j * 3;
                if (baseIdx + 2 < raw_position.GetLength(1))
                {
                    string name = j < pointNames.Length ? pointNames[j] : $"点{j}";
                    GUILayout.Label($"{name}: X={raw_position[i, baseIdx]:F2}, Y={raw_position[i, baseIdx+1]:F2}, Z={raw_position[i, baseIdx+2]:F2}");
                }
            }
            
            // 如果有更多点，显示提示
            if (raw_position.GetLength(1) / 3 > guiMaxPoints)
            {
                GUILayout.Label($"... 还有 {raw_position.GetLength(1)/3 - guiMaxPoints} 个点未显示");
            }
        }
        else
        {
            GUILayout.Label("无可用数据");
        }
        
        // 结束滚动视图
        GUILayout.EndScrollView();
        
        // 控制按钮
        GUILayout.BeginHorizontal();
        if (GUILayout.Button("增加点数"))
        {
            guiMaxPoints = Mathf.Min(guiMaxPoints + 5, 33); // 最多显示33个点
        }
        
        if (GUILayout.Button("减少点数"))
        {
            guiMaxPoints = Mathf.Max(guiMaxPoints - 5, 5); // 最少显示5个点
        }
        
        if (GUILayout.Button("隐藏GUI"))
        {
            showDebugGUI = false;
        }
        GUILayout.EndHorizontal();
        
        // 让窗口可拖动
        GUI.DragWindow();
    }
    
    // Update is called once per frame
    void Update()
    {
        // 如果使用实时数据并有新数据，更新索引i为0以确保使用最新数据
        if (useRealTimeData && hasNewData)
        {
            i = 0;
            lock (dataLock)
            {
                hasNewData = false;
            }
        }
        
        // 更新骨骼动作（保持原有逻辑）
        UpdateSkeletonAnimation();
        
        // 对于文件数据，循环播放
        if (!useRealTimeData)
        {
            if (i >= 0 && i <= count - 2)
            {
                i++;
            }
            else if (i >= count - 1)
            {
                i = 0;
            }
            Thread.Sleep(30); // 每做完一帧动作，线程暂停30ms
        }
    }
    
    private void UpdateSkeletonAnimation()
    {
        // 检查数据是否有效
        if (raw_position == null)
        {
            Debug.LogError("【动画更新】raw_position 为空，无法更新骨骼动画");
            return;
        }
        
        if (i >= raw_position.GetLength(0))
        {
            Debug.LogError($"【动画更新】索引 i={i} 超出范围 raw_position.GetLength(0)={raw_position.GetLength(0)}");
            return;
        }
        
        // 检查关键数据点是否存在
        if (raw_position.GetLength(1) < 99)
        {
            Debug.LogError($"【动画更新】数据维度不足 raw_position.GetLength(1)={raw_position.GetLength(1)}, 需要至少99个元素");
            return;
        }
        
        // 记录开始更新
        Debug.Log($"【动画更新】开始更新骨骼，帧索引: i={i}");
        
        // 更新Hips位置
        Vector3 hipsPos = new Vector3((raw_position[i, 69] + raw_position[i, 72]) / 2.0f, 
                                    (raw_position[i, 70] + raw_position[i, 73]) / 2.0f, 
                                    (raw_position[i, 71] + raw_position[i, 74]) / 2.0f);
        BodyPart[0].position = hipsPos;
        Debug.Log($"【位置更新】Hips位置: {hipsPos}");
        
        // 估计火柴人模型中对应的LeftUpperLeg与RightUpperLeg位置
        PosLeftUpperLeg = new Vector3(4.0f / 5.0f * raw_position[i, 69] + 1.0f / 5.0f * raw_position[i, 75], 
                                    4.0f / 5.0f * raw_position[i, 70] + 1.0f / 5.0f * raw_position[i, 76], 
                                    4.0f / 5.0f * raw_position[i, 71] + 1.0f / 5.0f * raw_position[i, 77]);
                                    
        PosRightUpperLeg = new Vector3(4.0f / 5.0f * raw_position[i, 72] + 1.0f / 5.0f * raw_position[i, 78], 
                                     4.0f / 5.0f * raw_position[i, 73] + 1.0f / 5.0f * raw_position[i, 79], 
                                     4.0f / 5.0f * raw_position[i, 74] + 1.0f / 5.0f * raw_position[i, 80]);
                                     
        // 计算人体Hips的朝向
        forward = TriangleNormal(BodyPart[0].position, PosLeftUpperLeg, PosRightUpperLeg);
        
        // 更新人体Hips的rotation
        BodyPart[0].rotation = Quaternion.LookRotation(forward) * Quaternion.Inverse(MidHips);
        
        // 更新其他所有关节
        UpdateLowerBody();
        UpdateSpineAndChest();
        UpdateArms();
        UpdateHands();
        UpdateNeck();
        
        // 更新完成记录
        Debug.Log("【动画更新】骨骼更新完成");
    }
    
    private void UpdateLowerBody()
    {
        // 计算左上腿与其子物体的方向
        LUL_vec = PosLeftUpperLeg - new Vector3(raw_position[i, 75], raw_position[i, 76], raw_position[i, 77]);
        // 更新左上腿的rotation
        BodyPart[18].rotation = Quaternion.LookRotation(LUL_vec, forward) * Quaternion.Inverse(MidLeftUpperLeg);
        
        // 计算右上腿与其子物体的方向
        RUL_vec = PosRightUpperLeg - new Vector3(raw_position[i, 78], raw_position[i, 79], raw_position[i, 80]);
        // 更新右上腿的rotation
        BodyPart[21].rotation = Quaternion.LookRotation(RUL_vec, forward) * Quaternion.Inverse(MidRightUpperLeg);
        
        // 计算左下腿与其子物体的方向
        LLL_vec = new Vector3(raw_position[i, 75], raw_position[i, 76], raw_position[i, 77]) - 
                 new Vector3(raw_position[i, 81], raw_position[i, 82], raw_position[i, 83]);
        // 更新左下腿的rotation
        BodyPart[19].rotation = Quaternion.LookRotation(LLL_vec, forward) * Quaternion.Inverse(MidLeftLowerLeg);
        
        // 计算右下腿与其子物体的方向
        RLL_vec = new Vector3(raw_position[i, 78], raw_position[i, 79], raw_position[i, 80]) - 
                 new Vector3(raw_position[i, 84], raw_position[i, 85], raw_position[i, 86]);
        // 更新右下腿的rotation
        BodyPart[22].rotation = Quaternion.LookRotation(RLL_vec, forward) * Quaternion.Inverse(MidRightLowerLeg);
        
        // 计算左足与其子物体的方向
        LF_vec = new Vector3(raw_position[i, 81], raw_position[i, 82], raw_position[i, 83]) - 
                new Vector3(raw_position[i, 93], raw_position[i, 94], raw_position[i, 95]);
        // 更新左足的rotation
        BodyPart[20].rotation = Quaternion.LookRotation(LF_vec, forward) * Quaternion.Inverse(MidLeftFoot);
        
        // 计算右足与其子物体的方向
        RF_vec = new Vector3(raw_position[i, 84], raw_position[i, 85], raw_position[i, 86]) - 
                new Vector3(raw_position[i, 96], raw_position[i, 97], raw_position[i, 98]);
        // 更新右足的rotation
        BodyPart[23].rotation = Quaternion.LookRotation(RF_vec, forward) * Quaternion.Inverse(MidRightFoot);
    }
    
    private void UpdateSpineAndChest()
    {
        // 计算Spine与其子物体的方向
        Sp_vec = BodyPart[0].position - new Vector3((raw_position[i, 33] + raw_position[i, 36]) / 2.0f, 
                                                 (raw_position[i, 34] + raw_position[i, 37]) / 2.0f, 
                                                 (raw_position[i, 35] + raw_position[i, 38]) / 2.0f);
        // 计算Spine的rotation
        BodyPart[1].rotation = Quaternion.LookRotation(Sp_vec, forward) * Quaternion.Inverse(MidSpine);
        
        // 计算Chest与其子物体的方向
        Ch_vec = Sp_vec;
        // 计算Chest的rotation
        BodyPart[2].rotation = Quaternion.LookRotation(Ch_vec, forward) * Quaternion.Inverse(MidChest);
        
        // 计算UpperChest的朝向
        UC_vec = Ch_vec;
        // 计算UpperChest的rotation
        BodyPart[3].rotation = Quaternion.LookRotation(Ch_vec, forward) * Quaternion.Inverse(MidUpperChest);
    }
    
    private void UpdateArms()
    {
        // 计算左肩与其子物体方向
        LS_vec = new Vector3(raw_position[i, 36] - raw_position[i, 33], 
                           raw_position[i, 37] - raw_position[i, 34], 
                           raw_position[i, 38] - raw_position[i, 35]);
        // 计算LeftShoulder的rotation
        BodyPart[4].rotation = Quaternion.LookRotation(LS_vec, forward) * Quaternion.Inverse(MidLeftShoulder);
        
        // 计算右肩与其子物体方向
        RS_vec = new Vector3(raw_position[i, 33] - raw_position[i, 36], 
                           raw_position[i, 34] - raw_position[i, 37], 
                           raw_position[i, 35] - raw_position[i, 38]);
        // 计算RightShoulder的rotation
        BodyPart[11].rotation = Quaternion.LookRotation(RS_vec, forward) * Quaternion.Inverse(MidRightShoulder);
        
        // 计算左上臂与子物体的方向
        LUA_vec = new Vector3(raw_position[i, 33] - raw_position[i, 39], 
                            raw_position[i, 34] - raw_position[i, 40], 
                            raw_position[i, 35] - raw_position[i, 41]);
        // 计算LeftUpperArm的rotation
        BodyPart[5].rotation = Quaternion.LookRotation(LUA_vec, forward) * Quaternion.Inverse(MidLeftUpperArm);
        
        // 计算右上臂与子物体的方向
        RUA_vec = new Vector3(raw_position[i, 36] - raw_position[i, 42], 
                            raw_position[i, 37] - raw_position[i, 43], 
                            raw_position[i, 38] - raw_position[i, 44]);
        // 计算RightUpperArm的rotation
        BodyPart[12].rotation = Quaternion.LookRotation(RUA_vec, forward) * Quaternion.Inverse(MidRightUpperArm);
        
        // 计算左下臂与子物体的方向
        LLA_vec = new Vector3(raw_position[i, 39] - raw_position[i, 45], 
                            raw_position[i, 40] - raw_position[i, 46], 
                            raw_position[i, 41] - raw_position[i, 47]);
        // 计算LeftLowerArm的rotation
        BodyPart[6].rotation = Quaternion.LookRotation(LLA_vec, forward) * Quaternion.Inverse(MidLeftLowerArm);
        
        // 计算右下臂与子物体的方向
        RLA_vec = new Vector3(raw_position[i, 42] - raw_position[i, 48], 
                            raw_position[i, 43] - raw_position[i, 49], 
                            raw_position[i, 44] - raw_position[i, 50]);
        // 计算RightLowerArm的rotation
        BodyPart[13].rotation = Quaternion.LookRotation(RLA_vec, forward) * Quaternion.Inverse(MidRightLowerArm);
    }
    
    private void UpdateHands()
    {
        // 计算LHN
        PosLeftIndex = new Vector3(raw_position[i, 57], raw_position[i, 58], raw_position[i, 59]);
        PosLeftThumb = new Vector3(raw_position[i, 63], raw_position[i, 64], raw_position[i, 65]);
        PosLeftLittle = new Vector3(raw_position[i, 51], raw_position[i, 52], raw_position[i, 53]);
        LHN = TriangleNormal(new Vector3(raw_position[i, 45], raw_position[i, 46], raw_position[i, 47]), 
                           PosLeftIndex, PosLeftThumb);
        // 计算左手的rotation
        BodyPart[7].rotation = Quaternion.LookRotation(PosLeftThumb - PosLeftIndex, LHN) * 
                               Quaternion.Inverse(MidLeftHand);

        // 计算RHN
        PosRightIndex = new Vector3(raw_position[i, 60], raw_position[i, 61], raw_position[i, 62]);
        PosRightThumb = new Vector3(raw_position[i, 66], raw_position[i, 67], raw_position[i, 68]);
        PosRightLittle = new Vector3(raw_position[i, 54], raw_position[i, 55], raw_position[i, 56]);
        RHN = TriangleNormal(new Vector3(raw_position[i, 48], raw_position[i, 49], raw_position[i, 50]), 
                           PosRightThumb, PosRightIndex);
        // 计算右手的rotation
        BodyPart[14].rotation = Quaternion.LookRotation(PosRightThumb - PosRightIndex, RHN) * 
                                Quaternion.Inverse(MidRightHand);

        // 左拇指
        LT_vec = new Vector3(raw_position[i, 45], raw_position[i, 46], raw_position[i, 47]) - PosLeftThumb;
        BodyPart[8].rotation = Quaternion.LookRotation(LT_vec, LHN) * Quaternion.Inverse(MidLeftThumb);
        
        // 左食指
        LI_vec = new Vector3(raw_position[i, 45], raw_position[i, 46], raw_position[i, 47]) - PosLeftIndex;
        BodyPart[9].rotation = Quaternion.LookRotation(LI_vec, LHN) * Quaternion.Inverse(MidLeftIndex);
        
        // 左小指
        LL_vec = new Vector3(raw_position[i, 45], raw_position[i, 46], raw_position[i, 47]) - PosLeftLittle;
        BodyPart[10].rotation = Quaternion.LookRotation(LL_vec, LHN) * Quaternion.Inverse(MidLeftLittle);
        
        // 左中指
        BodyPart[33].rotation = BodyPart[9].rotation;
        
        // 左无名指
        BodyPart[35].rotation = BodyPart[9].rotation;

        // 右拇指
        RT_vec = new Vector3(raw_position[i, 48], raw_position[i, 49], raw_position[i, 50]) - PosRightThumb;
        BodyPart[15].rotation = Quaternion.LookRotation(RT_vec, RHN) * Quaternion.Inverse(MidRightThumb);
        
        // 右食指
        RI_vec = new Vector3(raw_position[i, 48], raw_position[i, 49], raw_position[i, 50]) - PosRightIndex;
        BodyPart[16].rotation = Quaternion.LookRotation(RI_vec, RHN) * Quaternion.Inverse(MidRightIndex);
        
        // 右小指
        RL_vec = new Vector3(raw_position[i, 48], raw_position[i, 49], raw_position[i, 50]) - PosRightLittle;
        BodyPart[17].rotation = Quaternion.LookRotation(RL_vec, RHN) * Quaternion.Inverse(MidRightLittle);
        
        // 右中指
        BodyPart[34].rotation = BodyPart[16].rotation;
        
        // 右无名指
        BodyPart[36].rotation = BodyPart[16].rotation;
    }
    
    private void UpdateNeck()
    {
        // 颈部
        PosHead = new Vector3((raw_position[i, 21] + raw_position[i, 24]) / 2.0f, 
                            (raw_position[i, 22] + raw_position[i, 25]) / 2.0f, 
                            (raw_position[i, 23] + raw_position[i, 26]) / 2.0f);
        gaze = new Vector3(raw_position[i, 0], raw_position[i, 1], raw_position[i, 2]) - PosHead;
        BodyPart[24].rotation = Quaternion.LookRotation(gaze) * Quaternion.Inverse(MidNeck);
    }
    
    // 三角法向函数
    Vector3 TriangleNormal(Vector3 a, Vector3 b, Vector3 c)
    {
        Vector3 d1 = a - b;
        Vector3 d2 = a - c;

        Vector3 dd = Vector3.Cross(d1, d2); // 向量叉乘
        dd.Normalize();

        return dd;
    }
    
    // 当组件被禁用时取消订阅事件
    private void OnDisable()
    {
        if (dataReceiver != null)
        {
            dataReceiver.OnLimbDataReceived -= OnLimbDataReceived;
        }
    }
}