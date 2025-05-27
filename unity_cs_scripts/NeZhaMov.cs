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
    
    // Start is called before the first frame update
    void Start()
    {
        // 如果没有指定FaceDataReceiver，尝试查找
        if (dataReceiver == null)
        {
            dataReceiver = FindObjectOfType<FaceDataReceiver>();
            if (dataReceiver == null)
            {
                Debug.LogWarning("未找到FaceDataReceiver组件，将使用文件数据");
                useRealTimeData = false;
            }
        }
        
        // 订阅肢体数据事件
        if (useRealTimeData && dataReceiver != null)
        {
            dataReceiver.OnLimbDataReceived += OnLimbDataReceived;
            Debug.Log("已订阅肢体数据事件");
            
            // 初始化一个空的当前帧数据
            currentFrameData = new float[99]; // 33点 * 3坐标 = 99
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
        if (limbData.Length < 33 * 4) // 检查数据是否完整 (33点 * 4值)
        {
            Debug.LogWarning($"接收到的肢体数据不完整: {limbData.Length} 个值 (应为 {33*4})");
            return;
        }

        lock (dataLock)
        {
            // 将肢体数据转换为需要的格式 (x,y,z)
            // 每4个值(x,y,z,visibility)提取前3个值(x,y,z)
            for (int j = 0; j < 33; j++)
            {
                int srcIdx = j * 4; // 源数据索引
                int destIdx = j * 3; // 目标数据索引
                
                // 复制x,y,z数据并应用缩放，忽略visibility
                // 接收到的数据单位是米，需要转换为毫米（乘以1000）
                currentFrameData[destIdx] = limbData[srcIdx] * 1000 / 100.0f;     // x
                currentFrameData[destIdx+1] = limbData[srcIdx+1] * 1000 / 100.0f; // y
                currentFrameData[destIdx+2] = limbData[srcIdx+2] * 1000 / 300.0f; // z
            }
            
            // 更新当前帧数据
            for (int j = 0; j < 99; j++)
            {
                raw_position[0, j] = currentFrameData[j];
            }
            
            hasNewData = true;
            // Debug.Log($"已更新肢体数据，应用米到毫米转换（×1000）");
        }
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
        // 更新Hips位置
        BodyPart[0].position = new Vector3((raw_position[i, 69] + raw_position[i, 72]) / 2.0f, 
                                         (raw_position[i, 70] + raw_position[i, 73]) / 2.0f, 
                                         (raw_position[i, 71] + raw_position[i, 74]) / 2.0f);
        
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