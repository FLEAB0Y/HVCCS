using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using UnityEngine;

public class BDCtrl : MonoBehaviour
{
    public GameObject NeZha;//哪吒人物物体
    
    // 添加根节点位置偏移变量，可在Inspector中调整
    [Header("根节点位置调整")]
    [Tooltip("用于手动调整哪吒根节点(Hips)的位置偏移")]
    public Vector3 hipsPositionOffset = Vector3.zero;
    
    Transform[] BodyPart;//avatar人体模型的各个身体部件
    Vector3 raw1;
    Vector3 raw2;
    Vector3 raw;
    Vector3 vec;
    Vector3 forward;//Hips的方向
    Vector3 LHN;//Left Hand Normal，左手的法向(LookRotation中为y方向)
    Vector3 RHN;//Right Hand Normal，右手的法向(LookRotation中为y方向)
    Vector3 gaze;//哪吒头部的朝向

    // 骨骼名称映射字典
    private Dictionary<int, string> boneNameMap = new Dictionary<int, string>()
    {
        {0, "Hips"},                     // 根节点
        {1, "Spine"},                    // 脊柱
        {2, "Spine1"},                   // 胸腔
        {3, "Spine2"},                   // 上胸腔
        {4, "LeftShoulder"},             // 左肩
        {5, "LeftArm"},                  // 左上臂
        {6, "LeftForeArm"},              // 左下臂
        {7, "LeftHand"},                 // 左手
        {8, "LeftHandThumb2"},           // 左拇指
        {9, "LeftHandIndex2"},           // 左食指
        {10, "LeftHandPinky2"},          // 左小指
        {11, "RightShoulder"},           // 右肩
        {12, "RightArm"},                // 右上臂
        {13, "RightForeArm"},            // 右下臂
        {14, "RightHand"},               // 右手
        {15, "RightHandThumb2"},         // 右拇指
        {16, "RightHandIndex2"},         // 右食指
        {17, "RightHandPinky2"},         // 右小指
        {18, "LeftUpLeg"},               // 左上腿
        {19, "LeftLeg"},                 // 左下腿
        {20, "LeftFoot"},                // 左足
        {21, "RightUpLeg"},              // 右上腿
        {22, "RightLeg"},                // 右下腿
        {23, "RightFoot"},               // 右足
        {24, "Neck"},                    // 颈部
        {25, "LeftToeBase"},             // 左脚趾
        {26, "RightToeBase"},            // 右脚趾
        {27, "LeftHandThumb3"},          // 左拇指尖
        {28, "RightHandThumb3"},         // 右拇指尖
        {29, "LeftHandIndex3"},          // 左食指尖
        {30, "RightHandIndex3"},         // 右食指尖
        {31, "LeftHandPinky3"},          // 左小指指尖
        {32, "RightHandPinky3"},         // 右小指指尖
        {33, "LeftHandMiddle2"},         // 左中指
        {34, "RightHandMiddle2"},        // 右中指
        {35, "LeftHandRing2"},           // 左无名指
        {36, "RightHandRing2"},          // 右无名指
        {37, "Head"}                     // 头部
    };

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
    
    // Start is called before the first frame update
    void Start()
    {
        BodyPart = new Transform[38];//人体模型的各个身体关节
        
        // 通过递归查找所有骨骼
        Transform[] allBones = NeZha.GetComponentsInChildren<Transform>();
        
        // 先打印模型的所有骨骼名称，帮助调试
        Debug.Log("===== 模型所有骨骼列表开始 =====");
        for (int i = 0; i < allBones.Length; i++)
        {
            Debug.Log($"骨骼[{i}]: {allBones[i].name}, 父节点: {(allBones[i].parent ? allBones[i].parent.name : "无")}");
        }
        Debug.Log("===== 模型所有骨骼列表结束 =====");
        
        // 第一步：尝试标准匹配方式
        foreach (Transform bone in allBones)
        {
            foreach (var entry in boneNameMap)
            {
                if (bone.name.IndexOf(entry.Value, System.StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    BodyPart[entry.Key] = bone;
                    Debug.Log($"标准匹配到骨骼: {entry.Value} -> {bone.name}");
                    break;
                }
            }
        }
        
        // 第二步：针对每个未找到的骨骼进行精确查找
        for (int i = 0; i < boneNameMap.Count; i++)
        {
            if (BodyPart[i] == null)
            {
                string boneName = boneNameMap[i];
                Debug.Log($"尝试精确查找骨骼: {boneName}");
                
                // 创建可能的变体名称列表
                List<string> possibleNames = new List<string> {
                    boneName,                  // 精确匹配
                    boneName.ToLower(),        // 全小写
                    "mixamorig:" + boneName,   // mixamo前缀
                    boneName + "_end",         // _end后缀
                    boneName.Replace("Hand", "")  // 特殊处理手部骨骼
                };
                
                // 尝试所有可能的名称
                foreach (Transform bone in allBones)
                {
                    foreach (string possibleName in possibleNames)
                    {
                        if (bone.name.IndexOf(possibleName, System.StringComparison.OrdinalIgnoreCase) >= 0)
                        {
                            BodyPart[i] = bone;
                            Debug.Log($"精确查找到骨骼: {boneName} -> {bone.name}");
                            break;
                        }
                    }
                    
                    if (BodyPart[i] != null) break;
                }
            }
        }
        
        // 第三步：针对指定骨骼查找备选匹配方案
        // 例如Spine1可能有特殊命名
        if (BodyPart[2] == null) // Spine1
        {
            FindBoneWithNameAndAssign(allBones, "Spine1", 2);
            // 备选名称
            if (BodyPart[2] == null) FindBoneWithNameAndAssign(allBones, "Chest", 2);
        }
        
        // 特殊手指骨骼匹配
        if (BodyPart[8] == null) FindBoneWithNameAndAssign(allBones, "Thumb", 8);
        if (BodyPart[9] == null) FindBoneWithNameAndAssign(allBones, "Index", 9);
        if (BodyPart[10] == null) FindBoneWithNameAndAssign(allBones, "Little", 10);
        if (BodyPart[15] == null) FindBoneWithNameAndAssign(allBones, "Thumb", 15);
        if (BodyPart[16] == null) FindBoneWithNameAndAssign(allBones, "Index", 16);
        if (BodyPart[17] == null) FindBoneWithNameAndAssign(allBones, "Little", 17);
        
        // 检查是否所有骨骼都已找到
        bool allBonesFound = true;
        for (int i = 0; i < BodyPart.Length; i++)
        {
            if (BodyPart[i] == null)
            {
                Debug.LogError($"未找到骨骼: {boneNameMap[i]}");
                allBonesFound = false;
            }
        }
        
        if (!allBonesFound)
        {
            Debug.LogError("未找到所有必需的骨骼，请检查模型骨骼结构或骨骼名称映射");
            // 打印所有找到的骨骼，帮助调试
            PrintFoundBones();
            return;
        }

        // 初始化中间矩阵
        InitializeMiddleMatrices();
    }

    // 处理肢体数据的公共方法 - 由FaceDataReceiver直接调用
    public void ProcessLimbData(float[] limbData, long timestamp)
    {
        if (limbData == null || limbData.Length != 99)
        {
            Debug.LogWarning($"接收到的数据长度不匹配: 预期99，实际{limbData?.Length ?? 0}");
            return;
        }

        Debug.Log("直接处理肢体数据，时间戳: " + timestamp);
        
        // 直接处理数据，不再缓存
        UpdateModel(limbData);
    }
    
    // 初始化中间矩阵
    private void InitializeMiddleMatrices()
    {
        // 初始化朝向
        forward = TriangleNormal(BodyPart[0].position, BodyPart[18].position, BodyPart[21].position);
        
        //LowerBody
        //Hips中间矩阵
        MidHips = Quaternion.Inverse(BodyPart[0].rotation)*Quaternion.LookRotation(forward);
        //LeftUpperLeg中间矩阵，父对象-子对象
        MidLeftUpperLeg = Quaternion.Inverse(BodyPart[18].rotation) * Quaternion.LookRotation((BodyPart[18].position - BodyPart[19].position),forward);
        //RightUpperLeg中间矩阵
        MidRightUpperLeg = Quaternion.Inverse(BodyPart[21].rotation) * Quaternion.LookRotation((BodyPart[21].position - BodyPart[22].position),forward);
        //LeftLowerLeg中间矩阵
        MidLeftLowerLeg = Quaternion.Inverse(BodyPart[19].rotation) * Quaternion.LookRotation((BodyPart[19].position - BodyPart[20].position),forward);
        //RightLowerLeg中间矩阵
        MidRightLowerLeg = Quaternion.Inverse(BodyPart[22].rotation) * Quaternion.LookRotation((BodyPart[22].position - BodyPart[23].position), forward);
        //LeftFoot中间矩阵
        MidLeftFoot = Quaternion.Inverse(BodyPart[20].rotation) * Quaternion.LookRotation((BodyPart[20].position - BodyPart[25].position), forward);
        //RightFoot中间矩阵
        MidRightFoot = Quaternion.Inverse(BodyPart[23].rotation) * Quaternion.LookRotation((BodyPart[23].position - BodyPart[26].position), forward);

        //UpperBody
        //Spine中间矩阵
        MidSpine = Quaternion.Inverse(BodyPart[1].rotation) * Quaternion.LookRotation((BodyPart[1].position - BodyPart[2].position), forward);
        //Chest中间矩阵
        MidChest = Quaternion.Inverse(BodyPart[2].rotation) * Quaternion.LookRotation((BodyPart[2].position - BodyPart[3].position), forward);
        //UpperChest中间矩阵
        MidUpperChest = Quaternion.Inverse(BodyPart[3].rotation) * Quaternion.LookRotation((BodyPart[3].position - BodyPart[24].position), forward);
        //LeftShoulder
        MidLeftShoulder = Quaternion.Inverse(BodyPart[4].rotation) * Quaternion.LookRotation((BodyPart[4].position - BodyPart[5].position), forward);
        //LeftUpperArm
        MidLeftUpperArm = Quaternion.Inverse(BodyPart[5].rotation) * Quaternion.LookRotation((BodyPart[5].position - BodyPart[6].position), forward);
        //LeftLowerArm
        MidLeftLowerArm = Quaternion.Inverse(BodyPart[6].rotation) * Quaternion.LookRotation((BodyPart[6].position - BodyPart[7].position), forward);
        //LeftHand
        LHN = TriangleNormal(BodyPart[7].position, BodyPart[9].position, BodyPart[8].position);
        //拇指减去食指，thumb-index
        MidLeftHand = Quaternion.Inverse(BodyPart[7].rotation) * Quaternion.LookRotation((BodyPart[8].position - BodyPart[9].position), LHN);
        //RightShoulder
        MidRightShoulder = Quaternion.Inverse(BodyPart[11].rotation) * Quaternion.LookRotation((BodyPart[11].position - BodyPart[12].position), forward);
        //RightUpperArm
        MidRightUpperArm = Quaternion.Inverse(BodyPart[12].rotation) * Quaternion.LookRotation((BodyPart[12].position - BodyPart[13].position), forward);
        //RightLowerArm
        MidRightLowerArm = Quaternion.Inverse(BodyPart[13].rotation) * Quaternion.LookRotation((BodyPart[13].position - BodyPart[14].position), forward);
        //RightHand
        RHN = TriangleNormal(BodyPart[14].position, BodyPart[15].position, BodyPart[16].position);
        //拇指减去食指，thumb-index
        MidRightHand = Quaternion.Inverse(BodyPart[14].rotation) * Quaternion.LookRotation((BodyPart[15].position - BodyPart[16].position), RHN);
        //左拇指
        MidLeftThumb = Quaternion.Inverse(BodyPart[8].rotation) * Quaternion.LookRotation((BodyPart[8].position - BodyPart[27].position), LHN);
        //左食指
        MidLeftIndex = Quaternion.Inverse(BodyPart[9].rotation) * Quaternion.LookRotation((BodyPart[9].position - BodyPart[29].position), LHN);
        //左小指
        MidLeftLittle = Quaternion.Inverse(BodyPart[10].rotation) * Quaternion.LookRotation((BodyPart[10].position - BodyPart[31].position), LHN);
        //右拇指
        MidRightThumb = Quaternion.Inverse(BodyPart[15].rotation) * Quaternion.LookRotation((BodyPart[15].position - BodyPart[28].position), RHN);
        //右食指
        MidRightIndex = Quaternion.Inverse(BodyPart[16].rotation) * Quaternion.LookRotation((BodyPart[16].position - BodyPart[30].position), RHN);
        //右小指
        MidRightLittle = Quaternion.Inverse(BodyPart[17].rotation) * Quaternion.LookRotation((BodyPart[17].position - BodyPart[32].position), RHN);

        //Neck
        gaze = forward;
        MidNeck = Quaternion.Inverse(BodyPart[24].rotation) * Quaternion.LookRotation(gaze);
    }

    // 更新模型的方法
    private void UpdateModel(float[] data)
    {
        // 创建缩放后的数据数组
        float[] scaledData = new float[data.Length];
        
        // 对所有坐标点进行缩放
        for (int i = 0; i < data.Length; i += 3)
        {
            if (i + 2 < data.Length) // 确保有完整的xyz三个值
            {
                scaledData[i] = data[i] / 100f;      // x坐标缩放
                scaledData[i+1] = data[i+1] / 100f;  // y坐标缩放
                scaledData[i+2] = data[i+2] / 300f;  // z坐标缩放
            }
        }
        
        // 使用缩放后的数据更新模型
        // 更新Hips位置
        BodyPart[0].position = new Vector3((scaledData[69] + scaledData[72]) / 2.0f, 
                                  (scaledData[70] + scaledData[73]) / 2.0f, 
                                  (scaledData[71] + scaledData[74]) / 2.0f) + hipsPositionOffset;
        
        // 估计火柴人模型中对应的LeftUpperLeg与RightUpperLeg位置
        PosLeftUpperLeg = new Vector3(4.0f / 5.0f * scaledData[69] + 1.0f / 5.0f * scaledData[75], 4.0f / 5.0f * scaledData[70] + 1.0f / 5.0f * scaledData[76], 4.0f / 5.0f * scaledData[71] + 1.0f / 5.0f * scaledData[77]);
        PosRightUpperLeg = new Vector3(4.0f / 5.0f * scaledData[72] + 1.0f / 5.0f * scaledData[78], 4.0f / 5.0f * scaledData[73] + 1.0f / 5.0f * scaledData[79], 4.0f / 5.0f * scaledData[74] + 1.0f / 5.0f * scaledData[80]);
        
        // 计算人体Hips的朝向
        forward = TriangleNormal(BodyPart[0].position, PosLeftUpperLeg, PosRightUpperLeg);
        
        // 更新人体Hips的rotation
        BodyPart[0].rotation = Quaternion.LookRotation(forward) * Quaternion.Inverse(MidHips);
        
        // 计算左上腿与其子物体的方向
        LUL_vec = PosLeftUpperLeg - new Vector3(scaledData[75], scaledData[76], scaledData[77]);
        
        // 更新左上腿的rotation
        BodyPart[18].rotation = Quaternion.LookRotation(LUL_vec, forward) * Quaternion.Inverse(MidLeftUpperLeg);
        
        // 计算右上腿与其子物体的方向
        RUL_vec = PosRightUpperLeg - new Vector3(scaledData[78], scaledData[79], scaledData[80]);
        
        // 更新右上腿的rotation
        BodyPart[21].rotation = Quaternion.LookRotation(RUL_vec, forward) * Quaternion.Inverse(MidRightUpperLeg);
        
        // 计算左下腿与其子物体的方向
        LLL_vec = new Vector3(scaledData[75], scaledData[76], scaledData[77]) - new Vector3(scaledData[81], scaledData[82], scaledData[83]);
        
        // 更新左下腿的rotation
        BodyPart[19].rotation = Quaternion.LookRotation(LLL_vec, forward) * Quaternion.Inverse(MidLeftLowerLeg);
        
        // 计算右下腿与其子物体的方向
        RLL_vec = new Vector3(scaledData[78], scaledData[79], scaledData[80]) - new Vector3(scaledData[84], scaledData[85], scaledData[86]);
        
        // 更新右下腿的rotation
        BodyPart[22].rotation = Quaternion.LookRotation(RLL_vec, forward) * Quaternion.Inverse(MidRightLowerLeg);
        
        // 计算左足与其子物体的方向
        LF_vec = new Vector3(scaledData[81], scaledData[82], scaledData[83]) - new Vector3(scaledData[93], scaledData[94], scaledData[95]);
        
        // 更新左足的rotation
        BodyPart[20].rotation = Quaternion.LookRotation(LF_vec, forward) * Quaternion.Inverse(MidLeftFoot);
        
        // 计算右足与其子物体的方向
        RF_vec = new Vector3(scaledData[84], scaledData[85], scaledData[86]) - new Vector3(scaledData[96], scaledData[97], scaledData[98]);
        
        // 更新右足的rotation
        BodyPart[23].rotation = Quaternion.LookRotation(RF_vec, forward) * Quaternion.Inverse(MidRightFoot);

        // 计算Spine与其子物体的方向 - 修正为从Spine到Chest的方向
        Sp_vec = new Vector3((scaledData[33] + scaledData[36]) / 2.0f, 
                    (scaledData[34] + scaledData[37]) / 2.0f, 
                    (scaledData[35] + scaledData[38]) / 2.0f) - BodyPart[0].position;
        
        // 计算Spine的rotation - 保持向上的方向
        BodyPart[1].rotation = Quaternion.LookRotation(Sp_vec, forward) * Quaternion.Inverse(MidSpine);
        
        // 计算Chest与其子物体的方向 - 使用肩部中点到颈部的方向
        Vector3 shoulderMidpoint = new Vector3((scaledData[33] + scaledData[36]) / 2.0f,
                                     (scaledData[34] + scaledData[37]) / 2.0f,
                                     (scaledData[35] + scaledData[38]) / 2.0f);
        Vector3 neckPos = new Vector3((scaledData[21] + scaledData[24]) / 2.0f, 
                             (scaledData[22] + scaledData[25]) / 2.0f, 
                             (scaledData[23] + scaledData[26]) / 2.0f);
        Ch_vec = neckPos - shoulderMidpoint;
        
        // 计算Chest的rotation
        BodyPart[2].rotation = Quaternion.LookRotation(Ch_vec, forward) * Quaternion.Inverse(MidChest);
        
        // 计算UpperChest的方向 - 使用与Chest不同的方向
        UC_vec = neckPos - shoulderMidpoint;
        
        // 计算UpperChest的rotation
        BodyPart[3].rotation = Quaternion.LookRotation(UC_vec, forward) * Quaternion.Inverse(MidUpperChest);

        // 计算左肩与其子物体方向
        LS_vec = new Vector3(scaledData[36] - scaledData[33], scaledData[37] - scaledData[34], scaledData[38] - scaledData[35]);
        
        // 计算LeftShoulder的rotation
        BodyPart[4].rotation = Quaternion.LookRotation(LS_vec, forward) * Quaternion.Inverse(MidLeftShoulder);
        
        // 计算右肩与其子物体方向
        RS_vec = new Vector3(scaledData[33] - scaledData[36], scaledData[34] - scaledData[37], scaledData[35] - scaledData[38]);
        
        // 计算RightShoulder的rotation
        BodyPart[11].rotation = Quaternion.LookRotation(RS_vec, forward) * Quaternion.Inverse(MidRightShoulder);
        
        // 计算左上臂与子物体的方向
        LUA_vec = new Vector3(scaledData[33] - scaledData[39], scaledData[34] - scaledData[40], scaledData[35] - scaledData[41]);
        
        // 计算LeftUpperArm的rotation
        BodyPart[5].rotation = Quaternion.LookRotation(LUA_vec, forward) * Quaternion.Inverse(MidLeftUpperArm);
        
        // 计算右上臂与子物体的方向
        RUA_vec = new Vector3(scaledData[36] - scaledData[42], scaledData[37] - scaledData[43], scaledData[38] - scaledData[44]);
        
        // 计算RightUpperArm的rotation
        BodyPart[12].rotation = Quaternion.LookRotation(RUA_vec, forward) * Quaternion.Inverse(MidRightUpperArm);
        
        // 计算左下臂与子物体的方向
        LLA_vec = new Vector3(scaledData[39] - scaledData[45], scaledData[40] - scaledData[46], scaledData[41] - scaledData[47]);
        
        // 计算LeftLowerArm的rotation
        BodyPart[6].rotation = Quaternion.LookRotation(LLA_vec, forward) * Quaternion.Inverse(MidLeftLowerArm);
        
        // 计算右下臂与子物体的方向
        RLA_vec = new Vector3(scaledData[42] - scaledData[48], scaledData[43] - scaledData[49], scaledData[44] - scaledData[50]);
        
        // 计算RightLowerArm的rotation
        BodyPart[13].rotation = Quaternion.LookRotation(RLA_vec, forward) * Quaternion.Inverse(MidRightLowerArm);

        // 计算LHN
        PosLeftIndex = new Vector3(scaledData[57], scaledData[58], scaledData[59]);
        PosLeftThumb = new Vector3(scaledData[63], scaledData[64], scaledData[65]);
        PosLeftLittle = new Vector3(scaledData[51], scaledData[52], scaledData[53]);
        LHN = TriangleNormal(new Vector3(scaledData[45], scaledData[46], scaledData[47]), PosLeftIndex, PosLeftThumb);
        
        // 计算左手的rotation
        BodyPart[7].rotation = Quaternion.LookRotation(PosLeftThumb - PosLeftIndex, LHN) * Quaternion.Inverse(MidLeftHand);

        // 计算RHN
        PosRightIndex = new Vector3(scaledData[60], scaledData[61], scaledData[62]);
        PosRightThumb = new Vector3(scaledData[66], scaledData[67], scaledData[68]);
        PosRightLittle = new Vector3(scaledData[54], scaledData[55], scaledData[56]);
        RHN = TriangleNormal(new Vector3(scaledData[48], scaledData[49], scaledData[50]), PosRightThumb, PosRightIndex);
        
        // 计算右手的rotation
        BodyPart[14].rotation = Quaternion.LookRotation(PosRightThumb - PosRightIndex, RHN) * Quaternion.Inverse(MidRightHand);

        // 左拇指
        LT_vec = new Vector3(scaledData[45], scaledData[46], scaledData[47]) - PosLeftThumb;
        BodyPart[8].rotation = Quaternion.LookRotation(LT_vec,LHN)*Quaternion.Inverse(MidLeftThumb);
        
        // 左食指
        LI_vec = new Vector3(scaledData[45], scaledData[46], scaledData[47]) - PosLeftIndex;
        BodyPart[9].rotation = Quaternion.LookRotation(LI_vec, LHN) * Quaternion.Inverse(MidLeftIndex);
        
        // 左小指
        LL_vec = new Vector3(scaledData[45], scaledData[46], scaledData[47]) - PosLeftLittle;
        BodyPart[10].rotation = Quaternion.LookRotation(LL_vec, LHN) * Quaternion.Inverse(MidLeftLittle);
        
        // 左中指
        BodyPart[33].rotation = BodyPart[9].rotation;
        
        // 左无名指
        BodyPart[35].rotation = BodyPart[9].rotation;

        // 右拇指
        RT_vec = new Vector3(scaledData[48], scaledData[49], scaledData[50]) - PosRightThumb;
        BodyPart[15].rotation = Quaternion.LookRotation(RT_vec, RHN) * Quaternion.Inverse(MidRightThumb);
        
        // 右食指
        RI_vec = new Vector3(scaledData[48], scaledData[49], scaledData[50]) - PosRightIndex;
        BodyPart[16].rotation = Quaternion.LookRotation(RI_vec, RHN) * Quaternion.Inverse(MidRightIndex);
        
        // 右小指
        RL_vec = new Vector3(scaledData[48], scaledData[49], scaledData[50]) - PosRightLittle;
        BodyPart[17].rotation = Quaternion.LookRotation(RL_vec, RHN) * Quaternion.Inverse(MidRightLittle);
        
        // 右中指
        BodyPart[34].rotation = BodyPart[16].rotation;
        
        // 右无名指
        BodyPart[36].rotation = BodyPart[16].rotation;

        // 颈部
        PosHead = new Vector3((scaledData[21] + scaledData[24]) / 2.0f, (scaledData[22] + scaledData[25]) / 2.0f, (scaledData[23] + scaledData[26]) / 2.0f);
        gaze = new Vector3(scaledData[0], scaledData[1], scaledData[2]) - PosHead;
        BodyPart[24].rotation = Quaternion.LookRotation(gaze) * Quaternion.Inverse(MidNeck);
    }
    
    // 三角法向函数
    Vector3 TriangleNormal(Vector3 a, Vector3 b, Vector3 c)
    {
        Vector3 d1 = a - b;
        Vector3 d2 = a - c;

        Vector3 dd = Vector3.Cross(d1, d2);//向量叉乘
        dd.Normalize();

        return dd;
    }
    
    // 帮助调试用：显示找到的骨骼
    private void PrintFoundBones()
    {
        for (int i = 0; i < BodyPart.Length; i++)
        {
            if (BodyPart[i] != null)
            {
                Debug.Log($"找到骨骼 [{i}]: {boneNameMap[i]} -> {BodyPart[i].name}");
            }
            else
            {
                Debug.LogWarning($"未找到骨骼 [{i}]: {boneNameMap[i]}");
            }
        }
    }

    // 辅助方法：根据名称查找骨骼并赋值
    private void FindBoneWithNameAndAssign(Transform[] allBones, string nameToFind, int index)
    {
        foreach (Transform bone in allBones)
        {
            if (bone.name.IndexOf(nameToFind, System.StringComparison.OrdinalIgnoreCase) >= 0)
            {
                BodyPart[index] = bone;
                Debug.Log($"特殊查找到骨骼: {nameToFind} -> {bone.name} (索引:{index})");
                break;
            }
        }
    }
}