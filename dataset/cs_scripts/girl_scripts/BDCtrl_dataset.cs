using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using UnityEngine;

public class BDCtrl_dataset : MonoBehaviour
{
    public GameObject NeZha;//哪吒人物物体
    
    // 添加数据偏移设置
    [Header("数据偏移设置")]
    [Tooltip("X轴数据偏移量")]
    public float offsetX = 0f;
    [Tooltip("Y轴数据偏移量")]
    public float offsetY = 0f;
    [Tooltip("Z轴数据偏移量")]
    public float offsetZ = 0f;
    
    private Animator ani;//挂在哪吒上的动画组件
    
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
    
    // Start is called before the first frame update
    void Start()
    {
        BodyPart = new Transform[38];//unity中的25个身体关节
        ani = NeZha.GetComponent<Animator>();
        BodyPart[0] = ani.GetBoneTransform(HumanBodyBones.Hips);//*根节点*
        BodyPart[1] = ani.GetBoneTransform(HumanBodyBones.Spine);//脊柱
        BodyPart[2] = ani.GetBoneTransform(HumanBodyBones.Chest);//胸腔
        BodyPart[3] = ani.GetBoneTransform(HumanBodyBones.UpperChest);//上胸腔
        BodyPart[4] = ani.GetBoneTransform(HumanBodyBones.LeftShoulder);//左肩
        BodyPart[5] = ani.GetBoneTransform(HumanBodyBones.LeftUpperArm);//左上臂
        BodyPart[6] = ani.GetBoneTransform(HumanBodyBones.LeftLowerArm);//左下臂
        BodyPart[7] = ani.GetBoneTransform(HumanBodyBones.LeftHand);//左手
        BodyPart[8] = ani.GetBoneTransform(HumanBodyBones.LeftThumbIntermediate);//左拇指
        BodyPart[9] = ani.GetBoneTransform(HumanBodyBones.LeftIndexIntermediate);//左食指
        BodyPart[10] = ani.GetBoneTransform(HumanBodyBones.LeftLittleIntermediate);//左小指
        BodyPart[11] = ani.GetBoneTransform(HumanBodyBones.RightShoulder);//右肩
        BodyPart[12] = ani.GetBoneTransform(HumanBodyBones.RightUpperArm);//右上臂
        BodyPart[13] = ani.GetBoneTransform(HumanBodyBones.RightLowerArm);//右下臂
        BodyPart[14] = ani.GetBoneTransform(HumanBodyBones.RightHand);//右手
        BodyPart[15] = ani.GetBoneTransform(HumanBodyBones.RightThumbIntermediate);//右拇指
        BodyPart[16] = ani.GetBoneTransform(HumanBodyBones.RightIndexIntermediate);//右食指
        BodyPart[17] = ani.GetBoneTransform(HumanBodyBones.RightLittleIntermediate);//右小指
        BodyPart[18] = ani.GetBoneTransform(HumanBodyBones.LeftUpperLeg);//左上腿
        BodyPart[19] = ani.GetBoneTransform(HumanBodyBones.LeftLowerLeg);//左下腿
        BodyPart[20] = ani.GetBoneTransform(HumanBodyBones.LeftFoot);//左足
        BodyPart[21] = ani.GetBoneTransform(HumanBodyBones.RightUpperLeg);//右上腿
        BodyPart[22] = ani.GetBoneTransform(HumanBodyBones.RightLowerLeg);//右下腿
        BodyPart[23] = ani.GetBoneTransform(HumanBodyBones.RightFoot);//右足
        BodyPart[24] = ani.GetBoneTransform(HumanBodyBones.Neck);//颈部
        BodyPart[25] = ani.GetBoneTransform(HumanBodyBones.LeftToes);//左脚趾
        BodyPart[26] = ani.GetBoneTransform(HumanBodyBones.RightToes);//右脚趾
        BodyPart[27] = ani.GetBoneTransform(HumanBodyBones.LeftThumbDistal);//左拇指尖
        BodyPart[28]=ani.GetBoneTransform(HumanBodyBones.RightThumbDistal);//右拇指尖
        BodyPart[29] = ani.GetBoneTransform(HumanBodyBones.LeftIndexDistal);//左食指尖
        BodyPart[30] = ani.GetBoneTransform(HumanBodyBones.RightIndexDistal);//右食指尖
        BodyPart[31]= ani.GetBoneTransform(HumanBodyBones.LeftLittleDistal);//左小指指尖
        BodyPart[32] = ani.GetBoneTransform(HumanBodyBones.RightLittleDistal);//右小指指尖
        BodyPart[33] = ani.GetBoneTransform(HumanBodyBones.LeftMiddleIntermediate);//左中指
        BodyPart[34] = ani.GetBoneTransform(HumanBodyBones.RightMiddleIntermediate);//右中指
        BodyPart[35] = ani.GetBoneTransform(HumanBodyBones.LeftRingIntermediate);//左无名指
        BodyPart[36] = ani.GetBoneTransform(HumanBodyBones.RightRingIntermediate);//右无名指
        BodyPart[37] = ani.GetBoneTransform(HumanBodyBones.Head);//头部

        // 初始化中间矩阵
        InitializeMiddleMatrices();
    }

    // 处理肢体数据的公共方法 - 由FaceDataReceiver直接调用
    public void ProcessLimbData(float[] limbData, long timestamp)
    {
        if (limbData == null || limbData.Length != 99)
        {
            Debug.LogError($"接收到的数据长度不匹配: 预期99，实际{limbData?.Length ?? 0}");
            return;
        }

        // 删除不必要的日志输出
        
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
        
        // 对所有坐标点进行缩放并应用偏移量
        for (int i = 0; i < data.Length; i += 3)
        {
            if (i + 2 < data.Length) // 确保有完整的xyz三个值
            {
                scaledData[i] = data[i] / 100f + offsetX;      // x坐标缩放并添加偏移
                scaledData[i+1] = data[i+1] / 100f + offsetY;  // y坐标缩放并添加偏移
                scaledData[i+2] = data[i+2] / 300f + offsetZ;  // z坐标缩放并添加偏移
            }
        }
        
        // 使用缩放后的数据更新模型
        // 更新Hips位置
        BodyPart[0].position = new Vector3((scaledData[69] + scaledData[72]) / 2.0f, 
                                  (scaledData[70] + scaledData[73]) / 2.0f, 
                                  (scaledData[71] + scaledData[74]) / 2.0f);
        
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
        BodyPart[18].rotation = Quaternion.LookRotation(LUL_vec, forward) * Quaternion.Inverse(MidLeftUpperLeg) * Quaternion.Euler(0, 180, 0);
        
        // 计算右上腿与其子物体的方向
        RUL_vec = PosRightUpperLeg - new Vector3(scaledData[78], scaledData[79], scaledData[80]);
        
        // 更新右上腿的rotation
        BodyPart[21].rotation = Quaternion.LookRotation(RUL_vec, forward) * Quaternion.Inverse(MidRightUpperLeg) * Quaternion.Euler(0, 180, 0);
        
        // 计算左下腿与其子物体的方向
        LLL_vec = new Vector3(scaledData[75], scaledData[76], scaledData[77]) - new Vector3(scaledData[81], scaledData[82], scaledData[83]);
        
        // 更新左下腿的rotation
        BodyPart[19].rotation = Quaternion.LookRotation(LLL_vec, forward) * Quaternion.Inverse(MidLeftLowerLeg) * Quaternion.Euler(0, -180, 0);
        
        // 计算右下腿与其子物体的方向
        RLL_vec = new Vector3(scaledData[78], scaledData[79], scaledData[80]) - new Vector3(scaledData[84], scaledData[85], scaledData[86]);
        
        // 更新右下腿的rotation
        BodyPart[22].rotation = Quaternion.LookRotation(RLL_vec, forward) * Quaternion.Inverse(MidRightLowerLeg) * Quaternion.Euler(0, -180, 0);
        
        // 计算左足与其子物体的方向
        LF_vec = new Vector3(scaledData[81], scaledData[82], scaledData[83]) - new Vector3(scaledData[93], scaledData[94], scaledData[95]);
        
        // 更新左足的rotation
        BodyPart[20].rotation = Quaternion.LookRotation(LF_vec, forward) * Quaternion.Inverse(MidLeftFoot) * Quaternion.Euler(-45, 180, 0);
        
        // 计算右足与其子物体的方向
        RF_vec = new Vector3(scaledData[84], scaledData[85], scaledData[86]) - new Vector3(scaledData[96], scaledData[97], scaledData[98]);
        
        // 更新右足的rotation
        BodyPart[23].rotation = Quaternion.LookRotation(RF_vec, forward) * Quaternion.Inverse(MidRightFoot) * Quaternion.Euler(-45, 180, 0);

        // 计算Spine与其子物体的方向
        Sp_vec = BodyPart[0].position - new Vector3((scaledData[33] + scaledData[36]) / 2.0f, (scaledData[34] + scaledData[37]) / 2.0f, (scaledData[35] + scaledData[38]) / 2.0f);
        
        // 计算Spine的rotation
        BodyPart[1].rotation = Quaternion.LookRotation(Sp_vec,forward) * Quaternion.Inverse(MidSpine);
        
        // 计算Chest与其子物体的方向
        Ch_vec = Sp_vec;
        
        // 计算Chest的rotation
        BodyPart[2].rotation = Quaternion.LookRotation(Ch_vec,forward) * Quaternion.Inverse(MidChest);
        
        // 计算UpperChest的朝向
        UC_vec = Ch_vec;
        
        // 计算UpperChest的rotation
        BodyPart[3].rotation = Quaternion.LookRotation(Ch_vec, forward) * Quaternion.Inverse(MidUpperChest);

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
}