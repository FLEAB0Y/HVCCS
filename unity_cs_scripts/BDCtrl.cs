using UnityEngine;
using System.Collections.Generic;
using System.Text;

public class BDCtrl : MonoBehaviour
{
    public GameObject fbxModel; // 拖放要检查的FBX模型到这里

    // MediaPipe完整的33个关键点
    private readonly string[] mediapipeJoints = new string[] {
        // 面部关键点 (0-10)
        "nose", 
        "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear",
        "mouth_left", "mouth_right",
        
        // 上半身关键点 (11-22)
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_pinky", "right_pinky", 
        "left_index", "right_index",
        "left_thumb", "right_thumb",
        
        // 下半身关键点 (23-32)
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
        "left_heel", "right_heel",
        "left_foot_index", "right_foot_index"
    };

    // MediaPipe关节到Unity人形骨骼的可能映射
    private readonly Dictionary<string, HumanBodyBones> mediapipeToHumanoid = new Dictionary<string, HumanBodyBones>
    {
        // 面部映射
        {"nose", HumanBodyBones.Head},
        {"left_eye", HumanBodyBones.LeftEye},
        {"right_eye", HumanBodyBones.RightEye},
        {"left_ear", HumanBodyBones.LeftEye}, // 近似
        {"right_ear", HumanBodyBones.RightEye}, // 近似
        {"mouth_left", HumanBodyBones.Jaw}, // 近似
        {"mouth_right", HumanBodyBones.Jaw}, // 近似
        
        // 上肢映射
        {"left_shoulder", HumanBodyBones.LeftShoulder},
        {"right_shoulder", HumanBodyBones.RightShoulder},
        {"left_elbow", HumanBodyBones.LeftLowerArm},
        {"right_elbow", HumanBodyBones.RightLowerArm},
        {"left_wrist", HumanBodyBones.LeftHand},
        {"right_wrist", HumanBodyBones.RightHand},
        {"left_index", HumanBodyBones.LeftIndexProximal},
        {"right_index", HumanBodyBones.RightIndexProximal},
        {"left_thumb", HumanBodyBones.LeftThumbProximal},
        {"right_thumb", HumanBodyBones.RightThumbProximal},
        {"left_pinky", HumanBodyBones.LeftLittleProximal},
        {"right_pinky", HumanBodyBones.RightLittleProximal},
        
        // 下肢映射
        {"left_hip", HumanBodyBones.LeftUpperLeg},
        {"right_hip", HumanBodyBones.RightUpperLeg},
        {"left_knee", HumanBodyBones.LeftLowerLeg},
        {"right_knee", HumanBodyBones.RightLowerLeg},
        {"left_ankle", HumanBodyBones.LeftFoot},
        {"right_ankle", HumanBodyBones.RightFoot},
        {"left_heel", HumanBodyBones.LeftFoot}, // 近似
        {"right_heel", HumanBodyBones.RightFoot}, // 近似
        {"left_foot_index", HumanBodyBones.LeftToes},
        {"right_foot_index", HumanBodyBones.RightToes}
    };

    void Start()
    {
        if (fbxModel == null)
        {
            Debug.LogError("请在Inspector中指定要检查的FBX模型!");
            return;
        }
        
        CheckModelSkeleton();
    }

    // 添加UI按钮调用此方法进行骨骼检查
    public void CheckModelSkeleton()
    {
        Animator animator = fbxModel.GetComponent<Animator>();
        if (animator == null)
        {
            Debug.LogError("FBX模型没有Animator组件!");
            return;
        }

        StringBuilder report = new StringBuilder();
        report.AppendLine("======== FBX模型骨骼检查报告 ========");
        report.AppendLine($"模型名称: {fbxModel.name}");
        report.AppendLine($"MediaPipe关键点数量: {mediapipeJoints.Length}");
        
        // 检查Avatar是否有效
        if (animator.avatar == null || !animator.avatar.isValid)
        {
            report.AppendLine("警告: 模型的Avatar无效或未配置!");
        }
        
        // 检查是否是Humanoid类型
        if (animator.isHuman)
        {
            report.AppendLine("模型类型: Humanoid (人形)");
            CheckHumanoidSkeleton(animator, report);
        }
        else
        {
            report.AppendLine("模型类型: Generic (通用)");
            report.AppendLine("警告: MediaPipe通常与Humanoid类型的模型配合更好。建议在导入设置中将模型设置为Humanoid类型。");
            
            // 打印骨骼层级结构
            Transform rootBone = FindRootBone(fbxModel.transform);
            if (rootBone != null)
            {
                report.AppendLine("\n骨骼层级结构:");
                PrintBoneHierarchy(rootBone, 0, report);
                
                // 检查是否有可能匹配MediaPipe输出的骨骼
                CheckGenericSkeletonCompatibility(rootBone, report);
            }
            else
            {
                report.AppendLine("无法找到根骨骼! 模型可能没有正确的骨骼结构。");
            }
        }
        
        report.AppendLine("\n=== MediaPipe与FBX骨骼映射建议 ===");
        report.AppendLine("要在Unity中使用MediaPipe数据控制此FBX模型:");
        report.AppendLine("1. 确保模型设置为Humanoid类型");
        report.AppendLine("2. 根据上面的骨骼映射关系编写适配脚本");
        report.AppendLine("3. 如有缺失骨骼，考虑使用相邻骨骼作为近似");
        
        report.AppendLine("\n======================================");
        Debug.Log(report.ToString());
    }
    
    // 检查人形骨骼
    void CheckHumanoidSkeleton(Animator animator, StringBuilder report)
    {
        report.AppendLine("\n人形骨骼映射:");
        
        // 用于跟踪已找到的骨骼
        Dictionary<HumanBodyBones, bool> foundBones = new Dictionary<HumanBodyBones, bool>();
        foreach (HumanBodyBones bone in System.Enum.GetValues(typeof(HumanBodyBones)))
        {
            if (bone != HumanBodyBones.LastBone)
            {
                foundBones[bone] = false;
            }
        }
        
        // 检查所有人形骨骼
        foreach (HumanBodyBones bone in System.Enum.GetValues(typeof(HumanBodyBones)))
        {
            if (bone != HumanBodyBones.LastBone)
            {
                Transform boneTransform = animator.GetBoneTransform(bone);
                if (boneTransform != null)
                {
                    report.AppendLine($"  {bone}: {boneTransform.name}");
                    foundBones[bone] = true;
                }
            }
        }
        
        // 检查MediaPipe兼容性
        report.AppendLine("\nMediaPipe兼容性分析:");
        
        foreach (var kvp in mediapipeToHumanoid)
        {
            string mediapipeJoint = kvp.Key;
            HumanBodyBones humanoidBone = kvp.Value;
            
            Transform boneTransform = animator.GetBoneTransform(humanoidBone);
            if (boneTransform != null)
            {
                report.AppendLine($"  MediaPipe '{mediapipeJoint}' → 对应骨骼 '{humanoidBone}' ({boneTransform.name})");
            }
            else
            {
                report.AppendLine($"  警告: MediaPipe '{mediapipeJoint}' 对应的骨骼 '{humanoidBone}' 未在模型中定义");
            }
        }
        
        // 检查缺失的关键骨骼
        report.AppendLine("\n缺失的关键骨骼:");
        bool missingAny = false;
        
        HumanBodyBones[] criticalBones = {
            HumanBodyBones.Hips, HumanBodyBones.Spine, HumanBodyBones.Head,
            HumanBodyBones.LeftShoulder, HumanBodyBones.LeftUpperArm, HumanBodyBones.LeftLowerArm, HumanBodyBones.LeftHand,
            HumanBodyBones.RightShoulder, HumanBodyBones.RightUpperArm, HumanBodyBones.RightLowerArm, HumanBodyBones.RightHand,
            HumanBodyBones.LeftUpperLeg, HumanBodyBones.LeftLowerLeg, HumanBodyBones.LeftFoot,
            HumanBodyBones.RightUpperLeg, HumanBodyBones.RightLowerLeg, HumanBodyBones.RightFoot
        };
        
        foreach (HumanBodyBones bone in criticalBones)
        {
            if (!foundBones[bone])
            {
                report.AppendLine($"  缺少: {bone}");
                missingAny = true;
            }
        }
        
        if (!missingAny)
        {
            report.AppendLine("  无缺失关键骨骼，基本结构适合MediaPipe动作控制");
        }
    }
    
    // 检查通用骨骼的兼容性
    void CheckGenericSkeletonCompatibility(Transform rootBone, StringBuilder report)
    {
        List<string> allBoneNames = new List<string>();
        CollectAllBoneNames(rootBone, allBoneNames);
        
        report.AppendLine("\nMediaPipe关节可能的映射:");
        
        foreach (string joint in mediapipeJoints)
        {
            string bestMatch = FindBestMatch(joint, allBoneNames);
            if (!string.IsNullOrEmpty(bestMatch))
            {
                report.AppendLine($"  MediaPipe '{joint}' → 可能映射到 '{bestMatch}'");
            }
            else
            {
                report.AppendLine($"  警告: MediaPipe '{joint}' 没有找到匹配骨骼");
            }
        }
    }
    
    // 找到根骨骼
    Transform FindRootBone(Transform root)
    {
        // 尝试找到常见的根骨骼名称
        foreach (Transform child in root)
        {
            string childName = child.name.ToLower();
            if (childName.Contains("armature") || childName.Contains("skeleton") || 
                childName.Contains("rig") || childName.Contains("root") || childName.Contains("hips"))
            {
                return child;
            }
        }
        
        // 如果没找到明确的骨骼根节点，返回第一个有子物体的对象
        foreach (Transform child in root)
        {
            if (child.childCount > 0)
            {
                return child;
            }
        }
        
        return null;
    }
    
    // 打印骨骼层级结构
    void PrintBoneHierarchy(Transform bone, int depth, StringBuilder report)
    {
        string indent = new string(' ', depth * 2);
        report.AppendLine($"{indent}+ {bone.name}");
        
        foreach (Transform child in bone)
        {
            PrintBoneHierarchy(child, depth + 1, report);
        }
    }
    
    // 收集所有骨骼名称
    void CollectAllBoneNames(Transform bone, List<string> names)
    {
        names.Add(bone.name);
        
        foreach (Transform child in bone)
        {
            CollectAllBoneNames(child, names);
        }
    }
    
    // 找到最佳匹配的骨骼名称
    string FindBestMatch(string mediapipeJoint, List<string> boneNames)
    {
        // 创建几种可能的变体
        string[] variants = {
            mediapipeJoint,
            mediapipeJoint.Replace("_", ""),
            mediapipeJoint.Replace("left_", "l_"),
            mediapipeJoint.Replace("right_", "r_"),
            mediapipeJoint.Replace("left_", "left"),
            mediapipeJoint.Replace("right_", "right")
        };
        
        foreach (string variant in variants)
        {
            foreach (string boneName in boneNames)
            {
                if (boneName.ToLower().Contains(variant.ToLower()))
                {
                    return boneName;
                }
            }
        }
        
        return string.Empty;
    }

    // 此方法可从Editor按钮或其他组件调用
    public void GenerateCompatibilityReport()
    {
        CheckModelSkeleton();
    }
}