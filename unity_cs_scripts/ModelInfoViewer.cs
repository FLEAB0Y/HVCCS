using UnityEngine;
using System.Collections.Generic;
using System.Linq;
using System.Text;
#if UNITY_EDITOR
using UnityEditor;
#endif

public class ModelInfoViewer : MonoBehaviour
{
    [Tooltip("目标模型")]
    public GameObject targetModel;
    
    [Tooltip("是否在启动时自动获取信息")]
    public bool autoGetInfoOnStart = false;
    
    [Tooltip("是否在控制台输出详细信息")]
    public bool logToConsole = true;
    
    [Tooltip("是否显示GUI界面")]
    public bool showGUI = true;
    
    [Tooltip("GUI窗口位置")]
    public Rect windowRect = new Rect(20, 20, 350, 500);
    
    // 保存 blendshapes 信息
    private List<string> blendshapes = new List<string>();
    
    // 保存骨骼信息
    private List<BoneInfo> bonesInfo = new List<BoneInfo>();
    
    // 骨骼变换GUI相关变量
    private int selectedBoneIndex = -1;
    private Transform selectedBone = null;
    private Vector3 originalPosition;
    private Quaternion originalRotation;
    private Vector3 originalScale;
    private Vector3 positionOffset = Vector3.zero;
    private Vector3 rotationOffset = Vector3.zero;
    private Vector3 scaleOffset = Vector3.one;
    private Vector2 scrollPosition = Vector2.zero;
    
    // 用于存储骨骼信息的结构
    public class BoneInfo
    {
        public string name;
        public List<string> properties = new List<string>();
        public Transform transform;
        
        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine($"  - {name}");
            string props = string.Join(", ", properties);
            sb.AppendLine($"    属性: {(props.Length > 0 ? props : "无")}");
            return sb.ToString();
        }
    }
    
    void Start()
    {
        if (autoGetInfoOnStart && targetModel != null)
        {
            GetModelInfo();
        }
    }
    
    void OnGUI()
    {
        if (showGUI && targetModel != null)
        {
            windowRect = GUI.Window(0, windowRect, DrawWindow, $"骨骼调整器 - {targetModel.name}");
        }
    }
    
    void DrawWindow(int windowID)
    {
        const float padding = 10f;
        const float controlHeight = 25f;
        const float sliderHeight = 20f;
        const float spacing = 5f;
        
        GUILayout.BeginVertical(GUILayout.ExpandHeight(true));
        
        // 选择骨骼
        GUILayout.Label("选择骨骼:", GUILayout.Height(controlHeight));
        
        // 骨骼列表
        if (bonesInfo.Count > 0)
        {
            // 使用滚动视图显示骨骼列表
            scrollPosition = GUILayout.BeginScrollView(scrollPosition, GUILayout.Height(150));
            
            // 当前选中的骨骼索引
            int newSelectedIndex = GUILayout.SelectionGrid(selectedBoneIndex, 
                bonesInfo.Select(b => b.name).ToArray(), 1);
            
            if (newSelectedIndex != selectedBoneIndex)
            {
                // 骨骼选择变更
                if (selectedBone != null)
                {
                    // 保存当前骨骼的原始变换
                    ResetBoneTransform();
                }
                
                selectedBoneIndex = newSelectedIndex;
                if (selectedBoneIndex >= 0 && selectedBoneIndex < bonesInfo.Count)
                {
                    selectedBone = bonesInfo[selectedBoneIndex].transform;
                    if (selectedBone != null)
                    {
                        // 保存新选中骨骼的原始变换
                        originalPosition = selectedBone.localPosition;
                        originalRotation = selectedBone.localRotation;
                        originalScale = selectedBone.localScale;
                        
                        // 重置偏移值
                        positionOffset = Vector3.zero;
                        rotationOffset = Vector3.zero;
                        scaleOffset = Vector3.one;
                    }
                }
            }
            
            GUILayout.EndScrollView();
        }
        else
        {
            GUILayout.Label("没有找到骨骼", GUILayout.Height(controlHeight));
        }
        
        GUILayout.Space(spacing);
        
        // 显示选中骨骼的控制界面
        if (selectedBone != null)
        {
            GUILayout.Label($"调整骨骼: {selectedBone.name}", GUILayout.Height(controlHeight));
            
            GUILayout.Space(spacing);
            
            // 位置控制
            GUILayout.Label("位置 (本地):", GUILayout.Height(controlHeight));
            
            GUILayout.BeginHorizontal();
            GUILayout.Label("X:", GUILayout.Width(20));
            positionOffset.x = GUILayout.HorizontalSlider(positionOffset.x, -1f, 1f);
            GUILayout.Label(positionOffset.x.ToString("F2"), GUILayout.Width(40));
            GUILayout.EndHorizontal();
            
            GUILayout.BeginHorizontal();
            GUILayout.Label("Y:", GUILayout.Width(20));
            positionOffset.y = GUILayout.HorizontalSlider(positionOffset.y, -1f, 1f);
            GUILayout.Label(positionOffset.y.ToString("F2"), GUILayout.Width(40));
            GUILayout.EndHorizontal();
            
            GUILayout.BeginHorizontal();
            GUILayout.Label("Z:", GUILayout.Width(20));
            positionOffset.z = GUILayout.HorizontalSlider(positionOffset.z, -1f, 1f);
            GUILayout.Label(positionOffset.z.ToString("F2"), GUILayout.Width(40));
            GUILayout.EndHorizontal();
            
            // 应用位置变换
            selectedBone.localPosition = originalPosition + positionOffset;
            
            GUILayout.Space(spacing);
            
            // 旋转控制
            GUILayout.Label("旋转 (本地):", GUILayout.Height(controlHeight));
            
            GUILayout.BeginHorizontal();
            GUILayout.Label("X:", GUILayout.Width(20));
            rotationOffset.x = GUILayout.HorizontalSlider(rotationOffset.x, -180f, 180f);
            GUILayout.Label(rotationOffset.x.ToString("F0") + "°", GUILayout.Width(40));
            GUILayout.EndHorizontal();
            
            GUILayout.BeginHorizontal();
            GUILayout.Label("Y:", GUILayout.Width(20));
            rotationOffset.y = GUILayout.HorizontalSlider(rotationOffset.y, -180f, 180f);
            GUILayout.Label(rotationOffset.y.ToString("F0") + "°", GUILayout.Width(40));
            GUILayout.EndHorizontal();
            
            GUILayout.BeginHorizontal();
            GUILayout.Label("Z:", GUILayout.Width(20));
            rotationOffset.z = GUILayout.HorizontalSlider(rotationOffset.z, -180f, 180f);
            GUILayout.Label(rotationOffset.z.ToString("F0") + "°", GUILayout.Width(40));
            GUILayout.EndHorizontal();
            
            // 应用旋转变换
            selectedBone.localRotation = originalRotation * Quaternion.Euler(rotationOffset);
            
            GUILayout.Space(spacing);
            
            // 缩放控制
            GUILayout.Label("缩放 (本地):", GUILayout.Height(controlHeight));
            
            GUILayout.BeginHorizontal();
            GUILayout.Label("X:", GUILayout.Width(20));
            scaleOffset.x = GUILayout.HorizontalSlider(scaleOffset.x, 0.5f, 1.5f);
            GUILayout.Label(scaleOffset.x.ToString("F2"), GUILayout.Width(40));
            GUILayout.EndHorizontal();
            
            GUILayout.BeginHorizontal();
            GUILayout.Label("Y:", GUILayout.Width(20));
            scaleOffset.y = GUILayout.HorizontalSlider(scaleOffset.y, 0.5f, 1.5f);
            GUILayout.Label(scaleOffset.y.ToString("F2"), GUILayout.Width(40));
            GUILayout.EndHorizontal();
            
            GUILayout.BeginHorizontal();
            GUILayout.Label("Z:", GUILayout.Width(20));
            scaleOffset.z = GUILayout.HorizontalSlider(scaleOffset.z, 0.5f, 1.5f);
            GUILayout.Label(scaleOffset.z.ToString("F2"), GUILayout.Width(40));
            GUILayout.EndHorizontal();
            
            // 应用缩放变换
            selectedBone.localScale = new Vector3(
                originalScale.x * scaleOffset.x,
                originalScale.y * scaleOffset.y,
                originalScale.z * scaleOffset.z
            );
            
            GUILayout.Space(spacing);
            
            // 重置按钮
            if (GUILayout.Button("重置变换", GUILayout.Height(controlHeight)))
            {
                ResetBoneTransform();
                positionOffset = Vector3.zero;
                rotationOffset = Vector3.zero;
                scaleOffset = Vector3.one;
            }
        }
        
        GUILayout.Space(spacing);
        
        // 刷新按钮
        if (GUILayout.Button("刷新模型信息", GUILayout.Height(controlHeight)))
        {
            GetModelInfo();
        }
        
        GUILayout.EndVertical();
        
        // 允许窗口拖动
        GUI.DragWindow();
    }
    
    // 重置骨骼变换为原始状态
    private void ResetBoneTransform()
    {
        if (selectedBone != null)
        {
            selectedBone.localPosition = originalPosition;
            selectedBone.localRotation = originalRotation;
            selectedBone.localScale = originalScale;
        }
    }
    
    /// <summary>
    /// 获取模型的 Blendshapes 和骨骼信息
    /// </summary>
    public void GetModelInfo()
    {
        if (targetModel == null)
        {
            Debug.LogError("请先指定目标模型!");
            return;
        }
        
        Debug.Log($"开始分析模型: {targetModel.name}");
        
        // 清除之前的数据
        blendshapes.Clear();
        bonesInfo.Clear();
        
        // 获取 SkinnedMeshRenderer 组件
        SkinnedMeshRenderer[] renderers = targetModel.GetComponentsInChildren<SkinnedMeshRenderer>();
        Debug.Log($"找到 {renderers.Length} 个 SkinnedMeshRenderer 组件:");
        foreach (SkinnedMeshRenderer renderer in renderers)
        {
            Debug.Log($"  - {renderer.name}");
        }
        
        // 获取 Blendshapes
        GetBlendshapes();
        Debug.Log($"找到 {blendshapes.Count} 个 Blendshapes:");
        if (blendshapes.Count > 0)
        {
            foreach (string blendshape in blendshapes)
            {
                Debug.Log($"  - Blendshape: {blendshape}");
            }
        }
        else
        {
            Debug.Log("  无 Blendshapes");
        }
        
        // 获取骨骼信息
        GetBonesInfo();
        Debug.Log($"找到 {bonesInfo.Count} 个骨骼:");
        if (bonesInfo.Count > 0)
        {
            foreach (BoneInfo bone in bonesInfo)
            {
                string properties = string.Join(", ", bone.properties);
                Debug.Log($"  - 骨骼: {bone.name} (属性: {(properties.Length > 0 ? properties : "无")})");
            }
        }
        else
        {
            Debug.Log("  无骨骼");
        }
        
        // 输出详细信息到控制台
        if (logToConsole)
        {
            LogModelInfo();
        }
    }
    
    /// <summary>
    /// 获取模型的 Blendshapes
    /// </summary>
    private void GetBlendshapes()
    {
        // 查找所有 SkinnedMeshRenderer 组件
        SkinnedMeshRenderer[] renderers = targetModel.GetComponentsInChildren<SkinnedMeshRenderer>();
        
        foreach (SkinnedMeshRenderer renderer in renderers)
        {
            Mesh mesh = renderer.sharedMesh;
            if (mesh != null && mesh.blendShapeCount > 0)
            {
                for (int i = 0; i < mesh.blendShapeCount; i++)
                {
                    string blendshapeName = mesh.GetBlendShapeName(i);
                    blendshapes.Add(blendshapeName);
                }
            }
        }
    }
    
    /// <summary>
    /// 获取模型的骨骼信息
    /// </summary>
    private void GetBonesInfo()
    {
        // 获取所有骨骼 (Transform)
        Transform[] allTransforms = targetModel.GetComponentsInChildren<Transform>();
        
        // 查找所有 SkinnedMeshRenderer 组件以确定哪些 Transform 是骨骼
        SkinnedMeshRenderer[] renderers = targetModel.GetComponentsInChildren<SkinnedMeshRenderer>();
        HashSet<Transform> boneTransforms = new HashSet<Transform>();
        
        // 存储骨骼控制的顶点数据
        Dictionary<Transform, int> boneVertexInfluences = new Dictionary<Transform, int>();
        
        foreach (SkinnedMeshRenderer renderer in renderers)
        {
            if (renderer.bones != null)
            {
                foreach (Transform bone in renderer.bones)
                {
                    if (bone != null)
                    {
                        boneTransforms.Add(bone);
                        
                        // 初始化骨骼顶点影响计数
                        if (!boneVertexInfluences.ContainsKey(bone))
                        {
                            boneVertexInfluences[bone] = 0;
                        }
                    }
                }
                
                // 尝试分析骨骼权重数据
                Mesh mesh = renderer.sharedMesh;
                if (mesh != null)
                {
                    // 分析骨骼绑定信息
                    Matrix4x4[] bindposes = mesh.bindposes;
                    if (bindposes != null && bindposes.Length > 0)
                    {
                        for (int i = 0; i < renderer.bones.Length && i < bindposes.Length; i++)
                        {
                            Transform bone = renderer.bones[i];
                            if (bone != null && boneVertexInfluences.ContainsKey(bone))
                            {
                                // 累加这个骨骼影响的顶点数量 (估计值)
                                boneVertexInfluences[bone] += mesh.vertexCount / 4; // 平均估算，每个顶点通常受到约4个骨骼影响
                            }
                        }
                    }
                }
            }
        }
        
        // 处理每个骨骼
        foreach (Transform boneTransform in boneTransforms)
        {
            BoneInfo boneInfo = new BoneInfo
            {
                name = boneTransform.name,
                transform = boneTransform  // 保存骨骼的Transform引用
            };
            
            // 确定骨骼属性
            if (boneTransform.childCount > 0)
            {
                boneInfo.properties.Add("有子骨骼");
            }
            
            // 添加父骨骼信息
            if (boneTransform.parent != null && boneTransforms.Contains(boneTransform.parent))
            {
                boneInfo.properties.Add($"父骨骼: {boneTransform.parent.name}");
            }
            
            // 控制的变换参数
            boneInfo.properties.Add($"位置: {boneTransform.localPosition}");
            boneInfo.properties.Add($"旋转: {boneTransform.localEulerAngles}");
            boneInfo.properties.Add($"缩放: {boneTransform.localScale}");
            
            // 检查是否被用于蒙皮以及影响的顶点
            foreach (SkinnedMeshRenderer renderer in renderers)
            {
                if (renderer.bones.Contains(boneTransform))
                {
                    // 添加影响的顶点数量估计
                    if (boneVertexInfluences.ContainsKey(boneTransform) && boneVertexInfluences[boneTransform] > 0)
                    {
                        boneInfo.properties.Add($"影响约 {boneVertexInfluences[boneTransform]} 个顶点");
                    }
                    
                    boneInfo.properties.Add("用于蒙皮");
                    break;
                }
            }
            
            // 检查动画属性
            Animator animator = targetModel.GetComponent<Animator>();
            if (animator != null)
            {
                // 检查这个骨骼是否为人形骨骼的一部分
                if (animator.isHuman)
                {
                    for (int i = 0; i < (int)HumanBodyBones.LastBone; i++)
                    {
                        HumanBodyBones humanBone = (HumanBodyBones)i;
                        Transform humanBoneTransform = animator.GetBoneTransform(humanBone);
                        if (humanBoneTransform == boneTransform)
                        {
                            boneInfo.properties.Add($"人形骨骼: {humanBone}");
                            break;
                        }
                    }
                }
            }
            
            bonesInfo.Add(boneInfo);
        }
        
        // 按层次结构排序
        bonesInfo = bonesInfo.OrderBy(b => GetBoneHierarchyDepth(b.transform)).ToList();
        
        // 重置选择状态
        selectedBoneIndex = -1;
        selectedBone = null;
    }
    
    // 获取骨骼在层次结构中的深度
    private int GetBoneHierarchyDepth(Transform bone)
    {
        int depth = 0;
        Transform current = bone;
        while (current != null && current != targetModel.transform)
        {
            depth++;
            current = current.parent;
        }
        return depth;
    }
    
    /// <summary>
    /// 在控制台输出模型信息
    /// </summary>
    public void LogModelInfo()
    {
        StringBuilder sb = new StringBuilder();
        
        sb.AppendLine($"模型: {targetModel.name}");
        
        sb.AppendLine("\nBlendshapes:");
        if (blendshapes.Count > 0)
        {
            foreach (string blendshape in blendshapes)
            {
                sb.AppendLine($"  - {blendshape}");
            }
        }
        else
        {
            sb.AppendLine("  无 Blendshapes");
        }
        
        sb.AppendLine("\n骨骼:");
        if (bonesInfo.Count > 0)
        {
            foreach (BoneInfo bone in bonesInfo)
            {
                sb.Append(bone.ToString());
            }
        }
        else
        {
            sb.AppendLine("  无骨骼");
        }
        
        Debug.Log(sb.ToString());
    }
    
    // 获取 Blendshapes 数量
    public int GetBlendshapeCount()
    {
        return blendshapes.Count;
    }
    
    // 获取骨骼数量
    public int GetBoneCount()
    {
        return bonesInfo.Count;
    }
    
    // 获取所有 Blendshapes 名称
    public List<string> GetAllBlendshapes()
    {
        return new List<string>(blendshapes);
    }
    
    // 获取所有骨骼信息
    public List<BoneInfo> GetAllBones()
    {
        return new List<BoneInfo>(bonesInfo);
    }
}

#if UNITY_EDITOR
// 编辑器扩展，添加一个按钮用于获取信息
[CustomEditor(typeof(ModelInfoViewer))]
public class ModelInfoViewerEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();
        
        ModelInfoViewer viewer = (ModelInfoViewer)target;
        
        EditorGUILayout.Space();
        
        if (GUILayout.Button("获取模型信息"))
        {
            viewer.GetModelInfo();
        }
    }
}
#endif