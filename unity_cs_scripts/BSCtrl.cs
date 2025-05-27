using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class BSCtrl : MonoBehaviour
{
    // 引用带有BlendShape的SkinnedMeshRenderer组件
    public SkinnedMeshRenderer skinnedMeshRenderer;
    
    // 存储所有BlendShape的权重值
    private Dictionary<string, float> blendShapeWeights = new Dictionary<string, float>();
    
    // 存储BlendShape的索引映射
    private Dictionary<string, int> blendShapeIndices = new Dictionary<string, int>();
    
    // 标准的52个BlendShape名称
    private Dictionary<int, string> blendShapeMap = new Dictionary<int, string>();
    
    // 用于存储时间戳和延时信息
    private long receivedTimestamp = 0;
    private float latency = 0f;
    private bool hasLatencyData = false;
    private bool needToCalculateLatency = false;  // 新增：标记是否需要计算延迟
    private Camera mainCamera;  // 新增：引用主相机

    // 公开属性供UI访问
    public float Latency { get { return latency; } }
    public bool HasLatencyData { get { return hasLatencyData; } }
    public SkinnedMeshRenderer SkinnedMeshRenderer { get { return skinnedMeshRenderer; } }
    
    // Start is called before the first frame update
    void Start()
    {
        // 初始化标准BlendShape映射
        InitBlendShapeMap();
        
        // 确保已经分配了SkinnedMeshRenderer
        if (skinnedMeshRenderer == null)
        {
            // 尝试从当前游戏对象获取SkinnedMeshRenderer
            skinnedMeshRenderer = GetComponent<SkinnedMeshRenderer>();
            
            // 如果仍然为空，尝试在子对象中查找
            if (skinnedMeshRenderer == null)
            {
                skinnedMeshRenderer = GetComponentInChildren<SkinnedMeshRenderer>();
                
                // 如果仍然找不到，记录错误
                if (skinnedMeshRenderer == null)
                {
                    Debug.LogError("无法找到SkinnedMeshRenderer组件，请手动分配");
                    return;
                }
            }
        }
        
        // 获取主相机引用
        mainCamera = Camera.main;
        if (mainCamera == null)
        {
            Debug.LogWarning("未找到主相机，将使用LateUpdate计算延迟");
        }
        
        // 查找并初始化标准BlendShape值
        FindAndInitializeBlendShapes();
        
        Debug.Log($"已初始化{blendShapeIndices.Count}/52个标准BlendShape控制器");
    }
    
    // 初始化52个标准BlendShape映射
    private void InitBlendShapeMap()
    {
        blendShapeMap.Add(0, "_neutral");
        blendShapeMap.Add(1, "browDownLeft");
        blendShapeMap.Add(2, "browDownRight");
        blendShapeMap.Add(3, "browInnerUp");
        blendShapeMap.Add(4, "browOuterUpLeft");
        blendShapeMap.Add(5, "browOuterUpRight");
        blendShapeMap.Add(6, "cheekPuff");
        blendShapeMap.Add(7, "cheekSquintLeft");
        blendShapeMap.Add(8, "cheekSquintRight");
        blendShapeMap.Add(9, "eyeBlinkLeft");
        blendShapeMap.Add(10, "eyeBlinkRight");
        blendShapeMap.Add(11, "eyeLookDownLeft");
        blendShapeMap.Add(12, "eyeLookDownRight");
        blendShapeMap.Add(13, "eyeLookInLeft");
        blendShapeMap.Add(14, "eyeLookInRight");
        blendShapeMap.Add(15, "eyeLookOutLeft");
        blendShapeMap.Add(16, "eyeLookOutRight");
        blendShapeMap.Add(17, "eyeLookUpLeft");
        blendShapeMap.Add(18, "eyeLookUpRight");
        blendShapeMap.Add(19, "eyeSquintLeft");
        blendShapeMap.Add(20, "eyeSquintRight");
        blendShapeMap.Add(21, "eyeWideLeft");
        blendShapeMap.Add(22, "eyeWideRight");
        blendShapeMap.Add(23, "jawForward");
        blendShapeMap.Add(24, "jawLeft");
        blendShapeMap.Add(25, "jawOpen");
        blendShapeMap.Add(26, "jawRight");
        blendShapeMap.Add(27, "mouthClose");
        blendShapeMap.Add(28, "mouthDimpleLeft");
        blendShapeMap.Add(29, "mouthDimpleRight");
        blendShapeMap.Add(30, "mouthFrownLeft");
        blendShapeMap.Add(31, "mouthFrownRight");
        blendShapeMap.Add(32, "mouthFunnel");
        blendShapeMap.Add(33, "mouthLeft");
        blendShapeMap.Add(34, "mouthLowerDownLeft");
        blendShapeMap.Add(35, "mouthLowerDownRight");
        blendShapeMap.Add(36, "mouthPressLeft");
        blendShapeMap.Add(37, "mouthPressRight");
        blendShapeMap.Add(38, "mouthPucker");
        blendShapeMap.Add(39, "mouthRight");
        blendShapeMap.Add(40, "mouthRollLower");
        blendShapeMap.Add(41, "mouthRollUpper");
        blendShapeMap.Add(42, "mouthShrugLower");
        blendShapeMap.Add(43, "mouthShrugUpper");
        blendShapeMap.Add(44, "mouthSmileLeft");
        blendShapeMap.Add(45, "mouthSmileRight");
        blendShapeMap.Add(46, "mouthStretchLeft");
        blendShapeMap.Add(47, "mouthStretchRight");
        blendShapeMap.Add(48, "mouthUpperUpLeft");
        blendShapeMap.Add(49, "mouthUpperUpRight");
        blendShapeMap.Add(50, "noseSneerLeft");
        blendShapeMap.Add(51, "noseSneerRight");
    }
    
    // 查找并初始化BlendShape
    private void FindAndInitializeBlendShapes()
    {
        if (skinnedMeshRenderer == null || skinnedMeshRenderer.sharedMesh == null)
            return;
            
        int blendShapeCount = skinnedMeshRenderer.sharedMesh.blendShapeCount;
        
        // 为每个标准BlendShape查找对应的索引
        foreach (var kvp in blendShapeMap)
        {
            int index = -1;
            string blendShapeName = kvp.Value;
            
            // 在模型中查找对应名称的BlendShape
            for (int i = 0; i < blendShapeCount; i++)
            {
                string name = skinnedMeshRenderer.sharedMesh.GetBlendShapeName(i);
                if (name == blendShapeName)
                {
                    index = i;
                    break;
                }
            }
            
            // 如果找到了，添加到映射和权重字典中
            if (index != -1)
            {
                blendShapeIndices[blendShapeName] = index;
                blendShapeWeights[blendShapeName] = skinnedMeshRenderer.GetBlendShapeWeight(index);
                Debug.Log($"找到BlendShape: {blendShapeName}, 索引: {index}");
            }
            else
            {
                Debug.LogWarning($"模型中未找到BlendShape: {blendShapeName}");
            }
        }
    }

    // 处理BlendShape数据的公共接口(老方法，保留兼容性)
    public void ProcessBlendShapeData(string data)
    {
        try
        {
            // 解析数据字符串
            string[] entries = data.Split(';');
            int successCount = 0;
            bool foundTimestamp = false;
            
            foreach (string entry in entries)
            {
                if (string.IsNullOrEmpty(entry)) 
                    continue;
                
                string[] parts = entry.Split(',');
                
                // 检查是否是时间戳数据
                if (parts.Length == 2 && parts[0] == "timestamp")
                {
                    if (long.TryParse(parts[1], out long timestamp))
                    {
                        receivedTimestamp = timestamp;
                        foundTimestamp = true;
                        needToCalculateLatency = true;
                        
                        // 添加调试信息，确认时间戳被正确接收
                        Debug.Log($"接收到时间戳: {timestamp}ms，将在下一帧计算延迟");
                    }
                    continue;
                }
                
                // 处理BlendShape数据
                if (parts.Length == 2)
                {
                    if (int.TryParse(parts[0], out int id) && float.TryParse(parts[1], out float value))
                    {
                        // 将接收到的值(0-1)乘以100转换为BlendShape权重值(0-100)
                        SetBlendShapeWeight(id, value * 100f);
                        successCount++;
                    }
                    else
                    {
                        Debug.LogWarning($"无法解析数值: {entry}");
                    }
                }
                else
                {
                    Debug.LogWarning($"无效数据格式: {entry}");
                }
            }
            
            // 不在这里计算延迟，而是等待渲染完成后
            if (foundTimestamp)
            {
                Debug.Log($"成功处理 {successCount}/{entries.Length - 1} 个BlendShape数据项，等待渲染完成后计算延迟");
            }
            else
            {
                Debug.Log($"成功处理 {successCount}/{entries.Length} 个BlendShape数据项");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"解析BlendShape数据错误: {e.Message}");
        }
    }
    
    // 处理数组形式的BlendShape数据(新方法)
    public void ProcessBlendShapeDataArray(float[] faceData, long timestamp)
    {
        try
        {
            int successCount = 0;
            
            // 设置时间戳
            receivedTimestamp = timestamp;
            needToCalculateLatency = true;
            
            // 处理最多52个BlendShape值
            int maxValues = Mathf.Min(faceData.Length, 52);
            for (int i = 0; i < maxValues; i++)
            {
                // 将接收到的值(0-1)乘以100转换为BlendShape权重值(0-100)
                SetBlendShapeWeight(i, faceData[i] * 100f);
                successCount++;
            }
            
            Debug.Log($"成功处理 {successCount}/52 个BlendShape数据项，等待渲染完成后计算延迟");
        }
        catch (Exception e)
        {
            Debug.LogError($"处理BlendShape数组数据错误: {e.Message}");
        }
    }
    
    // 使用Update而非LateUpdate，确保每帧都检查一次
    void Update()
    {
        // 立即计算延迟，不依赖于相机回调
        if (needToCalculateLatency && receivedTimestamp > 0)
        {
            CalculateLatencyNow();
        }
    }
    
    // 在渲染完成后计算延迟
    private void CalculateLatencyNow()
    {
        if (needToCalculateLatency && receivedTimestamp > 0)
        {
            // 计算当前延迟
            long currentTimestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            long delayMs = currentTimestamp - receivedTimestamp; // 保留毫秒级别
            latency = delayMs / 1000f; // 转换为秒，但保存为秒方便其他代码使用
            hasLatencyData = true;
            
            // 输出延迟信息到控制台，同时显示毫秒和秒
            Debug.Log($"[延迟计算] 数据时间戳: {receivedTimestamp}ms, 当前时间戳: {currentTimestamp}ms, 延迟: {delayMs}ms ({latency.ToString("F3")}秒)");
            
            // 重置标记
            needToCalculateLatency = false;
        }
    }
    
    // 设置特定BlendShape的权重（通过BlendShape名称）
    public void SetBlendShapeWeight(string name, float weight)
    {
        if (skinnedMeshRenderer == null || !blendShapeIndices.ContainsKey(name))
            return;
            
        // 限制权重在0-100范围内
        weight = Mathf.Clamp(weight, 0f, 100f);
        
        // 更新字典中的值
        blendShapeWeights[name] = weight;
        
        // 应用到模型
        int index = blendShapeIndices[name];
        skinnedMeshRenderer.SetBlendShapeWeight(index, weight);
    }
    
    // 设置特定BlendShape的权重（通过映射ID）
    public void SetBlendShapeWeight(int mapId, float weight)
    {
        if (skinnedMeshRenderer == null)
        {
            Debug.LogError("SkinnedMeshRenderer为空");
            return;
        }
        
        if (!blendShapeMap.ContainsKey(mapId))
        {
            Debug.LogWarning($"映射ID不存在: {mapId}");
            return;
        }
        
        string name = blendShapeMap[mapId];
        if (!blendShapeIndices.ContainsKey(name))
        {
            Debug.LogWarning($"BlendShape不存在: {name}");
            return;
        }
        
        // 限制权重在0-100范围内
        weight = Mathf.Clamp(weight, 0f, 100f);
        
        // 更新字典中的值
        blendShapeWeights[name] = weight;
        
        // 应用到模型
        int index = blendShapeIndices[name];
        skinnedMeshRenderer.SetBlendShapeWeight(index, weight);
    }
    
    // 重置所有BlendShape权重为0
    public void ResetAllBlendShapes()
    {
        if (skinnedMeshRenderer == null)
            return;
            
        foreach (var name in blendShapeIndices.Keys)
        {
            SetBlendShapeWeight(name, 0f);
        }
    }
}