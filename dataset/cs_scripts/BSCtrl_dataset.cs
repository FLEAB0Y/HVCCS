using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class BSCtrl_dataset : MonoBehaviour
{
    // 引用带有BlendShape的SkinnedMeshRenderer组件
    public SkinnedMeshRenderer skinnedMeshRenderer;
    
    // 存储所有BlendShape的权重值
    private Dictionary<string, float> blendShapeWeights = new Dictionary<string, float>();
    
    // 存储BlendShape的索引映射
    private Dictionary<string, int> blendShapeIndices = new Dictionary<string, int>();
    
    // 标准的52个BlendShape名称
    private Dictionary<int, string> blendShapeMap = new Dictionary<int, string>();
    
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
            
            foreach (string entry in entries)
            {
                if (string.IsNullOrEmpty(entry)) 
                    continue;
                
                string[] parts = entry.Split(',');
                
                // 检查是否是时间戳数据 - 忽略时间戳
                if (parts.Length == 2 && parts[0] == "timestamp")
                {
                    continue;
                }
                
                // 处理BlendShape数据
                if (parts.Length == 2)
                {
                    string blendShapeName = parts[0];
                    if (blendShapeIndices.TryGetValue(blendShapeName, out int index))
                    {
                        if (float.TryParse(parts[1], out float weight))
                        {
                            // 设置BlendShape权重
                            SetBlendShapeWeight(index, weight);
                            successCount++;
                        }
                    }
                }
            }
            
            Debug.Log($"成功处理 {successCount} 个BlendShape数据项");
        }
        catch (Exception e)
        {
            Debug.LogError($"处理BlendShape数据错误: {e.Message}");
        }
    }
    
    // 处理BlendShape数据的优化接口
    public void ProcessBlendShapeDataArray(float[] faceData, long timestamp)
    {
        try
        {
            int successCount = 0;
            
            // 处理最多52个BlendShape值
            int maxValues = Mathf.Min(faceData.Length, 52);
            for (int i = 0; i < maxValues; i++)
            {
                // 将接收到的值(0-1)乘以100转换为BlendShape权重值(0-100)
                SetBlendShapeWeight(i, faceData[i] * 100f);
                successCount++;
            }
            
            Debug.Log($"成功处理 {successCount}/52 个BlendShape数据项");
        }
        catch (Exception e)
        {
            Debug.LogError($"处理BlendShape数组数据错误: {e.Message}");
        }
    }

    // 设置BlendShape权重
    private void SetBlendShapeWeight(int index, float weight)
    {
        if (skinnedMeshRenderer == null)
            return;
        
        // 限制权重在0到100之间
        weight = Mathf.Clamp(weight, 0f, 100f);
        
        // 设置BlendShape权重
        skinnedMeshRenderer.SetBlendShapeWeight(index, weight);
        
        // 更新权重字典
        foreach (var kvp in blendShapeIndices)
        {
            if (kvp.Value == index)
            {
                blendShapeWeights[kvp.Key] = weight;
                break;
            }
        }
    }
}