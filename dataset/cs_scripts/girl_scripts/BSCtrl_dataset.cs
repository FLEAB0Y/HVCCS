using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

namespace DatasetScripts
{
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
            
            // 添加调试代码，打印出实际的BlendShape数量和名称
            Debug.Log($"模型中共有 {blendShapeCount} 个BlendShape");
            
            // 创建模型中所有BlendShape的映射
            Dictionary<string, int> modelBlendShapeMap = new Dictionary<string, int>();
            for (int i = 0; i < blendShapeCount; i++)
            {
                string name = skinnedMeshRenderer.sharedMesh.GetBlendShapeName(i);
                Debug.Log($"找到BlendShape: 索引 {i}, 名称 \"{name}\"");
                modelBlendShapeMap[name] = i;
            }
            
            // 检测是否有共同前缀
            string prefix = DetectCommonPrefix(modelBlendShapeMap.Keys);
            if (!string.IsNullOrEmpty(prefix))
            {
                Debug.Log($"检测到共同前缀: \"{prefix}\"");
            }
            
            // 遍历标准BlendShape映射
            foreach (var kvp in blendShapeMap)
            {
                int index = -1;
                string blendShapeName = kvp.Value;
                string searchName = prefix + blendShapeName;
                
                // 尝试精确匹配（包含前缀）
                if (modelBlendShapeMap.TryGetValue(searchName, out index))
                {
                    blendShapeIndices[blendShapeName] = index;
                    blendShapeWeights[blendShapeName] = skinnedMeshRenderer.GetBlendShapeWeight(index);
                    continue;
                }
                
                // 尝试无前缀精确匹配
                if (modelBlendShapeMap.TryGetValue(blendShapeName, out index))
                {
                    blendShapeIndices[blendShapeName] = index;
                    blendShapeWeights[blendShapeName] = skinnedMeshRenderer.GetBlendShapeWeight(index);
                    continue;
                }
                
                // 如果精确匹配失败，尝试后缀匹配
                foreach (var modelName in modelBlendShapeMap.Keys)
                {
                    if (modelName.EndsWith(blendShapeName))
                    {
                        index = modelBlendShapeMap[modelName];
                        blendShapeIndices[blendShapeName] = index;
                        blendShapeWeights[blendShapeName] = skinnedMeshRenderer.GetBlendShapeWeight(index);
                        Debug.Log($"通过后缀匹配: 标准名称 \"{blendShapeName}\" 匹配到模型BlendShape \"{modelName}\"");
                        break;
                    }
                }
                
                // 如果仍然没有找到匹配项
                if (!blendShapeIndices.ContainsKey(blendShapeName))
                {
                    Debug.LogWarning($"模型中未找到BlendShape: {blendShapeName}");
                }
            }
        }

        // 检测共同前缀
        private string DetectCommonPrefix(IEnumerable<string> names)
        {
            string prefix = "";
            bool foundPrefix = false;
            
            // 查找第一个包含标准BlendShape名称的模型名称
            foreach (var name in names)
            {
                foreach (var standardName in blendShapeMap.Values)
                {
                    if (standardName == "_neutral") continue; // 跳过特殊情况
                    
                    if (name.EndsWith(standardName) && name != standardName)
                    {
                        // 找到前缀
                        prefix = name.Substring(0, name.Length - standardName.Length);
                        foundPrefix = true;
                        break;
                    }
                }
                
                if (foundPrefix) break;
            }
            
            return prefix;
        }

        // 处理BlendShape数据的公共接口(老方法，保留兼容性)
        public void ProcessBlendShapeData(string data)
        {
            try
            {
                // 解析数据字符串
                string[] entries = data.Split(';');
                
                foreach (string entry in entries)
                {
                    if (string.IsNullOrEmpty(entry)) 
                        continue;
                    
                    string[] parts = entry.Split(',');
                    
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
                            }
                        }
                    }
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"处理BlendShape数据错误: {e.Message}");
            }
        }
        
        // 处理BlendShape数据的优化接口
        public void ProcessBlendShapeDataArray(float[] faceData, long _)
        {
            try
            {
                // 处理最多52个BlendShape值
                int maxValues = Mathf.Min(faceData.Length, 52);
                for (int i = 0; i < maxValues; i++)
                {
                    // 将接收到的值(0-1)乘以100转换为BlendShape权重值(0-100)
                    SetBlendShapeWeight(i, faceData[i] * 100f);
                }
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
}