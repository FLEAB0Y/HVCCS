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
    
    // 调试GUI开关
    public bool showDebugGUI = true;
    
    // 滚动视图位置
    private Vector2 scrollPosition;

    // 用于存储时间戳和延时信息
    private long receivedTimestamp = 0;
    private float latency = 0f;
    private bool hasLatencyData = false;

    // 用于存储时延历史数据
    private List<float> latencyHistory = new List<float>();
    // 最大存储30秒的数据，假设每秒5个采样点(每200ms一个点)
    private int maxHistoryPoints = 150;
    // 平均时延值
    private float averageLatency = 0f;
    // 用于控制采样率的计时器
    private float latencySampleTimer = 0f;
    // 采样间隔(秒)
    private float latencySampleInterval = 0.2f;

    // GUI 位置偏移量 - 用于多实例显示
    public Vector2 guiOffset = Vector2.zero;
    
    // 显示名称 - 用于标识不同实例
    public string displayName = "";

    private bool guiCollapsed = false; // 控制GUI折叠状态
    
    // Start is called before the first frame update
    void Start()
    {
        // 如果未设置显示名称，则使用游戏对象名称
        if (string.IsNullOrEmpty(displayName))
        {
            displayName = gameObject.name;
        }
        
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

    // 处理BlendShape数据的公共接口
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
                        
                        // 计算延迟（当前时间 - 发送时间）
                        long currentTimestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
                        latency = (currentTimestamp - receivedTimestamp) / 1000f; // 转换为秒
                        hasLatencyData = true;
                        
                        Debug.Log($"接收到时间戳: {receivedTimestamp}, 延迟: {latency.ToString("F3")}秒");
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
            
            if (foundTimestamp)
            {
                Debug.Log($"成功处理 {successCount}/{entries.Length - 1} 个BlendShape数据项，延迟: {latency.ToString("F3")}秒");
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

    // Update方法用于定期添加时延数据到历史记录
    void Update()
    {
        // 记录时延数据
        if (hasLatencyData)
        {
            latencySampleTimer += Time.deltaTime;
            if (latencySampleTimer >= latencySampleInterval)
            {
                latencySampleTimer = 0f;
                
                // 添加时延数据到历史记录
                latencyHistory.Add(latency);
                
                // 如果超过最大点数，移除最早的数据点
                if (latencyHistory.Count > maxHistoryPoints)
                {
                    latencyHistory.RemoveAt(0);
                }
                
                // 计算平均时延
                float sum = 0f;
                foreach (float value in latencyHistory)
                {
                    sum += value;
                }
                averageLatency = latencyHistory.Count > 0 ? sum / latencyHistory.Count : 0f;
            }
        }
    }
    
    // 根据时延值返回对应颜色
    private Color GetLatencyColor(float latencyValue)
    {
        if (latencyValue > 0.25f) // 大于250ms
            return Color.red;
        else if (latencyValue > 0.09f) // 90ms到250ms之间
            return Color.yellow;
        else // 小于90ms
            return Color.green;
    }

    // 用于调试的GUI
    void OnGUI()
    {
        if (!showDebugGUI || skinnedMeshRenderer == null)
            return;
        
        // 保存原始字体大小
        int originalLabelSize = GUI.skin.label.fontSize;
        int originalBoxSize = GUI.skin.box.fontSize;
        int originalButtonSize = GUI.skin.button.fontSize;
        
        // 设置较大字体大小
        GUI.skin.label.fontSize = 16;
        GUI.skin.box.fontSize = 18;
        GUI.skin.button.fontSize = 16;
        
        // 创建标题栏区域（始终显示）
        GUILayout.BeginArea(new Rect(10 + guiOffset.x, 10 + guiOffset.y, 400, 40));
        GUILayout.BeginHorizontal();
        GUILayout.Label($"{displayName}传输状态", GUI.skin.box, GUILayout.Width(300));
        
        // 添加折叠按钮
        if (GUILayout.Button(guiCollapsed ? "▼" : "▲", GUILayout.Width(40)))
        {
            guiCollapsed = !guiCollapsed;
        }
        GUILayout.EndHorizontal();
        GUILayout.EndArea();
        
        // 如果GUI没有折叠，则显示详细信息
        if (!guiCollapsed)
        {
            // 创建详细信息区域
            GUILayout.BeginArea(new Rect(10 + guiOffset.x, 50 + guiOffset.y, 400, 40));
            
            // 添加控制按钮和延迟显示
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("重置所有"))
            {
                ResetAllBlendShapes();
            }
            
            // 显示延迟信息
            if (hasLatencyData)
            {
                // 根据时延值设置颜色
                GUI.color = GetLatencyColor(latency);
                GUILayout.Label($"当前延迟: {latency.ToString("F3")}秒", GUILayout.Width(180));
                GUI.color = GetLatencyColor(averageLatency);
                GUILayout.Label($"平均延迟: {averageLatency.ToString("F3")}秒", GUILayout.Width(180));
                GUI.color = Color.white;
            }
            GUILayout.EndHorizontal();
            
            GUILayout.EndArea();
            
            // 绘制时延曲线
            if (hasLatencyData)
            {
                // 为时延曲线创建区域
                Rect graphRect = new Rect(10 + guiOffset.x, 90 + guiOffset.y, 400, 120);
                GUI.Box(graphRect, "时延曲线 (30秒)");
                
                // 绘制曲线
                DrawLatencyGraph(graphRect);
            }
        }
        
        // 恢复原始字体大小
        GUI.skin.label.fontSize = originalLabelSize;
        GUI.skin.box.fontSize = originalBoxSize;
        GUI.skin.button.fontSize = originalButtonSize;
    }
    
    // 绘制时延曲线
    private void DrawLatencyGraph(Rect rect)
    {
        // 设置绘图区域的内边距
        float padding = 10f;
        Rect drawArea = new Rect(
            rect.x + padding, 
            rect.y + padding, 
            rect.width - padding * 2, 
            rect.height - padding * 2);
        
        // 如果没有数据点，不绘制
        if (latencyHistory.Count == 0)
            return;
        
        // 找出最大时延值，用于缩放
        float maxLatency = 0.3f; // 默认最大时延300ms
        foreach (float value in latencyHistory)
        {
            if (value > maxLatency)
                maxLatency = value;
        }
        maxLatency = Mathf.Ceil(maxLatency * 10) / 10f; // 向上取整到下一个100ms
        
        // 绘制网格线和标签
        DrawGraphGrid(drawArea, maxLatency);
        
        // 绘制时延数据（使用垂直线）
        int lineWidth = Mathf.Max(1, Mathf.FloorToInt(drawArea.width / maxHistoryPoints));
        
        for (int i = 0; i < latencyHistory.Count; i++)
        {
            float x = drawArea.x + (i / (float)(maxHistoryPoints - 1)) * drawArea.width;
            float height = (latencyHistory[i] / maxLatency) * drawArea.height;
            float y = drawArea.y + drawArea.height - height;
            
            // 根据时延值选择颜色
            GUI.color = GetLatencyColor(latencyHistory[i]);
            
            // 画垂直线（用矩形表示）
            GUI.DrawTexture(new Rect(x, y, lineWidth, height), Texture2D.whiteTexture);
        }
        
        // 重置颜色
        GUI.color = Color.white;
    }
    
    // 绘制网格线和标签
    private void DrawGraphGrid(Rect area, float maxLatency)
    {
        // 绘制水平网格线
        float[] thresholds = { 0f, 0.09f, 0.25f }; // 重要阈值
        string[] labels = { "0ms", "90ms", "250ms" };
        
        for (int i = 0; i < thresholds.Length; i++)
        {
            float normalizedY = thresholds[i] / maxLatency;
            if (normalizedY > 1) continue; // 如果超出范围则跳过
            
            float y = area.y + area.height - normalizedY * area.height;
            
            // 绘制网格线
            GUI.color = new Color(0.5f, 0.5f, 0.5f, 0.5f);
            GUI.DrawTexture(new Rect(area.x, y, area.width, 1), Texture2D.whiteTexture);
            
            // 标签
            GUI.color = Color.white;
            GUI.Label(new Rect(area.x - 40, y - 10, 40, 20), labels[i]);
        }
        
        // 绘制垂直网格线（时间刻度）
        int verticalSegments = 6; // 分6段，每段5秒
        for (int i = 0; i <= verticalSegments; i++)
        {
            float x = area.x + (i / (float)verticalSegments) * area.width;
            
            // 绘制网格线
            GUI.color = new Color(0.5f, 0.5f, 0.5f, 0.5f);
            GUI.DrawTexture(new Rect(x, area.y, 1, area.height), Texture2D.whiteTexture);
            
            // 标签 - 显示秒数
            int seconds = (verticalSegments - i) * 30 / verticalSegments;
            if (i < verticalSegments) // 不在最右边显示标签
            {
                GUI.color = Color.white;
                GUI.Label(new Rect(x - 10, area.y + area.height + 5, 30, 20), $"{seconds}s");
            }
        }
        
        // 重置颜色
        GUI.color = Color.white;
    }
}
