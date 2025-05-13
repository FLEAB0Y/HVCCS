using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class UI : MonoBehaviour
{
    // 引用多个BlendShape控制器
    [SerializeField] private List<BSCtrl> blendShapeControllers = new List<BSCtrl>();

    // 调试GUI开关
    public bool showDebugGUI = true;
    
    // 每个控制器的GUI显示配置
    [System.Serializable]
    public class ControllerUIConfig
    {
        public BSCtrl controller;
        public string displayName = "";
        public Vector2 guiOffset = Vector2.zero;
        public bool guiCollapsed = false;
    }
    
    [SerializeField] private List<ControllerUIConfig> controllerConfigs = new List<ControllerUIConfig>();
    
    // 每个控制器的延时数据
    private Dictionary<BSCtrl, List<float>> latencyHistories = new Dictionary<BSCtrl, List<float>>();
    private Dictionary<BSCtrl, float> averageLatencies = new Dictionary<BSCtrl, float>();
    private Dictionary<BSCtrl, float> latencySampleTimers = new Dictionary<BSCtrl, float>();
    
    // 最大存储30秒的数据，假设每秒5个采样点(每200ms一个点)
    private int maxHistoryPoints = 150;
    // 采样间隔(秒)
    private float latencySampleInterval = 0.2f;
    
    // Start is called before the first frame update
    void Start()
    {
        // 如果控制器配置列表为空，从控制器列表初始化
        if (controllerConfigs.Count == 0 && blendShapeControllers.Count > 0)
        {
            foreach (var controller in blendShapeControllers)
            {
                if (controller != null)
                {
                    ControllerUIConfig config = new ControllerUIConfig
                    {
                        controller = controller,
                        displayName = controller.gameObject.name
                    };
                    controllerConfigs.Add(config);
                }
            }
        }
        
        // 初始化延时数据存储
        foreach (var config in controllerConfigs)
        {
            if (config.controller != null)
            {
                latencyHistories[config.controller] = new List<float>();
                averageLatencies[config.controller] = 0f;
                latencySampleTimers[config.controller] = 0f;
            }
        }
    }

    // Update方法用于定期添加时延数据到历史记录
    void Update()
    {
        foreach (var config in controllerConfigs)
        {
            BSCtrl controller = config.controller;
            if (controller == null || !controller.HasLatencyData)
                continue;
                
            // 记录时延数据
            latencySampleTimers[controller] += Time.deltaTime;
            if (latencySampleTimers[controller] >= latencySampleInterval)
            {
                latencySampleTimers[controller] = 0f;
                
                // 添加时延数据到历史记录
                latencyHistories[controller].Add(controller.Latency);
                
                // 如果超过最大点数，移除最早的数据点
                if (latencyHistories[controller].Count > maxHistoryPoints)
                {
                    latencyHistories[controller].RemoveAt(0);
                }
                
                // 计算平均时延
                float sum = 0f;
                foreach (float value in latencyHistories[controller])
                {
                    sum += value;
                }
                averageLatencies[controller] = latencyHistories[controller].Count > 0 ? 
                    sum / latencyHistories[controller].Count : 0f;
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
        if (!showDebugGUI)
            return;
        
        // 保存原始字体大小
        int originalLabelSize = GUI.skin.label.fontSize;
        int originalBoxSize = GUI.skin.box.fontSize;
        int originalButtonSize = GUI.skin.button.fontSize;
        
        // 设置较大字体大小
        GUI.skin.label.fontSize = 16;
        GUI.skin.box.fontSize = 18;
        GUI.skin.button.fontSize = 16;
        
        // 为每个控制器绘制UI
        foreach (var config in controllerConfigs)
        {
            BSCtrl controller = config.controller;
            if (controller == null || controller.SkinnedMeshRenderer == null)
                continue;
                
            DrawControllerUI(config);
        }
        
        // 恢复原始字体大小
        GUI.skin.label.fontSize = originalLabelSize;
        GUI.skin.box.fontSize = originalBoxSize;
        GUI.skin.button.fontSize = originalButtonSize;
    }
    
    // 为单个控制器绘制UI
    private void DrawControllerUI(ControllerUIConfig config)
    {
        BSCtrl controller = config.controller;
        Vector2 offset = config.guiOffset;
        string name = !string.IsNullOrEmpty(config.displayName) ? config.displayName : controller.gameObject.name;
        
        // 创建标题栏区域（始终显示）
        GUILayout.BeginArea(new Rect(10 + offset.x, 10 + offset.y, 400, 40));
        GUILayout.BeginHorizontal();
        GUILayout.Label($"{name}传输状态", GUI.skin.box, GUILayout.Width(300));
        
        // 添加折叠按钮
        if (GUILayout.Button(config.guiCollapsed ? "▼" : "▲", GUILayout.Width(40)))
        {
            config.guiCollapsed = !config.guiCollapsed;
        }
        GUILayout.EndHorizontal();
        GUILayout.EndArea();
        
        // 如果GUI没有折叠，则显示详细信息
        if (!config.guiCollapsed)
        {
            // 创建详细信息区域
            GUILayout.BeginArea(new Rect(10 + offset.x, 50 + offset.y, 400, 40));
            
            // 添加控制按钮和延迟显示
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("重置所有"))
            {
                controller.ResetAllBlendShapes();
            }
            
            // 显示延迟信息
            if (controller.HasLatencyData)
            {
                // 根据时延值设置颜色
                GUI.color = GetLatencyColor(controller.Latency);
                GUILayout.Label($"当前延迟: {controller.Latency.ToString("F3")}秒", GUILayout.Width(180));
                GUI.color = GetLatencyColor(averageLatencies[controller]);
                GUILayout.Label($"平均延迟: {averageLatencies[controller].ToString("F3")}秒", GUILayout.Width(180));
                GUI.color = Color.white;
            }
            GUILayout.EndHorizontal();
            
            GUILayout.EndArea();
            
            // 绘制时延曲线
            if (controller.HasLatencyData && latencyHistories.ContainsKey(controller))
            {
                // 为时延曲线创建区域
                Rect graphRect = new Rect(10 + offset.x, 90 + offset.y, 400, 120);
                GUI.Box(graphRect, "时延曲线 (30秒)");
                
                // 绘制曲线
                DrawLatencyGraph(graphRect, controller);
            }
        }
    }
    
    // 绘制时延曲线
    private void DrawLatencyGraph(Rect rect, BSCtrl controller)
    {
        // 设置绘图区域的内边距
        float padding = 10f;
        Rect drawArea = new Rect(
            rect.x + padding, 
            rect.y + padding, 
            rect.width - padding * 2, 
            rect.height - padding * 2);
        
        List<float> history = latencyHistories[controller];
        
        // 如果没有数据点，不绘制
        if (history.Count == 0)
            return;
        
        // 找出最大时延值，用于缩放
        float maxLatency = 0.3f; // 默认最大时延300ms
        foreach (float value in history)
        {
            if (value > maxLatency)
                maxLatency = value;
        }
        maxLatency = Mathf.Ceil(maxLatency * 10) / 10f; // 向上取整到下一个100ms
        
        // 绘制网格线和标签
        DrawGraphGrid(drawArea, maxLatency);
        
        // 绘制时延数据（使用垂直线）
        int lineWidth = Mathf.Max(1, Mathf.FloorToInt(drawArea.width / maxHistoryPoints));
        
        for (int i = 0; i < history.Count; i++)
        {
            float x = drawArea.x + (i / (float)(maxHistoryPoints - 1)) * drawArea.width;
            float height = (history[i] / maxLatency) * drawArea.height;
            float y = drawArea.y + drawArea.height - height;
            
            // 根据时延值选择颜色
            GUI.color = GetLatencyColor(history[i]);
            
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
    
    // 添加控制器到监控列表
    public void AddController(BSCtrl controller, string name = "", Vector2 offset = new Vector2())
    {
        if (controller == null)
            return;
            
        // 检查是否已存在
        foreach (var config in controllerConfigs)
        {
            if (config.controller == controller)
                return;
        }
        
        // 添加新配置
        ControllerUIConfig newConfig = new ControllerUIConfig
        {
            controller = controller,
            displayName = !string.IsNullOrEmpty(name) ? name : controller.gameObject.name,
            guiOffset = offset
        };
        
        controllerConfigs.Add(newConfig);
        
        // 初始化延时数据
        latencyHistories[controller] = new List<float>();
        averageLatencies[controller] = 0f;
        latencySampleTimers[controller] = 0f;
    }
    
    // 移除控制器
    public void RemoveController(BSCtrl controller)
    {
        if (controller == null)
            return;
            
        // 移除配置
        for (int i = 0; i < controllerConfigs.Count; i++)
        {
            if (controllerConfigs[i].controller == controller)
            {
                controllerConfigs.RemoveAt(i);
                break;
            }
        }
        
        // 移除延时数据
        if (latencyHistories.ContainsKey(controller))
            latencyHistories.Remove(controller);
            
        if (averageLatencies.ContainsKey(controller))
            averageLatencies.Remove(controller);
            
        if (latencySampleTimers.ContainsKey(controller))
            latencySampleTimers.Remove(controller);
    }
}