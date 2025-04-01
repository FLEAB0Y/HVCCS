using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using UnityEngine;

public class BlendShapeController : MonoBehaviour
{
    public GameObject targetModel;
    public string modelName = "nezha_with_backgoud_spotlight";
    public int listenPort = 5556; // 不同于相机流的端口
    public bool printDebugInfo = true; // 控制是否打印每次设置的详细信息
    
    private TcpListener server;
    private Thread listenerThread;
    private bool isRunning = false;
    private Dictionary<int, string> blendShapeMap = new Dictionary<int, string>();
    private SkinnedMeshRenderer[] renderers;
    
    // 用于存储最新的BlendShape数据
    private Dictionary<int, float> blendShapeValues = new Dictionary<int, float>();
    private bool hasNewData = false;
    
    // 用于线程安全的数据交换
    private object lockObject = new object();
    
    void Start()
    {
        // 初始化BlendShape映射
        InitBlendShapeMap();
        
        // 查找目标模型
        if (targetModel == null)
        {
            targetModel = GameObject.Find(modelName);
            if (targetModel == null)
            {
                Debug.LogError($"未找到名为 {modelName} 的游戏对象");
                return;
            }
        }
        
        // 获取SkinnedMeshRenderer组件
        renderers = targetModel.GetComponentsInChildren<SkinnedMeshRenderer>(true);
        if (renderers.Length == 0)
        {
            Debug.LogError("没有找到SkinnedMeshRenderer组件");
            return;
        }
        
        // 启动TCP监听线程
        isRunning = true;
        listenerThread = new Thread(new ThreadStart(ListenForData));
        listenerThread.IsBackground = true;
        listenerThread.Start();
        
        Debug.Log($"BlendShape控制器已启动，监听端口: {listenPort}");
    }
    
    void Update()
    {
        // 在主线程中应用BlendShape权重
        if (hasNewData)
        {
            ApplyBlendShapeValues();
            hasNewData = false;
        }
    }
    
    void OnDestroy()
    {
        // 停止线程和网络连接
        isRunning = false;
        if (server != null)
        {
            server.Stop();
        }
        
        if (listenerThread != null && listenerThread.IsAlive)
        {
            listenerThread.Abort();
        }
    }
    
    private void ListenForData()
    {
        try
        {
            server = new TcpListener(IPAddress.Any, listenPort);
            server.Start();
            
            byte[] buffer = new byte[8192]; // 增大缓冲区
            
            Debug.Log("开始监听BlendShape数据...");
            
            while (isRunning)
            {
                // 非阻塞检查是否有新连接
                if (server.Pending())
                {
                    TcpClient client = server.AcceptTcpClient();
                    ProcessClientData(client, buffer);
                }
                
                // 在循环中短暂休眠，避免CPU占用过高
                Thread.Sleep(5);
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"BlendShape监听器错误: {e.Message}");
        }
    }
    
    private void ProcessClientData(TcpClient client, byte[] buffer)
    {
        try
        {
            using (client)
            {
                NetworkStream stream = client.GetStream();
                
                // 为此客户端设置短暂的超时，确保不会无限阻塞
                stream.ReadTimeout = 1000;
                
                int bytesRead;
                while (client.Connected && (bytesRead = stream.Read(buffer, 0, buffer.Length)) > 0)
                {
                    string data = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                    ProcessBlendShapeData(data);
                    
                    // 非阻塞模式下，检查是否还有更多数据
                    if (!stream.DataAvailable)
                        break;
                }
            }
        }
        catch (Exception e)
        {
            if (!(e is System.IO.IOException)) // 忽略连接关闭的异常
            {
                Debug.LogError($"处理客户端数据时出错: {e.Message}");
            }
        }
    }
    
    private void ProcessBlendShapeData(string data)
    {
        try
        {
            // 直接解析 "index,value;index,value;..." 格式的数据
            string[] pairs = data.Split(';');
            
            lock (lockObject)
            {
                foreach (string pair in pairs)
                {
                    if (!string.IsNullOrEmpty(pair))
                    {
                        string[] values = pair.Split(',');
                        if (values.Length == 2)
                        {
                            if (int.TryParse(values[0], out int index) && 
                                float.TryParse(values[1], out float value))
                            {
                                // 存储 BlendShape 值
                                blendShapeValues[index] = value * 100; // 可能需要乘以100，取决于您的模型权重范围
                                
                                if (printDebugInfo)
                                {
                                    string shapeName = blendShapeMap.ContainsKey(index) ? blendShapeMap[index] : $"Unknown_{index}";
                                    Debug.Log($"收到 BlendShape: {shapeName} (索引 {index}) = {value}");
                                }
                            }
                        }
                    }
                }
                hasNewData = true;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"处理BlendShape数据错误: {e.Message}, 数据: {data}");
        }
    }
    
    private void ApplyBlendShapeValues()
    {
        if (renderers == null || renderers.Length == 0)
            return;
            
        foreach (SkinnedMeshRenderer renderer in renderers)
        {
            Mesh mesh = renderer.sharedMesh;
            if (mesh != null && mesh.blendShapeCount > 0)
            {
                Dictionary<int, float> valuesToApply;
                
                lock (lockObject)
                {
                    // 创建一个副本，避免在应用过程中被修改
                    valuesToApply = new Dictionary<int, float>(blendShapeValues);
                }
                
                foreach (var pair in valuesToApply)
                {
                    int index = pair.Key;
                    float weight = pair.Value;
                    
                    if (index < mesh.blendShapeCount)
                    {
                        renderer.SetBlendShapeWeight(index, weight);
                        
                        // 只在需要时打印调试信息
                        if (printDebugInfo)
                        {
                            string blendShapeName = mesh.GetBlendShapeName(index);
                            Debug.Log($"设置BlendShape: {blendShapeName} (索引 {index}) 到 {weight}");
                        }
                    }
                }
            }
        }
    }
    
    // BlendShapeMap 初始化方法保持不变
    private void InitBlendShapeMap()
    {
        // 已有代码，保持不变
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
}