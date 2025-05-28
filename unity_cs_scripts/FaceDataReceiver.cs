using UnityEngine;
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Collections.Generic;

public class FaceDataReceiver : MonoBehaviour
{
    // Socket配置
    [SerializeField] private string ipAddress = "127.0.0.1";
    [SerializeField] private int port = 8890;
    [SerializeField] private int bufferSize = 8192;
    
    // 新增：延迟反馈配置
    [SerializeField] private int feedbackPort = 9890; // 反馈端口
    [SerializeField] private float feedbackInterval = 1.0f; // 发送反馈的时间间隔(秒)
    private float lastFeedbackTime = 0f;
    
    // 引用BlendShape控制器
    [SerializeField] private BSCtrl blendShapeController;
    
    // 添加NeZhaMov控制器引用
    [SerializeField] private NeZhaMov neZhaMov;

    // Socket对象
    private TcpListener tcpListener;
    private TcpClient clientConnection;
    private Thread serverThread;
    private bool isRunning = false;
    
    // 消息队列
    private readonly Queue<string> messageQueue = new Queue<string>();
    private readonly object queueLock = new object();

    void Start()
    {
        // 如果没有手动指定BlendShape控制器，则尝试查找
        if (blendShapeController == null)
        {
            blendShapeController = GetComponent<BSCtrl>();
            if (blendShapeController == null)
            {
                blendShapeController = FindObjectOfType<BSCtrl>();
                if (blendShapeController == null)
                {
                    Debug.LogError("未找到BlendShape控制器，请手动指定");
                }
            }
        }
        
        // 如果没有手动指定NeZhaMov控制器，则尝试查找
        if (neZhaMov == null)
        {
            neZhaMov = GetComponent<NeZhaMov>();
            if (neZhaMov == null)
            {
                neZhaMov = FindObjectOfType<NeZhaMov>();
                if (neZhaMov == null)
                {
                    Debug.LogError("未找到NeZhaMov控制器，请手动指定");
                }
            }
        }
    }

    // 当脚本启用时开始监听
    void OnEnable()
    {
        StartServer();
    }

    // 当脚本禁用时关闭连接
    void OnDisable()
    {
        StopServer();
    }

    // 当应用退出时确保关闭连接
    void OnApplicationQuit()
    {
        StopServer();
    }
    
    // 在主线程中处理消息队列
    void Update()
    {
        // 处理消息队列
        if (messageQueue.Count > 0)
        {
            lock (queueLock)
            {
                while (messageQueue.Count > 0)
                {
                    string message = messageQueue.Dequeue();
                    ProcessReceivedData(message);
                }
            }
        }
        
        // 新增：定期发送延迟反馈
        if (blendShapeController != null && blendShapeController.HasLatencyData)
        {
            if (Time.time - lastFeedbackTime >= feedbackInterval)
            {
                SendLatencyFeedback();
                lastFeedbackTime = Time.time;
            }
        }
    }
    
    // 新增：发送延迟反馈的方法
    private void SendLatencyFeedback()
    {
        if (blendShapeController == null || !blendShapeController.HasLatencyData)
            return;
            
        try
        {
            float latency = blendShapeController.Latency;
            string feedbackData = $"latency:{latency.ToString("F3")}";
            
            using (TcpClient client = new TcpClient())
            {
                client.Connect(ipAddress, feedbackPort);
                
                if (client.Connected)
                {
                    NetworkStream stream = client.GetStream();
                    byte[] data = Encoding.UTF8.GetBytes(feedbackData);
                    stream.Write(data, 0, data.Length);
                    Debug.Log($"【延迟反馈】已发送延迟数据: {latency.ToString("F3")}秒");
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"【延迟反馈】发送延迟数据失败: {e.Message}");
        }
    }
    
    // 处理接收到的数据
    private void ProcessReceivedData(string data)
    {
        try
        {
            Debug.Log($"【数据接收】收到原始数据，长度: {data?.Length}");
            if (!string.IsNullOrEmpty(data) && data.Length > 100)
            {
                Debug.Log($"【数据样本】数据前100字符: {data.Substring(0, 100)}...");
            }
            
            // 解析逗号分隔的数据字符串
            string[] parts = data.Split(',');
            
            if (parts.Length < 152) // 1(时间戳) + 52(面部数据) + 99(姿势数据) = 152
            {
                Debug.LogWarning($"【数据不足】数据项不足，期望至少152项，实际为{parts.Length}项");
                return;
            }
            
            Debug.Log($"【数据分段】共分割出 {parts.Length} 个数据项");
            
            // 提取时间戳（第一个数据项）
            if (!long.TryParse(parts[0], out long timestamp))
            {
                Debug.LogWarning($"【时间戳错误】无法解析时间戳: {parts[0]}");
                return;
            }
            
            Debug.Log($"【时间戳】接收到时间戳: {timestamp}ms");
            
            // 提取面部表情数据（接下来的52个数据项）
            float[] faceData = new float[52];
            for (int i = 0; i < 52; i++)
            {
                if (float.TryParse(parts[i + 1], out float value))
                {
                    faceData[i] = value;
                }
                else
                {
                    Debug.LogWarning($"【解析错误】无法解析面部数据项 {i}: {parts[i + 1]}");
                    faceData[i] = 0f;
                }
            }
            
            // 提取姿势数据（最后的99个数据项）
            float[] limbData = new float[99];
            for (int i = 0; i < 99 && i + 53 < parts.Length; i++)
            {
                if (float.TryParse(parts[i + 53], out float value))
                {
                    limbData[i] = value;
                }
                else
                {
                    Debug.LogWarning($"【解析错误】无法解析姿势数据项 {i}: {parts[i + 53]}");
                    limbData[i] = 0f;
                }
            }
            
            // 将面部数据传递给BlendShape控制器
            if (blendShapeController != null)
            {
                blendShapeController.ProcessBlendShapeDataArray(faceData, timestamp);
                Debug.Log($"【面部数据】已处理52个面部数据项");
            }
            else
            {
                Debug.LogError("【控制器缺失】BlendShape控制器未找到");
            }
            
            // 直接处理姿势数据 - 不使用事件
            if (neZhaMov != null)
            {
                neZhaMov.ProcessLimbData(limbData, timestamp);
                Debug.Log($"【姿势数据】已直接处理{limbData.Length}个姿势数据项");
            }
            else
            {
                Debug.LogError("【控制器缺失】NeZhaMov控制器未找到");
            }
            
            Debug.Log($"【处理完成】处理了52个面部数据和{limbData.Length}个姿势数据");
        }
        catch (Exception e)
        {
            Debug.LogError($"【解析错误】解析数据错误: {e.Message}\n{e.StackTrace}");
        }
    }
    
    // 启动服务器并开始监听客户端连接
    private void StartServer()
    {
        try
        {
            isRunning = true;
            Debug.Log($"启动服务器: {ipAddress}:{port}");
            
            serverThread = new Thread(new ThreadStart(ServerListen));
            serverThread.IsBackground = true;
            serverThread.Start();
        }
        catch (Exception e)
        {
            Debug.LogError("启动服务器失败: " + e.Message);
        }
    }

    // 服务器监听方法
    private void ServerListen()
    {
        try
        {
            // 创建并启动TCP监听器
            tcpListener = new TcpListener(IPAddress.Parse(ipAddress), port);
            tcpListener.Start();
            Debug.Log("服务器已启动，等待客户端连接...");
            
            while (isRunning)
            {
                // 等待客户端连接
                if (tcpListener.Pending())
                {
                    // 接受客户端连接
                    clientConnection = tcpListener.AcceptTcpClient();
                    Debug.Log("客户端已连接");
                    
                    // 开始接收数据
                    HandleClientConnection(clientConnection);
                }
                else
                {
                    Thread.Sleep(100); // 等待一小段时间再次检查
                }
            }
        }
        catch (SocketException socketException)
        {
            Debug.LogError("服务器Socket错误: " + socketException.Message);
        }
        catch (Exception e)
        {
            Debug.LogError("服务器发生错误: " + e.Message);
        }
        finally
        {
            StopServer();
        }
    }
    
    // 处理客户端连接和数据接收
    private void HandleClientConnection(TcpClient client)
    {
        try
        {
            NetworkStream stream = client.GetStream();
            byte[] buffer = new byte[bufferSize];
            int bytesRead;
            
            // 读取客户端发送的数据
            while (isRunning && client.Connected && (bytesRead = stream.Read(buffer, 0, buffer.Length)) > 0)
            {
                string data = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                // 将数据添加到队列
                lock (queueLock)
                {
                    messageQueue.Enqueue(data);
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError("处理客户端数据错误: " + e.Message);
        }
        finally
        {
            if (client != null)
            {
                client.Close();
            }
        }
    }

    // 关闭服务器
    private void StopServer()
    {
        isRunning = false;
        
        if (clientConnection != null)
        {
            clientConnection.Close();
            clientConnection = null;
        }
        
        if (tcpListener != null)
        {
            tcpListener.Stop();
            tcpListener = null;
        }
        
        if (serverThread != null)
        {
            serverThread.Abort();
            serverThread = null;
        }
        
        Debug.Log("服务器已关闭");
    }
}