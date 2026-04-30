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
    
    // 引用BlendShape控制器
    [SerializeField] private BSCtrl blendShapeController;
    
    // 添加BDCtrl控制器引用
    [SerializeField] private BDCtrl neZhaMov;

    // Socket对象
    private TcpListener tcpListener;
    private TcpClient clientConnection;
    private Thread serverThread;
    private bool isRunning = false;
    
    // 消息队列
    private class QueuedMessage
    {
        public string Data;
        public long tTransitMs;
    }
    private readonly Queue<QueuedMessage> messageQueue = new Queue<QueuedMessage>();
    private readonly object queueLock = new object();
    
    // 待反馈队列：存储 (t_begin, t_transit) 对，待 LateUpdate 补充 t_final
    private class PendingFeedback
    {
        public long tBeginMs;
        public long tTransitMs;
    }
    private readonly Queue<PendingFeedback> pendingFeedbacks = new Queue<PendingFeedback>();
    private readonly object feedbackLock = new object();

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
        
        // 如果没有手动指定BDCtrl控制器，则尝试查找
        if (neZhaMov == null)
        {
            neZhaMov = GetComponent<BDCtrl>();
            if (neZhaMov == null)
            {
                neZhaMov = FindObjectOfType<BDCtrl>();
                if (neZhaMov == null)
                {
                    Debug.LogError("未找到BDCtrl控制器，请手动指定");
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
                    QueuedMessage message = messageQueue.Dequeue();
                    ProcessReceivedData(message.Data, message.tTransitMs);
                }
            }
        }
    }
    
    // 在帧渲染结束后发送反馈
    void LateUpdate()
    {
        // 在帧渲染结束时补充 t_final 并发送待反馈的数据
        lock (feedbackLock)
        {
            while (pendingFeedbacks.Count > 0)
            {
                PendingFeedback pending = pendingFeedbacks.Dequeue();
                long tFinalMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
                SendFrameTimingFeedback(pending.tBeginMs, pending.tTransitMs, tFinalMs);
            }
        }
    }

    // 严格逐帧回传：timing:t_begin,t_transit,t_final
    private void SendFrameTimingFeedback(long tBeginMs, long tTransitMs, long tFinalMs)
    {
        try
        {
            string feedbackData = $"timing:{tBeginMs},{tTransitMs},{tFinalMs}";
            
            using (TcpClient client = new TcpClient())
            {
                client.Connect(ipAddress, feedbackPort);
                
                if (client.Connected)
                {
                    NetworkStream stream = client.GetStream();
                    byte[] data = Encoding.UTF8.GetBytes(feedbackData);
                    stream.Write(data, 0, data.Length);
                    Debug.Log($"【延迟反馈】已发送逐帧时间戳: t_begin={tBeginMs}, t_transit={tTransitMs}, t_final={tFinalMs}");
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"【延迟反馈】发送逐帧时间戳失败: {e.Message}");
        }
    }
    
    // 处理接收到的数据
    private void ProcessReceivedData(string data, long tTransitMs)
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
                Debug.LogError("【控制器缺失】BDCtrl控制器未找到");
            }
            
            // 将待反馈数据加入队列，等待 LateUpdate 在帧渲染结束时补充 t_final
            lock (feedbackLock)
            {
                pendingFeedbacks.Enqueue(new PendingFeedback { tBeginMs = timestamp, tTransitMs = tTransitMs });
            }

            Debug.Log($"【处理完成】处理了52个面部数据和{limbData.Length}个姿势数据，等待帧渲染结束后发送反馈");
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
                TcpClient client;
                try
                {
                    // 阻塞式 Accept，无需轮询 Sleep
                    client = tcpListener.AcceptTcpClient();
                }
                catch (Exception)
                {
                    if (!isRunning) break;
                    continue;
                }
                Debug.Log("客户端已连接");
                HandleClientConnection(client);
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
    
    // 处理客户端连接和数据接收（持久化连接，换行符分隔消息帧）
    private void HandleClientConnection(TcpClient client)
    {
        try
        {
            NetworkStream stream = client.GetStream();
            byte[] buffer = new byte[bufferSize * 4];
            System.Text.StringBuilder lineBuffer = new System.Text.StringBuilder();

            while (isRunning)
            {
                int bytesRead;
                try
                {
                    bytesRead = stream.Read(buffer, 0, buffer.Length);
                }
                catch (Exception)
                {
                    break;
                }
                if (bytesRead == 0) break;

                // t_transit: 数据刚从 socket 读到的时刻
                long tTransitMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
                lineBuffer.Append(Encoding.UTF8.GetString(buffer, 0, bytesRead));

                // 按换行符切分完整消息帧
                string bufStr = lineBuffer.ToString();
                int start = 0, idx;
                while ((idx = bufStr.IndexOf('\n', start)) >= 0)
                {
                    string message = bufStr.Substring(start, idx - start).Trim();
                    if (message.Length > 0)
                    {
                        lock (queueLock)
                        {
                            messageQueue.Enqueue(new QueuedMessage { Data = message, tTransitMs = tTransitMs });
                        }
                    }
                    start = idx + 1;
                }
                lineBuffer.Clear();
                if (start < bufStr.Length)
                    lineBuffer.Append(bufStr.Substring(start));
            }
        }
        catch (Exception e)
        {
            Debug.LogError("处理客户端数据错误: " + e.Message);
        }
        finally
        {
            if (client != null) client.Close();
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