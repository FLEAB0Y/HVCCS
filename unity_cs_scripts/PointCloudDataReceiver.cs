using System;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Text;
using UnityEngine;

public class PointCloudDataReceiver : MonoBehaviour
{
    [Header("Socket设置")]
    public string serverIP = "127.0.0.1";
    public int serverPort = 8894;  // 默认使用点云数据端口
    public int feedbackPort = 9894;  // 反馈端口
    public float feedbackInterval = 1.0f;  // 发送反馈的时间间隔(秒)
    
    [Header("点云设置")]
    public PointCloudCSHelper pointCloudHelper;
    
    private TcpListener tcpListener;
    private Thread serverThread;
    private bool isRunning = false;
    private bool isServerStarted = false; // 标记服务器是否已启动
    
    // 点云数据缓冲区
    private Queue<PointCloudData> dataBuffer = new Queue<PointCloudData>();
    private readonly object bufferLock = new object();
    private int maxBufferSize = 10; // 最大缓冲区大小
    
    // 延迟测量
    private float lastFeedbackTime = 0f;
    private float latency = 0f;
    private bool hasLatencyData = false;
    private long lastFrameTimestamp = 0;
    
    // 点云数据结构
    private class PointCloudData
    {
        public Vector3[] positions;
        public Color[] colors;
        public int pointCount;
        public long timestamp;
    }

    void Start()
    {
        if (pointCloudHelper == null)
        {
            pointCloudHelper = GetComponent<PointCloudCSHelper>();
            if (pointCloudHelper == null)
            {
                Debug.LogError("【点云客户端】未指定PointCloudCSHelper组件！");
                return;
            }
        }
        
        StartServer();
    }
    
    void OnEnable()
    {
        // 如果服务器已经启动，就不再重复启动
        if (!isServerStarted)
        {
            StartServer();
        }
    }
    
    void OnDisable()
    {
        StopServer();
    }
    
    void OnDestroy()
    {
        StopServer();
    }
    
    void Update()
    {
        // 处理点云数据缓冲区
        ProcessDataBuffer();
        
        // 发送延迟反馈
        if (hasLatencyData && Time.time - lastFeedbackTime >= feedbackInterval)
        {
            SendLatencyFeedback();
            lastFeedbackTime = Time.time;
        }
    }
    
    // 处理缓冲区中的点云数据
    private void ProcessDataBuffer()
    {
        PointCloudData data = null;
        
        lock(bufferLock)
        {
            if (dataBuffer.Count > 0)
            {
                data = dataBuffer.Dequeue();
            }
        }
        
        if (data != null)
        {
            // 将点云数据传递给PointCloudCSHelper
            pointCloudHelper.UpdatePointCloudData(data.positions, data.colors);
            Debug.Log($"【点云客户端】处理点云数据，点数: {data.pointCount}");
            
            // 计算延迟
            long currentTime = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            latency = (currentTime - data.timestamp) / 1000.0f;
            hasLatencyData = true;
        }
    }
    
    // 发送延迟反馈的方法
    private void SendLatencyFeedback()
    {
        if (!hasLatencyData)
            return;
            
        try
        {
            string feedbackData = $"latency:{latency.ToString("F3")}";
            
            using (TcpClient client = new TcpClient())
            {
                client.Connect(serverIP, feedbackPort);
                
                if (client.Connected)
                {
                    NetworkStream stream = client.GetStream();
                    byte[] data = Encoding.UTF8.GetBytes(feedbackData);
                    stream.Write(data, 0, data.Length);
                    Debug.Log($"【点云延迟反馈】已发送延迟数据: {latency.ToString("F3")}秒");
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"【点云延迟反馈】发送延迟数据失败: {e.Message}");
        }
    }
    
    // 启动服务器
    private void StartServer()
    {
        // 如果服务器已经在运行，直接返回
        if (isServerStarted)
        {
            Debug.Log("【点云服务器】服务器已经在运行中");
            return;
        }
        
        try
        {
            isRunning = true;
            isServerStarted = true;
            Debug.Log($"【点云服务器】启动点云监听服务: {serverIP}:{serverPort}");
            
            serverThread = new Thread(new ThreadStart(ServerListen));
            serverThread.IsBackground = true;
            serverThread.Start();
        }
        catch (Exception e)
        {
            isServerStarted = false;
            Debug.LogError($"【点云服务器】启动失败: {e.Message}");
        }
    }

    // 服务器监听方法
    private void ServerListen()
    {
        int retryCount = 0;
        int maxRetries = 3;
        int currentPort = serverPort;
        
        while (retryCount < maxRetries && isRunning)
        {
            try
            {
                // 创建Socket并设置地址重用选项
                Socket socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
                socket.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);
                
                // 创建绑定点
                IPEndPoint localEndPoint = new IPEndPoint(IPAddress.Parse(serverIP), currentPort);
                
                // 创建TcpListener并绑定Socket
                tcpListener = new TcpListener(localEndPoint);
                tcpListener.Start();
                Debug.Log($"【点云服务器】已启动，监听端口: {currentPort}");
                
                while (isRunning)
                {
                    // 等待客户端连接
                    if (tcpListener.Pending())
                    {
                        // 处理客户端连接
                        TcpClient client = tcpListener.AcceptTcpClient();
                        Debug.Log("【点云服务器】接收到来自grpc2socket的连接");
                        
                        try
                        {
                            // 处理接收到的数据
                            NetworkStream stream = client.GetStream();
                            
                            // 记录接收时间戳（用于计算延迟）
                            long receiveTimestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
                            
                            // 首先读取点的数量（4字节整数）
                            byte[] countBuffer = new byte[4];
                            int bytesRead = stream.Read(countBuffer, 0, 4);
                            if (bytesRead < 4)
                            {
                                Debug.LogError("【点云服务器】无法读取点的数量");
                                client.Close();
                                continue;
                            }
                            
                            // 转换字节序（网络字节序是大端，需要转换）
                            if (BitConverter.IsLittleEndian)
                                Array.Reverse(countBuffer);
                            
                            int pointCount = BitConverter.ToInt32(countBuffer, 0);
                            Debug.Log($"【点云服务器】将接收 {pointCount} 个点");
                            
                            // 创建点的位置和颜色数组
                            Vector3[] positions = new Vector3[pointCount];
                            Color[] colors = new Color[pointCount];
                            
                            // 读取每个点的数据（每个点6个float，共24字节）
                            byte[] pointBuffer = new byte[24];
                            
                            for (int i = 0; i < pointCount; i++)
                            {
                                if (stream.Read(pointBuffer, 0, 24) < 24)
                                {
                                    Debug.LogError($"【点云服务器】读取点数据不完整，只读取了 {i} 个点");
                                    pointCount = i; // 更新点的数量
                                    break;
                                }
                                
                                // 解析X, Y, Z坐标（每个都是float，需要转换字节序）
                                float x = ConvertBigEndianToFloat(pointBuffer, 0);
                                float y = ConvertBigEndianToFloat(pointBuffer, 4);
                                float z = ConvertBigEndianToFloat(pointBuffer, 8);
                                
                                // 解析R, G, B颜色（每个都是float，已经归一化到0-1范围）
                                float r = ConvertBigEndianToFloat(pointBuffer, 12);
                                float g = ConvertBigEndianToFloat(pointBuffer, 16);
                                float b = ConvertBigEndianToFloat(pointBuffer, 20);
                                
                                // 存储点的位置和颜色
                                positions[i] = new Vector3(x, y, z);
                                colors[i] = new Color(r, g, b, 1.0f);
                            }
                            
                            // 创建点云数据对象
                            PointCloudData pointCloudData = new PointCloudData
                            {
                                positions = positions,
                                colors = colors,
                                pointCount = pointCount,
                                timestamp = receiveTimestamp
                            };
                            
                            // 添加到缓冲区
                            lock (bufferLock)
                            {
                                // 如果缓冲区已满，移除最旧的数据
                                if (dataBuffer.Count >= maxBufferSize)
                                {
                                    dataBuffer.Dequeue();
                                    Debug.LogWarning("【点云服务器】缓冲区已满，丢弃最旧的点云数据");
                                }
                                
                                dataBuffer.Enqueue(pointCloudData);
                                Debug.Log($"【点云服务器】添加点云数据到缓冲区，点数: {pointCloudData.pointCount}");
                            }
                            
                            // 计算帧间延迟
                            if (lastFrameTimestamp > 0)
                            {
                                latency = (receiveTimestamp - lastFrameTimestamp) / 1000.0f;
                                hasLatencyData = true;
                            }
                            
                            // 更新最后一帧的时间戳
                            lastFrameTimestamp = receiveTimestamp;
                        }
                        catch (Exception e)
                        {
                            Debug.LogError($"【点云服务器】处理数据错误: {e.Message}");
                        }
                        finally
                        {
                            client.Close();
                        }
                    }
                    else
                    {
                        Thread.Sleep(10); // 短暂休眠，减少CPU使用
                    }
                }
                
                // 如果执行到这里，说明正常退出循环，跳出重试循环
                break;
            }
            catch (SocketException se)
            {
                // 端口已被占用，尝试使用下一个端口
                retryCount++;
                
                if (retryCount < maxRetries)
                {
                    currentPort = serverPort + retryCount;
                    Debug.LogWarning($"【点云服务器】端口 {serverPort} 已被占用，尝试使用端口 {currentPort}");
                    
                    // 如果需要修改反馈端口，也可以在这里调整
                    feedbackPort = 9894 + retryCount;
                    
                    // 短暂休眠后重试
                    Thread.Sleep(1000);
                }
                else
                {
                    Debug.LogError($"【点云服务器】Socket错误: {se.Message}，已达到最大重试次数");
                    isServerStarted = false;
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"【点云服务器】运行错误: {e.Message}");
                isServerStarted = false;
                break;
            }
        }
        
        StopServer();
    }
    
    private float ConvertBigEndianToFloat(byte[] buffer, int startIndex)
    {
        byte[] floatBytes = new byte[4];
        Array.Copy(buffer, startIndex, floatBytes, 0, 4);
        
        // 如果系统是小端字节序，需要反转字节
        if (BitConverter.IsLittleEndian)
            Array.Reverse(floatBytes);
            
        return BitConverter.ToSingle(floatBytes, 0);
    }
    
    // 停止服务器
    public void StopServer()
    {
        isRunning = false;
        
        if (tcpListener != null)
        {
            try {
                tcpListener.Stop();
            } catch (Exception e) {
                Debug.LogWarning($"【点云服务器】关闭监听器时出错: {e.Message}");
            }
            tcpListener = null;
        }
        
        if (serverThread != null && serverThread.IsAlive)
        {
            try {
                serverThread.Abort();
            } catch (Exception e) {
                Debug.LogWarning($"【点云服务器】中止线程时出错: {e.Message}");
            }
            serverThread = null;
        }
        
        isServerStarted = false;
        Debug.Log("【点云服务器】已关闭");
    }
    
    // 获取当前缓冲区大小
    public int GetBufferSize()
    {
        lock (bufferLock)
        {
            return dataBuffer.Count;
        }
    }
    
    // 清空缓冲区
    public void ClearBuffer()
    {
        lock (bufferLock)
        {
            dataBuffer.Clear();
            Debug.Log("【点云客户端】缓冲区已清空");
        }
    }
}