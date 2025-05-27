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
    [SerializeField] private int port = 8888;
    [SerializeField] private int bufferSize = 8192;
    
    // 引用BlendShape控制器
    [SerializeField] private BSCtrl blendShapeController;
    
    // 动作数据事件，其他组件可以订阅此事件接收肢体数据
    public delegate void LimbDataHandler(float[] limbData, long timestamp);
    public event LimbDataHandler OnLimbDataReceived;

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
        if (messageQueue.Count > 0 && blendShapeController != null)
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
    }
    
    // 处理接收到的数据
    private void ProcessReceivedData(string data)
    {
        try
        {
            // 解析数据字符串
            string[] entries = data.Split(';');
            List<float> allValues = new List<float>();
            long timestamp = 0;
            
            // 首先提取所有数值和时间戳
            foreach (string entry in entries)
            {
                if (string.IsNullOrEmpty(entry)) 
                    continue;
                
                string[] parts = entry.Split(',');
                
                // 检查是否是时间戳数据
                if (parts.Length == 2 && parts[0] == "timestamp")
                {
                    if (long.TryParse(parts[1], out timestamp))
                    {
                        Debug.Log($"接收到时间戳: {timestamp}ms");
                    }
                    continue;
                }
                
                // 处理普通数据
                if (parts.Length == 2)
                {
                    if (float.TryParse(parts[1], out float value))
                    {
                        allValues.Add(value);
                    }
                }
            }
            
            // 分离面部数据和肢体数据
            if (allValues.Count >= 52)
            {
                // 提取前52个值作为面部数据
                float[] faceData = allValues.GetRange(0, 52).ToArray();
                
                // 将面部数据传递给BlendShape控制器
                blendShapeController.ProcessBlendShapeDataArray(faceData, timestamp);
                
                // 如果有剩余数据，作为肢体数据处理
                if (allValues.Count > 52)
                {
                    float[] limbData = allValues.GetRange(52, allValues.Count - 52).ToArray();
                    // 触发肢体数据事件
                    OnLimbDataReceived?.Invoke(limbData, timestamp);
                    Debug.Log($"处理了52个面部数据和{limbData.Length}个肢体数据");
                }
                else
                {
                    Debug.Log($"仅处理了52个面部数据，无肢体数据");
                }
            }
            else
            {
                Debug.LogWarning($"接收到的数据不足52个: {allValues.Count}");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"解析数据错误: {e.Message}\n{e.StackTrace}");
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