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
                    Debug.Log("接收到数据: " + message);
                    
                    // 将数据传递给BlendShape控制器处理
                    blendShapeController.ProcessBlendShapeData(message);
                }
            }
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