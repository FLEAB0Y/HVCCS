using UnityEngine;
using System.Collections.Concurrent;
using System.Net.Sockets;
using System.Threading;
using System;
using System.IO;

// 确保你已经安装了Draco for Unity包
using Draco;

public class DracoReceiver : MonoBehaviour
{
    [Header("Network Settings")]
    [Tooltip("The IP address of the Python server.")]
    public string serverHost = "127.0.0.1";
    [Tooltip("The port the Python server is listening on for Unity.")]
    public int serverPort = 8894;

    [Header("Rendering Settings")]
    [Tooltip("The GameObject that will display the point cloud.")]
    public GameObject pointCloudObject;
    [Tooltip("The material to use for rendering the point cloud.")]
    public Material pointCloudMaterial;

    private TcpClient socketConnection;
    private Thread clientReceiveThread;
    
    // 线程安全的队列，用于从网络线程向Unity主线程传递接收到的Draco二进制数据
    private static ConcurrentQueue<byte[]> receivedDataQueue = new ConcurrentQueue<byte[]>();

    // 记录上一次接收数据的时间，用于计算延迟
    private float lastReceiveTime = 0f;

    void Start()
    {
        // 配置渲染对象
        if (pointCloudObject == null)
        {
            pointCloudObject = new GameObject("PointCloudRenderer");
        }
        if (pointCloudObject.GetComponent<MeshFilter>() == null)
            pointCloudObject.AddComponent<MeshFilter>();
        if (pointCloudObject.GetComponent<MeshRenderer>() == null)
        {
            var renderer = pointCloudObject.AddComponent<MeshRenderer>();
            renderer.material = pointCloudMaterial;
        }

        // 启动网络连接线程
        ConnectToServer();
    }

    void Update()
    {
        // 在主线程中检查队列，看是否有新的数据需要解码和渲染
        if (receivedDataQueue.TryDequeue(out byte[] dracoData))
        {
            if (dracoData != null && dracoData.Length > 0)
            {
                // 异步解码和应用Mesh，防止主线程卡顿
                DecodeAndApplyMesh(dracoData);
            }
        }
    }

    private async void DecodeAndApplyMesh(byte[] dracoData)
    {
        // 使用静态API进行解码
        Mesh decodedMesh = await DracoDecoder.DecodeMesh(dracoData);

        if (decodedMesh != null && pointCloudObject != null)
        {
            var meshFilter = pointCloudObject.GetComponent<MeshFilter>();
            if (meshFilter.mesh != null)
            {
                Destroy(meshFilter.mesh);
            }
            meshFilter.mesh = decodedMesh;
            // Debug.Log($"Applied new mesh with {decodedMesh.vertexCount} vertices.");
        }
        else
        {
            Debug.LogError("Draco decoding failed or resulted in a null mesh.");
        }
    }

    private void ConnectToServer()
    {
        try
        {
            clientReceiveThread = new Thread(new ThreadStart(ListenForData));
            clientReceiveThread.IsBackground = true;
            clientReceiveThread.Start();
        }
        catch (Exception e)
        {
            Debug.LogError("On client connect exception " + e);
        }
    }

    private void ListenForData()
    {
        while (true) // 自动重连循环
        {
            try
            {
                socketConnection = new TcpClient(serverHost, serverPort);
                Debug.Log("Successfully connected to Python server.");
                using (var stream = socketConnection.GetStream())
                {
                    var reader = new BinaryReader(stream);
                    while (true)
                    {
                        // 1. 读取4字节的长度前缀
                        byte[] lengthPrefix = reader.ReadBytes(4);
                        if (lengthPrefix.Length < 4) break; // 连接断开
                        
                        // 将网络字节序（大端）转换为主机字节序
                        if (BitConverter.IsLittleEndian)
                        {
                            Array.Reverse(lengthPrefix);
                        }
                        int dataLength = BitConverter.ToInt32(lengthPrefix, 0);

                        // 2. 根据长度读取完整的Draco数据
                        byte[] dracoData = reader.ReadBytes(dataLength);
                        if (dracoData.Length < dataLength) break; // 连接断开

                        // 3. 将数据放入线程安全队列
                        receivedDataQueue.Enqueue(dracoData);

                        // 记录接收时间
                        lastReceiveTime = Time.realtimeSinceStartup;
                    }
                }
            }
            catch (SocketException ex)
            {
                Debug.LogWarning($"Socket exception: {ex.Message}. Retrying in 5 seconds...");
            }
            catch (Exception ex)
            {
                Debug.LogError($"Error in network thread: {ex.Message}");
            }
            finally
            {
                socketConnection?.Close();
                Thread.Sleep(5000); // 等待5秒后尝试重连
            }
        }
    }

    private void SendLatencyFeedback()
    {
        try
        {
            using (TcpClient client = new TcpClient())
            {
                client.Connect("127.0.0.1", serverPort + 1000); // 反馈端口
                using (NetworkStream stream = client.GetStream())
                {
                    string feedback = $"latency:{Time.realtimeSinceStartup - lastReceiveTime}";
                    byte[] data = System.Text.Encoding.ASCII.GetBytes(feedback);
                    stream.Write(data, 0, data.Length);
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"Failed to send latency feedback: {e.Message}");
        }
    }

    void OnApplicationQuit()
    {
        // 确保在退出时关闭线程和连接
        if (clientReceiveThread != null)
        {
            clientReceiveThread.Abort();
        }
        if (socketConnection != null)
        {
            socketConnection.Close();
        }
    }
}