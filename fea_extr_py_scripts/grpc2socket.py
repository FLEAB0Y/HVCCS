from server import THStreamServiceServicer, serve
import threading
import time
import json
import socket

def send_blendshape_data(data_list, timestamp, socket_port, debug=False):
    """使用socket直接发送blendshape数据"""
    # 格式化索引和值，并添加时间戳
    data_str = ";".join([f"{idx},{val}" for idx, val in data_list])
    data_str += f";timestamp,{timestamp}"  # 添加时间戳
    
    if debug:
        print(f"[DEBUG] 发送数据: {data_str}")
    
    # 建立TCP连接
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", socket_port))
    
    # 发送数据
    client.send(data_str.encode('utf-8'))
    
    # 关闭连接
    client.close()

def grpc_thread(grpc_port, socket_port, debug=False):
    """gRPC线程处理函数"""
    servicer = THStreamServiceServicer()
    server_thread = threading.Thread(target=serve, args=(servicer, grpc_port))
    server_thread.start()

    while True:    
        # 缓冲区空了就等待
        buffer_size = servicer.receive_data_buffer.get_size()
        while buffer_size < 1:
            time.sleep(0.1)
            buffer_size = servicer.receive_data_buffer.get_size()
        # 从缓冲区获取数据
        payload_rec = servicer.receive_data_buffer.get_items()
        if payload_rec:
            try:
                face_data_bytes = payload_rec.faceData
                limb_data_bytes = payload_rec.limbData  # 新增 limbData 的处理
                timestamp = payload_rec.extDesc  # 获取时间戳
                latency = int(time.time() * 1000) - int(timestamp)  # 计算延迟
                print(f"[gRPC Port {grpc_port}] 延迟: {latency}ms")   
                
                # 解析 faceData 和 limbData
                face_data_list = json.loads(face_data_bytes.decode('utf-8'))  # 将接收到的 JSON 数据转换为列表
                limb_data_list = json.loads(limb_data_bytes.decode('utf-8'))  # 新增 limbData 的解析
                
                if debug:
                    print(f"[DEBUG] 接收到的 faceData: {face_data_list}")
                    print(f"[DEBUG] 接收到的 limbData: {limb_data_list}")
                
                # 分别发送 faceData 和 limbData
                send_blendshape_data(face_data_list, timestamp, socket_port, debug)  # 发送 faceData 和时间戳到对应的 socket 端口
                send_blendshape_data(limb_data_list, timestamp, socket_port + 1, debug)  # 假设 limbData 使用 socket_port + 1

            except AttributeError as e:
                print(f"[gRPC Port {grpc_port}] AttributeError: {e}")

if __name__ == '__main__':
    # 定义 gRPC 和对应的 socket 端口
    port_mappings = [
        (50051, 8890),
        (50052, 8891),
        (50053, 8892),
        (50054, 8893)
    ]

    # 是否启用 debug 模式
    debug = True

    # 为每对端口启动一个线程
    threads = []
    for grpc_port, socket_port in port_mappings:
        t = threading.Thread(target=grpc_thread, args=(grpc_port, socket_port, debug))
        t.start()
        threads.append(t)

    # 等待所有线程完成
    for t in threads:
        t.join()