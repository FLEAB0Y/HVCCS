from server import THStreamServiceServicer, serve
import threading
import time
import json
import socket

def send_combined_data(face_data_list, limb_data_list, timestamp, socket_port, debug=False):
    """将facedata和limbdata合并为一个数据帧发送"""
    # 合并数据，前52个是facedata，后33个是limbdata
    combined_data = []
    
    # 添加facedata（已经是纯数值列表）
    for val in face_data_list:
        combined_data.append((len(combined_data), val))
    
    # 添加limbdata
    for val in limb_data_list:
        combined_data.append((len(combined_data), val))
    
    # 格式化数据，并添加时间戳
    data_str = ";".join([f"{idx},{val}" for idx, val in combined_data])
    data_str += f";timestamp,{timestamp}"  # 添加时间戳
    
    if debug:
        print(f"[DEBUG] 发送合并数据: 总长度={len(combined_data)}")
        print(f"[DEBUG] 数据内容: {data_str[:]}...") 
        
    
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
            time.sleep(0.01)
            buffer_size = servicer.receive_data_buffer.get_size()
        # 从缓冲区获取数据
        payload_rec = servicer.receive_data_buffer.get_items()
        if payload_rec:
            try:
                face_data_bytes = payload_rec.faceData
                limb_data_bytes = payload_rec.limbData
                timestamp = payload_rec.extDesc  # 获取时间戳
                latency = int(time.time() * 1000) - int(timestamp)  # 计算延迟
                print(f"[gRPC Port {grpc_port}] 延迟: {latency}ms")   
                
                # 解析 faceData 和 limbData
                face_data_list = json.loads(face_data_bytes.decode('utf-8'))  # 现在是纯数值列表
                limb_data_list = json.loads(limb_data_bytes.decode('utf-8'))
                
                if debug:
                    print(f"[DEBUG] 接收到的 faceData 长度: {len(face_data_list)}")
                    print(f"[DEBUG] 接收到的 limbData 长度: {len(limb_data_list)}")
                
                # 发送合并后的数据
                send_combined_data(face_data_list, limb_data_list, timestamp, socket_port, debug)

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