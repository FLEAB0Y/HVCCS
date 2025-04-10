from server import THStreamServiceServicer, serve
import threading
import time
import json
import socket

def send_blendshape_data(data_list, timestamp):
    """使用socket直接发送blendshape数据"""
    # 格式化索引和值，并添加时间戳
    data_str = ";".join([f"{idx},{val}" for idx, val in data_list])
    data_str += f";timestamp,{timestamp}"  # 添加时间戳
    # print(f"发送数据: {data_str}")
    
    # 建立TCP连接
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", 8888))
    
    # 发送数据
    client.send(data_str.encode('utf-8'))
    
    # 关闭连接
    client.close()

if __name__ == '__main__':
    # 开启服务器线程
    servicer = THStreamServiceServicer()
    server_thread = threading.Thread(target=serve, args=(servicer,))
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
                timestamp = payload_rec.extDesc  # 获取时间戳
                latency = int(time.time() * 1000) - int(timestamp)  # 计算延迟
                print(f"延迟: {latency}ms")   
                data_list = json.loads(face_data_bytes.decode('utf-8'))  # 将接收到的 JSON 数据转换为列表
                send_blendshape_data(data_list, timestamp)  # 发送数据和时间戳到unity

                #  test 打印接收到的数据列表
                # print(f"数据列表长度: {len(data_list)}")
                # print(data_list[:10] + ['...'] if len(data_list) > 10 else data_list)
            except AttributeError as e:
                print(f"AttributeError: {e}")