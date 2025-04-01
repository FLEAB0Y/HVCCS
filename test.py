import socket
import time

def send_blendshape_data(data_list):
    # 格式化数据为需要的字符串格式
    data_str = "blendshape_data: [" + ", ".join([f"({idx}, {val})" for idx, val in data_list]) + "]"
    
    # 建立TCP连接
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", 5556))
    
    # 发送数据
    client.send(data_str.encode('utf-8'))
    
    # 关闭连接
    client.close()

# 示例数据
blendshape_data = [(0, 0.0), (1, 0.009668582119047642), (2, 0.007418482098728418), (3, 0.03089664876461029)]
send_blendshape_data(blendshape_data)