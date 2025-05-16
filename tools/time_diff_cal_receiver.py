# time_diff_cal_receiver.py
import socket
import time
import json

# 配置参数
RECEIVER_IP = '192.168.1.11'  # 接收端IP地址
RECEIVER_PORT = 9876          # 接收端端口

def start_receiver():
    """启动接收端服务器"""
    # 创建UDP套接字
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((RECEIVER_IP, RECEIVER_PORT))
    
    print(f"接收端已启动，监听 {RECEIVER_IP}:{RECEIVER_PORT}...")
    
    try:
        while True:
            # 接收数据
            data, addr = server_socket.recvfrom(1024)
            
            # 记录接收时间戳 T2
            t2 = int(time.time() * 1000)
            
            # 解析收到的数据
            message = json.loads(data.decode())
            t1 = message.get("sender_time")
            
            # 创建响应，包含发送时间和接收时间
            response = json.dumps({
                "sender_time": t1,
                "receiver_time": t2
            })
            
            # 发送响应
            server_socket.sendto(response.encode(), addr)
            print(f"响应已发送到 {addr}, 时间差: {t2-t1}ms")
    
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        server_socket.close()
        print("接收端已关闭")

if __name__ == "__main__":
    start_receiver()

