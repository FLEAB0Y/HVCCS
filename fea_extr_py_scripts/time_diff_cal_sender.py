import socket
import time
import json
import statistics

# 配置参数
RECEIVER_IP = '192.168.1.11'  # 接收端IP地址
RECEIVER_PORT = 9876          # 接收端端口
SENDER_IP = '192.168.1.10'    # 发送端IP地址
SAMPLE_COUNT = 10             # 采样次数，用于计算平均时间差
TIMEOUT = 5                   # 等待响应的超时时间（秒）

def calculate_time_diff():
    """计算两台计算机之间的时间差"""
    time_diffs = []
    
    # 创建UDP套接字
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.bind((SENDER_IP, 0))  # 绑定到发送端IP，端口随机
    client_socket.settimeout(TIMEOUT)
    
    print(f"开始计算与 {RECEIVER_IP}:{RECEIVER_PORT} 之间的时间差...")
    
    try:
        for i in range(SAMPLE_COUNT):
            # 记录发送时间戳 T1
            t1 = int(time.time() * 1000)  # 毫秒级时间戳
            
            # 发送包含当前时间戳的消息
            message = json.dumps({"sender_time": t1})
            client_socket.sendto(message.encode(), (RECEIVER_IP, RECEIVER_PORT))
            
            try:
                # 接收响应
                data, addr = client_socket.recvfrom(1024)
                
                # 记录接收时间戳 T3
                t3 = int(time.time() * 1000)
                
                # 解析响应中的时间戳
                response = json.loads(data.decode())
                t1 = response.get("sender_time")  # 发送时的时间戳
                t2 = response.get("receiver_time")  # 接收端接收到时的时间戳
                
                # 计算往返时间
                rtt = t3 - t1
                
                # 估算时间差 = ((T2-T1) - (T3-T2))/2
                # 即 ((T2-T1) - (RTT-(T2-T1)))/2
                # 简化为 (T2-T1) - RTT/2
                time_diff = (t2 - t1) - (rtt / 2)
                
                print(f"样本 {i+1}: 往返时间 = {rtt}ms, 估算时间差 = {time_diff}ms")
                time_diffs.append(time_diff)
                
                # 短暂暂停，避免过快发送
                time.sleep(0.5)
                
            except socket.timeout:
                print(f"样本 {i+1}: 接收响应超时")
    
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        client_socket.close()
    
    # 计算平均时间差和标准差
    if time_diffs:
        avg_time_diff = statistics.mean(time_diffs)
        std_dev = statistics.stdev(time_diffs) if len(time_diffs) > 1 else 0
        
        print(f"\n===== 结果 =====")
        print(f"有效样本数: {len(time_diffs)}/{SAMPLE_COUNT}")
        print(f"平均时间差: {avg_time_diff:.2f}ms")
        print(f"标准差: {std_dev:.2f}ms")
        print(f"这表示接收端时间比发送端时间快 {avg_time_diff:.2f}ms" if avg_time_diff > 0 
              else f"这表示发送端时间比接收端时间快 {abs(avg_time_diff):.2f}ms")
        
        return avg_time_diff
    return None

if __name__ == "__main__":
    time_diff = calculate_time_diff()
    if time_diff is not None:
        print(f"\n要将本地时间转换为接收端时间，请将本地时间加上 {time_diff:.2f}ms")
        print(f"换算公式: receiver_time = sender_time + {time_diff:.2f}ms")

