import os
import time
import threading
import glob
from client import THStreamClient
from THStreamData import THStreamDataPayload, THDataWarehouse
import sys

def run_client(client):
    """运行客户端线程"""
    client.run()

def send_ply_files(client, ply_directory, interval=0.033, debug=False):
    """
    从指定目录扫描并发送PLY文件
    
    参数:
        client: THStreamClient 客户端实例
        ply_directory: 包含PLY文件的目录路径
        interval: 两次发送之间的时间间隔(秒)，默认约30fps
        debug: 是否打印调试信息
    """
    # 检查目录是否存在
    if not os.path.exists(ply_directory):
        print(f"错误: 目录 {ply_directory} 不存在!")
        return
    
    # 获取所有PLY文件并按名称排序
    ply_files = sorted(glob.glob(os.path.join(ply_directory, "*.ply")))
    
    if not ply_files:
        print(f"警告: 在 {ply_directory} 中未找到PLY文件")
        return
    
    print(f"找到 {len(ply_files)} 个PLY文件，准备发送...")
    
    # 循环读取并发送PLY文件
    frame_count = 0
    for ply_file in ply_files:
        frame_count += 1
        
        # 等待缓冲区有空间
        buffer_size = client.send_data_buffer.get_size()
        while buffer_size >= 5:
            time.sleep(0.01)
            buffer_size = client.send_data_buffer.get_size()
        
        try:
            # 读取PLY文件二进制数据
            with open(ply_file, 'rb') as f:
                ply_binary = f.read()
            
            # 获取当前时间戳
            timestamp_ms = str(int(time.time() * 1000))
            
            # 创建数据负载 - PLY数据放在point_data字段中
            payload_send = THStreamDataPayload(
                rgb_data=b'\x00', 
                point_data=ply_binary,  # PLY二进制数据放在point_data字段中
                face_data=b'\x00',  # 不发送面部数据
                limb_data=b'\x00',  # limb_data设为空
                ext_data=b'\x00', 
                ext_desc=timestamp_ms  # 时间戳作为描述
            )
            
            # 发送数据
            client.send_data_buffer.add_item(payload_send)
            
            if debug:
                file_name = os.path.basename(ply_file)
                file_size = len(ply_binary)
                print(f"已发送: {file_name}, 大小: {file_size} 字节, 时间戳: {timestamp_ms}")
            
            # 控制发送速率
            time.sleep(interval)
            
        except Exception as e:
            print(f"发送文件 {ply_file} 时出错: {e}")
    
    print(f"完成发送 {frame_count} 个PLY文件")

def main(server_addr='127.0.0.1', port_num=50051, ply_dir='C:/Users/24150/plydata', debug=False):
    """
    主函数：设置并运行PLY文件发送器
    
    参数:
        server_addr: gRPC服务器地址
        port_num: gRPC服务器端口
        ply_dir: PLY文件目录
        debug: 是否打印调试信息
    """
    # 创建客户端
    client = THStreamClient(host=server_addr, port=port_num)
    
    # 启动客户端线程
    client_thread = threading.Thread(target=run_client, args=(client,))
    client_thread.daemon = True  # 设置为守护线程，主线程结束时自动结束
    client_thread.start()
    
    # 等待客户端连接
    print(f"连接到gRPC服务器 {server_addr}:{port_num}...")
    time.sleep(1)
    
    try:
        # 开始发送PLY文件
        send_ply_files(client, ply_dir, debug=debug)
    except KeyboardInterrupt:
        print('程序被用户中断')
    finally:
        # 给客户端一些时间处理最后的数据
        time.sleep(1)
        print('资源已释放')

if __name__ == "__main__":
    # 从命令行获取参数，如果有的话 python ply_grpc.py 127.0.0.1 50051 C:/Users/24150/plydata true
    server = '127.0.0.1'
    port = 50055
    ply_directory = 'C:/Users/24150/plydata'
    debug_mode = False
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        server = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    if len(sys.argv) > 3:
        ply_directory = sys.argv[3]
    if len(sys.argv) > 4 and sys.argv[4].lower() in ('true', 't', '1', 'yes', 'y'):
        debug_mode = True
    
    main(
        server_addr=server,
        port_num=port,
        ply_dir=ply_directory,
        debug=debug_mode
    )