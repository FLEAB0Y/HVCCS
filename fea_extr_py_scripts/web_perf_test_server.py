import grpc
from concurrent import futures
import threading
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import data_stream_pb2
import data_stream_pb2_grpc
from THStreamData import THStreamDataPayload, THDataWarehouse

# 添加全局变量来控制服务器
server_running = True
server_instance = None

class THStreamServiceServicer(data_stream_pb2_grpc.THStreamServiceServicer):
    def __init__(self):
        self.receive_data_buffer = THDataWarehouse(capacity=100)

    def BidirectionalStream(self, request_iterator, context):
        try:
            for request in request_iterator:
                print(f"***********Received request seqNo:{request.seqNo}***********")
                # 缓冲区满了就等待
                buffer_size = self.receive_data_buffer.get_size()
                while buffer_size > 10:
                    time.sleep(1./100.)
                    print("Buffer is full, waiting...")
                    buffer_size = self.receive_data_buffer.get_size()
                
                self.receive_data_buffer.add_item(request)
                yield data_stream_pb2.THStreamResponse(retCode=0, retMsg=f"{request.seqNo} Data received..")

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Server error: {str(e)}")
            yield data_stream_pb2.THStreamResponse(retCode=1, retMsg="Internal server error")

def serve(servicer, port):
    global server_instance
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    data_stream_pb2_grpc.add_THStreamServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    server_instance = server
    print(f"Server started, listening on port {port}")
    
    # 使用事件循环检查是否应该关闭
    global server_running
    while server_running:
        time.sleep(0.5)  # 每0.5秒检查一次
        
    print("Server stopping...")
    server.stop(grace=5)  # 优雅地关闭，给5秒处理中的请求
    print("Server stopped.")

def plot_latency_curve(data_sizes, latencies, frame_rate):
    """绘制数据大小-时延曲线并保存，包含帧率信息"""
    plt.figure(figsize=(12, 8))
    plt.plot(data_sizes, latencies, 'bo-', linewidth=2)
    plt.xscale('log')  # 使用对数刻度显示数据大小
    plt.grid(True)
    plt.xlabel('Data Size (Bytes)', fontsize=14)
    plt.ylabel('Transmission Latency (ms)', fontsize=14)
    title = f'Data Size vs Transmission Latency (Avg. Frame Rate: {frame_rate:.2f} fps)'
    plt.title(title, fontsize=16)
    
    # 在图上添加帧率文本
    plt.annotate(f'Average Frame Rate: {frame_rate:.2f} fps', 
                 xy=(0.02, 0.95), xycoords='axes fraction',
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
    
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建保存目录路径
    res_path = os.path.join(script_dir, "..", "res", "web_test_res")
    
    # 创建保存目录
    os.makedirs(res_path, exist_ok=True)
    
    # 保存图表
    plt.savefig(os.path.join(res_path, 'size_vs_latency.png'), dpi=300)
    plt.savefig(os.path.join(res_path, 'size_vs_latency.svg'), format='svg')
    
    print(f"Charts saved to {os.path.join(res_path, 'size_vs_latency.png')} and {os.path.join(res_path, 'size_vs_latency.svg')}")
    
    # 保存原始数据为CSV，包含帧率信息
    csv_path = os.path.join(res_path, 'latency_data.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Data_Size(Bytes)', 'Latency(ms)', 'Frame_Rate(fps)'])
        for size, latency in zip(data_sizes, latencies):
            writer.writerow([size, latency, frame_rate])
    
    print(f"Raw data saved to {csv_path}")

if __name__ == '__main__':
    # 初始化时间差
    diff = 54
    print(f"时间差: {diff} ms")

    # 存储测试结果 - 使用字典按大小分组
    latency_by_size = {}  # {数据大小: [延迟1, 延迟2, ...]}
    expected_sizes = set(np.logspace(2, np.log10(3600000), 100).astype(int))
    repeat_times = 10  # 每个大小期望接收10次

    servicer = THStreamServiceServicer()
    custom_port = 50051
    
    # 启动服务器线程
    server_process = threading.Thread(target=serve, args=(servicer, custom_port))
    server_process.daemon = False  # 设为非守护线程
    server_process.start()

    print("开始等待测试数据...")

    try:
        # 用于计算帧率
        start_time = time.time()
        total_received = 0
        
        # 持续接收数据，直到所有大小的数据都收到了足够的样本
        total_expected_packets = len(expected_sizes) * repeat_times
        
        while total_received < total_expected_packets:
            cnt = servicer.receive_data_buffer.get_size()
            if cnt < 1:
                time.sleep(1./100.)
                continue
                
            data = servicer.receive_data_buffer.get_items()
            current_timestamp_ms = int(time.time() * 1000)
            
            try:
                # 解析JSON格式的extDesc
                metadata = json.loads(data.extDesc)
                data_size = metadata["size"]
                sent_timestamp_ms = metadata["timestamp"]
                
                # 计算传输耗时
                transmission_time_ms = current_timestamp_ms - sent_timestamp_ms - diff
                
                if transmission_time_ms < 0:
                    print(f"Warning: Time synchronization issue, adjusting to positive value")
                    transmission_time_ms = abs(transmission_time_ms)
                
                # 记录到对应大小分组中
                if data_size not in latency_by_size:
                    latency_by_size[data_size] = []
                
                latency_by_size[data_size].append(transmission_time_ms)
                total_received += 1
                
                # 显示进度
                print(f"Received: {total_received}/{total_expected_packets} packets")
                
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                print(f"Error processing data: {e}")
        
        # 计算总帧率
        end_time = time.time()
        total_time = end_time - start_time
        average_frame_rate = total_received / total_time
        print(f"\nTotal time: {total_time:.2f} seconds")
        print(f"Average frame rate: {average_frame_rate:.2f} fps")
                
        # 所有测试完成后，计算每组的平均延迟
        print("\nAll tests completed! Calculating average latencies...")

        # 计算各大小的平均延迟
        sizes = []
        avg_latencies = []
        
        for size in sorted(latency_by_size.keys()):
            latencies = latency_by_size[size]
            avg_latency = sum(latencies) / len(latencies)
            sizes.append(size)
            avg_latencies.append(avg_latency)
            print(f"Size: {size} bytes, Avg Latency: {avg_latency:.2f} ms")
        
        # 绘制延迟曲线，包含帧率信息
        plot_latency_curve(sizes, avg_latencies, average_frame_rate)
        
    except KeyboardInterrupt:
        print("Test interrupted by user.")
    finally:
        # 停止服务器
        print("Shutting down server...")
        server_running = False
        server_process.join(timeout=10)  # 等待服务器线程完成
        print("Server shutdown complete.")





