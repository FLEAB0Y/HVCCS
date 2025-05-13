import grpc
from concurrent import futures
import data_stream_pb2
import data_stream_pb2_grpc
from THStreamData import THStreamDataPayload, THDataWarehouse
import threading
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

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
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    data_stream_pb2_grpc.add_THStreamServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"Server started, listening on port {port}")
    server.wait_for_termination()

def plot_latency_curve(data_sizes, latencies):
    """绘制数据大小-时延曲线并保存"""
    plt.figure(figsize=(12, 8))
    plt.plot(data_sizes, latencies, 'bo-', linewidth=2)
    plt.xscale('log')  # 使用对数刻度显示数据大小
    plt.grid(True)
    plt.xlabel('Data Size (Bytes)', fontsize=14)
    plt.ylabel('Transmission Latency (ms)', fontsize=14)
    plt.title('Data Size vs Transmission Latency', fontsize=16)
    
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
    
    # 保存原始数据为CSV
    csv_path = os.path.join(res_path, 'latency_data.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Data_Size(Bytes)', 'Latency(ms)'])
        for size, latency in zip(data_sizes, latencies):
            writer.writerow([size, latency])
    
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
    server_process = threading.Thread(target=serve, args=(servicer, custom_port))
    server_process.daemon = True
    server_process.start()

    print("开始等待测试数据...")

    try:
        # 持续接收数据，直到所有大小的数据都收到了足够的样本
        total_expected_packets = len(expected_sizes) * repeat_times
        total_received = 0
        
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
                
                # 计算已收到的数据包中，每个大小完成的数量
                completed = 0
                for size, latencies in latency_by_size.items():
                    if len(latencies) >= repeat_times:
                        completed += 1
                
                total_received += 1
                print(f"[{total_received}/{total_expected_packets}] Size: {data_size} bytes, Latency: {transmission_time_ms} ms, Completed sizes: {completed}/{len(expected_sizes)}")
                
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                print(f"Failed to parse data: {data.extDesc}, Error: {e}")
    
        # 所有测试完成后，计算每组的平均延迟
        print("\nAll tests completed! Calculating average latencies...")
        avg_data_sizes = []
        avg_latencies = []
        
        for size in sorted(latency_by_size.keys()):
            latencies = latency_by_size[size]
            # 确保每个大小都有足够的样本
            if len(latencies) >= repeat_times:
                # 取前repeat_times个样本计算平均值
                avg_latency = sum(latencies[:repeat_times]) / repeat_times
                avg_data_sizes.append(size)
                avg_latencies.append(avg_latency)
                print(f"Size: {size} bytes, Average Latency: {avg_latency:.2f} ms")
        
        # 绘制图表并保存结果
        plot_latency_curve(avg_data_sizes, avg_latencies)
        
    except KeyboardInterrupt:
        print("\nTest interrupted")
        if latency_by_size:
            print("Saving collected data...")
            # 计算已收集数据的平均值
            avg_data_sizes = []
            avg_latencies = []
            
            for size in sorted(latency_by_size.keys()):
                latencies = latency_by_size[size]
                if len(latencies) > 0:  # 只要有数据就计算平均值
                    avg_latency = sum(latencies) / len(latencies)
                    avg_data_sizes.append(size)
                    avg_latencies.append(avg_latency)
            
            plot_latency_curve(avg_data_sizes, avg_latencies)





