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
                    time.sleep(0.1)
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
    
    # 创建保存目录
    os.makedirs('results', exist_ok=True)
    
    # 保存图表
    plt.savefig('results/size_vs_latency.png', dpi=300)
    plt.savefig('results/size_vs_latency.svg', format='svg')
    
    print(f"图表已保存到 results/size_vs_latency.png 和 results/size_vs_latency.svg")
    
    # 保存原始数据为CSV
    with open('results/latency_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Data_Size(Bytes)', 'Latency(ms)'])
        for size, latency in zip(data_sizes, latencies):
            writer.writerow([size, latency])
    
    print(f"原始数据已保存到 results/latency_data.csv")

if __name__ == '__main__':
    # 初始化时间差（如果需要的话）
    diff = -71039
    print(f"时间差: {diff} ms")

    # 存储测试结果
    data_sizes = []
    latencies = []
    expected_tests = 100
    received_tests = 0

    servicer = THStreamServiceServicer()
    custom_port = 50051
    server_process = threading.Thread(target=serve, args=(servicer, custom_port))
    server_process.daemon = True
    server_process.start()

    print("开始等待测试数据...")

    try:
        while received_tests < expected_tests:
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
                    print(f"警告：系统时间不同步，传输耗时计算为负数！调整为正值")
                    transmission_time_ms = abs(transmission_time_ms)
                
                print(f"[{received_tests+1}/{expected_tests}] 数据大小: {data_size} 字节, 传输耗时: {transmission_time_ms} ms")
                
                # 记录结果
                data_sizes.append(data_size)
                latencies.append(transmission_time_ms)
                received_tests += 1
                
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                print(f"无法解析数据: {data.extDesc}, 错误: {e}")
    
        # 所有测试完成后，绘制图表并保存结果
        print("\n所有测试完成! 正在生成图表...")
        plot_latency_curve(data_sizes, latencies)
        
    except KeyboardInterrupt:
        print("\n测试被中断")
        if data_sizes and latencies:
            print("正在保存已收集的数据...")
            plot_latency_curve(data_sizes, latencies)





