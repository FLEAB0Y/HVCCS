import grpc
import data_stream_pb2
import data_stream_pb2_grpc
import time
import threading
import numpy as np
from THStreamData import THStreamDataPayload, THDataWarehouse
import json
import os

class THStreamClient:
    def __init__(self, host='127.0.0.1', port=50051):
        self.channel = grpc.insecure_channel(f'{host}:{port}')
        self.stub = data_stream_pb2_grpc.THStreamServiceStub(self.channel)
        self.seq_no = 0
        # 数据缓存
        self.send_data_buffer = THDataWarehouse(capacity=100)
        self.lock = threading.Lock()
        self.test_completed = False

    def next_seq_no(self):
        with self.lock:
            self.seq_no += 1
        return str(self.seq_no)

    def send_data(self):
        try:
            send_data = self.request_generator()
            if not send_data:
                return
            response_iterator = self.stub.BidirectionalStream(send_data)
            for response in response_iterator:
                print(f"Received response: retCode={response.retCode}, retMsg={response.retMsg}")
        except grpc.RpcError as e:
            print(f"gRPC error: {e}")

    def request_generator(self):
        if self.send_data_buffer.get_size() == 0:
            return None
        else:
            for i in range(self.send_data_buffer.get_size()):
                one_data = self.send_data_buffer.get_items()
                seq_no = self.next_seq_no()
                yield data_stream_pb2.THStreamRequest(seqNo=seq_no,
                                                      rgbData=one_data.rgb_data,
                                                      pointData=one_data.point_data,
                                                      faceData=one_data.face_data,
                                                      limbData=one_data.limb_data,
                                                      extData=one_data.ext_data,
                                                      extDesc=one_data.ext_desc)

    def run(self, interval=1./30.):
        """
        :param interval: 发送间隔时间
        :return:
        """
        try:
            while not self.test_completed:
                self.send_data()
                # time.sleep(interval)
        except KeyboardInterrupt:
            print("Client stopped")

def run_client(client, frame_rate):
    interval = 1.0 / frame_rate
    client.run(interval=interval)

def generate_test_sizes(min_size, max_size, num_sizes):
    """生成从min_size到max_size字节的num_sizes个指数间隔点"""
    return np.logspace(np.log10(min_size), np.log10(max_size), num_sizes).astype(int)

if __name__ == '__main__':
    # ===== 配置参数 =====
    # 服务器连接配置
    SERVER_HOST = '127.0.0.1'
    SERVER_PORT = 50051
    
    # 测试数据配置
    MIN_DATA_SIZE = 100          # 最小数据大小(字节)
    MAX_DATA_SIZE = 3600000      # 最大数据大小(字节)
    NUM_TEST_SIZES = 100         # 测试大小点的数量
    REPEAT_TIMES = 10            # 每个大小重复发送次数
    
    # 性能参数
    FRAME_RATE = 240             # 每秒发送帧数
    BUFFER_LIMIT = 5            # 缓冲区大小限制
    BUFFER_WAIT_TIME = 0.01      # 缓冲区满时等待时间(秒)
    
    # 保存结果
    SAVE_RESULTS = True
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "res", "web_test_res")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ===== 创建客户端 =====
    client = THStreamClient(host=SERVER_HOST, port=SERVER_PORT)
    client_thread = threading.Thread(target=run_client, args=(client, FRAME_RATE))
    client_thread.daemon = True  # 确保主线程退出时，此线程也退出
    client_thread.start()
    
    # ===== 生成测试数据大小 =====
    test_sizes = generate_test_sizes(MIN_DATA_SIZE, MAX_DATA_SIZE, NUM_TEST_SIZES)
    print(f"即将发送{len(test_sizes)}个不同大小的数据包，每个大小重复{REPEAT_TIMES}次")
    print(f"数据包大小范围：{test_sizes[0]}字节 - {test_sizes[-1]}字节")
    
    # ===== 等待确认开始测试 =====
    input("按回车键开始测试...")
    
    # ===== 发送测试数据 =====
    for i, size in enumerate(test_sizes):
        for repeat in range(REPEAT_TIMES):
            # 等待缓冲区有空间
            while client.send_data_buffer.get_size() >= BUFFER_LIMIT:
                time.sleep(BUFFER_WAIT_TIME)
            
            # 生成指定大小的数据
            test_data = b'\x00' * size
            
            # 获取当前时间戳（毫秒）
            timestamp_ms = int(time.time() * 1000)
            
            # 将大小和时间戳编码为JSON字符串
            metadata = json.dumps({
                "size": int(size),
                "timestamp": timestamp_ms
            })
            
            # 创建数据负载
            payload = THStreamDataPayload(
                rgb_data=b'\x01',
                point_data=test_data,  # 这里放大数据块
                face_data=b'\x03',
                limb_data=b'\x04',
                ext_data=b'\x05',
                ext_desc=metadata
            )
            
            # 添加到发送缓冲区
            client.send_data_buffer.add_item(payload)
            
            print(f"[{i+1}/{len(test_sizes)}] 大小: {size} 字节, 重复: {repeat+1}/{REPEAT_TIMES}")
    
    # ===== 等待所有数据被发送和处理 =====
    print("等待所有数据发送完成...")
    time.sleep(5)
    print("测试完成!")
    client.test_completed = True





