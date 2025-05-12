import grpc
import data_stream_pb2
import data_stream_pb2_grpc
import time
import threading
import numpy as np
from THStreamData import THStreamDataPayload, THDataWarehouse
import json

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
        :param interval: 1秒30帧数据
        :return:
        """
        try:
            while not self.test_completed:
                self.send_data()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("Client stopped")

def run_client(client):
    client.run()

def generate_test_sizes():
    """生成从100到3600000字节的100个指数间隔点"""
    return np.logspace(2, np.log10(3600000), 100).astype(int)

if __name__ == '__main__':
    # 创建客户端实例
    client = THStreamClient(host='192.168.1.11', port=50051)
    client_thread = threading.Thread(target=run_client, args=(client,))
    client_thread.daemon = True  # 确保主线程退出时，此线程也退出
    client_thread.start()
    
    # 生成测试数据大小
    test_sizes = generate_test_sizes()
    print(f"即将发送100个不同大小的数据包，大小范围：{test_sizes[0]}字节 - {test_sizes[-1]}字节")
    
    # 等待确认开始测试
    input("按回车键开始测试...")
    
    # 发送不同大小的数据包
    for i, size in enumerate(test_sizes):
        # 等待缓冲区有空间
        while client.send_data_buffer.get_size() >= 10:
            time.sleep(0.1)
        
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
        
        print(f"[{i+1}/100] 已发送数据：{size} 字节")
        
        # 每发送一个数据包后等待一段时间，确保数据被完全处理
        time.sleep(0.5)
    
    # 等待所有数据被发送和处理
    time.sleep(5)
    print("测试完成!")
    client.test_completed = True





