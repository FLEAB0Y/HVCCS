import grpc
import data_stream_pb2
import data_stream_pb2_grpc
import time
import threading
from THStreamData import THStreamDataPayload, THDataWarehouse
# from linux_calculate_latency import init_time

class THStreamClient:
    def __init__(self, host='127.0.0.1', port=50051):
        self.channel = grpc.insecure_channel(f'{host}:{port}')
        self.stub = data_stream_pb2_grpc.THStreamServiceStub(self.channel)
        self.seq_no = 0
        # 数据缓存
        self.send_data_buffer = THDataWarehouse(capacity= 100)
        self.lock = threading.Lock()

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

    # 这是计算数据处理延迟的发送函数，在数据发送后获取一个时间戳，与数据包中的时间戳做差实现
    # def send_data(self):
    #     try:
    #         send_data = self.request_generator()
    #         if not send_data:
    #             return
            
    #         for request in send_data:
    #             # 从请求的ext_desc字段获取发送时的时间戳
    #             send_timestamp = int(request.extDesc)
                
    #             # 发送请求并获取响应
    #             response_iterator = self.stub.BidirectionalStream(iter([request]))
                
    #             # 计算发送完成后的时间戳
    #             current_timestamp = int(time.time() * 1000)
    #             delay = current_timestamp - send_timestamp
                
    #             print(f"数据包延迟: {delay} ms")
                
    #             # 处理响应
    #             for response in response_iterator:
    #                 print(f"Received response: retCode={response.retCode}, retMsg={response.retMsg}")
    #     except grpc.RpcError as e:
    #         print(f"gRPC error: {e}")

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

    def run(self, interval=1./30.): # interval = 1./30.
        """
        :param interval: 1秒30帧数据
        :return:
        """
        try:
            while True:
                self.send_data()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("Client stopped")

def run_client(client):
    client.run()

if __name__ == '__main__':
    # 初始化时间差
    # diff = init_time()
    # print(f"时间差: {diff} ms")
    
    # Labserver: 101.6.65.237
    # local: 127.0.0.1
    client = THStreamClient(host='127.0.0.1', port=50051)
    client_thread = threading.Thread(target=run_client, args=(client,))
    client_thread.start()

    while True:
        
        # 缓冲区满了就等待
        buffer_size = client.send_data_buffer.get_size()
        while buffer_size >= 10:
            time.sleep(0.1)
            buffer_size = client.send_data_buffer.get_size()
        # 生成250,0000字节的数据
        large_data = b'\x00' * 2500000
        # 获取当前时间戳（毫秒）
        timestamp_ms = int(time.time() * 1000) 
        # 创建数据负载
        payload = THStreamDataPayload(rgb_data=b'\x01', 
                                       point_data=large_data, 
                                       face_data=b'\x03', 
                                       limb_data=b'\x04',
                                       ext_data=b'\x05', 
                                       ext_desc=f"{str(timestamp_ms)}")
        client.send_data_buffer.add_item(payload)





