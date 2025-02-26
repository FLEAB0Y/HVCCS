import grpc
import data_stream_pb2
import data_stream_pb2_grpc
import time
import multiprocessing
from THStreamData import THStreamDataPayload, THDataWarehouse
import os

class THStreamClient:
    def __init__(self, host='127.0.0.1', port=50051, data_queue=None):
        self.channel = grpc.insecure_channel(f'{host}:{port}')
        self.stub = data_stream_pb2_grpc.THStreamServiceStub(self.channel)
        self.seq_no = 0
        # 使用进程间共享队列代替内部缓存
        self.data_queue = data_queue if data_queue else multiprocessing.Queue(maxsize=100)
        self.lock = multiprocessing.Lock()

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
        if self.data_queue.empty():
            return None
        else:
            # 从队列中获取数据并生成请求
            items_to_process = []
            # 尝试从队列获取所有可用数据
            while not self.data_queue.empty() and len(items_to_process) < 100:
                try:
                    items_to_process.append(self.data_queue.get(block=False))
                except:
                    break  # 队列为空或其他错误

            if not items_to_process:
                return None
                
            for one_data in items_to_process:
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
            while True:
                self.send_data()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("Client stopped")


def data_sender_process(host, port, data_queue):
    """发送数据的独立进程"""
    client = THStreamClient(host=host, port=port, data_queue=data_queue)
    client.run()


def data_collector_process(data_queue, max_queue_size=100):
    """收集数据的独立进程"""
    i = 0
    try:
        while True:
            i += 1
            time_sample_ms = int(time.time() * 1000)
            # 生成1500字节的随机数据
            random_data = os.urandom(1500)
            # 将随机数据分配到不同字段
            chunk_size = 300  # 每个字段分配300字节，共5个字段
            payload = THStreamDataPayload(
                rgb_data=random_data[0:chunk_size],
                point_data=random_data[chunk_size:chunk_size*2],
                face_data=random_data[chunk_size*2:chunk_size*3],
                limb_data=random_data[chunk_size*3:chunk_size*4],
                ext_data=random_data[chunk_size*4:chunk_size*5],
                ext_desc=f"{str(time_sample_ms)+ 'No.' +str(i).zfill(5)}"
            )
            
            # 如果队列长度小于80%的最大长度，将数据放入队列
            if data_queue.qsize() <= max_queue_size * 0.8:
                # 将数据放入队列
                data_queue.put(payload)
                continue
            
            
            # 控制数据生成速率
            time.sleep(1./30.)  # 每秒约30帧
    except KeyboardInterrupt:
        print("Data collector stopped")


if __name__ == '__main__':
    # 创建进程间共享队列
    shared_queue = multiprocessing.Queue(maxsize=100)
    
    # 创建并启动数据发送进程
    sender_process = multiprocessing.Process(
        target=data_sender_process, 
        args=('127.0.0.1', 50051, shared_queue)
    )
    sender_process.daemon = True
    sender_process.start()
    
    # 创建并启动数据收集进程
    collector_process = multiprocessing.Process(
        target=data_collector_process,
        args=(shared_queue, 100)
    )
    collector_process.daemon = True
    collector_process.start()
    
    try:
        # 主进程等待，可以接收键盘中断
        sender_process.join()
        collector_process.join()
    except KeyboardInterrupt:
        print("主程序退出")

