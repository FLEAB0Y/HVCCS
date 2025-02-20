import grpc
import data_stream_pb2
import data_stream_pb2_grpc
import time
import multiprocessing
from THStreamData import THStreamDataPayload, THDataWarehouse

class THStreamClient:
    def __init__(self, host='127.0.0.1', port=50051):
        self.channel = grpc.insecure_channel(f'{host}:{port}')
        self.stub = data_stream_pb2_grpc.THStreamServiceStub(self.channel)
        self.seq_no = 0
        # 数据缓存
        self.send_data_buffer = THDataWarehouse(capacity= 100)
        self.lock = multiprocessing.Lock()

    def next_seq_no(self):
        with self.lock:
            self.seq_no += 1
        return str(self.seq_no)

    def send_data(self):
        try:
            send_data = self.request_generator()
            time.sleep(10.) # test
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
    client = THStreamClient(host='127.0.0.1', port=50051)
    client_process = multiprocessing.Process(target=run_client, args=(client,))
    client_process.start()
    i = 0
    while True:
        payload1 = THStreamDataPayload(rgb_data=b'\x01', point_data=b'\x02', face_data=b'\x03', limb_data=b'\x04',
                                   ext_data=b'\x05', ext_desc=f"{str(i)}")
        client.send_data_buffer.add_item(payload1)
        # 缓冲区满了就等待
        buffer_size = client.send_data_buffer.get_size()
        while buffer_size >= 10:
            time.sleep(0.1)
            buffer_size = client.send_data_buffer.get_size()
        time.sleep(1./30.)
        i += 1



