import grpc
from concurrent import futures
import data_stream_pb2
import data_stream_pb2_grpc
from THStreamData import THStreamDataPayload, THDataWarehouse
import threading
import time

class THStreamServiceServicer(data_stream_pb2_grpc.THStreamServiceServicer):
    def __init__(self):
        self.receive_data_buffer = THDataWarehouse(capacity=100)

    def BidirectionalStream(self, request_iterator, context):
        try:
            for request in request_iterator:
                # print(f"***********Received request seqNo:{request.seqNo}***********")
                # 缓冲区满了就等待
                buffer_size = self.receive_data_buffer.get_size()
                while buffer_size > 30:
                    time.sleep(0.01)
                    print("Buffer is full, waiting...")
                    buffer_size = self.receive_data_buffer.get_size()
                
                self.receive_data_buffer.add_item(request)
                # if request.rgbData:
                #     print(f"Received RGB data of length {len(request.rgbData)}")
                # if request.pointData:
                #     print(f"Received point data of length {len(request.pointData)}")
                # if request.faceData:
                #     print(f"Received face data of length {len(request.faceData)}")
                # if request.limbData:
                #     print(f"Received limb data of length {len(request.limbData)}")
                # if request.extData:
                #     print(f"Received ext data of length {len(request.extData)}")
                # if request.extDesc:
                #     print(f"Received ext desc: {request.extDesc}")
                # 发送响应给客户端
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

if __name__ == '__main__':
    # 初始化时间差
    diff = 0
    print(f"时间差: {diff} ms")

    servicer = THStreamServiceServicer()
    custom_port = 50051  # 替换为您需要的端口号
    server_process = threading.Thread(target=serve, args=(servicer, custom_port))
    server_process.start()

    while True:    
        cnt = servicer.receive_data_buffer.get_size()
        while cnt < 1:
            time.sleep(1./100.)
            cnt = servicer.receive_data_buffer.get_size()
        data = servicer.receive_data_buffer.get_items()

        # 获取当前时间戳（毫秒）
        current_timestamp_ms = int(time.time() * 1000) 

        # 计算传输耗时
        try:
            sent_timestamp_ms = int(data.extDesc)  # 将 extDesc 转换为整数
            transmission_time_ms = current_timestamp_ms - sent_timestamp_ms - diff
            print(f"收到的faceData: {data.faceData[:5]}...")  # 打印前5个元素
            print(f"收到的limbData: {data.limbData[:5]}...")  # 打印前5个元素
            # 检查传输时间是否为负数
            if transmission_time_ms < 0:
                print(f"警告：系统时间不同步，传输耗时计算为负数！")
                print(f"当前时间戳：{current_timestamp_ms} ms，发送时间戳：{sent_timestamp_ms} ms")
            else:
                print(f"传输耗时：{transmission_time_ms} ms")
        except ValueError:
            print(f"无法解析 extDesc: {data.extDesc}")





