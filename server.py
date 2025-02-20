import grpc
from concurrent import futures
import data_stream_pb2
import data_stream_pb2_grpc
from THStreamData import THStreamDataPayload, THDataWarehouse
import multiprocessing
import time

class THStreamServiceServicer(data_stream_pb2_grpc.THStreamServiceServicer):
    def __init__(self):
        self.receive_data_buffer = THDataWarehouse(capacity=100)

    def BidirectionalStream(self, request_iterator, context):
        try:
            for request in request_iterator:
                print(f"***********Received request seqNo:{request.seqNo}***********")
                self.receive_data_buffer.add_item(request)
                if request.rgbData:
                    print(f"Received RGB data of length {len(request.rgbData)}")
                if request.pointData:
                    print(f"Received point data of length {len(request.pointData)}")
                if request.faceData:
                    print(f"Received face data of length {len(request.faceData)}")
                if request.limbData:
                    print(f"Received limb data of length {len(request.limbData)}")
                if request.extData:
                    print(f"Received ext data of length {len(request.extData)}")
                if request.extDesc:
                    print(f"Received ext desc: {request.extDesc}")

                # 发送响应给客户端
                yield data_stream_pb2.THStreamResponse(retCode=0, retMsg=f"{request.seqNo} Data received..")

            # 注意：在双向流式 RPC 中，通常不需要在循环结束后返回单个响应
            # 但如果需要发送一个结束信号或总结信息，可以在这里添加

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Server error: {str(e)}")
            yield data_stream_pb2.THStreamResponse(retCode=1, retMsg="Internal server error")

def serve(servicer):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    data_stream_pb2_grpc.add_THStreamServiceServicer_to_server(servicer, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started, listening on port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    servicer = THStreamServiceServicer()
    server_process = multiprocessing.Process(target=serve, args=(servicer,))
    server_process.start()

    while True:    
        cnt = servicer.receive_data_buffer.get_size()
        while cnt < 1:
            time.sleep(1./30.)
            cnt = servicer.receive_data_buffer.get_size()
        data = servicer.receive_data_buffer.get_items()
        cnt = servicer.receive_data_buffer.get_size()
        print(cnt)
        print(str(data.extDesc).zfill(5))
        time.sleep(1./30.)




