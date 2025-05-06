import grpc
from concurrent import futures
import data_stream_pb2
import data_stream_pb2_grpc
import time
import threading
import subprocess
import os
import torch
import datetime

class THStreamServiceServicer(data_stream_pb2_grpc.THStreamServiceServicer):
    def __init__(self):
        self.running = True
        self.seq_no = 0
        self.lock = threading.Lock()
        self.ffmpeg_process = None
        self.streaming_thread = None
        self.start_streaming()

    def start_streaming(self):
        """Start FFmpeg streaming in a separate thread"""
        with self.lock:
            if self.streaming_thread is None or not self.streaming_thread.is_alive():
                self.streaming_thread = threading.Thread(target=self.start_ffmpeg_stream)
                self.streaming_thread.start()
                print("FFmpeg streaming thread started")

    def next_seq_no(self):
        self.seq_no += 1
        return str(self.seq_no)

    def start_ffmpeg_stream(self):
        """Start FFmpeg streaming pipeline"""

        ffmpeg_cmd = [
            'ffmpeg',
            '-use_wallclock_as_timestamps', '1',  # 使用系统时钟作为时间戳
            '-avioflags', 'direct',  # 减少I/O缓冲
            
            # 摄像头输入 - 修改分辨率为1280x720
            '-f', 'dshow',
            '-rtbufsize', '1k',  # 修正：去掉空格
            '-thread_queue_size', '2',  # 增大队列大小
            '-video_size', '1280x720',  # 修改为720p分辨率
            '-framerate', '30',
            '-i',
            'video=@device_pnp_\\\\?\\usb#vid_5986&pid_2169&mi_00#6&17e2b06b&1&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\\global',
            
            # 视频处理 - 保持720p分辨率
            '-vf', 'setpts=N/(30*TB)',  # 只保留时间戳处理
            
            # 视频编码参数（软件编码器 libx265）
            '-c:v', 'libx265',  # 使用H.265编码
            '-preset', 'ultrafast',  # 使用最快的预设以减少CPU负载
            '-tune', 'zerolatency',
            '-x265-params', 'bframes=0:no-scenecut=1',  # 低延迟参数
            '-b:v', '2M',  # 码率2Mbps
            '-pix_fmt', 'yuv420p',
            '-g', '30',
            '-keyint_min', '30',
            '-sc_threshold', '0',
            '-r', '30',  # 明确设置输出帧率
            
            # 输出流（RTP over UDP）
            '-f', 'rtp',
            '-sdp_file', 'D:\\HVCCS\\fea_extr_py_scripts\\stream.sdp',  # 自动生成SDP文件
            'rtp://183.173.139.132:5005',  # 修改为Mac电脑IP地址
        ]
        
        try:
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
            )
            print("FFmpeg streaming started")

            # Monitor FFmpeg output
            while self.running:
                output = self.ffmpeg_process.stderr.readline()
                if output == b'' and self.ffmpeg_process.poll() is not None:
                    break
                if output:
                    print(output.strip())

        except Exception as e:
            print(f"Error in FFmpeg streaming: {e}")
        finally:
            if self.ffmpeg_process:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait()
                print("FFmpeg streaming stopped")

    def BidirectionalStream(self, request_iterator, context):
        """Handle bidirectional streaming - just maintain connection"""
        try:
            for request in request_iterator:
                time.sleep(0.1)
        except Exception as e:
            print(f"Streaming connection error: {e}")
        finally:
            print("Client disconnected")

    def run(self):
        """Start the server"""
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        data_stream_pb2_grpc.add_THStreamServiceServicer_to_server(self, server)
        server.add_insecure_port('[::]:50051')
        server.start()
        print("Server started, listening on port 50051")

        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.running = False
            print("Shutting down...")

            # Stop FFmpeg if running
            if self.ffmpeg_process:
                self.ffmpeg_process.terminate()
            # Wait for streaming thread to finish
            if self.streaming_thread and self.streaming_thread.is_alive():
                self.streaming_thread.join()

        finally:
            server.stop(0)
            print("Server stopped")


if __name__ == '__main__':
    server = THStreamServiceServicer()
    server.run()