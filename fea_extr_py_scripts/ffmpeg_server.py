import grpc
from concurrent import futures
import data_stream_pb2
import data_stream_pb2_grpc
import time
import threading
import subprocess
import os
import platform
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
        # 为不同操作系统设置摄像头参数
        if platform.system() == 'Windows':
            # Windows使用dshow
            input_params = [
                '-f', 'dshow',
                '-rtbufsize', '1k',
                '-thread_queue_size', '2',
                '-video_size', '1280x720',
                '-framerate', '30',
                '-i', 'video=@device_pnp_\\\\?\\usb#vid_5986&pid_2169&mi_00#6&17e2b06b&1&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\\global'
            ]
        else:
            # macOS使用avfoundation
            input_params = [
                '-f', 'avfoundation',
                '-rtbufsize', '1k',
                '-thread_queue_size', '2',
                '-video_size', '1280x720',
                '-framerate', '30',
                '-i', '0'  # 通常0是第一个摄像头
            ]
        
        # 设置SDP文件路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sdp_file_path = os.path.join(script_dir, 'stream.sdp')
        
        # 基本ffmpeg命令参数
        ffmpeg_cmd = [
            'ffmpeg',
            '-use_wallclock_as_timestamps', '1',
            '-avioflags', 'direct',
        ]
        
        # 添加输入参数
        ffmpeg_cmd.extend(input_params)
        
        # 添加视频处理和编码参数
        ffmpeg_cmd.extend([
            '-vf', 'setpts=N/(30*TB)',
            '-c:v', 'libx265',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-x265-params', 'bframes=0:no-scenecut=1',
            '-b:v', '2M',
            '-pix_fmt', 'yuv420p',
            '-g', '30',
            '-keyint_min', '30',
            '-sc_threshold', '0',
            '-r', '30',
            
            # 输出流（RTP over UDP）
            '-f', 'rtp',
            '-sdp_file', sdp_file_path,
            'rtp://127.0.0.1:5005',  # 修改为接收端IP地址
        ])
        
        try:
            # 根据操作系统决定是否使用creationflags
            if platform.system() == 'Windows':
                self.ffmpeg_process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:  # macOS 和 Linux
                self.ffmpeg_process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            
            print(f"FFmpeg streaming started with command: {' '.join(ffmpeg_cmd)}")

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