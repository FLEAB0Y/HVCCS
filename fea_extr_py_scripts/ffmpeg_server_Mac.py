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
        
        # 获取SDP文件的绝对路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sdp_file_path = os.path.join(script_dir, "stream.sdp")
        
        if platform.system() == 'Darwin':  # macOS
            # macOS 下使用 AVFoundation 捕获摄像头
            ffmpeg_cmd = [
                'ffmpeg',
                '-use_wallclock_as_timestamps', '1',
                '-avioflags', 'direct',
                
                # macOS 摄像头输入 (AVFoundation)
                '-f', 'avfoundation',
                '-framerate', '30',
                '-video_size', '1280x720',  # 修改为摄像头支持的分辨率
                '-i', '0:none',  # 0:none 表示首个视频设备，无音频
                
                # 视频处理
                '-vf', 'setpts=N/(30*TB)',
                
                # 视频编码参数
                '-c:v', 'libx265',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-x265-params', 'lossless=0:bframes=0',
                '-b:v', '2M',  # 降低比特率以匹配较低的分辨率
                '-pix_fmt', 'yuv420p',
                '-g', '30',
                '-keyint_min', '30',
                '-sc_threshold', '0',
                '-r', '30',
                
                # 输出流
                '-f', 'rtp',
                '-sdp_file', sdp_file_path,
                'rtp://127.0.0.1:5005',
            ]
        else:  # Windows
            ffmpeg_cmd = [
                'ffmpeg',
                '-use_wallclock_as_timestamps', '1',
                '-avioflags', 'direct',
                
                # Windows DirectShow 摄像头输入
                '-f', 'dshow',
                '-rtbufsize', '1k',
                '-thread_queue_size', '2',
                '-video_size', '1920x1080',
                '-framerate', '30',
                '-i', 'video=@device_pnp_\\\\?\\usb#vid_5986&pid_2169&mi_00#6&17e2b06b&1&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\\global',
                
                '-vf', 'setpts=N/(30*TB)',
                
                '-c:v', 'libx265',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-x265-params', 'lossless=0:bframes=0',
                '-b:v', '4M',
                '-pix_fmt', 'yuv420p',
                '-g', '30',
                '-keyint_min', '30',
                '-sc_threshold', '0',
                '-r', '30',
                
                '-f', 'rtp',
                '-sdp_file', sdp_file_path,
                'rtp://127.0.0.1:5005',
            ]
        
        try:
            # 根据操作系统决定是否使用 creationflags
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
            
            print("FFmpeg streaming started")

            # 监控 FFmpeg 输出
            while self.running:
                output = self.ffmpeg_process.stderr.readline()
                if output == b'' and self.ffmpeg_process.poll() is not None:
                    break
                if output:
                    print(output.strip().decode('utf-8', errors='ignore'))  # 解码输出并处理非UTF-8字符

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