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
            #'-async', '1',  # 启用输入流同步（关键参数）
            #'-vsync', '0',  # 使用帧率同步模式（避免丢帧导致错乱）
            #'-fflags', '+nobuffer+discardcorrupt',  # 减少输入缓冲
            '-avioflags', 'direct',  # 减少I/O缓冲
             #'-strict', 'experimental',
            #'-debug_ts',  # 打印时间戳调试信息
             #'-loglevel', 'debug',  # 输出完整日志
            # 摄像头1输入
            '-f', 'dshow',
            '-rtbufsize', '1k ',  # 增加缓冲区
            #'-max_delay', '100000',
            '-thread_queue_size', '2',  # 增大队列大小
            '-video_size', '1920x1080',
            '-framerate', '30',
           # '-pixel_format', 'yuyv422',
            '-i',
            'video=@device_pnp_\\\\?\\usb#vid_0c45&pid_636f&mi_00#7&457b37f&0&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\\global',
            # 摄像头2输入
            '-f', 'dshow',
            '-rtbufsize', '1k',  # 增加缓冲区
            '-thread_queue_size', '2',  # 增大队列大小
            '-video_size', '1920x1080',
            '-framerate', '30',
           # '-pixel_format', 'yuyv422',
            '-i',
            'video=@device_pnp_\\\\?\\usb#vid_0c45&pid_636f&mi_00#7&b738621&0&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\\global',

            '-filter_complex',
            # 摄像头1：裁剪 + 时间戳（带字体路径）
            '[0:v]crop=1024:1024:448:28,'
            #'drawtext=text="%{pts\\\\:hms}":fontfile=C\\\\:/Windows/Fonts/arial.ttf:fontsize=32:fontcolor=white:box=1:boxcolor=black@0.5:x=10:y=10,'
            'setpts=N/(30*TB)[sync1];'
            
            # 摄像头2：裁剪 + 时间戳（带字体路径）
            '[1:v]crop=1024:1024:448:28,'
            #'drawtext=text="%{pts\\\\:hms}":fontfile=C\\\\:/Windows/Fonts/arial.ttf:fontsize=32:fontcolor=white:box=1:boxcolor=black@0.5:x=10:y=10,'
            'setpts=N/(30*TB)[sync2];'

            # 拼接两路视频
            '[sync1][sync2]hstack=inputs=2[outv]',
            # 视频编码参数（HEVC NVENC）
            '-map', '[outv]',
            '-c:v', 'hevc_nvenc',
            '-preset', 'p1',
            '-tune', 'll',
            '-b:v', '2M',
            '-profile:v', 'main',
            '-rc', 'cbr',
            '-rc-lookahead', '0',
            '-forced-idr', '1',
            '-g', '30',
            '-bf', '0',
            '-zerolatency', '1',
            '-delay', '0',  # 显式设置零延迟
            '-r', '30',  # 明确设置输出帧率
            #'-fps_mode', 'passthrough',  # 禁用帧率同步
            #'-flush_packets', '1',
            # 输出流（RTP over UDP）

            '-f', 'rtp',
            'rtp://127.0.0.1:5005',#183.173.48.193  #我的电脑183.173.171.204
            #'-payload_type', '96',
            #'-sdp_file', 'stream.sdp'

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