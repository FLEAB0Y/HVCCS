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

# 调试开关
DEBUG = False

def log(message, level="INFO"):
    """
    日志输出函数，根据调试开关决定是否输出
    
    参数:
        message (str): 日志信息
        level (str): 日志级别 (DEBUG, INFO, ERROR 等)
    """
    if DEBUG or level == "ERROR":
        time_str = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        print(f"[SERVER][{level}][{time_str}] {message}")

class THStreamServiceServicer(data_stream_pb2_grpc.THStreamServiceServicer):
    """视频流服务，负责从摄像头捕获并编码视频流"""
    
    def __init__(self, config):
        """
        初始化流媒体服务
        
        参数:
            config (dict): 配置字典，包含服务参数
        """
        self.config = config
        self.debug = config.get("debug", DEBUG)
        log("初始化流媒体服务", "INFO")
        log(f"配置: {self.config}", "DEBUG")
        
        self.running = True
        self.seq_no = 0
        self.lock = threading.Lock()
        self.ffmpeg_process = None
        self.streaming_thread = None
        self.start_streaming()

    def start_streaming(self):
        """在单独的线程中启动FFmpeg流传输"""
        with self.lock:
            log("准备启动流传输线程", "DEBUG")
            if self.streaming_thread is None or not self.streaming_thread.is_alive():
                self.streaming_thread = threading.Thread(target=self.start_ffmpeg_stream)
                self.streaming_thread.start()
                log("FFmpeg流线程已启动", "INFO")

    def next_seq_no(self):
        """生成下一个序列号"""
        self.seq_no += 1
        return str(self.seq_no)

    def start_ffmpeg_stream(self):
        """启动FFmpeg流传输管道"""
        log("启动FFmpeg流传输...", "INFO")
        
        # 获取输入参数（根据操作系统）
        input_params = self.get_system_input_params()
        log(f"输入参数: {input_params}", "DEBUG")
        
        # 设置SDP文件路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sdp_file_path = os.path.join(script_dir, self.config["sdp_file"])
        log(f"SDP文件路径: {sdp_file_path}", "DEBUG")
        
        # 构建FFmpeg命令
        ffmpeg_cmd = self.build_ffmpeg_command(input_params, sdp_file_path)
        log(f"FFmpeg命令: {' '.join(ffmpeg_cmd)}", "DEBUG")
        
        try:
            # 启动FFmpeg进程
            log("启动FFmpeg进程...", "DEBUG")
            self.ffmpeg_process = self.start_ffmpeg_process(ffmpeg_cmd)
            
            log(f"FFmpeg流已启动，PID: {self.ffmpeg_process.pid}", "INFO")

            # 监控FFmpeg输出
            log("开始监控FFmpeg输出", "DEBUG")
            self.monitor_ffmpeg_output()

        except Exception as e:
            log(f"FFmpeg流错误: {e}", "ERROR")
        finally:
            if self.ffmpeg_process:
                log("终止FFmpeg进程", "DEBUG")
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait()
                log("FFmpeg流已停止", "INFO")
    
    def get_system_input_params(self):
        """根据操作系统获取适当的输入参数"""
        video_width = self.config["video_resolution"][0]
        video_height = self.config["video_resolution"][1]
        framerate = self.config["framerate"]
        
        system = platform.system()
        log(f"检测到操作系统: {system}", "DEBUG")
        
        if system == 'Windows':
            # Windows使用dshow
            return [
                '-f', 'dshow',
                '-rtbufsize', '1k',
                '-thread_queue_size', '2',
                '-video_size', f'{video_width}x{video_height}',
                '-framerate', str(framerate),
                '-i', self.config["video_device_win"]
            ]
        else:
            # macOS使用avfoundation
            return [
                '-f', 'avfoundation',
                '-rtbufsize', '1k',
                '-thread_queue_size', '2',
                '-video_size', f'{video_width}x{video_height}',
                '-framerate', str(framerate),
                '-i', self.config["video_device_mac"]  # 通常0是第一个摄像头
            ]
    
    def build_ffmpeg_command(self, input_params, sdp_file_path):
        """构建完整的FFmpeg命令"""
        # 基本ffmpeg命令参数
        ffmpeg_cmd = [
            'ffmpeg',
            '-use_wallclock_as_timestamps', '1',
            '-avioflags', 'direct',
        ]
        
        # 添加输入参数
        ffmpeg_cmd.extend(input_params)
        
        # 添加视频处理和编码参数
        bitrate = self.config["bitrate"]
        framerate = self.config["framerate"]
        rtp_target = self.config["rtp_target"]
        
        ffmpeg_cmd.extend([
            '-vf', f'setpts=N/({framerate}*TB)',
            '-c:v', 'libx265',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-x265-params', 'bframes=0:no-scenecut=1',
            '-b:v', f'{bitrate}',
            '-pix_fmt', 'yuv420p',
            '-g', str(framerate),
            '-keyint_min', str(framerate),
            '-sc_threshold', '0',
            '-r', str(framerate),
            
            # FFmpeg日志级别
            '-loglevel', 'debug' if self.debug else 'error',
            
            # 输出流（RTP over UDP）
            '-f', 'rtp',
            '-sdp_file', sdp_file_path,
            rtp_target,  # RTP目标地址
        ])
        
        return ffmpeg_cmd
    
    def start_ffmpeg_process(self, ffmpeg_cmd):
        """启动FFmpeg进程"""
        system = platform.system()
        log(f"在{system}系统上启动FFmpeg进程", "DEBUG")
        
        if system == 'Windows':
            return subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:  # macOS 和 Linux
            return subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
    
    def monitor_ffmpeg_output(self):
        """监控FFmpeg输出"""
        log("开始监控FFmpeg输出", "DEBUG")
        line_count = 0
        last_log_time = time.time()
        
        while self.running:
            output = self.ffmpeg_process.stderr.readline()
            if output == b'' and self.ffmpeg_process.poll() is not None:
                log("FFmpeg进程已退出", "INFO")
                break
            if output:
                line_count += 1
                if self.debug:
                    # 限制日志输出频率，避免过多输出
                    current_time = time.time()
                    if current_time - last_log_time >= 5.0:
                        log(f"FFmpeg最近输出({line_count}行): {output.strip().decode()}", "DEBUG")
                        line_count = 0
                        last_log_time = current_time

    def BidirectionalStream(self, request_iterator, context):
        """处理双向流 - 仅维持连接"""
        peer = context.peer()
        log(f"接收到来自 {peer} 的双向流连接", "INFO")
        try:
            request_count = 0
            for request in request_iterator:
                request_count += 1
                if self.debug and request_count % 10 == 0:  # 每10个请求记录一次
                    log(f"已接收 {request_count} 个请求", "DEBUG")
                time.sleep(0.1)
        except Exception as e:
            log(f"流连接错误: {e}", "ERROR")
        finally:
            log(f"客户端 {peer} 断开连接", "INFO")

    def run(self):
        """启动服务器"""
        log("启动gRPC服务器...", "INFO")
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        data_stream_pb2_grpc.add_THStreamServiceServicer_to_server(self, server)
        
        port = self.config["port"]
        server.add_insecure_port(f'[::]:{port}')
        server.start()
        log(f"服务器已启动，监听端口 {port}", "INFO")

        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.running = False
            log("收到中断信号，正在关闭服务器...", "INFO")

            # 停止FFmpeg（如果在运行）
            if self.ffmpeg_process:
                log("终止FFmpeg进程", "DEBUG")
                self.ffmpeg_process.terminate()
            # 等待流线程完成
            if self.streaming_thread and self.streaming_thread.is_alive():
                log("等待流线程完成", "DEBUG")
                self.streaming_thread.join()

        finally:
            server.stop(0)
            log("服务器已完全停止", "INFO")


if __name__ == '__main__':
    # 配置参数
    CONFIG = {
        "port": 50051,                       # 服务器端口
        "rtp_target": "rtp://183.173.139.132:5005", # RTP目标地址
        "sdp_file": "stream.sdp",            # SDP文件名
        "video_resolution": (1280, 720),     # 视频分辨率
        "framerate": 30,                     # 帧率
        "bitrate": "2M",                     # 视频码率
        "debug": DEBUG,                      # 调试模式
        
        # 视频设备标识（根据操作系统）
        "video_device_win": "video=@device_pnp_\\\\?\\usb#vid_5986&pid_2169&mi_00#6&17e2b06b&1&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\\global",
        "video_device_mac": "0"              # macOS通常使用索引
    }
    
    log("程序启动", "INFO")
    
    # 启动服务器
    server = THStreamServiceServicer(CONFIG)
    server.run()