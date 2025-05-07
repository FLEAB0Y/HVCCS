import subprocess
import cv2
import grpc
import data_stream_pb2
import data_stream_pb2_grpc
import time
import numpy as np
import threading
import queue
import torch
import sys
import traceback
import collections
import os
import platform
import datetime

class FFmpegClient:
    """视频流解码客户端，负责接收视频流并转发到UI"""
    
    def __init__(self, frame_callback=None, stats_callback=None, server_address="183.173.117.138:50051", 
                 sdp_file="stream.sdp", video_resolution=(1280, 720), history_length=300, debug=True):
        """
        初始化FFmpeg客户端
        
        参数:
            frame_callback (function): 帧更新回调函数
            stats_callback (function): 统计数据更新回调函数
            server_address (str): gRPC服务器地址和端口
            sdp_file (str): SDP文件名
            video_resolution (tuple): 视频分辨率 (width, height)
            history_length (int): 历史数据保存点数
            debug (bool): 是否开启调试模式
        """
        # 调试开关
        self.debug = debug
        self.log("客户端初始化中...", "INFO")
        
        # 基本设置
        self.server_address = server_address
        self.sdp_file = sdp_file
        self.video_width, self.video_height = video_resolution
        self.history_length = history_length
        
        # gRPC连接设置
        self.channel = grpc.insecure_channel(self.server_address)
        self.stub = data_stream_pb2_grpc.THStreamServiceStub(self.channel)
        self.seq_no = 0
        
        # 状态控制
        self.running = True
        self.is_streaming = True
        
        # 解码后帧队列
        self.decoded_frame_queue = queue.Queue(maxsize=5)
        
        # 回调函数，用于向UI传递数据
        self.frame_callback = frame_callback  # 帧更新回调
        self.stats_callback = stats_callback  # 统计数据更新回调
        
        # 性能数据收集
        self.fps_history = collections.deque(maxlen=self.history_length)
        self.bitrate_history = collections.deque(maxlen=self.history_length)
        self.latency_history = collections.deque(maxlen=self.history_length)
        self.time_axis = np.linspace(-30, 0, self.history_length)  # 时间轴从-30秒到0秒
        
        # 初始化时间和数据大小记录
        self.last_frame_time = time.time()
        self.last_data_size = 0
        self.frame_timestamps = {}  # 存储帧时间戳
        
        # 线程引用初始化
        self.ffmpeg_process = None
        self.stream_receiver_thread = None
        self.display_thread = None
        self.stats_thread = None
        self.grpc_thread = None
        
        self.log("客户端初始化完成")

    def log(self, message, level="INFO"):
        """
        日志输出函数，根据调试开关决定是否输出
        
        参数:
            message (str): 日志信息
            level (str): 日志级别 (DEBUG, INFO, ERROR 等)
        """
        if self.debug or level == "ERROR":
            time_str = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
            print(f"[CLIENT][{level}][{time_str}] {message}")

    def next_seq_no(self):
        """生成下一个序列号"""
        self.seq_no += 1
        return str(self.seq_no)

    def cleanup(self):
        """清理资源"""
        self.log("开始清理资源...", "INFO")
        self.running = False
        if self.ffmpeg_process:
            self.log("终止FFmpeg进程", "DEBUG")
            self.ffmpeg_process.terminate()
            self.ffmpeg_process.wait()
        if hasattr(self, 'cfx') and self.cfx:
            try:
                self.log("清理CUDA上下文", "DEBUG")
                self.cfx.pop()
                self.cfx = None
            except Exception as e:
                self.log(f"清理CUDA上下文错误: {e}", "ERROR")
        self.log("清理CUDA缓存", "DEBUG")
        torch.cuda.empty_cache()
        self.log("关闭所有OpenCV窗口", "DEBUG")
        cv2.destroyAllWindows()
        self.log("资源已清理完成", "INFO")

    def start_ffmpeg_receiver(self):
        """启动FFmpeg接收和解码RTP流"""
        self.log("启动FFmpeg接收器...", "INFO")
        sdp_file_path = os.path.join(os.path.dirname(__file__), self.sdp_file)
        
        # 检查SDP文件是否存在
        if not os.path.exists(sdp_file_path):
            self.log(f"错误: SDP文件不存在: {sdp_file_path}", "ERROR")
            return
        else:
            self.log(f"找到SDP文件: {sdp_file_path}", "DEBUG")
        
        # 构建FFmpeg命令
        ffmpeg_cmd = self.build_ffmpeg_command(sdp_file_path)
        self.log(f"FFmpeg命令: {' '.join(ffmpeg_cmd)}", "DEBUG")
        
        try:
            # 启动FFmpeg进程
            self.log("启动FFmpeg进程...", "DEBUG")
            self.ffmpeg_process = self.start_ffmpeg_process(ffmpeg_cmd)

            # 启动错误输出监控线程
            self.log("启动错误监控线程", "DEBUG")
            self.start_error_monitor()

            # 帧大小
            frame_size = self.video_width * self.video_height * 3
            self.log(f"帧大小: {frame_size} 字节", "DEBUG")

            # 开始读取帧数据
            self.log("开始处理视频帧...", "INFO")
            self.process_video_frames(frame_size)

        except Exception as e:
            self.log(f"FFmpeg接收器错误: {e}", "ERROR")
            traceback.print_exc()
        finally:
            if self.ffmpeg_process:
                self.log("终止FFmpeg进程", "DEBUG")
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait()
            self.log("FFmpeg接收器已停止", "INFO")
    
    def build_ffmpeg_command(self, sdp_file_path):
        """构建FFmpeg命令"""
        return [
            'ffmpeg',
            '-fflags', 'nobuffer',
            '-flags', 'low_delay',
            '-avioflags', 'direct',
            '-loglevel', 'debug' if self.debug else 'error',  # 根据调试标志调整日志级别
            '-c:v', 'hevc',  # 确保使用HEVC解码器
            '-protocol_whitelist', 'file,udp,rtp,sdp',
            '-i', sdp_file_path,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-flush_packets', '1',
            '-s', f'{self.video_width}x{self.video_height}',
            '-threads', '1',
            '-'
        ]
    
    def start_ffmpeg_process(self, ffmpeg_cmd):
        """启动FFmpeg进程"""
        if platform.system() == 'Windows':
            self.log("在Windows系统上启动FFmpeg", "DEBUG")
            return subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:  # macOS 和 Linux
            self.log(f"在{platform.system()}系统上启动FFmpeg", "DEBUG")
            return subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
    
    def start_error_monitor(self):
        """启动FFmpeg错误输出监控线程"""
        def monitor_errors():
            while self.running:
                error_line = self.ffmpeg_process.stderr.readline()
                if error_line:
                    if self.debug:  # 仅在调试模式下输出FFmpeg日志
                        self.log(f"FFmpeg: {error_line.decode().strip()}", "DEBUG")
                else:
                    break
        
        error_thread = threading.Thread(target=monitor_errors)
        error_thread.daemon = True
        error_thread.start()
        self.log("错误监控线程已启动", "DEBUG")
    
    def process_video_frames(self, frame_size):
        """处理视频帧"""
        frame_count = 0
        last_log_time = time.time()
        
        self.log("开始处理视频帧流", "INFO")
        while self.running:
            # 记录接收开始时间（用于计算时延）
            recv_start_time = time.time()
            
            # 读取原始帧数据
            raw_frame = self.ffmpeg_process.stdout.read(frame_size)
            if not raw_frame or len(raw_frame) != frame_size:
                time.sleep(0.01)
                continue

            # 记录接收结束时间
            recv_end_time = time.time()
            
            # 计算性能指标
            latency = (recv_end_time - recv_start_time) * 1000  # 毫秒
            
            # 计算码率 (Mbps)
            current_time = time.time()
            time_diff = current_time - self.last_frame_time
            if time_diff > 0:
                bitrate = (len(raw_frame) * 8) / (time_diff * 1000000)  # Mbps
                self.bitrate_history.append(bitrate)
                self.latency_history.append(latency)
                self.last_frame_time = current_time
                self.last_data_size = len(raw_frame)

            # 转换为numpy数组
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.video_height, self.video_width, 3))
            
            # 将帧和时间戳放入队列
            if self.decoded_frame_queue.full():
                self.decoded_frame_queue.get()
            self.decoded_frame_queue.put((frame, recv_end_time, bitrate, latency))
            
            # 每秒打印一次处理的帧数
            frame_count += 1
            if current_time - last_log_time >= 1.0 and self.debug:
                self.log(f"已处理 {frame_count} 帧，当前码率: {bitrate:.2f} Mbps，延迟: {latency:.2f} ms", "DEBUG")
                frame_count = 0
                last_log_time = current_time

    def display_frames(self):
        """从解码帧队列中取出帧并处理，计算FPS并通过回调发送到UI"""
        frame_count = 0
        start_time = time.time()
        fps = 0
        last_log_time = time.time()
        
        self.log("开始显示帧", "INFO")
        while self.running:
            if not self.decoded_frame_queue.empty() and self.is_streaming:
                # 获取帧、时间戳和性能数据
                frame, timestamp, bitrate, latency = self.decoded_frame_queue.get()
                
                # 创建可写的帧副本
                frame = frame.copy()

                # 计算 FPS
                frame_count += 1
                current_time = time.time()
                elapsed_time = current_time - start_time
                if elapsed_time >= 0.1:  # 每0.1秒更新一次FPS
                    fps = frame_count / elapsed_time
                    self.fps_history.append(fps)
                    frame_count = 0
                    start_time = current_time

                # 绘制性能信息到帧上
                self.draw_performance_info(frame, fps, bitrate, latency)

                # 通过回调将帧发送到UI
                if self.frame_callback:
                    self.frame_callback(frame, fps, bitrate, latency)
                    
                # 每秒打印一次FPS
                if current_time - last_log_time >= 1.0 and self.debug:
                    self.log(f"FPS: {fps:.2f}, 队列大小: {self.decoded_frame_queue.qsize()}", "DEBUG")
                    last_log_time = current_time
            else:
                time.sleep(0.01)
    
    def draw_performance_info(self, frame, fps, bitrate, latency):
        """在帧上绘制性能信息"""
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        bitrate_text = f"Bitrate: {bitrate:.2f} Mbps"
        cv2.putText(frame, bitrate_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        latency_text = f"Latency: {latency:.2f} ms"
        cv2.putText(frame, latency_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    def update_stats(self):
        """定期更新统计数据并通过回调发送到UI"""
        self.log("开始更新统计数据", "INFO")
        last_log_time = time.time()
        
        while self.running:
            if len(self.fps_history) > 0:
                # 计算平均值
                avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
                avg_bitrate = sum(self.bitrate_history) / len(self.bitrate_history) if self.bitrate_history else 0
                avg_latency = sum(self.latency_history) / len(self.latency_history) if self.latency_history else 0
                
                # 准备历史数据
                stats_data = self.prepare_stats_data()
                
                # 通过回调将统计数据发送到UI
                if self.stats_callback:
                    self.stats_callback(
                        avg_fps, avg_bitrate, avg_latency, 
                        stats_data["fps"], stats_data["bitrate"], stats_data["latency"], 
                        self.time_axis
                    )
                
                # 每5秒打印一次平均统计数据
                current_time = time.time()
                if current_time - last_log_time >= 5.0 and self.debug:
                    self.log(f"统计数据: 平均FPS={avg_fps:.2f}, 平均码率={avg_bitrate:.2f}Mbps, 平均延迟={avg_latency:.2f}ms", "DEBUG")
                    self.log(f"历史数据点: FPS={len(self.fps_history)}, 码率={len(self.bitrate_history)}, 延迟={len(self.latency_history)}", "DEBUG")
                    last_log_time = current_time
            
            time.sleep(0.1)  # 每100ms更新一次
    
    def prepare_stats_data(self):
        """准备统计数据，确保长度一致"""
        # 填充历史数据，确保长度一致
        fps_data = list(self.fps_history)
        bitrate_data = list(self.bitrate_history)
        latency_data = list(self.latency_history)
        
        # 如果数据不足，补充零
        while len(fps_data) < self.history_length:
            fps_data.insert(0, 0)
        while len(bitrate_data) < self.history_length:
            bitrate_data.insert(0, 0)
        while len(latency_data) < self.history_length:
            latency_data.insert(0, 0)
        
        return {
            "fps": fps_data,
            "bitrate": bitrate_data,
            "latency": latency_data
        }

    def maintain_connection(self):
        """维持gRPC连接而不发送任何数据"""
        self.log("维持与服务器的连接", "DEBUG")
        try:
            while self.running:
                time.sleep(1)
            self.log("连接维持循环结束", "DEBUG")
            yield from ()
        except Exception as e:
            self.log(f"连接错误: {e}", "ERROR")

    def establish_grpc_connection(self):
        """在单独的线程中处理gRPC连接"""
        self.log(f"建立与 {self.server_address} 的gRPC连接", "INFO")
        try:
            response_iterator = self.stub.BidirectionalStream(self.maintain_connection())
            for response in response_iterator:
                self.log(f"服务器响应: {response.retMsg}", "DEBUG")
                if not self.running:
                    break
        except Exception as e:
            self.log(f"gRPC连接错误: {e}", "ERROR")
            traceback.print_exc()
        self.log("gRPC连接终止", "INFO")

    def set_streaming_state(self, state):
        """设置流状态"""
        self.is_streaming = state
        status = "启动" if state else "暂停"
        self.log(f"流状态已切换: {status}", "INFO")

    def start(self):
        """启动所有线程"""
        self.log("启动所有客户端线程", "INFO")
        
        # 启动FFmpeg接收线程
        self.log("启动FFmpeg接收线程", "DEBUG")
        self.stream_receiver_thread = threading.Thread(target=self.start_ffmpeg_receiver)
        self.stream_receiver_thread.daemon = True
        self.stream_receiver_thread.start()

        # 启动帧处理线程
        self.log("启动帧处理线程", "DEBUG")
        self.display_thread = threading.Thread(target=self.display_frames)
        self.display_thread.daemon = True
        self.display_thread.start()

        # 启动统计更新线程
        self.log("启动统计更新线程", "DEBUG")
        self.stats_thread = threading.Thread(target=self.update_stats)
        self.stats_thread.daemon = True
        self.stats_thread.start()

        # 建立gRPC连接
        self.log("启动gRPC连接线程", "DEBUG")
        self.grpc_thread = threading.Thread(target=self.establish_grpc_connection)
        self.grpc_thread.daemon = True
        self.grpc_thread.start()
        
        self.log("所有客户端线程已启动", "INFO")

    def stop(self):
        """停止客户端"""
        self.log("停止客户端", "INFO")
        self.running = False
        self.cleanup()