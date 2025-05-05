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
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QWidget, QStatusBar, QGroupBox, QGridLayout)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import pyqtgraph as pg

class VideoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("1080p视频流显示与分析")
        self.setGeometry(100, 100, 1280, 800)  # 加高窗口以容纳图表
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 上半部分：视频和基本信息
        top_layout = QHBoxLayout()
        main_layout.addLayout(top_layout, 3)  # 占3份高度
        
        # 视频部分
        video_group = QGroupBox("视频流")
        video_layout = QVBoxLayout(video_group)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(960, 540)
        video_layout.addWidget(self.video_label)
        
        # 控制按钮
        self.control_btn = QPushButton("停止")
        self.control_btn.clicked.connect(self.toggle_streaming)
        video_layout.addWidget(self.control_btn)
        
        top_layout.addWidget(video_group, 7)  # 占7份宽度
        
        # 实时数据显示部分
        stats_group = QGroupBox("实时数据")
        stats_layout = QGridLayout(stats_group)
        
        # FPS显示
        stats_layout.addWidget(QLabel("当前FPS:"), 0, 0)
        self.fps_label = QLabel("0.00")
        stats_layout.addWidget(self.fps_label, 0, 1)
        
        # 码率显示
        stats_layout.addWidget(QLabel("当前码率:"), 1, 0)
        self.bitrate_label = QLabel("0.00 Mbps")
        stats_layout.addWidget(self.bitrate_label, 1, 1)
        
        # 时延显示
        stats_layout.addWidget(QLabel("当前时延:"), 2, 0)
        self.latency_label = QLabel("0.00 ms")
        stats_layout.addWidget(self.latency_label, 2, 1)
        
        # 平均值显示
        stats_layout.addWidget(QLabel("平均FPS:"), 3, 0)
        self.avg_fps_label = QLabel("0.00")
        stats_layout.addWidget(self.avg_fps_label, 3, 1)
        
        stats_layout.addWidget(QLabel("平均码率:"), 4, 0)
        self.avg_bitrate_label = QLabel("0.00 Mbps")
        stats_layout.addWidget(self.avg_bitrate_label, 4, 1)
        
        stats_layout.addWidget(QLabel("平均时延:"), 5, 0)
        self.avg_latency_label = QLabel("0.00 ms")
        stats_layout.addWidget(self.avg_latency_label, 5, 1)
        
        top_layout.addWidget(stats_group, 3)  # 占3份宽度
        
        # 下半部分：图表显示
        chart_group = QGroupBox("30秒历史数据")
        chart_layout = QVBoxLayout(chart_group)
        main_layout.addWidget(chart_group, 2)  # 占2份高度
        
        # 创建绘图部件
        self.graph_widget = pg.GraphicsLayoutWidget()
        chart_layout.addWidget(self.graph_widget)
        
        # 创建三个子图
        self.fps_plot = self.graph_widget.addPlot(row=0, col=0, title="FPS历史")
        self.fps_plot.setLabel('left', 'FPS')
        self.fps_plot.setLabel('bottom', '时间 (秒)')
        self.fps_curve = self.fps_plot.plot(pen='g')
        
        self.bitrate_plot = self.graph_widget.addPlot(row=0, col=1, title="码率历史 (Mbps)")
        self.bitrate_plot.setLabel('left', '码率 (Mbps)')
        self.bitrate_plot.setLabel('bottom', '时间 (秒)')
        self.bitrate_curve = self.bitrate_plot.plot(pen='r')
        
        self.latency_plot = self.graph_widget.addPlot(row=0, col=2, title="时延历史 (ms)")
        self.latency_plot.setLabel('left', '时延 (ms)')
        self.latency_plot.setLabel('bottom', '时间 (秒)')
        self.latency_curve = self.latency_plot.plot(pen='b')
        
        # 创建状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.status_label = QLabel("就绪")
        self.statusBar.addWidget(self.status_label)
        
        self.is_streaming = True
        
    def update_frame(self, frame):
        """更新UI界面中的视频帧"""
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_img)
        pixmap = pixmap.scaled(960, 540, Qt.KeepAspectRatio)
        self.video_label.setPixmap(pixmap)
        
    def update_stats(self, fps, bitrate, latency):
        """更新性能指标"""
        self.fps_label.setText(f"{fps:.2f}")
        self.bitrate_label.setText(f"{bitrate:.2f} Mbps")
        self.latency_label.setText(f"{latency:.2f} ms")
        
    def update_averages(self, avg_fps, avg_bitrate, avg_latency):
        """更新平均值"""
        self.avg_fps_label.setText(f"{avg_fps:.2f}")
        self.avg_bitrate_label.setText(f"{avg_bitrate:.2f} Mbps")
        self.avg_latency_label.setText(f"{avg_latency:.2f} ms")
        
    def update_plots(self, fps_data, bitrate_data, latency_data, time_axis):
        """更新图表"""
        self.fps_curve.setData(time_axis, fps_data)
        self.bitrate_curve.setData(time_axis, bitrate_data)
        self.latency_curve.setData(time_axis, latency_data)
        
    def toggle_streaming(self):
        """切换流状态"""
        self.is_streaming = not self.is_streaming
        if self.is_streaming:
            self.control_btn.setText("停止")
        else:
            self.control_btn.setText("启动")

class THStreamClient:
    def __init__(self, host='183.172.152.218', port=50051):
        self.channel = grpc.insecure_channel(f'{host}:{port}')
        self.stub = data_stream_pb2_grpc.THStreamServiceStub(self.channel)
        self.seq_no = 0
        self.running = True
        self.decoded_frame_queue = queue.Queue(maxsize=1)
        
        # 性能数据收集
        self.history_length = 30 * 10  # 30秒，每秒10个数据点
        self.fps_history = collections.deque(maxlen=self.history_length)
        self.bitrate_history = collections.deque(maxlen=self.history_length)
        self.latency_history = collections.deque(maxlen=self.history_length)
        self.time_axis = np.linspace(-30, 0, self.history_length)  # 时间轴从-30秒到0秒
        
        # 初始化时间和数据大小记录
        self.last_frame_time = time.time()
        self.last_data_size = 0
        self.frame_timestamps = {}  # 存储帧时间戳
        
        # 创建PyQt应用
        self.app = QApplication(sys.argv)
        self.window = VideoWindow()
        self.window.show()
        
        # 启动显示线程
        self.display_thread = threading.Thread(target=self.display_frames)
        self.display_thread.daemon = True
        self.display_thread.start()
        
        # 启动统计更新定时器
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(100)  # 每100ms更新一次

        print("Client initialized and UI ready")
        self.ffmpeg_process = None
        self.stream_receiver_thread = None

    def next_seq_no(self):
        self.seq_no += 1
        return str(self.seq_no)

    def cleanup(self):
        """Clean up resources on exit"""
        self.running = False
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
            self.ffmpeg_process.wait()
        if hasattr(self, 'cfx') and self.cfx:
            try:
                self.cfx.pop()
                self.cfx = None
            except:
                pass
        torch.cuda.empty_cache()
        cv2.destroyAllWindows()

    def start_ffmpeg_receiver(self):
        """Start FFmpeg to receive and decode the RTP stream"""
        sdp_file_path = os.path.join(os.path.dirname(__file__), "stream.sdp")
        
        # 检查SDP文件是否存在
        if not os.path.exists(sdp_file_path):
            print(f"错误: SDP文件不存在: {sdp_file_path}")
            return
        else:
            print(f"找到SDP文件: {sdp_file_path}")
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-fflags', 'nobuffer',
            '-flags', 'low_delay',
            '-avioflags', 'direct',
            '-loglevel', 'debug',

            '-c:v', 'hevc',

            '-protocol_whitelist', 'file,udp,rtp,sdp',
            '-i', sdp_file_path,

            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-flush_packets', '1',
            '-s', '1280x720',  # 修改为与服务器匹配的分辨率

            '-threads', '1',
            '-'
        ]
        
        try:
            # 根据操作系统选择是否使用 creationflags
            if platform.system() == 'Windows':
                self.ffmpeg_process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:  # macOS 和 Linux 不需要 creationflags
                self.ffmpeg_process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

            # 启动错误输出监控线程
            def monitor_errors():
                while self.running:
                    error_line = self.ffmpeg_process.stderr.readline()
                    if error_line:
                        print(f"FFmpeg: {error_line.decode().strip()}")
                    else:
                        break
            
            error_thread = threading.Thread(target=monitor_errors)
            error_thread.daemon = True
            error_thread.start()

            # 帧大小
            width, height = 1280, 720  # 更新为与服务器匹配的分辨率
            frame_size = width * height * 3

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
                
                # 计算时延 (毫秒)
                latency = (recv_end_time - recv_start_time) * 1000
                
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
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
                
                # 将帧和时间戳放入队列
                if self.decoded_frame_queue.full():
                    self.decoded_frame_queue.get()
                self.decoded_frame_queue.put((frame, recv_end_time, bitrate, latency))

        except Exception as e:
            print(f"Error in FFmpeg receiver: {e}")
            traceback.print_exc()
        finally:
            if self.ffmpeg_process:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait()
            print("FFmpeg receiver stopped")

    def display_frames(self):
        """从解码帧队列中取出帧并显示，并计算 FPS"""
        frame_count = 0
        start_time = time.time()
        fps = 0

        while self.running:
            if not self.decoded_frame_queue.empty() and self.window.is_streaming:
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

                # 在帧上绘制FPS
                fps_text = f"FPS: {fps:.2f}"
                cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 在帧上绘制码率
                bitrate_text = f"Bitrate: {bitrate:.2f} Mbps"
                cv2.putText(frame, bitrate_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # 在帧上绘制时延
                latency_text = f"Latency: {latency:.2f} ms"
                cv2.putText(frame, latency_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # 更新UI显示
                self.window.update_frame(frame)
                self.window.update_stats(fps, bitrate, latency)
            else:
                time.sleep(0.01)

    def update_stats(self):
        """更新统计数据"""
        if len(self.fps_history) > 0:
            # 计算平均值
            avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
            avg_bitrate = sum(self.bitrate_history) / len(self.bitrate_history) if self.bitrate_history else 0
            avg_latency = sum(self.latency_history) / len(self.latency_history) if self.latency_history else 0
            
            # 更新UI
            self.window.update_averages(avg_fps, avg_bitrate, avg_latency)
            
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
            
            # 更新图表
            self.window.update_plots(fps_data, bitrate_data, latency_data, self.time_axis)

    def maintain_connection(self):
        """Maintain gRPC connection without sending any data"""
        try:
            while self.running:
                time.sleep(1)
            yield from ()
        except Exception as e:
            print(f"Connection error: {e}")

    def run(self):
        """Run the client"""
        try:
            # 启动FFmpeg接收线程
            self.stream_receiver_thread = threading.Thread(target=self.start_ffmpeg_receiver)
            self.stream_receiver_thread.daemon = True
            self.stream_receiver_thread.start()

            # 建立gRPC连接
            grpc_thread = threading.Thread(target=self.establish_grpc_connection)
            grpc_thread.daemon = True
            grpc_thread.start()

            # 运行PyQt事件循环
            sys.exit(self.app.exec_())

        except KeyboardInterrupt:
            print("Client stopping...")
        except Exception as e:
            print(f"Client error: {e}")
        finally:
            self.cleanup()
            
    def establish_grpc_connection(self):
        """在单独的线程中处理gRPC连接"""
        try:
            response_iterator = self.stub.BidirectionalStream(self.maintain_connection())
            for response in response_iterator:
                print(f"Server response: {response.retMsg}")
                if not self.running:
                    break
        except Exception as e:
            print(f"gRPC connection error: {e}")

if __name__ == '__main__':
    client = THStreamClient()
    client.run()